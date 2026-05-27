import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import MERGE_FEATURE_COLUMNS, load_prediction_frame, run_backtest
from config import BEST_CONFIG


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "formal_model_comparison"
CURRENT_WALK_FORWARD_PATH = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"

MODEL_SPECS = [
    {
        "label": "LSTM sl20",
        "model_family": "lstm",
        "model_dir": ROOT_DIR / "app" / "model",
        "formal_candidate": "是",
        "notes": "当前正式默认主线",
    },
    {
        "label": "LightGBM",
        "model_family": "lightgbm",
        "model_dir": ROOT_DIR / "app" / "model" / "baseline_lightgbm_same_protocol",
        "formal_candidate": "否",
        "notes": "同特征集机器学习基线",
    },
    {
        "label": "Linear Regression",
        "model_family": "linear_regression",
        "model_dir": ROOT_DIR / "app" / "model" / "baseline_linear_same_protocol",
        "formal_candidate": "否",
        "notes": "同特征集线性基线",
    },
    {
        "label": "XGBoost",
        "model_family": "xgboost",
        "model_dir": ROOT_DIR / "app" / "model" / "xgboost_baseline",
        "formal_candidate": "否",
        "notes": "同特征集树模型对照",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the formal model comparison table for the report.")
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--momentum_signal", default="mom_5d")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_backtest_config(profile_name: str) -> dict:
    return {
        "profile_name": profile_name,
        "top_k": int(BEST_CONFIG["selection"]["top_k"]),
        "primary_candidate_size": int(BEST_CONFIG["selection"]["primary_candidate_size"]),
        "enable_risk_filters": int(bool(BEST_CONFIG["selection"]["enable_risk_filters"])),
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": float(BEST_CONFIG["risk_filter_thresholds"]["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(BEST_CONFIG["risk_filter_thresholds"]["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(BEST_CONFIG["risk_filter_thresholds"]["risk_penalty_weight"]),
        "weighting_scheme": BEST_CONFIG["selection"]["weighting_scheme"],
        "sort_strategy": BEST_CONFIG["selection"]["sort_strategy"],
        "transaction_cost": float(BEST_CONFIG["execution"]["transaction_cost"]),
        "max_turnover": float(BEST_CONFIG["execution"]["max_turnover"]),
    }


def compute_rank_ic(valid_df: pd.DataFrame) -> float:
    per_day = []
    for _, day_df in valid_df.groupby("date"):
        if day_df["pred_return"].nunique() <= 1 or day_df["target_return"].nunique() <= 1:
            continue
        corr = day_df["pred_return"].corr(day_df["target_return"], method="spearman")
        if pd.notna(corr):
            per_day.append(float(corr))
    return float(np.mean(per_day)) if per_day else 0.0


def compute_top5_mean_return(valid_df: pd.DataFrame) -> float:
    returns = []
    for _, day_df in valid_df.groupby("date"):
        top5 = day_df.sort_values("pred_return", ascending=False).head(5)
        if not top5.empty:
            returns.append(float(top5["target_return"].mean()))
    return float(np.mean(returns)) if returns else 0.0


def _dcg(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, values.size + 2))
    return float(np.sum(values * discounts))


def compute_ndcg_at_k(valid_df: pd.DataFrame, k: int) -> float:
    scores = []
    for _, day_df in valid_df.groupby("date"):
        ranked_pred = day_df.sort_values("pred_return", ascending=False).head(k)
        ranked_true = day_df.sort_values("target_return", ascending=False).head(k)
        pred_gain = np.clip(ranked_pred["target_return"].to_numpy(dtype=float), a_min=0.0, a_max=None)
        ideal_gain = np.clip(ranked_true["target_return"].to_numpy(dtype=float), a_min=0.0, a_max=None)
        ideal_dcg = _dcg(ideal_gain)
        if ideal_dcg <= 1e-12:
            continue
        scores.append(_dcg(pred_gain) / ideal_dcg)
    return float(np.mean(scores)) if scores else 0.0


def compute_hit_rate_at_k(valid_df: pd.DataFrame, k: int) -> float:
    scores = []
    for _, day_df in valid_df.groupby("date"):
        pred_top = set(day_df.sort_values("pred_return", ascending=False).head(k)["stock_id"].astype(str))
        true_top = set(day_df.sort_values("target_return", ascending=False).head(k)["stock_id"].astype(str))
        if not true_top:
            continue
        scores.append(len(pred_top & true_top) / float(min(k, len(true_top))))
    return float(np.mean(scores)) if scores else 0.0


def compute_walk_forward_summary(prediction_df: pd.DataFrame) -> dict:
    working = prediction_df.copy()
    working["target_return"] = pd.to_numeric(working["target_return"], errors="coerce")
    working["pred_return"] = pd.to_numeric(working["pred_return"], errors="coerce")
    working = working.dropna(subset=["target_return", "pred_return", "fold_id"])

    fold_rows = []
    for fold_id, fold_df in working.groupby("fold_id", sort=True):
        y_true = fold_df["target_return"].to_numpy()
        y_pred = fold_df["pred_return"].to_numpy()
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
                "rank_ic": compute_rank_ic(fold_df),
                "top5_mean_return": compute_top5_mean_return(fold_df),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    return {
        "rmse_mean": float(fold_df["rmse"].mean()) if not fold_df.empty else 0.0,
        "mae_mean": float(fold_df["mae"].mean()) if not fold_df.empty else 0.0,
        "rank_ic_mean": float(fold_df["rank_ic"].mean()) if not fold_df.empty else 0.0,
        "top5_mean_return_mean": float(fold_df["top5_mean_return"].mean()) if not fold_df.empty else 0.0,
        "ndcg_at_5": compute_ndcg_at_k(working, 5),
        "ndcg_at_10": compute_ndcg_at_k(working, 10),
        "ndcg_at_20": compute_ndcg_at_k(working, 20),
        "hit_rate_at_5": compute_hit_rate_at_k(working, 5),
        "hit_rate_at_10": compute_hit_rate_at_k(working, 10),
        "hit_rate_at_20": compute_hit_rate_at_k(working, 20),
        "fold_metrics": fold_df,
    }


def collect_model_row(spec: dict, feature_path: Path) -> dict:
    model_dir = Path(spec["model_dir"])
    metadata = load_json(model_dir / "model_meta.json")
    prediction_df = load_prediction_frame(model_dir / "walk_forward_predictions.csv", feature_path)
    walk_forward_summary = compute_walk_forward_summary(prediction_df)
    backtest_config = build_backtest_config(profile_name=f"{spec['model_family']}__formal_table")
    backtest_summary_df, _, _ = run_backtest(
        prediction_df=prediction_df,
        config=backtest_config,
        prediction_source="replay_walk_forward_predictions",
    )
    backtest_summary = backtest_summary_df.iloc[0].to_dict()
    feature_columns = metadata.get("feature_columns", [])
    return {
        "模型名": spec["label"],
        "模型族": spec["model_family"],
        "特征集": metadata.get("feature_set", ""),
        "sequence_length": metadata.get("sequence_length", ""),
        "MAE": float(walk_forward_summary["mae_mean"]),
        "RMSE": float(walk_forward_summary["rmse_mean"]),
        "RankIC": float(walk_forward_summary["rank_ic_mean"]),
        "NDCG@5": float(walk_forward_summary["ndcg_at_5"]),
        "NDCG@10": float(walk_forward_summary["ndcg_at_10"]),
        "NDCG@20": float(walk_forward_summary["ndcg_at_20"]),
        "HitRate@5": float(walk_forward_summary["hit_rate_at_5"]),
        "HitRate@10": float(walk_forward_summary["hit_rate_at_10"]),
        "HitRate@20": float(walk_forward_summary["hit_rate_at_20"]),
        "Top5平均收益": float(walk_forward_summary["top5_mean_return_mean"]),
        "回测累计收益": float(backtest_summary["cumulative_return_after_cost"]),
        "Sharpe": float(backtest_summary["sharpe_after_cost"]),
        "最大回撤": float(backtest_summary["max_drawdown_after_cost"]),
        "是否正式候选": spec["formal_candidate"],
        "说明": spec["notes"],
        "特征数": int(len(feature_columns)),
        "目标": metadata.get("target_mode", ""),
    }


def collect_momentum_row(feature_path: Path, signal_column: str) -> dict:
    prediction_df = pd.read_csv(CURRENT_WALK_FORWARD_PATH, dtype={"stock_id": str})
    feature_df = pd.read_csv(feature_path, dtype={"stock_id": str})
    merge_columns = ["stock_id", "date", signal_column, *MERGE_FEATURE_COLUMNS]
    merged = prediction_df.merge(
        feature_df[merge_columns].drop_duplicates(["stock_id", "date"]),
        on=["stock_id", "date"],
        how="left",
    )
    if signal_column not in merged.columns:
        raise ValueError(f"Missing momentum signal column: {signal_column}")
    merged["pred_return"] = pd.to_numeric(merged[signal_column], errors="coerce")
    merged = merged.dropna(subset=["pred_return", "target_return", "fold_id"]).copy()

    walk_forward_summary = compute_walk_forward_summary(merged)
    backtest_config = build_backtest_config(profile_name=f"momentum_{signal_column}__formal_table")
    merged_for_backtest = merged.copy()
    backtest_summary_df, _, _ = run_backtest(
        prediction_df=merged_for_backtest,
        config=backtest_config,
        prediction_source=f"feature_baseline_{signal_column}",
    )
    backtest_summary = backtest_summary_df.iloc[0].to_dict()
    return {
        "模型名": f"Momentum ({signal_column})",
        "模型族": "momentum_rule",
        "特征集": BEST_CONFIG["training"]["feature_set"],
        "sequence_length": "",
        "MAE": float(walk_forward_summary["mae_mean"]),
        "RMSE": float(walk_forward_summary["rmse_mean"]),
        "RankIC": float(walk_forward_summary["rank_ic_mean"]),
        "NDCG@5": float(walk_forward_summary["ndcg_at_5"]),
        "NDCG@10": float(walk_forward_summary["ndcg_at_10"]),
        "NDCG@20": float(walk_forward_summary["ndcg_at_20"]),
        "HitRate@5": float(walk_forward_summary["hit_rate_at_5"]),
        "HitRate@10": float(walk_forward_summary["hit_rate_at_10"]),
        "HitRate@20": float(walk_forward_summary["hit_rate_at_20"]),
        "Top5平均收益": float(walk_forward_summary["top5_mean_return_mean"]),
        "回测累计收益": float(backtest_summary["cumulative_return_after_cost"]),
        "Sharpe": float(backtest_summary["sharpe_after_cost"]),
        "最大回撤": float(backtest_summary["max_drawdown_after_cost"]),
        "是否正式候选": "否",
        "说明": "简单动量规则基线，直接以短期动量特征排序",
        "特征数": 1,
        "目标": "rule_based",
    }


def write_report(summary_df: pd.DataFrame, output_path: Path) -> None:
    ranked = summary_df.sort_values(["回测累计收益", "Sharpe", "RankIC"], ascending=[False, False, False]).reset_index(drop=True)
    lines = [
        "# 正式模型对比表",
        "",
        "## 对比口径",
        "",
        f"- 当前正式默认特征集：`{BEST_CONFIG['training']['feature_set']}`",
        f"- 当前正式默认排序策略：`{BEST_CONFIG['selection']['sort_strategy']}`",
        f"- 当前正式默认权重策略：`{BEST_CONFIG['selection']['weighting_scheme']}`",
        f"- 当前正式默认 `top_k`：`{BEST_CONFIG['selection']['top_k']}`",
        "- `NDCG@K`：按每日预测排序前 K 只股票的真实收益计算 DCG，再用真实收益理想排序归一化后求均值。",
        "- `HitRate@K`：按每日预测 Top-K 与真实收益 Top-K 的股票重合比例求均值。",
        "",
        "## 结果表",
        "",
        "| 模型 | 特征集 | sequence_length | MAE | RMSE | RankIC | NDCG@5 | NDCG@10 | NDCG@20 | HitRate@5 | HitRate@10 | HitRate@20 | Top5平均收益 | 回测累计收益 | Sharpe | 最大回撤 | 是否正式候选 | 说明 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in ranked.iterrows():
        seq = "" if pd.isna(row["sequence_length"]) or row["sequence_length"] == "" else int(row["sequence_length"])
        lines.append(
            f"| {row['模型名']} | {row['特征集']} | {seq} | "
            f"{row['MAE']:.6f} | {row['RMSE']:.6f} | {row['RankIC']:.6f} | "
            f"{row['NDCG@5']:.6f} | {row['NDCG@10']:.6f} | {row['NDCG@20']:.6f} | "
            f"{row['HitRate@5']:.6f} | {row['HitRate@10']:.6f} | {row['HitRate@20']:.6f} | "
            f"{row['Top5平均收益']:.6f} | "
            f"{row['回测累计收益']:.6f} | {row['Sharpe']:.6f} | {row['最大回撤']:.6f} | "
            f"{row['是否正式候选']} | {row['说明']} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    feature_path = Path(args.feature_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [collect_momentum_row(feature_path=feature_path, signal_column=args.momentum_signal)]
    for spec in MODEL_SPECS:
        rows.append(collect_model_row(spec=spec, feature_path=feature_path))

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["回测累计收益", "Sharpe", "RankIC"], ascending=[False, False, False]).reset_index(drop=True)

    csv_path = output_dir / "formal_model_comparison.csv"
    report_path = output_dir / "formal_model_comparison.md"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_report(summary_df, report_path)

    print(f"[formal_model_comparison] rows={len(summary_df)}")
    print(f"[formal_model_comparison] wrote {csv_path}")
    print(f"[formal_model_comparison] wrote {report_path}")


if __name__ == "__main__":
    main()
