import argparse
import json
from pathlib import Path

import pandas as pd

from backtest import load_or_generate_predictions, run_backtest
from config import BEST_CONFIG
from evaluate_rank_stability import append_stability_summary, summarize_prediction_rank_stability


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "model_comparison"
DEFAULT_MODEL_SPECS = [
    {
        "model_family": "lstm",
        "model_dir": ROOT_DIR / "app" / "model",
        "label": "LSTM sl20",
    },
    {
        "model_family": "lightgbm",
        "model_dir": ROOT_DIR / "app" / "model" / "baseline_lightgbm_same_protocol",
        "label": "LightGBM",
    },
    {
        "model_family": "xgboost",
        "model_dir": ROOT_DIR / "app" / "model" / "xgboost_baseline",
        "label": "XGBoost",
    },
    {
        "model_family": "transformer",
        "model_dir": ROOT_DIR / "app" / "model" / "transformer_baseline",
        "label": "Transformer",
    },
    {
        "model_family": "linear_regression",
        "model_dir": ROOT_DIR / "app" / "model" / "baseline_linear_same_protocol",
        "label": "Linear Regression",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified model comparison under the same evaluation protocol.")
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "model_meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_default_backtest_config(profile_name: str) -> dict:
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


def collect_single_model_summary(model_family: str, label: str, model_dir: Path, feature_path: Path) -> dict:
    metadata = load_metadata(model_dir)
    prediction_path = model_dir / "walk_forward_predictions.csv"
    prediction_df, prediction_source = load_or_generate_predictions(
        prediction_path=prediction_path,
        feature_path=feature_path,
        model_dir=model_dir,
    )
    profile_name = f"{model_family}__same_protocol"
    config = build_default_backtest_config(profile_name=profile_name)
    backtest_summary_df, _, _ = run_backtest(
        prediction_df=prediction_df,
        config=config,
        prediction_source=prediction_source,
    )
    backtest_summary = backtest_summary_df.iloc[0].to_dict()
    walk_forward_summary = metadata["walk_forward_summary"]
    stability_summary = summarize_prediction_rank_stability(
        prediction_df=prediction_df,
        experiment_name=f"model_comparison/{model_family}",
        extra_fields={
            "model_family": model_family,
            "model_label": label,
            "model_dir": str(model_dir),
        },
    )
    append_stability_summary({**stability_summary, **backtest_summary})

    return {
        "model_family": model_family,
        "model_label": label,
        "model_dir": str(model_dir),
        "backend": metadata.get("backend", ""),
        "feature_set": metadata.get("feature_set", ""),
        "feature_count": int(len(metadata.get("feature_columns", []))),
        "target_mode": metadata.get("target_mode", ""),
        "sequence_length": metadata.get("sequence_length", ""),
        "rank_ic_mean": float(stability_summary["rank_ic_mean"]),
        "rank_ic_std": float(stability_summary["rank_ic_std"]),
        "worst_fold_rank_ic": float(stability_summary["worst_fold_rank_ic"]),
        "negative_day_rank_ic_ratio": float(stability_summary["negative_day_rank_ic_ratio"]),
        "top5_mean_return_mean": float(walk_forward_summary["top5_mean_return_mean"]),
        "rmse_mean": float(walk_forward_summary["rmse_mean"]),
        "mae_mean": float(walk_forward_summary["mae_mean"]),
        "cumulative_return_after_cost": float(backtest_summary["cumulative_return_after_cost"]),
        "sharpe_after_cost": float(backtest_summary["sharpe_after_cost"]),
        "max_drawdown_after_cost": float(backtest_summary["max_drawdown_after_cost"]),
        "avg_turnover": float(backtest_summary["avg_turnover"]),
        "win_rate_after_cost": float(backtest_summary["win_rate_after_cost"]),
        "avg_cash_weight": float(backtest_summary["avg_cash_weight"]),
        "prediction_source": prediction_source,
        "sort_strategy": backtest_summary["sort_strategy"],
        "weighting_scheme": backtest_summary["weighting_scheme"],
        "max_turnover": float(backtest_summary["max_turnover"]),
    }


def write_report(summary_df: pd.DataFrame, output_path: Path) -> None:
    ranked = summary_df.sort_values(
        ["worst_fold_rank_ic", "negative_day_rank_ic_ratio", "cumulative_return_after_cost", "sharpe_after_cost"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    ranked["is_best"] = "否"
    if not ranked.empty:
        ranked.loc[0, "is_best"] = "是"

    lines = [
        "# 统一模型对比报告",
        "",
        "## 对比口径",
        "",
        f"- 特征集：`{BEST_CONFIG['training']['feature_set']}`",
        f"- 训练目标：`{BEST_CONFIG['training']['target_mode']}`",
        (
            "- 选股逻辑："
            f"`{BEST_CONFIG['selection']['sort_strategy']} sort + "
            f"{BEST_CONFIG['selection']['weighting_scheme']} weight + "
            f"max_turnover={BEST_CONFIG['execution']['max_turnover']:.2f}`"
        ),
        "- 回测口径：统一使用当前正式默认风险过滤与执行约束",
        "",
        "## 总结论",
        "",
    ]

    if ranked.empty:
        lines.extend(["当前没有可用模型结果。", ""])
    else:
        best = ranked.iloc[0]
        lines.extend(
            [
                f"- 当前同口径下综合表现最优模型：`{best['model_label']}`",
                f"- 成本后累计收益：`{best['cumulative_return_after_cost']:.6f}`",
                f"- 成本后夏普：`{best['sharpe_after_cost']:.6f}`",
                f"- 成本后最大回撤：`{best['max_drawdown_after_cost']:.6f}`",
                f"- 平均换手率：`{best['avg_turnover']:.6f}`",
                "",
                "## 模型排序",
                "",
                "| 排名 | 模型 | worst_fold_rank_ic | negative_day_rank_ic_ratio | rank_ic_mean | cumulative_return_after_cost | sharpe_after_cost | max_drawdown_after_cost | avg_turnover | 是否最优 |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for idx, (_, row) in enumerate(ranked.iterrows(), start=1):
            lines.append(
                f"| {idx} | {row['model_label']} | "
                f"{row['worst_fold_rank_ic']:.6f} | {row['negative_day_rank_ic_ratio']:.6f} | "
                f"{row['rank_ic_mean']:.6f} | "
                f"{row['cumulative_return_after_cost']:.6f} | {row['sharpe_after_cost']:.6f} | "
                f"{row['max_drawdown_after_cost']:.6f} | {row['avg_turnover']:.6f} | {row['is_best']} |"
            )

        lines.extend(
            [
                "",
                "## 结果解读",
                "",
                "- `rank_ic_mean` 和 `top5_mean_return_mean` 反映 walk-forward 预测排序能力。",
                "- `cumulative_return_after_cost`、`sharpe_after_cost`、`max_drawdown_after_cost`、`avg_turnover` 反映同一执行逻辑下的真实组合表现。",
                "- 如果某个模型回归误差不差，但成本后收益明显弱，通常说明它在 Top-K 排序稳定性上不如更优模型。",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    feature_path = Path(args.feature_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in DEFAULT_MODEL_SPECS:
        rows.append(
            collect_single_model_summary(
                model_family=spec["model_family"],
                label=spec["label"],
                model_dir=spec["model_dir"],
                feature_path=feature_path,
            )
        )

    summary_df = pd.DataFrame(rows).sort_values(
        ["worst_fold_rank_ic", "negative_day_rank_ic_ratio", "cumulative_return_after_cost", "sharpe_after_cost"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)

    summary_path = output_dir / "model_comparison_summary.csv"
    report_path = output_dir / "model_comparison_report.md"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary_df, report_path)

    best = summary_df.iloc[0]
    print(f"[model_comparison] models={len(summary_df)}")
    print(
        "[model_comparison] best_model="
        f"{best['model_family']} "
        f"cum_after={best['cumulative_return_after_cost']:.6f} "
        f"sharpe_after={best['sharpe_after_cost']:.6f}"
    )
    print(f"[model_comparison] wrote {summary_path}")
    print(f"[model_comparison] wrote {report_path}")


if __name__ == "__main__":
    main()
