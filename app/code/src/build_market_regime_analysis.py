import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import load_prediction_frame, run_backtest
from config import BEST_CONFIG


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_PREDICTION_PATH = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "market_regime_analysis"
DEFAULT_MODEL_DIR = ROOT_DIR / "app" / "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fold-level and market-regime-level analysis for walk-forward results.")
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--prediction_path", default=str(DEFAULT_PREDICTION_PATH))
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


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


def compute_rank_ic(df: pd.DataFrame) -> float:
    scores = []
    for _, day_df in df.groupby("date"):
        if day_df["pred_return"].nunique() <= 1 or day_df["target_return"].nunique() <= 1:
            continue
        corr = day_df["pred_return"].corr(day_df["target_return"], method="spearman")
        if pd.notna(corr):
            scores.append(float(corr))
    return float(np.mean(scores)) if scores else 0.0


def compute_top5_mean_return(df: pd.DataFrame) -> float:
    returns = []
    for _, day_df in df.groupby("date"):
        top5 = day_df.sort_values("pred_return", ascending=False).head(5)
        if not top5.empty:
            returns.append(float(top5["target_return"].mean()))
    return float(np.mean(returns)) if returns else 0.0


def summarize_regime_slice(
    stage_name: str,
    prediction_slice: pd.DataFrame,
    daily_backtest_slice: pd.DataFrame,
    sample_days: int,
    extra_note: str,
) -> dict:
    rank_ic = compute_rank_ic(prediction_slice)
    top5_mean_return = compute_top5_mean_return(prediction_slice)
    backtest_return = float((1.0 + daily_backtest_slice["net_return"]).prod() - 1.0) if not daily_backtest_slice.empty else 0.0
    win_rate = float((daily_backtest_slice["net_return"] > 0).mean()) if not daily_backtest_slice.empty else 0.0
    avg_turnover = float(daily_backtest_slice["turnover"].mean()) if not daily_backtest_slice.empty else 0.0

    if rank_ic >= 0.03 and backtest_return > 0.10:
        conclusion = "阶段内排序与收益转化都较强，可视为主线优势区间。"
    elif rank_ic < 0.0 and backtest_return <= 0.02:
        conclusion = "阶段内排序失真较明显，说明当前主线在该市场状态下稳定性偏弱。"
    elif rank_ic >= 0.0 and backtest_return > 0.0:
        conclusion = "阶段内仍能保持正向有效，但优势不如最强阶段稳定。"
    else:
        conclusion = "阶段内存在一定预测能力，但组合收益转化偏弱，需要结合执行与筛选约束继续看。"

    if extra_note:
        conclusion = f"{conclusion}{extra_note}"

    return {
        "阶段名": stage_name,
        "样本天数": int(sample_days),
        "样本行数": int(len(prediction_slice)),
        "RankIC": rank_ic,
        "Top5平均收益": top5_mean_return,
        "回测收益": backtest_return,
        "胜率": win_rate,
        "平均换手": avg_turnover,
        "结论": conclusion,
    }


def build_daily_market_features(feature_path: Path) -> pd.DataFrame:
    feature_df = pd.read_csv(feature_path, dtype={"stock_id": str})
    feature_df["date"] = pd.to_datetime(feature_df["date"])

    daily_market = (
        feature_df.groupby("date", as_index=False)
        .agg(
            market_volatility_20d=("volatility_20d", "mean"),
            market_volatility_5d=("volatility_5d", "mean"),
            market_turnover_rate=("turnover_rate", "mean"),
            trend_strength=("trend_persistence_score_10d_v2", lambda s: float(np.nanmean(np.abs(s)))),
            trend_direction=("trend_persistence_score_10d_v2", "mean"),
            accel_strength=("rel_strength_accel_5d_v2", lambda s: float(np.nanmean(np.abs(s)))),
            market_ret_5d=("ret_5d", "mean"),
        )
    )
    daily_market["volatility_regime"] = np.where(
        daily_market["market_volatility_20d"] >= daily_market["market_volatility_20d"].median(),
        "高波动",
        "低波动",
    )
    daily_market["trend_regime"] = np.where(
        daily_market["trend_strength"] >= daily_market["trend_strength"].median(),
        "趋势",
        "震荡",
    )
    return daily_market


def build_fold_stage_table(prediction_df: pd.DataFrame, backtest_daily_df: pd.DataFrame, daily_market: pd.DataFrame) -> pd.DataFrame:
    merged = prediction_df.merge(daily_market, on="date", how="left")
    rows = []
    for fold_id, fold_df in merged.groupby("fold_id", sort=True):
        fold_dates = sorted(fold_df["date"].drop_duplicates())
        backtest_slice = backtest_daily_df[backtest_daily_df["date"].isin(fold_dates)].copy()
        avg_vol = float(fold_df["market_volatility_20d"].mean())
        avg_trend = float(fold_df["trend_strength"].mean())
        extra = f" 平均20日波动率为 {avg_vol:.4f}，趋势强度为 {avg_trend:.4f}。"
        rows.append(
            summarize_regime_slice(
                stage_name=f"Fold {int(fold_id)}",
                prediction_slice=fold_df,
                daily_backtest_slice=backtest_slice,
                sample_days=len(fold_dates),
                extra_note=extra,
            )
        )
    return pd.DataFrame(rows)


def build_rule_stage_table(prediction_df: pd.DataFrame, backtest_daily_df: pd.DataFrame, daily_market: pd.DataFrame) -> pd.DataFrame:
    merged = prediction_df.merge(daily_market, on="date", how="left")
    rows = []

    for stage_name, stage_df in merged.groupby("volatility_regime", sort=False):
        dates = sorted(stage_df["date"].drop_duplicates())
        backtest_slice = backtest_daily_df[backtest_daily_df["date"].isin(dates)].copy()
        extra = (
            f" 对应样本的平均20日波动率为 {float(stage_df['market_volatility_20d'].mean()):.4f}。"
        )
        rows.append(
            summarize_regime_slice(
                stage_name=stage_name,
                prediction_slice=stage_df,
                daily_backtest_slice=backtest_slice,
                sample_days=len(dates),
                extra_note=extra,
            )
        )

    for stage_name, stage_df in merged.groupby("trend_regime", sort=False):
        dates = sorted(stage_df["date"].drop_duplicates())
        backtest_slice = backtest_daily_df[backtest_daily_df["date"].isin(dates)].copy()
        extra = (
            f" 对应样本的平均趋势强度为 {float(stage_df['trend_strength'].mean()):.4f}。"
        )
        rows.append(
            summarize_regime_slice(
                stage_name=stage_name,
                prediction_slice=stage_df,
                daily_backtest_slice=backtest_slice,
                sample_days=len(dates),
                extra_note=extra,
            )
        )

    combo_group = merged.groupby(["volatility_regime", "trend_regime"], sort=False)
    for (vol_regime, trend_regime), stage_df in combo_group:
        dates = sorted(stage_df["date"].drop_duplicates())
        backtest_slice = backtest_daily_df[backtest_daily_df["date"].isin(dates)].copy()
        stage_name = f"{vol_regime}-{trend_regime}"
        extra = (
            f" 平均20日波动率 {float(stage_df['market_volatility_20d'].mean()):.4f}，"
            f" 平均趋势强度 {float(stage_df['trend_strength'].mean()):.4f}。"
        )
        rows.append(
            summarize_regime_slice(
                stage_name=stage_name,
                prediction_slice=stage_df,
                daily_backtest_slice=backtest_slice,
                sample_days=len(dates),
                extra_note=extra,
            )
        )

    return pd.DataFrame(rows)


def write_report(
    fold_table: pd.DataFrame,
    regime_table: pd.DataFrame,
    daily_market: pd.DataFrame,
    output_path: Path,
) -> None:
    worst_fold = fold_table.sort_values(["RankIC", "回测收益"], ascending=[True, True]).iloc[0]
    best_fold = fold_table.sort_values(["RankIC", "回测收益"], ascending=[False, False]).iloc[0]
    high_vol = regime_table[regime_table["阶段名"] == "高波动"].iloc[0]
    low_vol = regime_table[regime_table["阶段名"] == "低波动"].iloc[0]
    trend_stage = regime_table[regime_table["阶段名"] == "趋势"].iloc[0]
    range_stage = regime_table[regime_table["阶段名"] == "震荡"].iloc[0]

    lines = [
        "# 市场阶段分析",
        "",
        "## 阶段划分规则",
        "",
        "- Fold 阶段：直接按 walk-forward 三折验证期拆分。",
        "- 波动阶段：按每日横截面平均 `volatility_20d` 相对样本中位数划分为高波动/低波动。",
        "- 趋势阶段：按每日横截面平均 `abs(trend_persistence_score_10d_v2)` 相对样本中位数划分为趋势/震荡。",
        "",
        "## Fold 表现",
        "",
        "| 阶段名 | 样本天数 | RankIC | Top5平均收益 | 回测收益 | 结论 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in fold_table.iterrows():
        lines.append(
            f"| {row['阶段名']} | {int(row['样本天数'])} | {row['RankIC']:.6f} | "
            f"{row['Top5平均收益']:.6f} | {row['回测收益']:.6f} | {row['结论']} |"
        )

    lines.extend(
        [
            "",
            "## 规则阶段表现",
            "",
            "| 阶段名 | 样本天数 | RankIC | Top5平均收益 | 回测收益 | 结论 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in regime_table.iterrows():
        lines.append(
            f"| {row['阶段名']} | {int(row['样本天数'])} | {row['RankIC']:.6f} | "
            f"{row['Top5平均收益']:.6f} | {row['回测收益']:.6f} | {row['结论']} |"
        )

    lines.extend(
        [
            "",
            "## 核心结论",
            "",
            f"- 最弱阶段是 `{worst_fold['阶段名']}`，其 `RankIC={worst_fold['RankIC']:.6f}`、阶段回测收益 `{worst_fold['回测收益']:.6f}`。",
            f"- 最强阶段是 `{best_fold['阶段名']}`，其 `RankIC={best_fold['RankIC']:.6f}`、阶段回测收益 `{best_fold['回测收益']:.6f}`。",
            f"- 高波动阶段相对低波动阶段，`RankIC` 从 `{low_vol['RankIC']:.6f}` 变化到 `{high_vol['RankIC']:.6f}`，"
            f"阶段回测收益从 `{low_vol['回测收益']:.6f}` 变化到 `{high_vol['回测收益']:.6f}`。",
            f"- 趋势阶段相对震荡阶段，`RankIC` 从 `{range_stage['RankIC']:.6f}` 变化到 `{trend_stage['RankIC']:.6f}`，"
            f"阶段回测收益从 `{range_stage['回测收益']:.6f}` 变化到 `{trend_stage['回测收益']:.6f}`。",
            "",
            "从解释上看，如果模型在高波动或震荡阶段的横截面排序明显变差，往往意味着短期收益、日内收益与拥挤度信号之间的联动更容易失真；"
            "而当趋势强度更高、收益方向更一致时，排序信号更容易转化为组合收益。",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    feature_path = Path(args.feature_path)
    prediction_path = Path(args.prediction_path)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_df = load_prediction_frame(prediction_path, feature_path)
    prediction_df["date"] = pd.to_datetime(prediction_df["date"])

    backtest_summary_df, backtest_daily_df, _ = run_backtest(
        prediction_df=prediction_df,
        config=build_backtest_config(profile_name="market_regime_analysis"),
        prediction_source="replay_walk_forward_predictions",
    )
    backtest_daily_df["date"] = pd.to_datetime(backtest_daily_df["date"])

    daily_market = build_daily_market_features(feature_path=feature_path)
    fold_table = build_fold_stage_table(prediction_df=prediction_df, backtest_daily_df=backtest_daily_df, daily_market=daily_market)
    regime_table = build_rule_stage_table(prediction_df=prediction_df, backtest_daily_df=backtest_daily_df, daily_market=daily_market)

    daily_market.to_csv(output_dir / "daily_market_regimes.csv", index=False, encoding="utf-8-sig")
    fold_table.to_csv(output_dir / "fold_stage_performance.csv", index=False, encoding="utf-8-sig")
    regime_table.to_csv(output_dir / "rule_stage_performance.csv", index=False, encoding="utf-8-sig")
    backtest_summary_df.to_csv(output_dir / "stage_analysis_backtest_summary.csv", index=False, encoding="utf-8-sig")
    backtest_daily_df.to_csv(output_dir / "stage_analysis_backtest_daily.csv", index=False, encoding="utf-8-sig")
    write_report(fold_table, regime_table, daily_market, output_dir / "market_regime_analysis.md")

    print(f"[market_regime_analysis] wrote {output_dir / 'fold_stage_performance.csv'}")
    print(f"[market_regime_analysis] wrote {output_dir / 'rule_stage_performance.csv'}")
    print(f"[market_regime_analysis] wrote {output_dir / 'market_regime_analysis.md'}")


if __name__ == "__main__":
    main()
