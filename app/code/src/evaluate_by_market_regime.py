from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import load_prediction_frame, run_backtest
from config import ROOT_DIR
from load_submission_config import build_default_inference_args, load_submission_config
from market_regime_split import DEFAULT_OUTPUT_PATH as DEFAULT_REGIME_PATH
from market_regime_split import load_and_split_market_regimes, resolve_path


DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_PRED_PATH = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "market_regime_analysis"

REGIME_FLAGS = [
    ("low_volatility", "is_low_volatility"),
    ("high_volatility", "is_high_volatility"),
    ("trend", "is_trend"),
    ("range", "is_range"),
    ("high_volatility_range", "is_high_volatility_range"),
    ("low_volatility_trend", "is_low_volatility_trend"),
]

SUMMARY_COLUMNS = [
    "regime",
    "sample_days",
    "rank_ic",
    "top5_return",
    "cost_after_return",
    "sharpe",
    "max_drawdown",
    "win_rate",
    "avg_turnover",
    "negative_rankic_ratio",
]


def build_backtest_config() -> dict:
    cfg = load_submission_config()
    args = build_default_inference_args(cfg)
    return {
        "profile_name": "market_regime_analysis",
        "top_k": int(args["top_k"]),
        "primary_candidate_size": int(args["primary_candidate_size"]),
        "enable_risk_filters": bool(args["enable_risk_filters"]),
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": float(args["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(args["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(args["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(args["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(args["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(args["risk_penalty_weight"]),
        "weighting_scheme": str(args["weighting_scheme"]),
        "weight_blend_alpha": float(args.get("weight_blend_alpha", 1.0)),
        "max_single_weight": args.get("max_single_weight"),
        "sort_strategy": str(args["sort_strategy"]),
        "transaction_cost": float(args["transaction_cost"]),
        "max_turnover": float(args["max_turnover"]),
    }


def daily_rank_ic(prediction_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    working = prediction_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    for date, day_df in working.groupby("date", sort=True):
        valid = day_df[["pred_return", "target_return"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(valid) < 2 or valid["pred_return"].nunique() <= 1 or valid["target_return"].nunique() <= 1:
            rank_ic = np.nan
        else:
            rank_ic = valid["pred_return"].corr(valid["target_return"], method="spearman")
        top5 = day_df.sort_values(["pred_return", "stock_id"], ascending=[False, True]).head(5)
        rows.append(
            {
                "date": date,
                "day_rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "day_top5_return": float(pd.to_numeric(top5["target_return"], errors="coerce").mean()) if not top5.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compound_return(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if values.empty:
        return 0.0
    return float((1.0 + values).prod() - 1.0)


def sharpe(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if len(values) < 2:
        return 0.0
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    return float(values.mean() / std * np.sqrt(len(values)))


def max_drawdown(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if values.empty:
        return 0.0
    equity = (1.0 + values).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min()) if not dd.empty else 0.0


def summarize_regime(
    regime: str,
    dates: set[pd.Timestamp],
    rank_daily: pd.DataFrame,
    backtest_daily: pd.DataFrame,
) -> dict:
    rank_slice = rank_daily[rank_daily["date"].isin(dates)].copy()
    bt_slice = backtest_daily[backtest_daily["date"].isin(dates)].copy()
    rank_ic = pd.to_numeric(rank_slice["day_rank_ic"], errors="coerce").dropna()
    net_return = pd.to_numeric(bt_slice.get("net_return", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "regime": regime,
        "sample_days": int(len(dates)),
        "rank_ic": float(rank_ic.mean()) if not rank_ic.empty else 0.0,
        "top5_return": float(pd.to_numeric(rank_slice["day_top5_return"], errors="coerce").mean()) if not rank_slice.empty else 0.0,
        "cost_after_return": compound_return(net_return),
        "sharpe": sharpe(net_return),
        "max_drawdown": max_drawdown(net_return),
        "win_rate": float((net_return > 0).mean()) if not net_return.empty else 0.0,
        "avg_turnover": float(pd.to_numeric(bt_slice.get("turnover", pd.Series(dtype=float)), errors="coerce").mean()) if not bt_slice.empty else 0.0,
        "negative_rankic_ratio": float((rank_ic < 0).mean()) if not rank_ic.empty else 0.0,
    }


def evaluate_by_regime(prediction_df: pd.DataFrame, backtest_daily_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    regimes = regime_df.copy()
    regimes["date"] = pd.to_datetime(regimes["date"], errors="coerce").dt.normalize()
    backtest_daily = backtest_daily_df.copy()
    backtest_daily["date"] = pd.to_datetime(backtest_daily["date"], errors="coerce").dt.normalize()
    rank_daily = daily_rank_ic(prediction_df)
    eval_dates = set(rank_daily["date"].dropna())

    rows = []
    rows.append(summarize_regime("all", eval_dates, rank_daily, backtest_daily))
    for regime, flag in REGIME_FLAGS:
        if flag not in regimes.columns:
            raise ValueError(f"Regime table missing required column `{flag}`")
        dates = set(regimes.loc[regimes[flag].astype(int).eq(1), "date"]) & eval_dates
        rows.append(summarize_regime(regime, dates, rank_daily, backtest_daily))
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def _fmt(value: object) -> str:
    return f"{float(value):.6f}"


def write_report(summary_df: pd.DataFrame, regime_df: pd.DataFrame, output_path: Path) -> None:
    scored = summary_df[summary_df["regime"].ne("all")].copy()
    scored = scored[scored["sample_days"] > 0].copy()
    strongest = scored.sort_values(["rank_ic", "top5_return", "cost_after_return"], ascending=[False, False, False]).iloc[0]
    weakest = scored.sort_values(["rank_ic", "top5_return", "cost_after_return"], ascending=[True, True, True]).iloc[0]
    high_range = summary_df[summary_df["regime"].eq("high_volatility_range")]
    high_range_row = high_range.iloc[0] if not high_range.empty else None
    low_trend = summary_df[summary_df["regime"].eq("low_volatility_trend")]
    low_trend_row = low_trend.iloc[0] if not low_trend.empty else None

    latest_thresholds = regime_df.iloc[-1]
    high_range_is_risk = bool(
        high_range_row is not None
        and (
            float(high_range_row["rank_ic"]) < float(summary_df.loc[summary_df["regime"].eq("all"), "rank_ic"].iloc[0])
            or float(high_range_row["top5_return"]) < 0
        )
    )
    recommend_switch = bool(
        low_trend_row is not None
        and high_range_row is not None
        and float(low_trend_row["rank_ic"]) > float(high_range_row["rank_ic"])
    )

    lines = [
        "# Market Regime Analysis Report",
        "",
        "## Simple Regime Rules",
        "",
        "- 低/高波动：每日股票池平均 `volatility_20d`，按全样本中位数切分。",
        "- 趋势/震荡：20 日市场平均收益绝对值高于中位数，且方向一致性高于中位数，才标为趋势；否则标为震荡。",
        "- 高波动震荡：同时满足高波动和震荡。",
        "- 低波动趋势：同时满足低波动和趋势。",
        "",
        "## Required Answers",
        "",
        f"1. 当前模型最强阶段：`{strongest['regime']}`，RankIC={_fmt(strongest['rank_ic'])}，Top5={_fmt(strongest['top5_return'])}。",
        f"2. 当前模型最弱阶段：`{weakest['regime']}`，RankIC={_fmt(weakest['rank_ic'])}，Top5={_fmt(weakest['top5_return'])}。",
        (
            "3. 高波动震荡阶段是主要风险来源。"
            if high_range_is_risk
            else "3. 高波动震荡阶段不是唯一风险来源，但仍应作为防守阶段单独监控。"
        ),
        (
            "4. 建议低波动/趋势使用 aggressive，高波动/震荡使用 robust，并先做离线 replay 验证。"
            if recommend_switch
            else "4. 暂不建议直接切换 aggressive/robust，先累计更多 regime 样本验证。"
        ),
        (
            "5. regime switching 阈值建议保持简单："
            f"`volatility_20d` 横截面均值 >= {float(latest_thresholds['volatility_threshold']):.6f} 判高波动；"
            f"`abs(20d_market_return)` >= {float(latest_thresholds['trend_strength_threshold']):.6f} 且 "
            f"`direction_consistency` >= {float(latest_thresholds['direction_consistency_threshold']):.6f} 判趋势。"
            "只用中位数阈值，不训练复杂 regime 模型，避免过拟合。"
        ),
        "",
        "## Regime Summary",
        "",
        "| regime | days | rank_ic | top5_return | cost_after_return | sharpe | max_drawdown | win_rate | avg_turnover | neg_rankic_ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['regime']} | {int(row['sample_days'])} | {_fmt(row['rank_ic'])} | "
            f"{_fmt(row['top5_return'])} | {_fmt(row['cost_after_return'])} | {_fmt(row['sharpe'])} | "
            f"{_fmt(row['max_drawdown'])} | {_fmt(row['win_rate'])} | {_fmt(row['avg_turnover'])} | "
            f"{_fmt(row['negative_rankic_ratio'])} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction and backtest performance by simple market regime.")
    parser.add_argument("--pred_path", "--prediction_path", dest="pred_path", default=str(DEFAULT_PRED_PATH))
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--regime_path", default=str(DEFAULT_REGIME_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regime_path = resolve_path(args.regime_path)

    regime_df = load_and_split_market_regimes(feature_path=args.feature_path, output_path=regime_path)
    prediction_df = load_prediction_frame(resolve_path(args.pred_path), resolve_path(args.feature_path))
    backtest_summary_df, backtest_daily_df, holdings_df = run_backtest(
        prediction_df=prediction_df,
        config=build_backtest_config(),
        prediction_source="walk_forward_predictions",
    )
    summary_df = evaluate_by_regime(prediction_df, backtest_daily_df, regime_df)

    summary_path = output_dir / "market_regime_summary.csv"
    report_path = output_dir / "market_regime_report.md"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    backtest_summary_df.to_csv(output_dir / "market_regime_backtest_summary.csv", index=False, encoding="utf-8-sig")
    backtest_daily_df.to_csv(output_dir / "market_regime_backtest_daily.csv", index=False, encoding="utf-8-sig")
    holdings_df.to_csv(output_dir / "market_regime_backtest_holdings.csv", index=False, encoding="utf-8-sig")
    write_report(summary_df, regime_df, report_path)

    scored = summary_df[summary_df["regime"].ne("all")].sort_values("rank_ic", ascending=False)
    print(f"[market_regime_eval] strongest={scored.iloc[0]['regime']} rank_ic={scored.iloc[0]['rank_ic']:.6f}")
    print(f"[market_regime_eval] weakest={scored.iloc[-1]['regime']} rank_ic={scored.iloc[-1]['rank_ic']:.6f}")
    print(f"[market_regime_eval] wrote {summary_path}")
    print(f"[market_regime_eval] wrote {report_path}")


if __name__ == "__main__":
    main()
