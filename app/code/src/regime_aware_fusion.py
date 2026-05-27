from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import run_backtest
from config import ROOT_DIR
from regime_rerank_switch import (
    add_switch_signal,
    build_backtest_config,
    load_prediction_with_features,
    load_regimes,
    resolve_path,
    run_profile,
    summarize_selection,
    write_result_from_backtest,
)


DEFAULT_PRED_PATH = ROOT_DIR / "app" / "model" / "v4_rerank_penalty_search" / "baseline_predictions_with_v4_risk_features.csv"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "regime_aware_fusion"
DEFAULT_REGIME_PATH = ROOT_DIR / "app" / "model" / "market_regime_analysis" / "daily_market_regimes.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal regime-aware fusion and candidate-selection experiments.")
    parser.add_argument("--pred_paths", nargs="+", default=[str(DEFAULT_PRED_PATH)])
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--regime_path", default=str(DEFAULT_REGIME_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--risk_weight", type=float, default=-0.05)
    return parser.parse_args()


def daily_rank_metrics(prediction_df: pd.DataFrame, top_k: int = 5) -> dict[str, float]:
    rows = []
    working = prediction_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    for date, day_df in working.groupby("date", sort=True):
        day = day_df.dropna(subset=["pred_return", "target_return"]).copy()
        if day.empty:
            continue
        valid = day[["pred_return", "target_return"]].apply(pd.to_numeric, errors="coerce").dropna()
        rank_ic = np.nan
        if len(valid) > 1 and valid["pred_return"].nunique() > 1 and valid["target_return"].nunique() > 1:
            rank_ic = valid["pred_return"].corr(valid["target_return"], method="spearman")
        pred_top = day.sort_values(["pred_return", "stock_id"], ascending=[False, True]).head(top_k)
        rows.append(
            {
                "date": date,
                "fold_id": int(day["fold_id"].iloc[0]) if "fold_id" in day.columns else 0,
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "top5_return": float(pd.to_numeric(pred_top["target_return"], errors="coerce").mean()),
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return {
            "rank_ic_mean": 0.0,
            "worst_fold_rank_ic": 0.0,
            "fold1_rank_ic": np.nan,
            "fold3_rank_ic": np.nan,
            "negative_day_rank_ic_ratio": 0.0,
            "top5_return_mean": 0.0,
        }
    fold = daily.groupby("fold_id", as_index=False).agg(rank_ic=("rank_ic", "mean"), top5_return=("top5_return", "mean"))
    rank_ic = pd.to_numeric(daily["rank_ic"], errors="coerce").dropna()
    return {
        "rank_ic_mean": float(fold["rank_ic"].mean()),
        "worst_fold_rank_ic": float(fold["rank_ic"].min()),
        "fold1_rank_ic": float(fold.loc[fold["fold_id"].eq(1), "rank_ic"].iloc[0]) if (fold["fold_id"].eq(1)).any() else np.nan,
        "fold3_rank_ic": float(fold.loc[fold["fold_id"].eq(3), "rank_ic"].iloc[0]) if (fold["fold_id"].eq(3)).any() else np.nan,
        "negative_day_rank_ic_ratio": float((rank_ic < 0).mean()) if not rank_ic.empty else 0.0,
        "top5_return_mean": float(daily["top5_return"].mean()),
    }


def build_average_rank_fusion(
    base_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    *,
    signal: str,
    regime_flag: str,
    weight: float,
    defensive_share: float = 0.5,
) -> pd.DataFrame:
    switched = add_switch_signal(
        base_df,
        regime_df,
        signal=signal,
        regime_flag=regime_flag,
        output_column="_switch_defensive_signal",
    )
    out = switched.merge(regime_df[["date", regime_flag]], on="date", how="left")
    active = out[regime_flag].fillna(0).astype(int).eq(1)
    base_rank = out.groupby("date")["pred_return"].rank(pct=True)
    defensive_score = base_rank + float(weight) * (out.groupby("date")["_switch_defensive_signal"].rank(pct=True) - 0.5)
    defensive_rank = defensive_score.groupby(out["date"]).rank(pct=True)
    out["pred_return"] = np.where(
        active,
        (1.0 - defensive_share) * base_rank + defensive_share * defensive_rank,
        base_rank,
    )
    return out.drop(columns=[regime_flag, "_switch_defensive_signal"])


def run_fusion_profile(
    base_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    output_dir: Path,
    *,
    profile_name: str,
    signal: str,
    regime_flag: str,
    weight: float,
) -> dict[str, Any]:
    profile_dir = output_dir / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    prediction_df = build_average_rank_fusion(
        base_df,
        regime_df,
        signal=signal,
        regime_flag=regime_flag,
        weight=weight,
    )
    config = build_backtest_config(profile_name, rerank_column=None, rerank_weight=0.0)
    summary_df, daily_df, holdings_df = run_backtest(prediction_df, config, prediction_source=profile_name)
    prediction_df.to_csv(profile_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(profile_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(profile_dir / "backtest_daily.csv", index=False, encoding="utf-8-sig")
    holdings_df.to_csv(profile_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    write_result_from_backtest(daily_df, profile_dir / "result.csv")
    bt = summary_df.iloc[0].to_dict()
    row = {
        "profile_name": profile_name,
        "experiment_type": "average_rank_fusion",
        "signal": signal,
        "regime_flag": regime_flag,
        "weight": float(weight),
        "cost_after_return": float(bt["cumulative_return_after_cost"]),
        "sharpe": float(bt["sharpe_after_cost"]),
        "max_drawdown": float(bt["max_drawdown_after_cost"]),
        "avg_turnover": float(bt["avg_turnover"]),
        "max_single_contribution_share": float(bt.get("max_single_contribution_share", 0.0)),
    }
    row.update(daily_rank_metrics(prediction_df))
    row.update(summarize_selection(prediction_df, daily_df, regime_df, profile_name))
    return row


def enrich_rows(rows: list[dict[str, Any]], base_df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(rows)
    if "experiment_type" not in summary.columns:
        summary["experiment_type"] = "regime_candidate_selection"
    summary["experiment_type"] = summary["experiment_type"].fillna("regime_candidate_selection")
    base_rank = daily_rank_metrics(base_df)
    for key, value in base_rank.items():
        if key not in summary.columns:
            summary[key] = value
        summary[key] = summary[key].fillna(value)

    baseline = summary[summary["profile_name"].eq("baseline")].iloc[0]
    delta_columns = [
        "rank_ic_mean",
        "worst_fold_rank_ic",
        "top5_return_mean",
        "cost_after_return",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "selected_top5_return_mean",
        "fold3_selected_top5_return",
        "high_volatility_selected_top5_return",
        "high_volatility_range_selected_top5_return",
        "poor_false_positives",
    ]
    for column in delta_columns:
        summary[f"delta_{column}"] = summary[column] - baseline[column]
    summary["passes_robust_rule"] = (
        (summary["profile_name"].ne("baseline"))
        & (
            (summary["delta_fold3_selected_top5_return"] > 0)
            | (summary["delta_high_volatility_selected_top5_return"] > 0)
        )
        & (summary["delta_cost_after_return"] > -0.02)
        & (summary["delta_sharpe"] > -0.05)
        & (summary["delta_poor_false_positives"] <= 0)
    )
    return summary


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    baseline = summary[summary["profile_name"].eq("baseline")].iloc[0]
    robust_candidates = summary[summary["passes_robust_rule"].astype(bool)].copy()
    if robust_candidates.empty:
        best = summary[summary["profile_name"].ne("baseline")].sort_values(
            ["high_volatility_selected_top5_return", "fold3_selected_top5_return", "cost_after_return"],
            ascending=[False, False, False],
        ).iloc[0]
        adopted = False
    else:
        best = robust_candidates.sort_values(
            [
                "high_volatility_selected_top5_return",
                "fold3_selected_top5_return",
                "selected_top5_return_mean",
                "sharpe",
                "cost_after_return",
            ],
            ascending=[False, False, False, False, False],
        ).iloc[0]
        adopted = True

    lines = [
        "# Regime-Aware Fusion Report",
        "",
        "This experiment keeps the current aggressive baseline as the default behavior and only tests defensive ranking/fusion on high-risk market regimes.",
        "",
        "## Decision",
        "",
        f"- robust_candidate: `{best['profile_name']}`",
        f"- passes_robust_rule: `{str(adopted).lower()}`",
        f"- aggressive_default_unchanged: `true`",
        f"- dual_config_recommended: `{str(adopted).lower()}`",
        "",
        "## Required Answers",
        "",
        f"1. Fold 3 improvement: `{best['delta_fold3_selected_top5_return']:.6f}` versus baseline `{baseline['fold3_selected_top5_return']:.6f}`.",
        f"2. High-volatility improvement: `{best['delta_high_volatility_selected_top5_return']:.6f}` versus baseline `{baseline['high_volatility_selected_top5_return']:.6f}`.",
        "3. Low-volatility/aggressive behavior is preserved for regime-switch profiles because the defensive signal is constant outside the target regime.",
        "4. `close_position_20d -0.05` is preferred over `reversal_risk_score -0.05` when it improves high-vol/Fold3 and keeps return metrics intact.",
        f"5. robust config generation is `{str(adopted).lower()}` under the conservative rule.",
        f"6. aggressive/robust dual config is `{str(adopted).lower()}`; default aggressive should remain untouched.",
        "",
        "## Summary Table",
        "",
        "| profile | type | signal | regime | cost_after | sharpe | max_dd | selected_top5 | fold3 | high_vol | high_vol_range | poor_fp | robust |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['profile_name']} | {row['experiment_type']} | {row['signal']} | {row['regime_flag']} | "
            f"{row['cost_after_return']:.6f} | {row['sharpe']:.6f} | {row['max_drawdown']:.6f} | "
            f"{row['selected_top5_return_mean']:.6f} | {row['fold3_selected_top5_return']:.6f} | "
            f"{row['high_volatility_selected_top5_return']:.6f} | {row['high_volatility_range_selected_top5_return']:.6f} | "
            f"{int(row['poor_false_positives'])} | {str(bool(row['passes_robust_rule'])).lower()} |"
        )
    (output_dir / "regime_aware_fusion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = resolve_path(args.feature_path)
    regime_path = resolve_path(args.regime_path)

    base_df = load_prediction_with_features(resolve_path(args.pred_paths[0]), feature_path)
    regime_df = load_regimes(feature_path, regime_path)
    weight = float(args.risk_weight)

    candidate_profiles: list[tuple[str, str | None, str | None]] = [
        ("baseline", None, None),
        ("global_close_position_20d_m005", "close_position_20d", None),
        ("global_reversal_risk_score_m005", "reversal_risk_score", None),
        ("hv_close_position_20d_m005", "close_position_20d", "is_high_volatility"),
        ("hvrange_close_position_20d_m005", "close_position_20d", "is_high_volatility_range"),
        ("hv_reversal_risk_score_m005", "reversal_risk_score", "is_high_volatility"),
        ("hvrange_reversal_risk_score_m005", "reversal_risk_score", "is_high_volatility_range"),
    ]

    rows: list[dict[str, Any]] = []
    for profile_name, signal, regime_flag in candidate_profiles:
        print(f"[regime_aware_fusion] running candidate_selection {profile_name}")
        row = run_profile(base_df, regime_df, output_dir, profile_name, signal, regime_flag, weight)
        row["experiment_type"] = "regime_candidate_selection"
        rows.append(row)

    fusion_profiles = [
        ("avg_rank_hv_close_position_20d_m005", "close_position_20d", "is_high_volatility"),
        ("avg_rank_hvrange_close_position_20d_m005", "close_position_20d", "is_high_volatility_range"),
        ("avg_rank_hv_reversal_risk_score_m005", "reversal_risk_score", "is_high_volatility"),
    ]
    for profile_name, signal, regime_flag in fusion_profiles:
        print(f"[regime_aware_fusion] running average_rank_fusion {profile_name}")
        rows.append(run_fusion_profile(base_df, regime_df, output_dir, profile_name=profile_name, signal=signal, regime_flag=regime_flag, weight=weight))

    summary = enrich_rows(rows, base_df)
    summary.to_csv(output_dir / "regime_aware_fusion_summary.csv", index=False, encoding="utf-8-sig")
    write_report(summary, output_dir)
    print(f"[regime_aware_fusion] wrote {output_dir / 'regime_aware_fusion_summary.csv'}")
    print(f"[regime_aware_fusion] wrote {output_dir / 'regime_aware_fusion_report.md'}")


if __name__ == "__main__":
    main()
