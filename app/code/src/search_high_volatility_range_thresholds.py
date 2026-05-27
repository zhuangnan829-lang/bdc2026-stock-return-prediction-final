from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from config import ROOT_DIR
from run_high_volatility_range_optimization import (
    DEFAULT_AGGRESSIVE_CONFIG,
    DEFAULT_FEATURE_PATH,
    DEFAULT_MAINLINE_CONFIG,
    DEFAULT_PRED_PATH,
    DEFAULT_ROBUST_CONFIG,
    add_decision_columns,
    resolve_path,
    run_named_profile,
)
from regime_switch_submission import build_simple_regimes, config_to_backtest_config
from backtest import load_prediction_frame


DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "high_volatility_range_threshold_search"


def fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def parse_float_list(raw: list[str]) -> list[float]:
    return [float(item) for item in raw]


def run_one_grid(
    prediction_df: pd.DataFrame,
    feature_path: Path,
    output_dir: Path,
    *,
    volatility_quantile: float,
    range_quantile: float,
    mainline_config: dict[str, Any],
    aggressive_config: dict[str, Any],
    robust_config: dict[str, Any],
) -> dict[str, Any]:
    grid_dir = output_dir / f"vol{volatility_quantile:.2f}_range{range_quantile:.2f}"
    regime_df = build_simple_regimes(
        feature_path,
        volatility_quantile=volatility_quantile,
        range_quantile=range_quantile,
        output_path=grid_dir / "daily_high_volatility_range_regimes.csv",
    )
    rows = [
        run_named_profile(
            prediction_df,
            regime_df,
            grid_dir,
            output_name="mainline_static",
            engine_profile_name="aggressive_static",
            aggressive_config=mainline_config,
            robust_config=robust_config,
        ),
        run_named_profile(
            prediction_df,
            regime_df,
            grid_dir,
            output_name="aggressive_static",
            engine_profile_name="aggressive_static",
            aggressive_config=aggressive_config,
            robust_config=robust_config,
        ),
        run_named_profile(
            prediction_df,
            regime_df,
            grid_dir,
            output_name="robust_static",
            engine_profile_name="robust_static",
            aggressive_config=aggressive_config,
            robust_config=robust_config,
        ),
        run_named_profile(
            prediction_df,
            regime_df,
            grid_dir,
            output_name="regime_switch",
            engine_profile_name="regime_switch",
            aggressive_config=aggressive_config,
            robust_config=robust_config,
        ),
    ]
    summary = add_decision_columns(pd.DataFrame(rows))
    mainline = summary[summary["profile_name"].eq("mainline_static")].iloc[0]
    switch = summary[summary["profile_name"].eq("regime_switch")].iloc[0]
    hvr_days = int(regime_df["is_high_volatility_range"].sum())
    row = {
        "volatility_quantile": volatility_quantile,
        "range_quantile": range_quantile,
        "hvr_days": hvr_days,
        "mainline_return": float(mainline["cost_after_return"]),
        "mainline_max_drawdown": float(mainline["max_drawdown"]),
        "mainline_avg_turnover": float(mainline["avg_turnover"]),
        "mainline_hvr_return": float(mainline["high_vol_range_cost_after_return"]),
        "mainline_hvr_max_drawdown": float(mainline["high_vol_range_max_drawdown"]),
        "switch_return": float(switch["cost_after_return"]),
        "switch_sharpe": float(switch["Sharpe"]),
        "switch_max_drawdown": float(switch["max_drawdown"]),
        "switch_avg_turnover": float(switch["avg_turnover"]),
        "switch_hvr_return": float(switch["high_vol_range_cost_after_return"]),
        "switch_hvr_max_drawdown": float(switch["high_vol_range_max_drawdown"]),
        "return_loss_vs_mainline": float(switch["return_loss_vs_mainline"]),
        "turnover_delta_vs_mainline": float(switch["avg_turnover"] - mainline["avg_turnover"]),
        "hvr_return_delta_vs_mainline": float(switch["high_vol_range_cost_after_return"] - mainline["high_vol_range_cost_after_return"]),
        "hvr_drawdown_delta_vs_mainline": float(switch["high_vol_range_max_drawdown"] - mainline["high_vol_range_max_drawdown"]),
        "recommended_for_next_step": bool(switch["recommended_for_next_step"]),
    }
    row["rule_pass"] = bool(
        row["return_loss_vs_mainline"] <= 0.05
        and row["turnover_delta_vs_mainline"] < 0
        and row["hvr_drawdown_delta_vs_mainline"] >= -1e-12
    )
    grid_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(grid_dir / "profile_comparison.csv", index=False, encoding="utf-8-sig")
    return row


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    candidates = summary[summary["rule_pass"].astype(bool)].copy()
    if candidates.empty:
        best = summary.sort_values(
            ["return_loss_vs_mainline", "turnover_delta_vs_mainline", "hvr_drawdown_delta_vs_mainline"],
            ascending=[True, True, False],
        ).iloc[0]
        decision = "No threshold combination passed the conservative rule. Keep sl20 mainline and robust as observation only."
    else:
        best = candidates.sort_values(
            ["return_loss_vs_mainline", "turnover_delta_vs_mainline", "hvr_return_delta_vs_mainline"],
            ascending=[True, True, False],
        ).iloc[0]
        decision = "At least one threshold passed the conservative rule. Promote the best row to deeper replay validation."

    lines = [
        "# High Volatility Range Threshold Search Report",
        "",
        "Conservative pass rule: return_loss_vs_mainline <= 0.05, turnover decreases, and high-volatility-range drawdown does not worsen.",
        "",
        "## Decision",
        "",
        f"- passed_rows: `{int(candidates.shape[0])}`",
        f"- best_volatility_quantile: `{fmt(best['volatility_quantile'])}`",
        f"- best_range_quantile: `{fmt(best['range_quantile'])}`",
        f"- best_return_loss_vs_mainline: `{fmt(best['return_loss_vs_mainline'])}`",
        f"- best_turnover_delta_vs_mainline: `{fmt(best['turnover_delta_vs_mainline'])}`",
        f"- best_hvr_return_delta_vs_mainline: `{fmt(best['hvr_return_delta_vs_mainline'])}`",
        f"- best_hvr_drawdown_delta_vs_mainline: `{fmt(best['hvr_drawdown_delta_vs_mainline'])}`",
        f"- decision: {decision}",
        "",
        "## Top Rows",
        "",
        "| vol_q | range_q | hvr_days | switch_return | return_loss | turnover_delta | hvr_return_delta | hvr_drawdown_delta | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    ranked = summary.sort_values(
        ["rule_pass", "return_loss_vs_mainline", "turnover_delta_vs_mainline"],
        ascending=[False, True, True],
    )
    for _, row in ranked.head(20).iterrows():
        lines.append(
            f"| {fmt(row['volatility_quantile'])} | {fmt(row['range_quantile'])} | {int(row['hvr_days'])} | "
            f"{fmt(row['switch_return'])} | {fmt(row['return_loss_vs_mainline'])} | "
            f"{fmt(row['turnover_delta_vs_mainline'])} | {fmt(row['hvr_return_delta_vs_mainline'])} | "
            f"{fmt(row['hvr_drawdown_delta_vs_mainline'])} | {str(bool(row['rule_pass'])).lower()} |"
        )
    (output_dir / "high_volatility_range_threshold_search_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search high-volatility range regime-switch thresholds.")
    parser.add_argument("--pred_path", default=str(DEFAULT_PRED_PATH))
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--mainline_config", default=str(DEFAULT_MAINLINE_CONFIG))
    parser.add_argument("--aggressive_config", default=str(DEFAULT_AGGRESSIVE_CONFIG))
    parser.add_argument("--robust_config", default=str(DEFAULT_ROBUST_CONFIG))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--volatility_quantiles", nargs="+", default=["0.60", "0.65", "0.70", "0.75"])
    parser.add_argument("--range_quantiles", nargs="+", default=["0.25", "0.30", "0.35", "0.40"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = resolve_path(args.feature_path)
    prediction_df = load_prediction_frame(resolve_path(args.pred_path), feature_path)
    mainline_config = config_to_backtest_config(resolve_path(args.mainline_config), "mainline_static")
    aggressive_config = config_to_backtest_config(resolve_path(args.aggressive_config), "aggressive_static")
    robust_config = config_to_backtest_config(resolve_path(args.robust_config), "robust_static")

    rows = []
    for volatility_quantile in parse_float_list(args.volatility_quantiles):
        for range_quantile in parse_float_list(args.range_quantiles):
            print(f"[hvr_threshold_search] vol={volatility_quantile:.2f} range={range_quantile:.2f}")
            rows.append(
                run_one_grid(
                    prediction_df,
                    feature_path,
                    output_dir,
                    volatility_quantile=volatility_quantile,
                    range_quantile=range_quantile,
                    mainline_config=mainline_config,
                    aggressive_config=aggressive_config,
                    robust_config=robust_config,
                )
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "high_volatility_range_threshold_search_summary.csv", index=False, encoding="utf-8-sig")
    write_report(summary, output_dir)
    print(summary.sort_values(["rule_pass", "return_loss_vs_mainline"], ascending=[False, True]).to_string(index=False))
    print(f"[hvr_threshold_search] wrote {output_dir / 'high_volatility_range_threshold_search_summary.csv'}")
    print(f"[hvr_threshold_search] wrote {output_dir / 'high_volatility_range_threshold_search_report.md'}")


if __name__ == "__main__":
    main()
