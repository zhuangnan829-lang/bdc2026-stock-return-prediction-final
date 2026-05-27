import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "app" / "code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import backtest as backtest_module
from config import BEST_CONFIG, BEST_PROFILE_NAME


DEFAULT_PREDICTION_PATH = "app/model/walk_forward_predictions.csv"
DEFAULT_FEATURE_PATH = "app/temp/train_features.csv"
DEFAULT_MODEL_DIR = "app/model"
DEFAULT_OUTPUT_DIR = "app/model/inference_local_upgrade_search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local search for upgrading the current default inference profile.")
    parser.add_argument("--prediction_path", default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--feature_path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_n_report", type=int, default=20)
    return parser.parse_args()


def build_local_grid() -> list[dict]:
    base = {
        "top_k": int(BEST_CONFIG["selection"]["top_k"]),
        "enable_risk_filters": 1,
        "allow_cash_fallback": 0,
        "turnover_rate_lower_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_ratio_upper_pct"]),
        "weighting_scheme": BEST_CONFIG["selection"]["weighting_scheme"],
        "sort_strategy": BEST_CONFIG["selection"]["sort_strategy"],
        "transaction_cost": float(BEST_CONFIG["execution"]["transaction_cost"]),
    }

    candidate_sizes = [45, 50, 55, 60]
    vol20_pcts = [0.82, 0.84, 0.86, 0.88]
    vol5_pcts = [0.90, 0.92, 0.94, 0.96]
    risk_penalty_weights = [0.14, 0.16, 0.18, 0.20, 0.22]
    max_turnovers = [0.50, 0.55, 0.60, 0.65]

    grid = []
    for candidate_size, vol20, vol5, rp, mt in itertools.product(
        candidate_sizes,
        vol20_pcts,
        vol5_pcts,
        risk_penalty_weights,
        max_turnovers,
    ):
        grid.append(
            {
                **base,
                "profile_name": (
                    f"local_cs{candidate_size}_v20{int(vol20*100)}_v5{int(vol5*100)}_"
                    f"rp{int(rp*100)}_mt{int(mt*100)}"
                ),
                "primary_candidate_size": candidate_size,
                "max_volatility_20d_pct": vol20,
                "max_volatility_5d_pct": vol5,
                "risk_penalty_weight": rp,
                "max_turnover": mt,
            }
        )
    return grid


def is_current_default(profile: dict) -> bool:
    return (
        profile["primary_candidate_size"] == BEST_CONFIG["selection"]["primary_candidate_size"]
        and abs(profile["max_volatility_20d_pct"] - BEST_CONFIG["risk_filter_thresholds"]["max_volatility_20d_pct"]) < 1e-12
        and abs(profile["max_volatility_5d_pct"] - BEST_CONFIG["risk_filter_thresholds"]["max_volatility_5d_pct"]) < 1e-12
        and abs(profile["risk_penalty_weight"] - BEST_CONFIG["risk_filter_thresholds"]["risk_penalty_weight"]) < 1e-12
        and abs(profile["max_turnover"] - BEST_CONFIG["execution"]["max_turnover"]) < 1e-12
    )


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_df, prediction_source = backtest_module.load_or_generate_predictions(
        prediction_path=ROOT / args.prediction_path,
        feature_path=ROOT / args.feature_path,
        model_dir=ROOT / args.model_dir,
    )

    rows = []
    grid = build_local_grid()
    for idx, profile in enumerate(grid, start=1):
        summary_df, _, _ = backtest_module.run_backtest(
            prediction_df=prediction_df,
            config=profile,
            prediction_source=prediction_source,
        )
        row = summary_df.iloc[0].to_dict()
        row.update(
            {
                "primary_candidate_size": profile["primary_candidate_size"],
                "max_volatility_20d_pct": profile["max_volatility_20d_pct"],
                "max_volatility_5d_pct": profile["max_volatility_5d_pct"],
                "risk_penalty_weight": profile["risk_penalty_weight"],
                "max_turnover": profile["max_turnover"],
                "grid_index": idx,
                "is_current_default_profile": is_current_default(profile),
            }
        )
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost", "avg_turnover", "max_drawdown_after_cost"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    result_df["rank_after_cost"] = result_df.index + 1
    result_df["is_best_profile"] = False
    if not result_df.empty:
        result_df.loc[0, "is_best_profile"] = True

    current_default_row = result_df[result_df["is_current_default_profile"]].head(1)
    best = result_df.iloc[0]

    summary_path = output_dir / "inference_local_upgrade_summary.csv"
    top_path = output_dir / "inference_local_upgrade_top.csv"
    result_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    result_df.head(args.top_n_report).to_csv(top_path, index=False, encoding="utf-8-sig")

    report_lines = [
        "# Local Inference Upgrade Search Report",
        "",
        f"- total_profiles: `{len(result_df)}`",
        f"- current_default_profile: `{BEST_PROFILE_NAME}`",
        "",
        "## Best Profile",
        "",
        f"- profile_name: `{best['profile_name']}`",
        f"- cumulative_return_after_cost: `{best['cumulative_return_after_cost']:.6f}`",
        f"- sharpe_after_cost: `{best['sharpe_after_cost']:.6f}`",
        f"- max_drawdown_after_cost: `{best['max_drawdown_after_cost']:.6f}`",
        f"- avg_turnover: `{best['avg_turnover']:.6f}`",
        "",
    ]
    if not current_default_row.empty:
        current = current_default_row.iloc[0]
        report_lines.extend(
            [
                "## Current Default",
                "",
                f"- profile_name: `{current['profile_name']}`",
                f"- cumulative_return_after_cost: `{current['cumulative_return_after_cost']:.6f}`",
                f"- sharpe_after_cost: `{current['sharpe_after_cost']:.6f}`",
                f"- max_drawdown_after_cost: `{current['max_drawdown_after_cost']:.6f}`",
                f"- avg_turnover: `{current['avg_turnover']:.6f}`",
                "",
                "## Best-vs-Current Delta",
                "",
                f"- delta_cumulative_return_after_cost: `{best['cumulative_return_after_cost'] - current['cumulative_return_after_cost']:.6f}`",
                f"- delta_sharpe_after_cost: `{best['sharpe_after_cost'] - current['sharpe_after_cost']:.6f}`",
                f"- delta_avg_turnover: `{best['avg_turnover'] - current['avg_turnover']:.6f}`",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Top Candidates",
            "",
            "| rank | profile_name | candidate_size | vol20_pct | vol5_pct | risk_penalty | max_turnover | cum_after_cost | sharpe_after_cost | avg_turnover | max_drawdown_after_cost |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in result_df.head(args.top_n_report).iterrows():
        report_lines.append(
            f"| {int(row['rank_after_cost'])} | {row['profile_name']} | "
            f"{int(row['primary_candidate_size'])} | {float(row['max_volatility_20d_pct']):.2f} | "
            f"{float(row['max_volatility_5d_pct']):.2f} | {float(row['risk_penalty_weight']):.2f} | "
            f"{float(row['max_turnover']):.2f} | {float(row['cumulative_return_after_cost']):.6f} | "
            f"{float(row['sharpe_after_cost']):.6f} | {float(row['avg_turnover']):.6f} | "
            f"{float(row['max_drawdown_after_cost']):.6f} |"
        )

    report_path = output_dir / "inference_local_upgrade_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(result_df.head(args.top_n_report).to_string(index=False))
    print(f"[local_upgrade_search] wrote {summary_path}")
    print(f"[local_upgrade_search] wrote {top_path}")
    print(f"[local_upgrade_search] wrote {report_path}")


if __name__ == "__main__":
    main()
