import argparse
import itertools
from pathlib import Path

import pandas as pd

import backtest as backtest_module
from config import BEST_CONFIG, BEST_PROFILE_NAME, ROOT_DIR


DEFAULT_PREDICTION_PATH = "app/model/walk_forward_predictions.csv"
DEFAULT_FEATURE_PATH = "app/temp/train_features.csv"
DEFAULT_MODEL_DIR = "app/model"
DEFAULT_OUTPUT_DIR = "app/model/backtest_stress_test"
DEFAULT_TRANSACTION_COSTS = [0.0, 0.001, 0.002, 0.003, 0.005]
DEFAULT_MAX_TURNOVERS = [0.50, 0.75, 1.00]
DEFAULT_WEIGHT_CAPS = [0.16, 0.18, 0.20]


def parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one numeric value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extreme backtest stress tests across cost, turnover, and weight-cap grids."
    )
    parser.add_argument("--prediction_path", default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--feature_path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--transaction_costs",
        type=parse_float_list,
        default=DEFAULT_TRANSACTION_COSTS,
        help="Comma-separated transaction_cost grid, e.g. 0,0.001,0.002,0.003,0.005",
    )
    parser.add_argument(
        "--max_turnovers",
        type=parse_float_list,
        default=DEFAULT_MAX_TURNOVERS,
        help="Comma-separated max_turnover grid, e.g. 0.5,0.75,1.0",
    )
    parser.add_argument(
        "--weight_caps",
        type=parse_float_list,
        default=DEFAULT_WEIGHT_CAPS,
        help="Comma-separated max_single_weight grid, e.g. 0.16,0.18,0.20",
    )
    parser.add_argument(
        "--write_daily",
        type=int,
        choices=[0, 1],
        default=0,
        help="Write combined daily backtest results for all stress scenarios.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def build_base_profile() -> dict:
    selection = BEST_CONFIG["selection"]
    risk = BEST_CONFIG["risk_filter_thresholds"]
    execution = BEST_CONFIG["execution"]
    return {
        "top_k": int(selection["top_k"]),
        "primary_candidate_size": int(selection["primary_candidate_size"]),
        "enable_risk_filters": int(selection.get("enable_risk_filters", 1)),
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": float(risk["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(risk["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(risk["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(risk["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(risk["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(risk["risk_penalty_weight"]),
        "weighting_scheme": str(selection["weighting_scheme"]),
        "weight_blend_alpha": float(selection.get("weight_blend_alpha", 1.0)),
        "max_single_weight": float(selection.get("max_single_weight", 1.0)),
        "sort_strategy": str(selection["sort_strategy"]),
        "transaction_cost": float(execution["transaction_cost"]),
        "max_turnover": float(execution["max_turnover"]),
        "rerank_signal_column": None,
        "rerank_signal_weight": 0.0,
        "secondary_candidate_size": 0,
        "secondary_screen_mode": "none",
        "secondary_screen_weight": 0.0,
        "local_tiebreak_start_rank": 8,
        "local_tiebreak_end_rank": 15,
    }


def build_profiles(
    transaction_costs: list[float],
    max_turnovers: list[float],
    weight_caps: list[float],
) -> list[dict]:
    base = build_base_profile()
    profiles = []
    for transaction_cost, max_turnover, weight_cap in itertools.product(
        transaction_costs, max_turnovers, weight_caps
    ):
        profiles.append(
            {
                **base,
                "profile_name": (
                    f"tc{int(round(transaction_cost * 10000)):04d}_"
                    f"mt{int(round(max_turnover * 100)):03d}_"
                    f"wc{int(round(weight_cap * 100)):02d}"
                ),
                "transaction_cost": float(transaction_cost),
                "max_turnover": float(max_turnover),
                "max_single_weight": float(weight_cap),
            }
        )
    return profiles


def add_stress_labels(result_df: pd.DataFrame) -> pd.DataFrame:
    out = result_df.copy()
    out["weight_cap"] = out["max_single_weight_param"].astype(float)
    out["is_positive_after_cost"] = out["cumulative_return_after_cost"] > 0.0
    out["cost_environment_positive_rate"] = out.groupby("transaction_cost")[
        "is_positive_after_cost"
    ].transform("mean")
    out["stress_rank"] = (
        out.sort_values(
            [
                "transaction_cost",
                "cumulative_return_after_cost",
                "sharpe_after_cost",
                "max_drawdown_after_cost",
                "avg_turnover",
            ],
            ascending=[False, False, False, False, True],
        )
        .reset_index()
        .reset_index()
        .set_index("index")["level_0"]
        + 1
    )
    return out


def write_report(result_df: pd.DataFrame, output_dir: Path) -> Path:
    report_path = output_dir / "backtest_stress_report.md"
    high_cost = float(result_df["transaction_cost"].max()) if not result_df.empty else 0.0
    high_cost_slice = result_df[result_df["transaction_cost"].sub(high_cost).abs() < 1e-12]
    high_cost_positive = bool(high_cost_slice["is_positive_after_cost"].any()) if not high_cost_slice.empty else False

    best_by_cost = (
        result_df.sort_values(
            ["transaction_cost", "cumulative_return_after_cost", "sharpe_after_cost", "avg_turnover"],
            ascending=[True, False, False, True],
        )
        .groupby("transaction_cost", as_index=False)
        .head(1)
        .sort_values("transaction_cost")
    )

    lines = [
        "# Backtest Stress Test",
        "",
        f"- current_default_profile: `{BEST_PROFILE_NAME}`",
        f"- transaction_cost_grid: `{', '.join(f'{v:.3f}' for v in sorted(result_df['transaction_cost'].unique()))}`",
        f"- max_turnover_grid: `{', '.join(f'{v:.2f}' for v in sorted(result_df['max_turnover'].unique()))}`",
        f"- weight_cap_grid: `{', '.join(f'{v:.2f}' for v in sorted(result_df['weight_cap'].unique()))}`",
        f"- total_profiles: `{len(result_df)}`",
        "",
        "## Cost Robustness Answer",
        "",
    ]
    if high_cost_positive:
        best_high = high_cost_slice.sort_values(
            ["cumulative_return_after_cost", "sharpe_after_cost", "avg_turnover"],
            ascending=[False, False, True],
        ).iloc[0]
        lines.append(
            f"- At transaction_cost `{high_cost:.3f}`, at least one stressed profile remains positive: "
            f"`{float(best_high['cumulative_return_after_cost']):.6f}` after cost."
        )
    else:
        lines.append(
            f"- At transaction_cost `{high_cost:.3f}`, no stressed profile remains positive after cost."
        )

    lines.extend(
        [
            "",
            "## Best Profile By Cost",
            "",
            "| transaction_cost | profile_name | max_turnover | weight_cap | return | Sharpe | max_drawdown | win_rate |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in best_by_cost.iterrows():
        lines.append(
            f"| {float(row['transaction_cost']):.3f} | {row['profile_name']} | "
            f"{float(row['max_turnover']):.2f} | {float(row['weight_cap']):.2f} | "
            f"{float(row['cumulative_return_after_cost']):.6f} | "
            f"{float(row['sharpe_after_cost']):.6f} | "
            f"{float(row['max_drawdown_after_cost']):.6f} | "
            f"{float(row['win_rate_after_cost']):.6f} |"
        )

    lines.extend(["", "Reference CSV: `backtest_stress_summary.csv`", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_df, prediction_source = backtest_module.load_or_generate_predictions(
        prediction_path=resolve_path(args.prediction_path),
        feature_path=resolve_path(args.feature_path),
        model_dir=resolve_path(args.model_dir),
    )

    rows = []
    daily_frames = []
    profiles = build_profiles(args.transaction_costs, args.max_turnovers, args.weight_caps)
    for grid_index, profile in enumerate(profiles, start=1):
        summary_df, daily_df, _ = backtest_module.run_backtest(
            prediction_df=prediction_df,
            config=profile,
            prediction_source=prediction_source,
        )
        row = summary_df.iloc[0].to_dict()
        row["grid_index"] = grid_index
        rows.append(row)
        if args.write_daily:
            daily_frames.append(daily_df)

    result_df = pd.DataFrame(rows)
    result_df = add_stress_labels(result_df)
    result_df = result_df.sort_values(
        ["transaction_cost", "max_turnover", "weight_cap"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    summary_path = output_dir / "backtest_stress_summary.csv"
    result_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path = write_report(result_df, output_dir)

    if args.write_daily and daily_frames:
        daily_path = output_dir / "backtest_stress_daily_results.csv"
        pd.concat(daily_frames, ignore_index=True).to_csv(daily_path, index=False, encoding="utf-8-sig")
        print(f"[backtest_stress_test] wrote {daily_path}")

    print(
        result_df[
            [
                "profile_name",
                "transaction_cost",
                "max_turnover",
                "weight_cap",
                "cumulative_return_after_cost",
                "sharpe_after_cost",
                "max_drawdown_after_cost",
                "win_rate_after_cost",
                "is_positive_after_cost",
            ]
        ].to_string(index=False)
    )
    print(f"[backtest_stress_test] prediction_source={prediction_source}")
    print(f"[backtest_stress_test] profiles_run={len(result_df)}")
    print(f"[backtest_stress_test] wrote {summary_path}")
    print(f"[backtest_stress_test] wrote {report_path}")


if __name__ == "__main__":
    main()
