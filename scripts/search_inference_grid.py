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
from config import BEST_PROFILE_NAME, BEST_CONFIG, build_default_inference_args


DEFAULT_PREDICTION_PATH = "app/model/walk_forward_predictions.csv"
DEFAULT_FEATURE_PATH = "app/temp/train_features.csv"
DEFAULT_MODEL_DIR = "app/model"
DEFAULT_OUTPUT_DIR = "app/model/inference_grid_search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-grained inference grid search.")
    parser.add_argument("--prediction_path", default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--feature_path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_n_report", type=int, default=20)
    return parser.parse_args()


def build_grid() -> list[dict]:
    defaults = build_default_inference_args()

    candidate_sizes = [35, 40, 45]
    vol20_pcts = [0.88, 0.90]
    vol5_pcts = [0.88, 0.90]
    turnover_low_pcts = [0.02, 0.03]
    turnover_high_pcts = [0.95]
    turnover_ratio_high_pcts = [0.93, 0.95]
    risk_penalty_weights = [0.20, 0.25]
    max_turnovers = [0.45, 0.50, 0.55]

    grid = []
    for (
        candidate_size,
        vol20,
        vol5,
        turnover_low,
        turnover_high,
        turnover_ratio_high,
        risk_penalty_weight,
        max_turnover,
    ) in itertools.product(
        candidate_sizes,
        vol20_pcts,
        vol5_pcts,
        turnover_low_pcts,
        turnover_high_pcts,
        turnover_ratio_high_pcts,
        risk_penalty_weights,
        max_turnovers,
    ):
        grid.append(
            {
                "profile_name": (
                    f"grid_cs{candidate_size}_v20{int(vol20*100)}_v5{int(vol5*100)}_"
                    f"tl{int(turnover_low*100)}_th{int(turnover_high*100)}_"
                    f"tr{int(turnover_ratio_high*100)}_rp{int(risk_penalty_weight*100)}_"
                    f"mt{int(max_turnover*100)}"
                ),
                "top_k": defaults["top_k"],
                "primary_candidate_size": candidate_size,
                "enable_risk_filters": 1,
                "allow_cash_fallback": 0,
                "max_volatility_20d_pct": vol20,
                "max_volatility_5d_pct": vol5,
                "turnover_rate_lower_pct": turnover_low,
                "turnover_rate_upper_pct": turnover_high,
                "turnover_ratio_upper_pct": turnover_ratio_high,
                "risk_penalty_weight": risk_penalty_weight,
                "weighting_scheme": BEST_CONFIG["selection"]["weighting_scheme"],
                "sort_strategy": BEST_CONFIG["selection"]["sort_strategy"],
                "transaction_cost": defaults["transaction_cost"],
                "max_turnover": max_turnover,
            }
        )
    return grid


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
    grid = build_grid()
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
                "turnover_rate_lower_pct": profile["turnover_rate_lower_pct"],
                "turnover_rate_upper_pct": profile["turnover_rate_upper_pct"],
                "turnover_ratio_upper_pct": profile["turnover_ratio_upper_pct"],
                "risk_penalty_weight": profile["risk_penalty_weight"],
            }
        )
        row["grid_index"] = idx
        row["is_current_default_profile"] = (
            profile["primary_candidate_size"] == BEST_CONFIG["selection"]["primary_candidate_size"]
            and abs(profile["max_volatility_20d_pct"] - BEST_CONFIG["risk_filter_thresholds"]["max_volatility_20d_pct"]) < 1e-12
            and abs(profile["max_volatility_5d_pct"] - BEST_CONFIG["risk_filter_thresholds"]["max_volatility_5d_pct"]) < 1e-12
            and abs(profile["turnover_rate_lower_pct"] - BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_lower_pct"]) < 1e-12
            and abs(profile["turnover_rate_upper_pct"] - BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_upper_pct"]) < 1e-12
            and abs(profile["turnover_ratio_upper_pct"] - BEST_CONFIG["risk_filter_thresholds"]["turnover_ratio_upper_pct"]) < 1e-12
            and abs(profile["risk_penalty_weight"] - BEST_CONFIG["risk_filter_thresholds"]["risk_penalty_weight"]) < 1e-12
            and abs(profile["max_turnover"] - BEST_CONFIG["execution"]["max_turnover"]) < 1e-12
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

    summary_path = output_dir / "inference_grid_search_summary.csv"
    top_path = output_dir / "inference_grid_search_top.csv"
    result_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    result_df.head(args.top_n_report).to_csv(top_path, index=False, encoding="utf-8-sig")

    best = result_df.iloc[0]
    report_lines = [
        "# Inference Grid Search Report",
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
        "## Top Candidates",
        "",
        "| rank | profile_name | candidate_size | vol20_pct | vol5_pct | turnover_low | turnover_high | turnover_ratio_high | risk_penalty | max_turnover | cum_after_cost | sharpe_after_cost | avg_turnover |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in result_df.head(args.top_n_report).iterrows():
        report_lines.append(
            f"| {int(row['rank_after_cost'])} | {row['profile_name']} | "
            f"{int(row['primary_candidate_size'])} | {float(row['max_volatility_20d_pct']):.2f} | "
            f"{float(row['max_volatility_5d_pct']):.2f} | {float(row['turnover_rate_lower_pct']):.2f} | "
            f"{float(row['turnover_rate_upper_pct']):.2f} | {float(row['turnover_ratio_upper_pct']):.2f} | "
            f"{float(row['risk_penalty_weight']):.2f} | {float(row['max_turnover']):.2f} | "
            f"{float(row['cumulative_return_after_cost']):.6f} | {float(row['sharpe_after_cost']):.6f} | "
            f"{float(row['avg_turnover']):.6f} |"
        )

    report_path = output_dir / "inference_grid_search_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(result_df.head(args.top_n_report).to_string(index=False))
    print(f"[grid_search] wrote {summary_path}")
    print(f"[grid_search] wrote {top_path}")
    print(f"[grid_search] wrote {report_path}")


if __name__ == "__main__":
    main()
