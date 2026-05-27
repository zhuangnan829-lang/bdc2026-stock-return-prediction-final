import argparse
import json
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
DEFAULT_OUTPUT_DIR = "app/model/turnover_pred_local_search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local turnover search around pred weighting + risk_adjusted sort.")
    parser.add_argument("--prediction_path", default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--feature_path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_profiles() -> list[dict]:
    base = {
        "top_k": int(BEST_CONFIG["selection"]["top_k"]),
        "primary_candidate_size": int(BEST_CONFIG["selection"]["primary_candidate_size"]),
        "enable_risk_filters": 1,
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": float(BEST_CONFIG["risk_filter_thresholds"]["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(BEST_CONFIG["risk_filter_thresholds"]["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(BEST_CONFIG["risk_filter_thresholds"]["risk_penalty_weight"]),
        "sort_strategy": "risk_adjusted",
        "weighting_scheme": "pred",
        "transaction_cost": float(BEST_CONFIG["execution"]["transaction_cost"]),
    }
    turnovers = [0.68, 0.70, 0.72, 0.74, 0.76]
    profiles = []
    for turnover in turnovers:
        profiles.append(
            {
                **base,
                "profile_name": f"turnover_pred_mt{int(turnover * 100):02d}",
                "max_turnover": turnover,
            }
        )
    return profiles


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
    for idx, profile in enumerate(build_profiles(), start=1):
        summary_df, _, _ = backtest_module.run_backtest(
            prediction_df=prediction_df,
            config=profile,
            prediction_source=prediction_source,
        )
        row = summary_df.iloc[0].to_dict()
        row["grid_index"] = idx
        row["is_current_default_profile"] = abs(profile["max_turnover"] - float(BEST_CONFIG["execution"]["max_turnover"])) < 1e-12
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost", "avg_turnover", "max_drawdown_after_cost"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    result_df["rank_after_cost"] = result_df.index + 1
    result_df["is_best_profile"] = False
    if not result_df.empty:
        result_df.loc[0, "is_best_profile"] = True

    summary_path = output_dir / "turnover_pred_local_summary.csv"
    result_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    best = result_df.iloc[0]
    current = result_df[result_df["is_current_default_profile"]].head(1)

    best_candidate = {
        "profile_name": str(best["profile_name"]),
        "sort_strategy": "risk_adjusted",
        "weighting_scheme": "pred",
        "max_turnover": float(best["max_turnover"]),
        "cumulative_return_after_cost": float(best["cumulative_return_after_cost"]),
        "sharpe_after_cost": float(best["sharpe_after_cost"]),
        "max_drawdown_after_cost": float(best["max_drawdown_after_cost"]),
        "avg_turnover": float(best["avg_turnover"]),
    }
    (output_dir / "turnover_pred_local_best_candidate.json").write_text(
        json.dumps(best_candidate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Turnover Local Search",
        "",
        f"- current_default_profile: `{BEST_PROFILE_NAME}`",
        "",
        "| rank | profile_name | max_turnover | cum_after_cost | sharpe_after_cost | max_drawdown_after_cost | avg_turnover |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in result_df.iterrows():
        report_lines.append(
            f"| {int(row['rank_after_cost'])} | {row['profile_name']} | {float(row['max_turnover']):.2f} | "
            f"{float(row['cumulative_return_after_cost']):.6f} | {float(row['sharpe_after_cost']):.6f} | "
            f"{float(row['max_drawdown_after_cost']):.6f} | {float(row['avg_turnover']):.6f} |"
        )
    if not current.empty:
        current_row = current.iloc[0]
        report_lines.extend(
            [
                "",
                "## Best-vs-Current Delta",
                "",
                f"- delta_cumulative_return_after_cost: `{float(best['cumulative_return_after_cost']) - float(current_row['cumulative_return_after_cost']):.6f}`",
                f"- delta_sharpe_after_cost: `{float(best['sharpe_after_cost']) - float(current_row['sharpe_after_cost']):.6f}`",
                f"- delta_max_drawdown_after_cost: `{float(best['max_drawdown_after_cost']) - float(current_row['max_drawdown_after_cost']):.6f}`",
            ]
        )
    (output_dir / "turnover_pred_local_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(result_df.to_string(index=False))
    print(f"[turnover_pred_local] wrote {summary_path}")


if __name__ == "__main__":
    main()
