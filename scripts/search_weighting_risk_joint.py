import argparse
import itertools
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
DEFAULT_OUTPUT_DIR = "app/model/weighting_risk_joint_search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint search for weighting scheme and risk penalty.")
    parser.add_argument("--prediction_path", default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--feature_path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_n_report", type=int, default=20)
    return parser.parse_args()


def build_grid() -> list[dict]:
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
        "sort_strategy": BEST_CONFIG["selection"]["sort_strategy"],
        "transaction_cost": float(BEST_CONFIG["execution"]["transaction_cost"]),
        "max_turnover": float(BEST_CONFIG["execution"]["max_turnover"]),
    }

    weighting_schemes = ["equal", "pred", "risk_adjusted"]
    risk_penalty_weights = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]

    grid = []
    for weighting_scheme, risk_penalty_weight in itertools.product(weighting_schemes, risk_penalty_weights):
        grid.append(
            {
                **base,
                "profile_name": f"wr_{weighting_scheme}_rp{int(risk_penalty_weight * 100):02d}",
                "weighting_scheme": weighting_scheme,
                "risk_penalty_weight": risk_penalty_weight,
            }
        )
    return grid


def is_current_default(profile: dict) -> bool:
    return (
        profile["weighting_scheme"] == BEST_CONFIG["selection"]["weighting_scheme"]
        and abs(profile["risk_penalty_weight"] - BEST_CONFIG["risk_filter_thresholds"]["risk_penalty_weight"]) < 1e-12
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
                "grid_index": idx,
                "weighting_scheme": profile["weighting_scheme"],
                "risk_penalty_weight": profile["risk_penalty_weight"],
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

    best = result_df.iloc[0]
    current_default_row = result_df[result_df["is_current_default_profile"]].head(1)

    summary_path = output_dir / "weighting_risk_joint_summary.csv"
    top_path = output_dir / "weighting_risk_joint_top.csv"
    result_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    result_df.head(args.top_n_report).to_csv(top_path, index=False, encoding="utf-8-sig")

    best_candidate = {
        "profile_name": str(best["profile_name"]),
        "feature_set": BEST_CONFIG["training"]["feature_set"],
        "sort_strategy": BEST_CONFIG["selection"]["sort_strategy"],
        "weighting_scheme": str(best["weighting_scheme"]),
        "risk_penalty_weight": float(best["risk_penalty_weight"]),
        "primary_candidate_size": int(BEST_CONFIG["selection"]["primary_candidate_size"]),
        "max_volatility_20d_pct": float(BEST_CONFIG["risk_filter_thresholds"]["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(BEST_CONFIG["risk_filter_thresholds"]["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_ratio_upper_pct"]),
        "max_turnover": float(BEST_CONFIG["execution"]["max_turnover"]),
        "transaction_cost": float(BEST_CONFIG["execution"]["transaction_cost"]),
        "cumulative_return_after_cost": float(best["cumulative_return_after_cost"]),
        "sharpe_after_cost": float(best["sharpe_after_cost"]),
        "max_drawdown_after_cost": float(best["max_drawdown_after_cost"]),
        "avg_turnover": float(best["avg_turnover"]),
    }
    best_candidate_path = output_dir / "weighting_risk_best_candidate.json"
    best_candidate_path.write_text(json.dumps(best_candidate, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Weighting And Risk Penalty Joint Search",
        "",
        f"- total_profiles: `{len(result_df)}`",
        f"- current_default_profile: `{BEST_PROFILE_NAME}`",
        "",
        "## Best Profile",
        "",
        f"- profile_name: `{best['profile_name']}`",
        f"- weighting_scheme: `{best['weighting_scheme']}`",
        f"- risk_penalty_weight: `{float(best['risk_penalty_weight']):.2f}`",
        f"- cumulative_return_after_cost: `{float(best['cumulative_return_after_cost']):.6f}`",
        f"- sharpe_after_cost: `{float(best['sharpe_after_cost']):.6f}`",
        f"- max_drawdown_after_cost: `{float(best['max_drawdown_after_cost']):.6f}`",
        f"- avg_turnover: `{float(best['avg_turnover']):.6f}`",
        "",
    ]

    if not current_default_row.empty:
        current = current_default_row.iloc[0]
        report_lines.extend(
            [
                "## Current Default Slice",
                "",
                f"- profile_name: `{current['profile_name']}`",
                f"- weighting_scheme: `{current['weighting_scheme']}`",
                f"- risk_penalty_weight: `{float(current['risk_penalty_weight']):.2f}`",
                f"- cumulative_return_after_cost: `{float(current['cumulative_return_after_cost']):.6f}`",
                f"- sharpe_after_cost: `{float(current['sharpe_after_cost']):.6f}`",
                "",
                "## Best-vs-Current Delta",
                "",
                f"- delta_cumulative_return_after_cost: `{float(best['cumulative_return_after_cost']) - float(current['cumulative_return_after_cost']):.6f}`",
                f"- delta_sharpe_after_cost: `{float(best['sharpe_after_cost']) - float(current['sharpe_after_cost']):.6f}`",
                f"- delta_max_drawdown_after_cost: `{float(best['max_drawdown_after_cost']) - float(current['max_drawdown_after_cost']):.6f}`",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Top Candidates",
            "",
            "| rank | profile_name | weighting_scheme | risk_penalty | cum_after_cost | sharpe_after_cost | max_drawdown_after_cost | avg_turnover |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in result_df.head(args.top_n_report).iterrows():
        report_lines.append(
            f"| {int(row['rank_after_cost'])} | {row['profile_name']} | {row['weighting_scheme']} | "
            f"{float(row['risk_penalty_weight']):.2f} | {float(row['cumulative_return_after_cost']):.6f} | "
            f"{float(row['sharpe_after_cost']):.6f} | {float(row['max_drawdown_after_cost']):.6f} | "
            f"{float(row['avg_turnover']):.6f} |"
        )

    report_path = output_dir / "weighting_risk_joint_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(result_df.head(args.top_n_report).to_string(index=False))
    print(f"[weighting_risk_joint] wrote {summary_path}")
    print(f"[weighting_risk_joint] wrote {top_path}")
    print(f"[weighting_risk_joint] wrote {best_candidate_path}")
    print(f"[weighting_risk_joint] wrote {report_path}")


if __name__ == "__main__":
    main()
