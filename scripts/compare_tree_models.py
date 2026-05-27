import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "app" / "code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import backtest as backtest_module
import train as train_module
from config import build_default_inference_args


DEFAULT_FEATURE_PATH = "app/temp/train_features.csv"
DEFAULT_OUTPUT_DIR = "app/model/model_comparison"
DEFAULT_VALID_DATES = 20
DEFAULT_NUM_FOLDS = 3
DEFAULT_TARGET_MODE = "cross_section_rank"
DEFAULT_FEATURE_SET = "base_alpha_v3_rs_crowding_mini4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LightGBM and XGBoost baselines.")
    parser.add_argument("--feature_path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--valid_dates", type=int, default=DEFAULT_VALID_DATES)
    parser.add_argument("--num_folds", type=int, default=DEFAULT_NUM_FOLDS)
    parser.add_argument("--target_mode", default=DEFAULT_TARGET_MODE)
    parser.add_argument("--feature_set", default=DEFAULT_FEATURE_SET)
    return parser.parse_args()


def prepare_prediction_frame(walk_predictions: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    merge_columns = ["stock_id", "date", *backtest_module.MERGE_FEATURE_COLUMNS]
    return walk_predictions.merge(
        feature_df[merge_columns].drop_duplicates(["stock_id", "date"]),
        on=["stock_id", "date"],
        how="left",
        validate="many_to_one",
    )


def make_profile(model_family: str) -> dict:
    defaults = build_default_inference_args()
    return {
        "profile_name": f"model_{model_family}",
        "top_k": defaults["top_k"],
        "primary_candidate_size": defaults["primary_candidate_size"],
        "enable_risk_filters": 1,
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": defaults["max_volatility_20d_pct"],
        "max_volatility_5d_pct": defaults["max_volatility_5d_pct"],
        "turnover_rate_lower_pct": defaults["turnover_rate_lower_pct"],
        "turnover_rate_upper_pct": defaults["turnover_rate_upper_pct"],
        "turnover_ratio_upper_pct": defaults["turnover_ratio_upper_pct"],
        "risk_penalty_weight": defaults["risk_penalty_weight"],
        "sort_strategy": defaults["sort_strategy"],
        "weighting_scheme": defaults["weighting_scheme"],
        "transaction_cost": defaults["transaction_cost"],
        "max_turnover": defaults["max_turnover"],
    }


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_module.set_seed(train_module.SEED)
    feature_df = train_module.load_training_frame(ROOT / args.feature_path)
    feature_df = train_module.add_training_target(feature_df, args.target_mode)
    feature_columns = train_module.resolve_feature_columns(args.feature_set)

    rows = []
    for model_family in ["lightgbm", "xgboost"]:
        fold_metrics, walk_predictions, backend = train_module.run_walk_forward(
            df=feature_df,
            feature_columns=feature_columns,
            valid_dates=args.valid_dates,
            num_folds=args.num_folds,
            model_family=model_family,
        )
        train_summary = train_module.summarise_metrics(fold_metrics)
        prediction_df = prepare_prediction_frame(walk_predictions, feature_df)
        profile = make_profile(model_family)
        backtest_summary_df, _, _ = backtest_module.run_backtest(
            prediction_df=prediction_df,
            config=profile,
            prediction_source=f"walk_forward_{model_family}",
        )
        row = {
            "model_family": model_family,
            "backend": backend,
            "feature_set": args.feature_set,
            "feature_count": len(feature_columns),
            "target_mode": args.target_mode,
            **train_summary,
            **backtest_summary_df.iloc[0].to_dict(),
        }
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost", "rank_ic_mean"],
        ascending=[False, False, False],
    )
    summary_path = output_dir / "tree_model_comparison.csv"
    result_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    report_lines = [
        "# Tree Model Comparison",
        "",
        "| model_family | rank_ic_mean | top5_mean_return_mean | cum_after_cost | sharpe_after_cost | max_dd_after_cost | avg_turnover |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in result_df.iterrows():
        report_lines.append(
            f"| {row['model_family']} | {float(row['rank_ic_mean']):.6f} | "
            f"{float(row['top5_mean_return_mean']):.6f} | {float(row['cumulative_return_after_cost']):.6f} | "
            f"{float(row['sharpe_after_cost']):.6f} | {float(row['max_drawdown_after_cost']):.6f} | "
            f"{float(row['avg_turnover']):.6f} |"
        )
    report_path = output_dir / "tree_model_comparison.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(result_df.to_string(index=False))
    print(f"[model_compare] wrote {summary_path}")
    print(f"[model_compare] wrote {report_path}")


if __name__ == "__main__":
    main()
