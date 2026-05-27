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
from config import BEST_CONFIG
from evaluate_rank_stability import append_stability_summary, summarize_prediction_rank_stability


DEFAULT_FEATURE_PATH = "app/temp/train_features.csv"
DEFAULT_OUTPUT_DIR = "app/model/ablation"
DEFAULT_TRANSACTION_COST = 0.001
DEFAULT_VALID_DATES = 20
DEFAULT_NUM_FOLDS = 3
DEFAULT_TARGET_MODE = "cross_section_rank"


BASE_PROFILE = {
    "profile_name": "baseline",
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
    "sort_strategy": BEST_CONFIG["selection"]["sort_strategy"],
    "weighting_scheme": BEST_CONFIG["selection"]["weighting_scheme"],
    "transaction_cost": float(BEST_CONFIG["execution"]["transaction_cost"]),
    "max_turnover": float(BEST_CONFIG["execution"]["max_turnover"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run systematic ablation study.")
    parser.add_argument("--feature_path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--valid_dates", type=int, default=DEFAULT_VALID_DATES)
    parser.add_argument("--num_folds", type=int, default=DEFAULT_NUM_FOLDS)
    parser.add_argument(
        "--target_mode",
        choices=["raw_return", "cross_section_zscore", "cross_section_rank"],
        default=DEFAULT_TARGET_MODE,
    )
    parser.add_argument("--transaction_cost", type=float, default=DEFAULT_TRANSACTION_COST)
    return parser.parse_args()


def prepare_prediction_frame(walk_predictions: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    merge_columns = ["stock_id", "date", *backtest_module.MERGE_FEATURE_COLUMNS]
    return walk_predictions.merge(
        feature_df[merge_columns].drop_duplicates(["stock_id", "date"]),
        on=["stock_id", "date"],
        how="left",
        validate="many_to_one",
    )


def run_training_for_feature_set(
    feature_df: pd.DataFrame,
    feature_set: str,
    valid_dates: int,
    num_folds: int,
    target_mode: str,
) -> tuple[pd.DataFrame, dict, str, list[str]]:
    df = train_module.add_training_target(feature_df, target_mode)
    feature_columns = train_module.resolve_feature_columns(feature_set)
    fold_metrics, walk_predictions, backend = train_module.run_walk_forward(
        df=df,
        feature_columns=feature_columns,
        valid_dates=valid_dates,
        num_folds=num_folds,
    )
    summary = train_module.summarise_metrics(fold_metrics)
    walk_predictions = prepare_prediction_frame(walk_predictions, feature_df)
    return walk_predictions, summary, backend, feature_columns


def run_backtest_summary(prediction_df: pd.DataFrame, profile: dict, prediction_source: str) -> dict:
    summary_df, _, _ = backtest_module.run_backtest(
        prediction_df=prediction_df,
        config=profile,
        prediction_source=prediction_source,
    )
    return summary_df.iloc[0].to_dict()


def summarize_ablation_stability(prediction_df: pd.DataFrame, experiment_name: str, extra_fields: dict) -> dict:
    return summarize_prediction_rank_stability(
        prediction_df=prediction_df,
        experiment_name=experiment_name,
        extra_fields=extra_fields,
    )


def main() -> None:
    args = parse_args()
    feature_path = ROOT / args.feature_path
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_module.set_seed(train_module.SEED)
    feature_df = train_module.load_training_frame(feature_path)

    rows: list[dict] = []

    # 1) Feature ablation: retrain per feature set.
    feature_variants = [
        ("base", "base"),
        ("base_technical", "base_technical"),
        ("base_technical_risk", "base_technical_risk"),
        ("base_technical_risk_alpha", "base_technical_risk_alpha"),
    ]
    full_prediction_df = None
    for variant_name, feature_set in feature_variants:
        walk_predictions, train_summary, backend, feature_columns = run_training_for_feature_set(
            feature_df=feature_df,
            feature_set=feature_set,
            valid_dates=args.valid_dates,
            num_folds=args.num_folds,
            target_mode=args.target_mode,
        )
        if feature_set == BEST_CONFIG["training"]["feature_set"]:
            full_prediction_df = walk_predictions.copy()

        profile = dict(BASE_PROFILE)
        profile["profile_name"] = f"feature_{variant_name}"
        profile["transaction_cost"] = args.transaction_cost
        backtest_summary = run_backtest_summary(
            prediction_df=walk_predictions,
            profile=profile,
            prediction_source=f"walk_forward_{feature_set}",
        )
        stability_summary = summarize_ablation_stability(
            prediction_df=walk_predictions,
            experiment_name=f"ablation/feature_{variant_name}",
            extra_fields={
                "ablation_type": "feature",
                "variant": variant_name,
                "feature_set": feature_set,
            },
        )
        append_stability_summary({**stability_summary, **backtest_summary})
        rows.append(
            {
                "ablation_type": "feature",
                "variant": variant_name,
                "feature_set": feature_set,
                "feature_count": len(feature_columns),
                "sort_strategy": profile["sort_strategy"],
                "weighting_scheme": profile["weighting_scheme"],
                "max_turnover": profile["max_turnover"],
                "backend": backend,
                **train_summary,
                "worst_fold_rank_ic": stability_summary["worst_fold_rank_ic"],
                "negative_day_rank_ic_ratio": stability_summary["negative_day_rank_ic_ratio"],
                **backtest_summary,
            }
        )

    if full_prediction_df is None:
        raise RuntimeError("Failed to prepare full-feature prediction frame for non-feature ablations")

    # 2) Sort strategy ablation.
    for sort_strategy in ["pure_prediction", "risk_adjusted"]:
        profile = dict(BASE_PROFILE)
        profile["profile_name"] = f"sort_{sort_strategy}"
        profile["sort_strategy"] = sort_strategy
        profile["transaction_cost"] = args.transaction_cost
        backtest_summary = run_backtest_summary(
            prediction_df=full_prediction_df,
            profile=profile,
            prediction_source=f"walk_forward_{BEST_CONFIG['training']['feature_set']}",
        )
        stability_summary = summarize_ablation_stability(
            prediction_df=full_prediction_df,
            experiment_name=f"ablation/sort_{sort_strategy}",
            extra_fields={
                "ablation_type": "sort_strategy",
                "variant": sort_strategy,
                "feature_set": BEST_CONFIG["training"]["feature_set"],
                "sort_strategy": sort_strategy,
            },
        )
        append_stability_summary({**stability_summary, **backtest_summary})
        rows.append(
            {
                "ablation_type": "sort_strategy",
                "variant": sort_strategy,
                "feature_set": BEST_CONFIG["training"]["feature_set"],
                "feature_count": len(train_module.resolve_feature_columns(BEST_CONFIG["training"]["feature_set"])),
                "sort_strategy": sort_strategy,
                "weighting_scheme": profile["weighting_scheme"],
                "max_turnover": profile["max_turnover"],
                "backend": "reused_full_feature_predictions",
                "rank_ic_mean": stability_summary["rank_ic_mean"],
                "rank_ic_std": stability_summary["rank_ic_std"],
                "worst_fold_rank_ic": stability_summary["worst_fold_rank_ic"],
                "negative_day_rank_ic_ratio": stability_summary["negative_day_rank_ic_ratio"],
                **backtest_summary,
            }
        )

    # 3) Weighting ablation.
    for weighting_scheme in ["equal", "pred", "risk_adjusted"]:
        profile = dict(BASE_PROFILE)
        profile["profile_name"] = f"weight_{weighting_scheme}"
        profile["weighting_scheme"] = weighting_scheme
        profile["transaction_cost"] = args.transaction_cost
        backtest_summary = run_backtest_summary(
            prediction_df=full_prediction_df,
            profile=profile,
            prediction_source=f"walk_forward_{BEST_CONFIG['training']['feature_set']}",
        )
        stability_summary = summarize_ablation_stability(
            prediction_df=full_prediction_df,
            experiment_name=f"ablation/weight_{weighting_scheme}",
            extra_fields={
                "ablation_type": "weighting",
                "variant": weighting_scheme,
                "feature_set": BEST_CONFIG["training"]["feature_set"],
                "weighting_scheme": weighting_scheme,
            },
        )
        append_stability_summary({**stability_summary, **backtest_summary})
        rows.append(
            {
                "ablation_type": "weighting",
                "variant": weighting_scheme,
                "feature_set": BEST_CONFIG["training"]["feature_set"],
                "feature_count": len(train_module.resolve_feature_columns(BEST_CONFIG["training"]["feature_set"])),
                "sort_strategy": profile["sort_strategy"],
                "weighting_scheme": weighting_scheme,
                "max_turnover": profile["max_turnover"],
                "backend": "reused_full_feature_predictions",
                "rank_ic_mean": stability_summary["rank_ic_mean"],
                "rank_ic_std": stability_summary["rank_ic_std"],
                "worst_fold_rank_ic": stability_summary["worst_fold_rank_ic"],
                "negative_day_rank_ic_ratio": stability_summary["negative_day_rank_ic_ratio"],
                **backtest_summary,
            }
        )

    # 4) Turnover ablation.
    turnover_values = sorted({1.0, 0.75, 0.5, float(BEST_CONFIG["execution"]["max_turnover"])}, reverse=True)
    for max_turnover in turnover_values:
        profile = dict(BASE_PROFILE)
        profile["profile_name"] = f"turnover_{str(max_turnover).replace('.', '_')}"
        profile["max_turnover"] = max_turnover
        profile["transaction_cost"] = args.transaction_cost
        backtest_summary = run_backtest_summary(
            prediction_df=full_prediction_df,
            profile=profile,
            prediction_source=f"walk_forward_{BEST_CONFIG['training']['feature_set']}",
        )
        stability_summary = summarize_ablation_stability(
            prediction_df=full_prediction_df,
            experiment_name=f"ablation/turnover_{str(max_turnover).replace('.', '_')}",
            extra_fields={
                "ablation_type": "turnover",
                "variant": f"max_turnover_{max_turnover}",
                "feature_set": BEST_CONFIG["training"]["feature_set"],
                "max_turnover": max_turnover,
            },
        )
        append_stability_summary({**stability_summary, **backtest_summary})
        rows.append(
            {
                "ablation_type": "turnover",
                "variant": f"max_turnover_{max_turnover}",
                "feature_set": BEST_CONFIG["training"]["feature_set"],
                "feature_count": len(train_module.resolve_feature_columns(BEST_CONFIG["training"]["feature_set"])),
                "sort_strategy": profile["sort_strategy"],
                "weighting_scheme": profile["weighting_scheme"],
                "max_turnover": max_turnover,
                "backend": "reused_full_feature_predictions",
                "rank_ic_mean": stability_summary["rank_ic_mean"],
                "rank_ic_std": stability_summary["rank_ic_std"],
                "worst_fold_rank_ic": stability_summary["worst_fold_rank_ic"],
                "negative_day_rank_ic_ratio": stability_summary["negative_day_rank_ic_ratio"],
                **backtest_summary,
            }
        )

    result_df = pd.DataFrame(rows)
    sort_columns = [
        "ablation_type",
        "variant",
        "feature_set",
        "feature_count",
        "backend",
        "rank_ic_mean",
        "rank_ic_std",
        "worst_fold_rank_ic",
        "negative_day_rank_ic_ratio",
        "top5_mean_return_mean",
        "cumulative_return_after_cost",
        "sharpe_after_cost",
        "max_drawdown_after_cost",
        "avg_turnover",
        "avg_execution_strength",
        "sort_strategy",
        "weighting_scheme",
        "max_turnover",
    ]
    for column in sort_columns:
        if column not in result_df.columns:
            result_df[column] = pd.NA

    summary_path = output_dir / "ablation_summary.csv"
    result_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    best_by_group = (
        result_df.sort_values(
            [
                "ablation_type",
                "worst_fold_rank_ic",
                "negative_day_rank_ic_ratio",
                "cumulative_return_after_cost",
                "sharpe_after_cost",
            ],
            ascending=[True, False, True, False, False],
        )
        .groupby("ablation_type", as_index=False)
        .head(1)
    )
    best_path = output_dir / "ablation_best_by_group.csv"
    best_by_group.to_csv(best_path, index=False, encoding="utf-8-sig")

    ranked = result_df.sort_values(
        [
            "ablation_type",
            "worst_fold_rank_ic",
            "negative_day_rank_ic_ratio",
            "cumulative_return_after_cost",
            "sharpe_after_cost",
        ],
        ascending=[True, False, True, False, False],
    ).copy()
    ranked["is_group_winner"] = ranked.groupby("ablation_type").cumcount() == 0
    report_lines = [
        "# Ablation Study Report",
        "",
        "## Winners By Group",
        "",
        "| ablation_type | variant | worst_fold_rank_ic | negative_day_rank_ic_ratio | cum_after_cost | sharpe_after_cost | winner |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in ranked.iterrows():
        report_lines.append(
            f"| {row['ablation_type']} | {row['variant']} | "
            f"{float(row.get('worst_fold_rank_ic', 0.0)):.6f} | "
            f"{float(row.get('negative_day_rank_ic_ratio', 0.0)):.6f} | "
            f"{float(row.get('cumulative_return_after_cost', 0.0)):.6f} | "
            f"{float(row.get('sharpe_after_cost', 0.0)):.6f} | "
            f"{'yes' if bool(row['is_group_winner']) else 'no'} |"
        )
    report_path = output_dir / "ablation_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(result_df[sort_columns].to_string(index=False))
    print(f"[ablation] wrote {summary_path}")
    print(f"[ablation] wrote {best_path}")
    print(f"[ablation] wrote {report_path}")


if __name__ == "__main__":
    main()
