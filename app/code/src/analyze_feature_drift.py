from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feature_importance_report import (
    DEFAULT_FEATURE_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_PREDICTION_PATH,
    build_feature_importance_table,
    compute_feature_ic_by_fold,
    load_fold_context,
    load_metadata,
    resolve_feature_columns,
    resolve_model_path,
    resolve_path,
)


DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR / "feature_drift_monitoring"
RAW_LABEL_COLUMN = "target_return"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze fold-level feature drift, feature IC stability, and model feature importance."
    )
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--metadata_path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--prediction_path", default=str(DEFAULT_PREDICTION_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top_n", type=int, default=12)
    return parser.parse_args()


def _safe_quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else 0.0


def build_fold_feature_stats(fold_context: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    ic_df = compute_feature_ic_by_fold(fold_context, feature_columns)
    ic_lookup = {
        (row["feature"], int(row["fold_id"])): row
        for _, row in ic_df.iterrows()
    }
    rows: list[dict[str, Any]] = []

    for fold_id, fold_df in fold_context.groupby("fold_id", sort=True):
        for feature in feature_columns:
            values = pd.to_numeric(fold_df[feature], errors="coerce")
            valid_values = values.dropna()
            ic_row = ic_lookup.get((feature, int(fold_id)), {})
            rows.append(
                {
                    "feature": feature,
                    "fold_id": int(fold_id),
                    "count": int(valid_values.count()),
                    "missing_rate": float(values.isna().mean()),
                    "mean": float(valid_values.mean()) if not valid_values.empty else 0.0,
                    "std": float(valid_values.std(ddof=0)) if len(valid_values) > 1 else 0.0,
                    "p05": _safe_quantile(valid_values, 0.05),
                    "p25": _safe_quantile(valid_values, 0.25),
                    "p50": _safe_quantile(valid_values, 0.50),
                    "p75": _safe_quantile(valid_values, 0.75),
                    "p95": _safe_quantile(valid_values, 0.95),
                    "stage_ic": float(ic_row.get("stage_ic", 0.0)),
                    "stage_ic_std": float(ic_row.get("stage_ic_std", 0.0)),
                    "stage_ic_positive_ratio": float(ic_row.get("stage_ic_positive_ratio", 0.0)),
                    "stage_ic_days": int(ic_row.get("stage_ic_days", 0)),
                }
            )
    return pd.DataFrame(rows)


def build_drift_summary(fold_stats: pd.DataFrame, importance_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fold_stats.groupby("feature", as_index=False)
        .agg(
            folds=("fold_id", "nunique"),
            count_min=("count", "min"),
            missing_rate_max=("missing_rate", "max"),
            mean_min=("mean", "min"),
            mean_max=("mean", "max"),
            std_mean=("std", "mean"),
            std_min=("std", "min"),
            std_max=("std", "max"),
            p50_min=("p50", "min"),
            p50_max=("p50", "max"),
            p95_min=("p95", "min"),
            p95_max=("p95", "max"),
            stage_ic_mean=("stage_ic", "mean"),
            stage_ic_min=("stage_ic", "min"),
            stage_ic_max=("stage_ic", "max"),
            stage_ic_std=("stage_ic", lambda s: float(np.std(s, ddof=0))),
            stage_ic_positive_ratio_mean=("stage_ic_positive_ratio", "mean"),
        )
    )
    summary["mean_range"] = summary["mean_max"] - summary["mean_min"]
    summary["std_range"] = summary["std_max"] - summary["std_min"]
    summary["p50_range"] = summary["p50_max"] - summary["p50_min"]
    summary["p95_range"] = summary["p95_max"] - summary["p95_min"]
    summary["stage_ic_range"] = summary["stage_ic_max"] - summary["stage_ic_min"]
    summary["relative_mean_range"] = summary["mean_range"] / (summary["std_mean"].abs() + 1e-12)
    summary["relative_p50_range"] = summary["p50_range"] / (summary["std_mean"].abs() + 1e-12)
    summary["drift_score"] = (
        summary["relative_mean_range"].abs()
        + 0.5 * summary["relative_p50_range"].abs()
        + 10.0 * summary["stage_ic_range"].abs()
        + summary["missing_rate_max"].abs()
    )

    importance_columns = [
        column
        for column in [
            "feature",
            "importance_rank",
            "gain_importance",
            "gain_importance_pct",
            "split_importance",
            "split_importance_pct",
        ]
        if column in importance_df.columns
    ]
    if importance_columns:
        summary = summary.merge(importance_df[importance_columns], on="feature", how="left")

    summary["stability_label"] = np.select(
        [
            (summary["drift_score"] <= summary["drift_score"].quantile(0.35))
            & (summary["stage_ic_range"].abs() <= summary["stage_ic_range"].abs().quantile(0.50)),
            (summary["drift_score"] >= summary["drift_score"].quantile(0.75))
            | (summary["stage_ic_range"].abs() >= summary["stage_ic_range"].abs().quantile(0.75)),
        ],
        ["stable", "unstable"],
        default="watch",
    )
    return summary.sort_values(["drift_score", "stage_ic_range"], ascending=[False, False]).reset_index(drop=True)


def build_fold_ic_table(fold_stats: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_stats.pivot(index="feature", columns="fold_id", values="stage_ic")
        .rename(columns=lambda fold_id: f"fold_{int(fold_id)}_stage_ic")
        .reset_index()
    )


def format_float(value: Any, digits: int = 6) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def top_feature_lines(df: pd.DataFrame, columns: list[str], top_n: int) -> list[str]:
    lines = ["| Feature | " + " | ".join(columns) + " |", "|---|" + "|".join(["---:"] * len(columns)) + "|"]
    for _, row in df.head(top_n).iterrows():
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, (float, np.floating)):
                cells.append(format_float(value))
            else:
                cells.append(str(value))
        lines.append(f"| `{row['feature']}` | " + " | ".join(cells) + " |")
    return lines


def write_report(
    report_path: Path,
    fold_stats: pd.DataFrame,
    drift_summary: pd.DataFrame,
    importance_df: pd.DataFrame,
    top_n: int,
) -> None:
    stable = drift_summary[drift_summary["stability_label"] == "stable"].sort_values(
        ["stage_ic_range", "drift_score"], ascending=[True, True]
    )
    unstable = drift_summary[drift_summary["stability_label"] == "unstable"].sort_values(
        ["drift_score", "stage_ic_range"], ascending=[False, False]
    )
    important_unstable = unstable.merge(
        importance_df[["feature", "importance_rank", "gain_importance_pct"]],
        on="feature",
        how="left",
        suffixes=("", "_importance"),
    ).sort_values(["importance_rank", "drift_score"], ascending=[True, False])

    fold_ranges = (
        fold_stats.groupby("fold_id")
        .agg(
            feature_count=("feature", "nunique"),
            avg_abs_stage_ic=("stage_ic", lambda s: float(np.mean(np.abs(s)))),
            avg_missing_rate=("missing_rate", "mean"),
        )
        .reset_index()
    )

    lines = [
        "# Feature Drift and Importance Report",
        "",
        "## Scope",
        "",
        f"- Validation folds: {', '.join(str(int(v)) for v in sorted(fold_stats['fold_id'].unique()))}",
        f"- Features analyzed: {int(drift_summary['feature'].nunique())}",
        "- Drift score combines fold-to-fold mean shift, median shift, missing-rate change, and stage IC range.",
        "",
        "## Fold Overview",
        "",
        "| Fold | Feature Count | Avg Abs IC | Avg Missing Rate |",
        "|---:|---:|---:|---:|",
    ]
    for _, row in fold_ranges.iterrows():
        lines.append(
            f"| {int(row['fold_id'])} | {int(row['feature_count'])} | "
            f"{row['avg_abs_stage_ic']:.6f} | {row['avg_missing_rate']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Stable Features",
            "",
            "These features have relatively small distribution movement across Fold 1/2/3 and comparatively stable stage IC.",
            "",
            *top_feature_lines(
                stable,
                ["drift_score", "stage_ic_mean", "stage_ic_range", "relative_mean_range", "importance_rank"],
                top_n,
            ),
            "",
            "## Unstable Features",
            "",
            "These features show larger fold-to-fold drift or IC reversal, so they should be watched when Fold 1/3 weakens.",
            "",
            *top_feature_lines(
                unstable,
                ["drift_score", "stage_ic_mean", "stage_ic_range", "relative_mean_range", "importance_rank"],
                top_n,
            ),
            "",
            "## Important But Unstable",
            "",
            "High-importance features in this list can explain why a fold fails even when the average model score looks acceptable.",
            "",
            *top_feature_lines(
                important_unstable,
                ["importance_rank", "gain_importance_pct", "drift_score", "stage_ic_range"],
                min(top_n, len(important_unstable)),
            ),
            "",
            "## Top LightGBM Importance",
            "",
            *top_feature_lines(
                importance_df.sort_values(["gain_importance", "split_importance"], ascending=[False, False]),
                ["gain_importance_pct", "split_importance_pct", "stage_ic_mean", "stage_ic_range"],
                top_n,
            ),
            "",
            "## Interpretation",
            "",
            "- Stable features are better candidates for the core signal set because their fold IC and cross-sectional distribution are less regime-dependent.",
            "- Unstable features are not automatically bad, but they need caps, ablation checks, or regime-aware usage if they also rank high in LightGBM importance.",
            "- If Fold 1 or Fold 3 underperforms, first inspect important-but-unstable features with large `stage_ic_range`; those are the most likely contributors to signal failure.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_dir = resolve_path(args.model_dir)
    metadata_path = resolve_path(args.metadata_path)
    metadata = load_metadata(metadata_path)
    model_path = resolve_model_path(model_dir, metadata, args.model_path)
    feature_path = resolve_path(args.feature_path)
    prediction_path = resolve_path(args.prediction_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_context = load_fold_context(feature_path, prediction_path)
    feature_columns = resolve_feature_columns(metadata, fold_context)
    if not feature_columns:
        raise ValueError("No feature columns were found for drift analysis.")

    importance_df = build_feature_importance_table(
        model_path=model_path,
        feature_path=feature_path,
        prediction_path=prediction_path,
        metadata=metadata,
    )
    fold_stats = build_fold_feature_stats(fold_context, feature_columns)
    drift_summary = build_drift_summary(fold_stats, importance_df)
    fold_ic = build_fold_ic_table(fold_stats)
    drift_summary = drift_summary.merge(fold_ic, on="feature", how="left")

    fold_stats.to_csv(output_dir / "feature_drift_by_fold.csv", index=False, encoding="utf-8-sig")
    drift_summary.to_csv(output_dir / "feature_drift_summary.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    write_report(
        report_path=output_dir / "feature_drift_report.md",
        fold_stats=fold_stats,
        drift_summary=drift_summary,
        importance_df=importance_df,
        top_n=args.top_n,
    )

    print(f"[analyze_feature_drift] wrote {output_dir / 'feature_drift_report.md'}")
    print(f"[analyze_feature_drift] wrote {output_dir / 'feature_importance.csv'}")
    print(f"[analyze_feature_drift] wrote {output_dir / 'feature_drift_summary.csv'}")


if __name__ == "__main__":
    main()
