from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import ROOT_DIR


MODEL_DIR = ROOT_DIR / "app" / "model"
DEFAULT_OUTPUT_PATH = MODEL_DIR / "experiment_leaderboard.csv"
DEFAULT_REPORT_PATH = MODEL_DIR / "experiment_leaderboard_report.md"

SCAN_DIRS = [
    MODEL_DIR / "experiments",
    MODEL_DIR / "weight_cap_search",
    MODEL_DIR / "weight_blend_search",
    MODEL_DIR / "turnover_stress_test",
    MODEL_DIR / "topk_objective_search",
    MODEL_DIR / "label_variant_search",
    MODEL_DIR / "rank_blend",
    MODEL_DIR / "sequence_length_search",
    MODEL_DIR / "transformer_lite",
]

SUMMARY_FILE_HINTS = (
    "summary",
    "comparison",
    "decision",
    "search",
    "recommended",
    "all",
    "shortlist",
    "performance",
)

SKIP_FILE_NAMES = {"experiment_leaderboard.csv"}

LABEL_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "experiment_name",
    "profile_name",
    "label",
    "model_name",
    "model",
    "feature_set",
    "stage_name",
    "config_name",
]

METRIC_ALIASES = {
    "rank_ic_mean": ["rank_ic_mean", "rank_ic", "avg_day_rank_ic"],
    "worst_fold": ["worst_fold", "worst_fold_rank_ic", "min_rank_ic"],
    "top5_mean_return": ["top5_return_mean", "top5_mean_return_mean", "top5_mean_return", "top5_weighted_return"],
    "top5_return_min_by_fold": ["top5_return_min_by_fold", "top5_min_return", "min_top5_return"],
    "cumulative_return_after_cost": [
        "cumulative_return_after_cost",
        "cost_after_return",
        "backtest_return",
        "cum_after_cost",
        "return_after_cost",
        "portfolio_return",
    ],
    "sharpe_after_cost": ["sharpe_after_cost", "sharpe"],
    "max_drawdown_after_cost": ["max_drawdown_after_cost", "max_drawdown", "max_dd_after_cost"],
    "avg_turnover": ["avg_turnover", "turnover", "turnover_mean"],
    "slice_score": ["score_self_case_slice", "case_slice_score", "slice_score", "slice_return"],
    "single_slice_score": ["single_slice_score", "top5_weighted_return", "slice_score"],
}

VALIDATION_ALIASES = [
    "result_valid",
    "pass_basic_validation",
    "result_validator_passed",
    "valid",
    "is_valid",
]

LEADERBOARD_COLUMNS = [
    "rank",
    "stable_alpha_score",
    "composite_score",
    "decision",
    "risk_flags",
    "aggressive_rank",
    "robust_rank",
    "candidate_id",
    "candidate_label",
    "source_kind",
    "experiment_dir",
    "source_file",
    "rank_ic_mean",
    "worst_fold",
    "top5_mean_return",
    "top5_return_min_by_fold",
    "cumulative_return_after_cost",
    "sharpe_after_cost",
    "max_drawdown_after_cost",
    "avg_turnover",
    "slice_score",
    "single_slice_score",
    "result_valid",
    "result_validator_status",
    "score_rank_ic",
    "score_worst_fold",
    "score_top5",
    "score_return",
    "score_sharpe",
    "drawdown_penalty",
    "turnover_penalty",
    "slice_score_component",
    "available_metric_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified experiment leaderboard.")
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report_path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--min_metrics", type=int, default=2)
    parser.add_argument(
        "--scan_all_model_dirs",
        action="store_true",
        help="Scan all app/model subdirectories instead of the fixed Prompt 20 scope.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def first_present(row: pd.Series, aliases: Iterable[str]) -> float:
    for column in aliases:
        if column in row.index:
            value = safe_float(row[column])
            if pd.notna(value):
                return value
    return np.nan


def first_text(row: pd.Series, columns: Iterable[str]) -> str:
    for column in columns:
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column]).strip()
    return ""


def first_bool(row: pd.Series, columns: Iterable[str]) -> bool | None:
    for column in columns:
        if column not in row.index or pd.isna(row[column]):
            continue
        value = row[column]
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "pass", "passed"}:
            return True
        if text in {"false", "0", "no", "n", "fail", "failed"}:
            return False
    return None


def short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def metric_count(row: dict) -> int:
    return int(sum(pd.notna(row.get(column)) for column in METRIC_ALIASES))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path)


def candidate_label_from_row(source_file: Path, row: pd.Series) -> str:
    if "profile_name" in row.index and pd.notna(row["profile_name"]) and str(row["profile_name"]).strip():
        return str(row["profile_name"]).strip()
    if "alpha" in row.index and pd.notna(row["alpha"]):
        cap = row["max_single_weight"] if "max_single_weight" in row.index and pd.notna(row["max_single_weight"]) else "none"
        return f"alpha_{row['alpha']}_cap_{cap}"
    if "cap" in row.index and pd.notna(row["cap"]):
        return f"cap_{row['cap']}"
    if "max_turnover" in row.index and "weight_strategy" in row.index:
        cap = row["max_single_weight"] if "max_single_weight" in row.index and pd.notna(row["max_single_weight"]) else "none"
        tc = row["transaction_cost"] if "transaction_cost" in row.index and pd.notna(row["transaction_cost"]) else "na"
        return f"mt{row['max_turnover']}_tc{tc}_{row['weight_strategy']}_cap{cap}"
    label = first_text(row, LABEL_COLUMNS)
    if label:
        return label
    parts = []
    for column in ["max_turnover", "transaction_cost", "weight_strategy", "max_single_weight", "alpha", "cap"]:
        if column in row.index and pd.notna(row[column]):
            parts.append(f"{column}={row[column]}")
    return "_".join(parts) if parts else source_file.parent.name


def normalize_summary_row(source_file: Path, row: pd.Series, source_kind: str) -> dict:
    label = candidate_label_from_row(source_file, row)
    normalized = {
        "candidate_label": label,
        "source_kind": source_kind,
        "experiment_dir": rel(source_file.parent),
        "source_file": rel(source_file),
    }
    for output_column, aliases in METRIC_ALIASES.items():
        normalized[output_column] = first_present(row, aliases)
    validation = first_bool(row, VALIDATION_ALIASES)
    normalized["result_valid"] = validation
    normalized["result_validator_status"] = "pass" if validation is True else "fail" if validation is False else "unknown"
    normalized["available_metric_count"] = metric_count(normalized)
    normalized["candidate_id"] = short_hash(
        "|".join([normalized["candidate_label"], normalized["source_kind"], normalized["source_file"]])
    )
    return normalized


def summarize_numeric_column(df: pd.DataFrame, aliases: Iterable[str], reducer: str = "mean") -> float:
    for column in aliases:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        if reducer == "min":
            return float(values.min())
        return float(values.mean())
    return np.nan


def first_backtest_row(experiment_dir: Path) -> pd.Series:
    for path in [experiment_dir / "backtest_summary.csv", experiment_dir / "backtest" / "backtest_summary.csv"]:
        if path.exists():
            df = read_csv(path)
            if not df.empty:
                return df.iloc[0]
    return pd.Series(dtype=object)


def collect_training_dir(experiment_dir: Path) -> dict | None:
    metrics_paths = [
        experiment_dir / "metrics.csv",
        experiment_dir / "walk_forward_metrics.csv",
        experiment_dir / "fold_results.csv",
        experiment_dir / "fold_diagnostics.csv",
    ]
    existing = [path for path in metrics_paths if path.exists()]
    if not existing:
        return None

    frames = []
    for path in existing:
        df = read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return None

    fold_like = pd.concat(frames, ignore_index=True, sort=False)
    row = normalize_summary_row(existing[0], fold_like.iloc[0], source_kind="training_dir")
    backtest_row = first_backtest_row(experiment_dir)

    row["candidate_label"] = experiment_dir.name
    row["experiment_dir"] = rel(experiment_dir)
    row["source_file"] = ";".join(rel(path) for path in existing)
    row["rank_ic_mean"] = summarize_numeric_column(fold_like, METRIC_ALIASES["rank_ic_mean"], "mean")
    row["worst_fold"] = summarize_numeric_column(fold_like, METRIC_ALIASES["rank_ic_mean"], "min")
    row["top5_mean_return"] = summarize_numeric_column(fold_like, METRIC_ALIASES["top5_mean_return"], "mean")
    row["top5_return_min_by_fold"] = summarize_numeric_column(fold_like, METRIC_ALIASES["top5_mean_return"], "min")

    for column in [
        "cumulative_return_after_cost",
        "sharpe_after_cost",
        "max_drawdown_after_cost",
        "avg_turnover",
        "slice_score",
        "single_slice_score",
    ]:
        if pd.isna(row[column]):
            row[column] = first_present(backtest_row, METRIC_ALIASES[column])

    validation = first_bool(backtest_row, VALIDATION_ALIASES)
    if validation is not None:
        row["result_valid"] = validation
        row["result_validator_status"] = "pass" if validation else "fail"
    row["available_metric_count"] = metric_count(row)
    row["candidate_id"] = short_hash(f"training_dir|{row['experiment_dir']}|{row['source_file']}")
    return row


def is_summary_file(path: Path) -> bool:
    name = path.name.lower()
    if name in SKIP_FILE_NAMES:
        return False
    return any(hint in name for hint in SUMMARY_FILE_HINTS)


def collect_summary_table(path: Path) -> list[dict]:
    try:
        df = read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []
    metric_columns = {alias for aliases in METRIC_ALIASES.values() for alias in aliases}
    if not (set(df.columns) & metric_columns):
        return []
    rows = []
    for _, source_row in df.iterrows():
        normalized = normalize_summary_row(path, source_row, source_kind="summary_table")
        if normalized["available_metric_count"] > 0:
            rows.append(normalized)
    return rows


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.zeros(len(values)), index=values.index)
    lower = float(valid.min())
    upper = float(valid.max())
    if abs(upper - lower) < 1e-12:
        score = pd.Series(np.where(values.notna(), 1.0, 0.0), index=values.index)
    else:
        score = (values - lower) / (upper - lower)
    if not higher_is_better:
        score = 1.0 - score
    return score.fillna(0.0).clip(0.0, 1.0)


def baseline_metrics(df: pd.DataFrame) -> dict[str, float]:
    mask = (
        df["candidate_label"].astype(str).str.contains("baseline_cross_section_rank|default|sl20", case=False, na=False)
        | df["source_file"].astype(str).str.contains("baseline_cross_section_rank|default|sl20", case=False, na=False)
    )
    baseline = df[mask].copy()
    if baseline.empty:
        baseline = df.copy()
    return {
        "worst_fold": float(pd.to_numeric(baseline["worst_fold"], errors="coerce").median()),
        "max_drawdown_after_cost": float(pd.to_numeric(baseline["max_drawdown_after_cost"], errors="coerce").median()),
        "avg_turnover": float(pd.to_numeric(baseline["avg_turnover"], errors="coerce").median()),
    }


def add_adoption_rules(scored: pd.DataFrame) -> pd.DataFrame:
    ruled = scored.copy()
    base = baseline_metrics(ruled)
    return_median = pd.to_numeric(ruled["cumulative_return_after_cost"], errors="coerce").median()
    slice_values = pd.to_numeric(ruled["single_slice_score"].fillna(ruled["slice_score"]), errors="coerce")
    slice_threshold = slice_values.quantile(0.80)

    decisions = []
    flags_all = []
    for _, row in ruled.iterrows():
        flags: list[str] = []
        hard_reject = False
        caution = False

        if row.get("result_valid") is False:
            flags.append("result_validator 不通过")
            hard_reject = True

        top5_min = safe_float(row.get("top5_return_min_by_fold"))
        if pd.notna(top5_min) and top5_min <= 0:
            flags.append("top5_return_min_by_fold <= 0，缺少合理解释")
            hard_reject = True

        worst_fold = safe_float(row.get("worst_fold"))
        if pd.notna(worst_fold) and pd.notna(base["worst_fold"]) and worst_fold < base["worst_fold"] - 0.02:
            flags.append("worst_fold_rank_ic 明显低于当前 sl20")
            caution = True

        drawdown = safe_float(row.get("max_drawdown_after_cost"))
        if pd.notna(drawdown) and pd.notna(base["max_drawdown_after_cost"]):
            if abs(drawdown) > abs(base["max_drawdown_after_cost"]) * 1.25:
                flags.append("max_drawdown 明显恶化")
                hard_reject = True

        turnover = safe_float(row.get("avg_turnover"))
        cost_return = safe_float(row.get("cumulative_return_after_cost"))
        if pd.notna(turnover) and pd.notna(base["avg_turnover"]):
            turnover_worse = turnover > base["avg_turnover"] * 1.15
            no_return_comp = pd.isna(cost_return) or pd.isna(return_median) or cost_return < return_median
            if turnover_worse and no_return_comp:
                flags.append("avg_turnover 明显升高且收益没有补偿")
                hard_reject = True

        slice_score = safe_float(row.get("single_slice_score"))
        if pd.isna(slice_score):
            slice_score = safe_float(row.get("slice_score"))
        rank_ic = safe_float(row.get("rank_ic_mean"))
        top5 = safe_float(row.get("top5_mean_return"))
        if (
            pd.notna(slice_score)
            and pd.notna(slice_threshold)
            and slice_score >= slice_threshold
            and ((pd.notna(rank_ic) and rank_ic < 0) or (pd.notna(top5) and top5 < 0))
        ):
            flags.append("single_slice_score 高但 walk-forward 变差，疑似过拟合")
            caution = True

        if hard_reject:
            decision = "reject"
        elif caution:
            decision = "caution"
        else:
            decision = "adopt"
        decisions.append(decision)
        flags_all.append("; ".join(flags) if flags else "通过硬性规则")

    ruled["decision"] = decisions
    ruled["risk_flags"] = flags_all
    return ruled


def dedupe_candidate_rows(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    severity = {"reject": 2, "caution": 1, "adopt": 0}
    deduped = scored.copy()
    deduped["_decision_severity"] = deduped["decision"].map(severity).fillna(0)
    deduped["_has_top5_min"] = pd.to_numeric(deduped["top5_return_min_by_fold"], errors="coerce").notna().astype(int)
    deduped = deduped.sort_values(
        [
            "candidate_label",
            "_decision_severity",
            "_has_top5_min",
            "available_metric_count",
            "stable_alpha_score",
        ],
        ascending=[True, False, False, False, False],
        na_position="last",
    )
    deduped = deduped.drop_duplicates(subset=["candidate_label"], keep="first")
    return deduped.drop(columns=["_decision_severity", "_has_top5_min"]).reset_index(drop=True)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["score_rank_ic"] = minmax_score(scored["rank_ic_mean"], True)
    scored["score_worst_fold"] = minmax_score(scored["worst_fold"], True)
    scored["score_top5"] = minmax_score(scored["top5_mean_return"], True)
    scored["score_return"] = minmax_score(scored["cumulative_return_after_cost"], True)
    scored["score_sharpe"] = minmax_score(scored["sharpe_after_cost"], True)
    scored["slice_score_component"] = minmax_score(scored["single_slice_score"].fillna(scored["slice_score"]), True)
    scored["drawdown_penalty"] = minmax_score(scored["max_drawdown_after_cost"].abs(), True)
    scored["turnover_penalty"] = minmax_score(scored["avg_turnover"], True)
    scored["stable_alpha_score"] = (
        0.25 * scored["score_top5"]
        + 0.20 * scored["score_return"]
        + 0.20 * scored["score_sharpe"]
        + 0.15 * scored["score_rank_ic"]
        + 0.10 * scored["score_worst_fold"]
        - 0.05 * scored["drawdown_penalty"]
        - 0.05 * scored["turnover_penalty"]
    )
    scored["composite_score"] = scored["stable_alpha_score"]
    scored = add_adoption_rules(scored)
    scored = dedupe_candidate_rows(scored)

    eligible = scored["decision"].isin(["adopt", "caution"])
    scored["aggressive_sort_score"] = scored["stable_alpha_score"] + 0.06 * scored["score_return"] + 0.04 * scored["slice_score_component"]
    scored["robust_sort_score"] = scored["stable_alpha_score"] - 0.08 * scored["drawdown_penalty"] - 0.08 * scored["turnover_penalty"]
    scored["aggressive_rank"] = np.nan
    scored["robust_rank"] = np.nan
    scored.loc[eligible, "aggressive_rank"] = scored.loc[eligible, "aggressive_sort_score"].rank(method="first", ascending=False)
    scored.loc[eligible, "robust_rank"] = scored.loc[eligible, "robust_sort_score"].rank(method="first", ascending=False)

    scored = scored.sort_values(
        ["stable_alpha_score", "cumulative_return_after_cost", "sharpe_after_cost", "rank_ic_mean"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    scored.insert(0, "rank", np.arange(1, len(scored) + 1))
    return scored


def collect_leaderboard_rows(scan_dirs: list[Path], min_metrics: int) -> pd.DataFrame:
    rows: list[dict] = []
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for experiment_dir in sorted(path for path in scan_dir.rglob("*") if path.is_dir()):
            row = collect_training_dir(experiment_dir)
            if row is not None and row["available_metric_count"] >= min_metrics:
                rows.append(row)
        for path in sorted(scan_dir.rglob("*.csv")):
            if not is_summary_file(path):
                continue
            for row in collect_summary_table(path):
                if row["available_metric_count"] >= min_metrics:
                    rows.append(row)

    if not rows:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    df = pd.DataFrame(rows).drop_duplicates(subset=["candidate_id"]).reset_index(drop=True)
    return add_scores(df)


def fmt(value: object, digits: int = 6) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_candidate_table(df: pd.DataFrame, max_rows: int = 10) -> list[str]:
    lines = [
        "| candidate | score | decision | top5 | return | sharpe | worst_ic | max_dd | turnover | reason |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in df.head(max_rows).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{str(row.get('candidate_label', ''))}`",
                    fmt(row.get("stable_alpha_score")),
                    str(row.get("decision", "")),
                    fmt(row.get("top5_mean_return")),
                    fmt(row.get("cumulative_return_after_cost")),
                    fmt(row.get("sharpe_after_cost")),
                    fmt(row.get("worst_fold")),
                    fmt(row.get("max_drawdown_after_cost")),
                    fmt(row.get("avg_turnover")),
                    str(row.get("risk_flags", "")),
                ]
            )
            + " |"
        )
    return lines


def render_report(leaderboard: pd.DataFrame, output_path: Path, scan_dirs: list[Path]) -> str:
    if leaderboard.empty:
        return "# Experiment Leaderboard Report\n\nNo eligible experiment rows were found.\n"

    eligible = leaderboard[leaderboard["decision"].isin(["adopt", "caution"])].copy()
    aggressive = eligible.sort_values(["aggressive_rank", "stable_alpha_score"], ascending=[True, False], na_position="last")
    robust = eligible.sort_values(["robust_rank", "stable_alpha_score"], ascending=[True, False], na_position="last")
    final = robust[robust["decision"] == "adopt"].head(1)
    if final.empty:
        final = eligible.head(1)
    risky = leaderboard[leaderboard["decision"].isin(["reject", "caution"])].sort_values("stable_alpha_score", ascending=False)

    lines = [
        "# Experiment Leaderboard Report",
        "",
        "## Scoring Rule",
        "",
        "`stable_alpha_score = 0.25 * normalized(top5_return_mean) + 0.20 * normalized(cost_after_return) + 0.20 * normalized(sharpe) + 0.15 * normalized(rank_ic_mean) + 0.10 * normalized(worst_fold_rank_ic) - 0.05 * drawdown_penalty - 0.05 * turnover_penalty`",
        "",
        "## Scan Scope",
        "",
    ]
    lines.extend(f"- `{path}`" for path in scan_dirs)
    lines.extend(["", f"CSV output: `{output_path}`", "", "## Top 10 aggressive 候选", ""])
    lines.extend(render_candidate_table(aggressive, 10))
    lines.extend(["", "## Top 10 robust 候选", ""])
    lines.extend(render_candidate_table(robust, 10))
    lines.extend(["", "## 推荐最终候选", ""])
    lines.extend(render_candidate_table(final, 1))
    lines.extend(["", "## 不推荐的高风险候选及原因", ""])
    lines.extend(render_candidate_table(risky, 20) if not risky.empty else ["No rejected or caution candidates found."])
    lines.extend(
        [
            "",
            "## Adoption Rules",
            "",
            "- `top5_return_min_by_fold <= 0` 且没有合理解释，淘汰。",
            "- `worst_fold_rank_ic` 明显低于当前 sl20，标记谨慎。",
            "- `max_drawdown` 明显恶化，淘汰。",
            "- `avg_turnover` 明显升高且收益没有补偿，淘汰。",
            "- `single_slice_score` 高但 walk-forward 变差，标记疑似过拟合。",
            "- `result_validator` 不通过，淘汰。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_path = resolve_path(args.output_path)
    report_path = resolve_path(args.report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    scan_dirs = [MODEL_DIR] if args.scan_all_model_dirs else SCAN_DIRS
    leaderboard = collect_leaderboard_rows(scan_dirs=scan_dirs, min_metrics=args.min_metrics)
    for column in LEADERBOARD_COLUMNS:
        if column not in leaderboard.columns:
            leaderboard[column] = np.nan
    leaderboard = leaderboard[LEADERBOARD_COLUMNS]
    leaderboard.to_csv(output_path, index=False, encoding="utf-8-sig")
    report_path.write_text(render_report(leaderboard, output_path, scan_dirs), encoding="utf-8")

    print(f"[experiment_leaderboard] rows={len(leaderboard)}")
    print(f"[experiment_leaderboard] wrote {output_path}")
    print(f"[experiment_leaderboard] wrote {report_path}")
    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        print(
            "[experiment_leaderboard] best="
            f"{best['candidate_label']} stable_alpha_score={best['stable_alpha_score']:.6f}"
        )


if __name__ == "__main__":
    main()
