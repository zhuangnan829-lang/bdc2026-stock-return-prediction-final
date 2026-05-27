from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from config import ROOT_DIR
except Exception:
    ROOT_DIR = Path(__file__).resolve().parents[3]


DEFAULT_PRED_PATH = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "misrank_diagnostics"
DEFAULT_TARGET_COL = "target_return"
DEFAULT_PRED_COL = "pred_return"

REQUESTED_FEATURES = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "intraday_return",
    "volume_ratio",
    "turnover_rate",
    "turnover_spike",
    "volatility_5d",
    "volatility_20d",
    "rel_hs300_ret_5d",
    "close_position_20d",
    "overheat_score",
    "reversal_risk_score",
]

FEATURE_ALIASES = {
    "volume_ratio": ["volume_ratio", "volume_ratio_5d", "volume_ratio_10d"],
    "turnover_spike": ["turnover_spike", "turnover_spike_5d"],
    "rel_hs300_ret_5d": ["rel_hs300_ret_5d", "rel_hs300_mean_ret_5d", "rel_ret_5d"],
    "close_position_20d": ["close_position_20d", "distance_to_20d_high", "close_to_ma_20d"],
    "overheat_score": ["overheat_score", "crowding_risk_5d", "amplitude_ratio_5d"],
    "reversal_risk_score": ["reversal_risk_score", "crowding_reversal_risk_5d"],
}

EXTRA_CONTEXT_FEATURES = [
    "volume_ratio_5d",
    "volume_ratio_10d",
    "turnover_spike_5d",
    "turnover_ratio_5d",
    "turnover_ratio_10d",
    "amplitude_ratio_5d",
    "crowding_risk_5d",
    "crowding_reversal_risk_5d",
    "distance_to_20d_high",
    "rebound_from_10d_low",
    "volume_price_divergence_5d",
]


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def normalize_stock_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_id": str})


def detect_columns(pred_df: pd.DataFrame, target_col: str) -> tuple[str, str, str, str]:
    stock_col = "stock_id" if "stock_id" in pred_df.columns else ""
    date_col = "date" if "date" in pred_df.columns else ("trade_date" if "trade_date" in pred_df.columns else "")
    pred_col = DEFAULT_PRED_COL if DEFAULT_PRED_COL in pred_df.columns else ("y_pred" if "y_pred" in pred_df.columns else "")
    true_col = target_col if target_col in pred_df.columns else ("y_true" if "y_true" in pred_df.columns else "")
    missing = [
        name
        for name, col in [
            ("stock_id", stock_col),
            ("date/trade_date", date_col),
            ("pred_return/y_pred", pred_col),
        ]
        if not col
    ]
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}. Existing columns: {list(pred_df.columns)}")
    return stock_col, date_col, pred_col, true_col


def build_feature_plan(feature_df: pd.DataFrame) -> tuple[list[str], dict[str, str], list[str], list[str]]:
    available: list[str] = []
    display_to_column: dict[str, str] = {}
    missing: list[str] = []
    used_aliases: list[str] = []
    for requested in REQUESTED_FEATURES:
        candidates = FEATURE_ALIASES.get(requested, [requested])
        matched = next((column for column in candidates if column in feature_df.columns), None)
        if matched:
            available.append(matched)
            display_to_column[requested] = matched
            if matched != requested:
                used_aliases.append(f"{requested} -> {matched}")
                missing.append(requested)
        else:
            missing.append(requested)
    for column in EXTRA_CONTEXT_FEATURES:
        if column in feature_df.columns and column not in available:
            available.append(column)
            display_to_column[column] = column
    return available, display_to_column, missing, used_aliases


def load_inputs(pred_path: Path, feature_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str], list[str], list[str]]:
    pred_df = read_csv(pred_path)
    feature_df = read_csv(feature_path)
    stock_col, date_col, pred_col, pred_target_col = detect_columns(pred_df, target_col)

    for frame, name in [(feature_df, "feature_path")]:
        if "stock_id" not in frame.columns or "date" not in frame.columns:
            raise ValueError(f"{name} must contain stock_id and date columns. Existing columns: {list(frame.columns)}")

    pred_df = pred_df.copy()
    pred_df["stock_id"] = normalize_stock_id(pred_df[stock_col])
    pred_df["date"] = normalize_date(pred_df[date_col])
    pred_df["pred_score"] = pd.to_numeric(pred_df[pred_col], errors="coerce")
    if pred_target_col:
        pred_df[target_col] = pd.to_numeric(pred_df[pred_target_col], errors="coerce")

    feature_df = feature_df.copy()
    feature_df["stock_id"] = normalize_stock_id(feature_df["stock_id"])
    feature_df["date"] = normalize_date(feature_df["date"])
    if target_col in feature_df.columns:
        feature_df[target_col] = pd.to_numeric(feature_df[target_col], errors="coerce")

    if target_col not in pred_df.columns or pred_df[target_col].isna().all():
        if target_col not in feature_df.columns:
            raise ValueError(
                f"Target column `{target_col}` was not found in predictions or features. "
                f"Prediction columns: {list(pred_df.columns)}; feature columns: {list(feature_df.columns)}"
            )
        pred_df = pred_df.merge(feature_df[["stock_id", "date", target_col]], on=["stock_id", "date"], how="left")

    if "fold_id" not in pred_df.columns:
        pred_df["fold_id"] = 0
    pred_df["fold_id"] = pd.to_numeric(pred_df["fold_id"], errors="coerce").fillna(0).astype(int)
    pred_df[target_col] = pd.to_numeric(pred_df[target_col], errors="coerce")
    pred_df = pred_df.dropna(subset=["stock_id", "date", "pred_score", target_col]).copy()

    available_features, display_to_column, missing_features, used_aliases = build_feature_plan(feature_df)
    keep_cols = ["stock_id", "date", *available_features]
    feature_keep = feature_df[keep_cols].drop_duplicates(["stock_id", "date"], keep="last")
    for column in available_features:
        feature_keep[column] = pd.to_numeric(feature_keep[column], errors="coerce")
    merged = pred_df.merge(feature_keep, on=["stock_id", "date"], how="left")
    return merged, feature_keep, available_features, display_to_column, missing_features, used_aliases


def assign_daily_ranks(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    ranked = df.copy()
    ranked["model_rank"] = ranked.groupby("date")["pred_score"].rank(method="first", ascending=False)
    ranked["true_rank"] = ranked.groupby("date")[target_col].rank(method="first", ascending=False)
    return ranked


def collect_misranked(df: pd.DataFrame, target_col: str, top_k: int, true_top_k: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missed_rows: list[pd.DataFrame] = []
    false_rows: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    for date, day_df in df.groupby("date", sort=True):
        day_df = day_df.copy()
        model_top = day_df.sort_values(["pred_score", "stock_id"], ascending=[False, True]).head(top_k)
        true_top = day_df.sort_values([target_col, "stock_id"], ascending=[False, True]).head(true_top_k)
        model_set = set(model_top["stock_id"])
        true_set = set(true_top["stock_id"])
        poor_cutoff = float(day_df[target_col].quantile(0.5))
        missed = true_top[~true_top["stock_id"].isin(model_set)].copy()
        false_positive = model_top[~model_top["stock_id"].isin(true_set)].copy()
        for case_df, case_type in [(missed, "missed_winners"), (false_positive, "false_positives")]:
            if case_df.empty:
                continue
            case_df["case_type"] = case_type
            case_df["date"] = date
            case_df["daily_true_return_median"] = poor_cutoff
            case_df["is_poor_return"] = case_df[target_col] <= poor_cutoff
            if case_type == "missed_winners":
                case_df["miss_margin"] = case_df["model_rank"] - float(top_k)
            else:
                case_df["return_gap_to_true_top_min"] = case_df[target_col] - float(true_top[target_col].min())
            if case_type == "missed_winners":
                missed_rows.append(case_df)
            else:
                false_rows.append(case_df)
        daily_rows.append(
            {
                "fold_id": int(day_df["fold_id"].iloc[0]),
                "date": pd.Timestamp(date).date().isoformat(),
                "model_top_return_mean": float(model_top[target_col].mean()),
                "true_top_return_mean": float(true_top[target_col].mean()),
                "top_hit_count": int(len(model_set & true_set)),
                "top_hit_rate": float(len(model_set & true_set) / max(1, min(top_k, true_top_k))),
                "missed_winners_count": int(len(missed)),
                "false_positives_count": int(len(false_positive)),
                "poor_false_positives_count": int(false_positive["is_poor_return"].sum()) if not false_positive.empty else 0,
            }
        )
    missed_df = pd.concat(missed_rows, ignore_index=True) if missed_rows else pd.DataFrame()
    false_df = pd.concat(false_rows, ignore_index=True) if false_rows else pd.DataFrame()
    daily_df = pd.DataFrame(daily_rows)
    return missed_df, false_df, daily_df


def feature_stats(
    all_df: pd.DataFrame,
    missed_df: pd.DataFrame,
    false_df: pd.DataFrame,
    available_features: list[str],
) -> pd.DataFrame:
    if not available_features:
        return pd.DataFrame()
    cases = [("missed_winners", missed_df), ("false_positives", false_df)]
    rows: list[dict[str, Any]] = []
    for fold_id, fold_all in all_df.groupby("fold_id"):
        for feature in available_features:
            baseline = pd.to_numeric(fold_all[feature], errors="coerce")
            baseline_mean = float(baseline.mean()) if baseline.notna().any() else np.nan
            baseline_median = float(baseline.median()) if baseline.notna().any() else np.nan
            baseline_std = float(baseline.std(ddof=0)) if baseline.notna().any() else 0.0
            for case_type, case_df in cases:
                fold_case = case_df[case_df["fold_id"] == fold_id] if not case_df.empty else pd.DataFrame()
                values = pd.to_numeric(fold_case[feature], errors="coerce") if feature in fold_case.columns else pd.Series(dtype=float)
                case_mean = float(values.mean()) if values.notna().any() else np.nan
                case_median = float(values.median()) if values.notna().any() else np.nan
                z_gap = (case_mean - baseline_mean) / baseline_std if pd.notna(case_mean) and pd.notna(baseline_mean) and baseline_std > 1e-12 else 0.0
                rows.append(
                    {
                        "fold_id": int(fold_id),
                        "case_type": case_type,
                        "feature": feature,
                        "sample_count": int(values.notna().sum()),
                        "case_mean": case_mean,
                        "case_median": case_median,
                        "fold_mean": baseline_mean,
                        "fold_median": baseline_median,
                        "fold_std": baseline_std,
                        "z_gap_vs_fold": float(z_gap),
                    }
                )
    return pd.DataFrame(rows)


def top_feature_lines(stats_df: pd.DataFrame, case_type: str, fold_id: int | None = None, top_n: int = 6) -> list[str]:
    subset = stats_df[(stats_df["case_type"] == case_type) & (stats_df["sample_count"] > 0)].copy()
    if fold_id is not None:
        subset = subset[subset["fold_id"] == fold_id]
    if subset.empty:
        return ["- No available feature statistics."]
    subset["abs_gap"] = subset["z_gap_vs_fold"].abs()
    subset = subset.sort_values("abs_gap", ascending=False).head(top_n)
    lines = []
    for _, row in subset.iterrows():
        direction = "higher" if row["z_gap_vs_fold"] > 0 else "lower"
        lines.append(
            f"- `{row['feature']}` is {direction} than fold baseline "
            f"(mean {row['case_mean']:.4f} vs {row['fold_mean']:.4f}, z-gap {row['z_gap_vs_fold']:.2f})."
        )
    return lines


def mean_gap(stats_df: pd.DataFrame, case_type: str, feature_candidates: list[str]) -> float:
    subset = stats_df[(stats_df["case_type"] == case_type) & (stats_df["feature"].isin(feature_candidates))]
    if subset.empty:
        return 0.0
    return float(subset["z_gap_vs_fold"].mean())


def build_report(
    missed_df: pd.DataFrame,
    false_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    missing_features: list[str],
    used_aliases: list[str],
    top_k: int,
    true_top_k: int,
) -> str:
    fold_summary = (
        daily_df.groupby("fold_id", as_index=False)
        .agg(
            dates=("date", "count"),
            model_top_return_mean=("model_top_return_mean", "mean"),
            true_top_return_mean=("true_top_return_mean", "mean"),
            top_hit_rate=("top_hit_rate", "mean"),
            missed_winners_count=("missed_winners_count", "sum"),
            false_positives_count=("false_positives_count", "sum"),
            poor_false_positives_count=("poor_false_positives_count", "sum"),
        )
        .sort_values("fold_id")
    )
    fp_overheat_gap = mean_gap(
        stats_df,
        "false_positives",
        ["ret_1d", "ret_3d", "ret_5d", "volume_ratio", "volume_ratio_5d", "turnover_rate", "turnover_spike", "turnover_spike_5d", "volatility_5d", "volatility_20d", "overheat_score", "crowding_risk_5d"],
    )
    reversal_gap = mean_gap(stats_df, "false_positives", ["reversal_risk_score", "crowding_reversal_risk_5d", "distance_to_20d_high"])
    missed_rebound_gap = mean_gap(stats_df, "missed_winners", ["rebound_from_10d_low", "ret_3d", "ret_5d", "volatility_5d", "volatility_20d"])

    lines = [
        "# Misranked Sample Diagnostics",
        "",
        f"Definitions: `missed_winners` = true Top{true_top_k} but not model Top{top_k}; `false_positives` = model Top{top_k} but not true Top{true_top_k}.",
        "",
        "## Required Answers",
        "",
        "1. 漏选赢家有什么共同特征？",
        *top_feature_lines(stats_df, "missed_winners", top_n=8),
        "",
        "2. 误选输家是否存在短期过热、放量、换手异常、高波动、长上影等特征？",
        f"- 过热/放量/高波动综合 z-gap 均值为 {fp_overheat_gap:.2f}；"
        f"{'存在较明显迹象' if fp_overheat_gap > 0.25 else '未形成单一强证据，但仍需结合 Fold 分组观察'}。",
        f"- 反转/位置风险相关 z-gap 均值为 {reversal_gap:.2f}。",
        "",
        "3. 是否应该加入反转保护特征？",
        f"- {'建议加入' if fp_overheat_gap > 0.20 or reversal_gap > 0.15 else '建议小步验证'}：false positives 中高波动、换手或位置风险信号偏强时，反转保护有助于抑制追高误选。",
        "",
        "4. 是否应该调整 risk penalty 或 candidate filter？",
        f"- {'建议先调 risk penalty，再做候选池过滤小实验' if fp_overheat_gap > 0.20 else '建议先保持 candidate filter，优先用诊断特征做排序侧修正'}；不能直接推翻当前主线。",
        "",
        "5. 哪些特征最值得在下一轮加入 base_alpha_v4_medium？",
    ]
    candidate_features = (
        stats_df.assign(abs_gap=stats_df["z_gap_vs_fold"].abs())
        .sort_values("abs_gap", ascending=False)["feature"]
        .drop_duplicates()
        .head(10)
        .tolist()
        if not stats_df.empty
        else []
    )
    if candidate_features:
        lines.extend([f"- `{feature}`" for feature in candidate_features])
    else:
        lines.append("- No feature candidates were available.")
    if missing_features:
        lines.extend(["", "## Missing Requested Features", "", *[f"- `{feature}`" for feature in missing_features]])
        lines.append("")
        lines.append("建议新增缺失特征，尤其是 `overheat_score`、`reversal_risk_score`、`close_position_20d` 这类显式风险保护字段。")
    if used_aliases:
        lines.extend(["", "## Alias Features Used", "", *[f"- {item}" for item in used_aliases]])

    lines.extend(
        [
            "",
            "## Fold Overview",
            "",
            "| fold | dates | model_top_ret | true_top_ret | hit_rate | missed | false_pos | poor_false_pos |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in fold_summary.iterrows():
        lines.append(
            f"| {int(row['fold_id'])} | {int(row['dates'])} | {row['model_top_return_mean']:.6f} | "
            f"{row['true_top_return_mean']:.6f} | {row['top_hit_rate']:.3f} | "
            f"{int(row['missed_winners_count'])} | {int(row['false_positives_count'])} | {int(row['poor_false_positives_count'])} |"
        )

    for fold_id in fold_summary["fold_id"].astype(int).tolist():
        lines.extend(["", f"## Fold {fold_id} Feature Gaps", "", "### Missed Winners"])
        lines.extend(top_feature_lines(stats_df, "missed_winners", fold_id=fold_id))
        lines.extend(["", "### False Positives"])
        lines.extend(top_feature_lines(stats_df, "false_positives", fold_id=fold_id))

    weakest = daily_df.sort_values("model_top_return_mean").head(10)
    lines.extend(
        [
            "",
            "## Weakest Dates",
            "",
            "| fold | date | model_top_ret | true_top_ret | hit_rate | missed | false_pos |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in weakest.iterrows():
        lines.append(
            f"| {int(row['fold_id'])} | {row['date']} | {row['model_top_return_mean']:.6f} | "
            f"{row['true_top_return_mean']:.6f} | {row['top_hit_rate']:.3f} | "
            f"{int(row['missed_winners_count'])} | {int(row['false_positives_count'])} |"
        )
    lines.append("")
    lines.append(f"Diagnostic note: missed rebound/momentum gap={missed_rebound_gap:.2f}.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build missed_winners and false_positives diagnostic sample libraries.")
    parser.add_argument("--pred_path", default=str(DEFAULT_PRED_PATH))
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--target_col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--true_top_k", type=int, default=5)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = resolve_path(args.pred_path)
    feature_path = resolve_path(args.feature_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_df, _, available_features, _, missing_features, used_aliases = load_inputs(pred_path, feature_path, args.target_col)
    ranked_df = assign_daily_ranks(all_df, args.target_col)
    missed_df, false_df, daily_df = collect_misranked(ranked_df, args.target_col, args.top_k, args.true_top_k)
    stats_df = feature_stats(ranked_df, missed_df, false_df, available_features)

    missed_path = output_dir / "missed_winners.csv"
    false_path = output_dir / "false_positives.csv"
    stats_path = output_dir / "misrank_feature_stats.csv"
    report_path = output_dir / "misrank_report.md"
    daily_path = output_dir / "misrank_daily_summary.csv"

    missed_df.to_csv(missed_path, index=False, encoding="utf-8-sig")
    false_df.to_csv(false_path, index=False, encoding="utf-8-sig")
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    report = build_report(
        missed_df=missed_df,
        false_df=false_df,
        daily_df=daily_df,
        stats_df=stats_df,
        missing_features=missing_features,
        used_aliases=used_aliases,
        top_k=args.top_k,
        true_top_k=args.true_top_k,
    )
    report_path.write_text(report, encoding="utf-8-sig")

    print(f"[misrank] missed_winners: {len(missed_df)} rows -> {missed_path}")
    print(f"[misrank] false_positives: {len(false_df)} rows -> {false_path}")
    print(f"[misrank] feature_stats: {len(stats_df)} rows -> {stats_path}")
    print(f"[misrank] report -> {report_path}")
    if missing_features:
        print(f"[misrank] missing requested features: {', '.join(missing_features)}")


if __name__ == "__main__":
    main()
