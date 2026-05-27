from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from load_submission_config import build_default_inference_args
from utils import build_portfolio_weights, select_top_candidates


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "performance_bottleneck"
SUMMARY_COLUMNS = [
    "date",
    "true_top5_return",
    "candidate_pool_true_top5_return",
    "model_top5_equal_return",
    "model_top5_weighted_return",
    "model_top5_hit_rate",
    "model_top10_hit_rate",
    "worst_stock_return_in_model_top5",
    "best_missed_stock_return",
    "weight_strategy_gain",
    "max_single_weight",
    "max_single_contribution_ratio",
]
CONTRIBUTION_COLUMNS = [
    "stock_id",
    "weight",
    "true_return",
    "weighted_contribution",
    "contribution_ratio",
]

COLUMN_ALIASES = {
    "stock_id": ["stock_id", "股票代码", "鑲＄エ浠ｇ爜", "code", "ticker", "symbol"],
    "date": ["date", "日期", "鏃ユ湡", "trade_date", "prediction_date"],
    "pred_return": ["pred_return", "predict_score", "prediction", "score", "pred", "y_pred"],
    "target_return": ["target_return", "future_return", "true_return", "return", "ret", "y_true"],
}

RISK_COLUMNS = [
    "volatility_5d",
    "volatility_20d",
    "turnover_rate",
    "turnover_ratio_10d",
    "amplitude_ratio_5d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether return bottlenecks come from pool, ranking, weighting, or concentration.")
    parser.add_argument("--pred_path", required=True, help="Prediction score CSV, e.g. app/model/walk_forward_predictions.csv")
    parser.add_argument("--data_path", required=True, help="CSV containing true future return, e.g. app/temp/train_features.csv")
    parser.add_argument("--result_path", help="Optional current result.csv for single-slice contribution recalculation")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_size", type=int, default=180)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_id": str})


def find_column(df: pd.DataFrame, canonical: str, required: bool = True) -> str | None:
    lower_map = {str(column).strip().lower(): column for column in df.columns}
    for alias in COLUMN_ALIASES[canonical]:
        key = alias.lower()
        if key in lower_map:
            return str(lower_map[key])
    if required:
        raise ValueError(
            f"Cannot detect required column `{canonical}`. "
            f"Tried aliases={COLUMN_ALIASES[canonical]}; available={list(df.columns)}"
        )
    return None


def normalize_frame(df: pd.DataFrame, *, need_pred: bool, need_target: bool, source_name: str) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for canonical, required in [
        ("stock_id", True),
        ("date", True),
        ("pred_return", need_pred),
        ("target_return", need_target),
    ]:
        column = find_column(df, canonical, required=required)
        if column and column != canonical:
            rename[column] = canonical
    out = df.rename(columns=rename).copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        bad = int(out["date"].isna().sum())
        raise ValueError(f"{source_name} contains {bad} invalid date rows")
    if need_pred:
        out["pred_return"] = pd.to_numeric(out["pred_return"], errors="coerce")
    if need_target:
        out["target_return"] = pd.to_numeric(out["target_return"], errors="coerce")
    return out


def merge_prediction_and_truth(pred_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    base = pred_df.copy()
    data_columns = ["stock_id", "date", "target_return", *[c for c in RISK_COLUMNS if c in data_df.columns]]
    if "target_return" in base.columns:
        data_columns = ["stock_id", "date", *[c for c in data_columns if c not in {"stock_id", "date", "target_return"}]]
    merged = base.merge(
        data_df[data_columns].drop_duplicates(["stock_id", "date"]),
        on=["stock_id", "date"],
        how="left",
        suffixes=("", "_data"),
    )
    if "target_return_data" in merged.columns and "target_return" not in merged.columns:
        merged["target_return"] = merged["target_return_data"]
    if "target_return" not in merged.columns:
        raise ValueError("No true future return column found after merging prediction and data files.")
    merged["target_return"] = pd.to_numeric(merged["target_return"], errors="coerce")
    merged["pred_return"] = pd.to_numeric(merged["pred_return"], errors="coerce")
    merged = merged.dropna(subset=["stock_id", "date", "pred_return", "target_return"]).copy()
    if merged.empty:
        raise ValueError("Merged prediction/data frame has no rows with both prediction and true future return.")
    return merged


def contribution_ratio(weights: pd.Series, returns: pd.Series) -> float:
    contributions = (weights.astype(float) * returns.astype(float)).abs()
    total = float(contributions.sum())
    if total <= 1e-12:
        return 0.0
    return float(contributions.max() / total)


def weighted_return(df: pd.DataFrame, weight_col: str = "weight") -> float:
    if df.empty or weight_col not in df:
        return 0.0
    return float((pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0) * df["target_return"]).sum())


def select_model_rows(day_df: pd.DataFrame, top_k: int, candidate_size: int, defaults: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    missing_risk = [column for column in RISK_COLUMNS if column not in day_df.columns]
    if missing_risk:
        selected = day_df.sort_values(["pred_return", "stock_id"], ascending=[False, True]).head(top_k).copy()
        selected["pred_rank_pct"] = selected["pred_return"].rank(pct=True)
        selected["risk_penalty"] = 0.0
        return selected, "pure_prediction_fallback_missing_" + ",".join(missing_risk)

    selected, _ = select_top_candidates(
        latest_df=day_df,
        top_k=top_k,
        primary_candidate_size=candidate_size,
        max_volatility_20d_pct=defaults["max_volatility_20d_pct"],
        max_volatility_5d_pct=defaults["max_volatility_5d_pct"],
        turnover_rate_lower_pct=defaults["turnover_rate_lower_pct"],
        turnover_rate_upper_pct=defaults["turnover_rate_upper_pct"],
        turnover_ratio_upper_pct=defaults["turnover_ratio_upper_pct"],
        risk_penalty_weight=defaults["risk_penalty_weight"],
        sort_strategy=defaults["sort_strategy"],
        enable_risk_filters=True,
        allow_cash_fallback=False,
    )
    return selected.copy(), defaults["sort_strategy"]


def selected_top_n(day_df: pd.DataFrame, n: int, candidate_size: int, defaults: dict[str, Any]) -> pd.DataFrame:
    selected, mode = select_model_rows(day_df, n, candidate_size, defaults)
    selected["selection_mode"] = mode
    return selected


def analyze_by_date(df: pd.DataFrame, top_k: int, candidate_size: int) -> tuple[pd.DataFrame, list[str]]:
    defaults = build_default_inference_args()
    notes: list[str] = []
    rows: list[dict[str, Any]] = []
    for trade_date, day_df in df.groupby("date"):
        day_df = day_df.copy().sort_values(["pred_return", "stock_id"], ascending=[False, True])
        true_top = day_df.sort_values(["target_return", "stock_id"], ascending=[False, True]).head(top_k)
        true_top10 = day_df.sort_values(["target_return", "stock_id"], ascending=[False, True]).head(max(10, top_k))
        candidate_pool = day_df.head(min(candidate_size, len(day_df)))
        candidate_true_top = candidate_pool.sort_values(["target_return", "stock_id"], ascending=[False, True]).head(top_k)

        model_top = selected_top_n(day_df, top_k, candidate_size, defaults)
        model_top10 = selected_top_n(day_df, max(10, top_k), candidate_size, defaults)
        if model_top["selection_mode"].iloc[0].startswith("pure_prediction_fallback"):
            notes.append(f"{trade_date.date()}: {model_top['selection_mode'].iloc[0]}")

        weighted = build_portfolio_weights(
            model_top,
            top_k=top_k,
            weighting_scheme=defaults["weighting_scheme"],
            max_single_weight=defaults["max_single_weight"],
            weight_blend_alpha=defaults["weight_blend_alpha"],
        )
        weighted = weighted.merge(
            model_top[["stock_id", "target_return"]].drop_duplicates("stock_id"),
            on="stock_id",
            how="left",
            suffixes=("", "_selected"),
        )
        if "target_return_selected" in weighted.columns:
            weighted["target_return"] = weighted["target_return"].fillna(weighted["target_return_selected"])

        true_top_ids = set(true_top["stock_id"].astype(str))
        true_top10_ids = set(true_top10["stock_id"].astype(str))
        model_top_ids = set(model_top["stock_id"].astype(str))
        model_top10_ids = set(model_top10["stock_id"].astype(str))
        missed_true_top = true_top[~true_top["stock_id"].astype(str).isin(model_top_ids)]

        equal_return = float(model_top["target_return"].mean()) if not model_top.empty else 0.0
        model_weighted_return = weighted_return(weighted)
        rows.append(
            {
                "date": trade_date.date().isoformat(),
                "true_top5_return": float(true_top["target_return"].mean()) if not true_top.empty else 0.0,
                "candidate_pool_true_top5_return": float(candidate_true_top["target_return"].mean()) if not candidate_true_top.empty else 0.0,
                "model_top5_equal_return": equal_return,
                "model_top5_weighted_return": model_weighted_return,
                "model_top5_hit_rate": len(model_top_ids & true_top_ids) / float(max(1, min(top_k, len(true_top_ids)))),
                "model_top10_hit_rate": len(model_top10_ids & true_top10_ids) / float(max(1, len(true_top10_ids))),
                "worst_stock_return_in_model_top5": float(model_top["target_return"].min()) if not model_top.empty else 0.0,
                "best_missed_stock_return": float(missed_true_top["target_return"].max()) if not missed_true_top.empty else 0.0,
                "weight_strategy_gain": model_weighted_return - equal_return,
                "max_single_weight": float(weighted["weight"].max()) if not weighted.empty else 0.0,
                "max_single_contribution_ratio": contribution_ratio(weighted["weight"], weighted["target_return"]) if not weighted.empty else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS), sorted(set(notes))


def analyze_result_contribution(result_path: Path, analysis_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    result = pd.read_csv(result_path, encoding="utf-8-sig", dtype={"stock_id": str})
    if not {"stock_id", "weight"}.issubset(result.columns):
        raise ValueError(f"{result_path} must contain stock_id and weight columns")
    latest_date = analysis_df["date"].max()
    latest_truth = analysis_df[analysis_df["date"] == latest_date][["stock_id", "target_return"]].drop_duplicates("stock_id")
    detail = result[["stock_id", "weight"]].copy()
    detail["stock_id"] = detail["stock_id"].astype(str).str.zfill(6)
    detail["weight"] = pd.to_numeric(detail["weight"], errors="coerce").fillna(0.0)
    detail = detail.merge(latest_truth, on="stock_id", how="left")
    missing = int(detail["target_return"].isna().sum())
    detail["true_return"] = pd.to_numeric(detail["target_return"], errors="coerce").fillna(0.0)
    detail["weighted_contribution"] = detail["weight"] * detail["true_return"]
    denom = float(detail["weighted_contribution"].abs().sum())
    detail["contribution_ratio"] = 0.0 if denom <= 1e-12 else detail["weighted_contribution"].abs() / denom
    note = (
        f"result.csv contribution uses latest analyzable date {latest_date.date()}; "
        f"missing_true_return_rows={missing}."
    )
    return detail[CONTRIBUTION_COLUMNS], note


def decide_bottleneck(summary: pd.DataFrame) -> tuple[str, dict[str, float]]:
    if summary.empty:
        return "ranking", {}
    gaps = {
        "candidate_gap": float((summary["true_top5_return"] - summary["candidate_pool_true_top5_return"]).mean()),
        "ranking_gap": float((summary["candidate_pool_true_top5_return"] - summary["model_top5_equal_return"]).mean()),
        "weighting_loss": float((-summary["weight_strategy_gain"]).clip(lower=0.0).mean()),
        "concentration": float(summary["max_single_contribution_ratio"].mean()),
    }
    if gaps["concentration"] >= 0.55:
        return "concentration", gaps
    candidates = {
        "candidate_pool": gaps["candidate_gap"],
        "ranking": gaps["ranking_gap"],
        "weighting": gaps["weighting_loss"],
    }
    return max(candidates, key=candidates.get), gaps


def yes_no(value: bool) -> str:
    return "是" if value else "否"


def render_report(
    *,
    pred_path: Path,
    data_path: Path,
    result_path: Path | None,
    summary: pd.DataFrame,
    bottleneck: str,
    gaps: dict[str, float],
    notes: list[str],
    missing_pred: bool = False,
    result_note: str = "",
) -> str:
    lines = ["# Performance Bottleneck Report", ""]
    lines.extend(
        [
            f"- pred_path: `{pred_path}`",
            f"- data_path: `{data_path}`",
            f"- result_path: `{result_path}`" if result_path else "- result_path: not provided",
            "",
        ]
    )
    if missing_pred:
        lines.extend(
            [
                "Prediction score file is missing, so the script cannot decompose candidate pool, ranking, or weighting yet.",
                "",
                f"Final judgment: `{bottleneck}`",
                "",
            ]
        )
        return "\n".join(lines)

    if summary.empty:
        lines.extend(["No analyzable rows were produced.", "", f"Final judgment: `{bottleneck}`", ""])
        return "\n".join(lines)

    avg = summary.mean(numeric_only=True)
    pool_leak = avg["candidate_pool_true_top5_return"] < avg["true_top5_return"] - 1e-12
    ranking_weak = avg["model_top5_equal_return"] < avg["candidate_pool_true_top5_return"] - 1e-12
    weighting_better = avg["weight_strategy_gain"] > 1e-12
    concentrated = avg["max_single_contribution_ratio"] >= 0.55

    lines.extend(
        [
            "## Summary Averages",
            "",
            f"- true_top5_return: `{avg['true_top5_return']:.6f}`",
            f"- candidate_pool_true_top5_return: `{avg['candidate_pool_true_top5_return']:.6f}`",
            f"- model_top5_equal_return: `{avg['model_top5_equal_return']:.6f}`",
            f"- model_top5_weighted_return: `{avg['model_top5_weighted_return']:.6f}`",
            f"- model_top5_hit_rate: `{avg['model_top5_hit_rate']:.6f}`",
            f"- model_top10_hit_rate: `{avg['model_top10_hit_rate']:.6f}`",
            f"- weight_strategy_gain: `{avg['weight_strategy_gain']:.6f}`",
            f"- max_single_contribution_ratio: `{avg['max_single_contribution_ratio']:.6f}`",
            "",
            "## Diagnosis",
            "",
            f"- 候选池是否漏掉真实高收益股票？{yes_no(pool_leak)}。candidate gap = `{gaps.get('candidate_gap', 0.0):.6f}`。",
            f"- 模型排序是否把候选池中的高收益股票排到前面？{'基本可以' if not ranking_weak else '仍有明显不足'}。ranking gap = `{gaps.get('ranking_gap', 0.0):.6f}`。",
            f"- 当前权重策略是否优于等权？{yes_no(weighting_better)}。平均增益 = `{avg['weight_strategy_gain']:.6f}`。",
            f"- 当前收益是否过度依赖单只股票？{yes_no(concentrated)}。平均最大单票贡献占比 = `{avg['max_single_contribution_ratio']:.6f}`。",
            f"- 下一步更应该改候选池、排序模型还是权重？建议优先看 `{bottleneck}`。",
            "",
            "## Bottleneck Gap Metrics",
            "",
            "| metric | value |",
            "|---|---:|",
        ]
    )
    for key, value in gaps.items():
        lines.append(f"| {key} | {value:.6f} |")
    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes[:20]:
            lines.append(f"- {note}")
        if len(notes) > 20:
            lines.append(f"- ... {len(notes) - 20} more similar notes omitted.")
    if result_note:
        lines.extend(["", "## Result Slice Contribution", "", f"- {result_note}"])
    lines.extend(["", f"Final judgment: `{bottleneck}`", ""])
    return "\n".join(lines)


def write_missing_pred_outputs(output_dir: Path, pred_path: Path, data_path: Path, result_path: Path | None) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(output_dir / "performance_bottleneck_summary.csv", index=False, encoding="utf-8-sig")
    report = render_report(
        pred_path=pred_path,
        data_path=data_path,
        result_path=result_path,
        summary=pd.DataFrame(columns=SUMMARY_COLUMNS),
        bottleneck="ranking",
        gaps={},
        notes=[],
        missing_pred=True,
    )
    (output_dir / "performance_bottleneck_report.md").write_text(report, encoding="utf-8-sig")
    return "ranking"


def main() -> None:
    args = parse_args()
    pred_path = resolve_path(args.pred_path)
    data_path = resolve_path(args.data_path)
    result_path = resolve_path(args.result_path) if args.result_path else None
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        bottleneck = write_missing_pred_outputs(output_dir, pred_path, data_path, result_path)
        print(f"[performance_bottleneck] prediction file missing: {pred_path}")
        print(f"[performance_bottleneck] wrote {output_dir / 'performance_bottleneck_report.md'}")
        print(f"[performance_bottleneck] bottleneck more likely in {bottleneck}")
        return
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data_path with true future return: {data_path}")

    pred_df = normalize_frame(read_csv(pred_path), need_pred=True, need_target=False, source_name=str(pred_path))
    data_df = normalize_frame(read_csv(data_path), need_pred=False, need_target=True, source_name=str(data_path))
    analysis_df = merge_prediction_and_truth(pred_df, data_df)
    summary, notes = analyze_by_date(analysis_df, top_k=int(args.top_k), candidate_size=int(args.candidate_size))
    bottleneck, gaps = decide_bottleneck(summary)

    summary_path = output_dir / "performance_bottleneck_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    result_note = ""
    if result_path is not None:
        if result_path.exists():
            contribution, result_note = analyze_result_contribution(result_path, analysis_df)
            contribution.to_csv(output_dir / "result_slice_contribution.csv", index=False, encoding="utf-8-sig")
        else:
            result_note = f"result.csv not found: {result_path}"
            pd.DataFrame(columns=CONTRIBUTION_COLUMNS).to_csv(
                output_dir / "result_slice_contribution.csv", index=False, encoding="utf-8-sig"
            )

    report = render_report(
        pred_path=pred_path,
        data_path=data_path,
        result_path=result_path,
        summary=summary,
        bottleneck=bottleneck,
        gaps=gaps,
        notes=notes,
        result_note=result_note,
    )
    report_path = output_dir / "performance_bottleneck_report.md"
    report_path.write_text(report, encoding="utf-8-sig")

    print(f"[performance_bottleneck] rows={len(summary)}")
    print(f"[performance_bottleneck] wrote {summary_path}")
    print(f"[performance_bottleneck] wrote {report_path}")
    if result_path is not None:
        print(f"[performance_bottleneck] wrote {output_dir / 'result_slice_contribution.csv'}")
    print(f"[performance_bottleneck] bottleneck more likely in {bottleneck}")


if __name__ == "__main__":
    main()
