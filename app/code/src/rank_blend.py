from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import make_config, run_backtest
from config import ROOT_DIR
from evaluate_rank_stability import build_daily_rank_ic, build_fold_rank_ic
from load_submission_config import build_default_inference_args, load_submission_config
from utils import ensure_dir, load_feature_frame


DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "rank_blend"
DEFAULT_BASE_CONFIG = ROOT_DIR / "app" / "model" / "default_submission_config.json"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_LSTM_PATH = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"
DEFAULT_LGB_PATH = ROOT_DIR / "app" / "model" / "baseline_lightgbm_same_protocol" / "walk_forward_predictions.csv"
DEFAULT_XGB_PATH = ROOT_DIR / "app" / "model" / "xgboost_baseline" / "walk_forward_predictions.csv"
DEFAULT_ENRICHED_LSTM_PATH = (
    ROOT_DIR / "app" / "model" / "v4_rerank_penalty_search" / "baseline_predictions_with_v4_risk_features.csv"
)

SUMMARY_COLUMNS = [
    "blend_name",
    "requested_components",
    "used_components",
    "missing_components",
    "rank_ic_mean",
    "worst_fold_rank_ic",
    "top5_return_mean",
    "top5_return_min_by_fold",
    "NDCG@5",
    "HitRate@5",
    "cost_after_return",
    "Sharpe",
    "max_drawdown",
    "avg_turnover",
    "single_slice_score",
    "max_single_contribution_ratio",
    "result_source_count",
    "strict_blend",
]

BLEND_SPECS: dict[str, dict[str, float]] = {
    "A_lstm70_lightgbm30": {"lstm": 0.7, "lightgbm": 0.3},
    "B_lstm60_lightgbm20_momentum20": {"lstm": 0.6, "lightgbm": 0.2, "momentum": 0.2},
    "C_lstm50_lightgbm30_reversal20": {"lstm": 0.5, "lightgbm": 0.3, "reversal": 0.2},
    "D_lstm50_lightgbm25_xgboost25": {"lstm": 0.5, "lightgbm": 0.25, "xgboost": 0.25},
    "E_lstm60_momentum40": {"lstm": 0.6, "momentum": 0.4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend model scores by daily rank percentile, not raw prediction values.")
    parser.add_argument("--lstm_path", default=str(DEFAULT_LSTM_PATH))
    parser.add_argument("--lightgbm_path", default=str(DEFAULT_LGB_PATH))
    parser.add_argument("--xgboost_path", default=str(DEFAULT_XGB_PATH))
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--enriched_lstm_path", default=str(DEFAULT_ENRICHED_LSTM_PATH))
    parser.add_argument("--base_config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def read_prediction_source(path: Path, component: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_id": str})
    required = {"stock_id", "date", "pred_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{component} score file missing columns {sorted(missing)}: {path}")
    out = df[["stock_id", "date", "pred_return"]].copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[f"{component}_score"] = pd.to_numeric(out["pred_return"], errors="coerce")
    return out.drop(columns=["pred_return"]).dropna(subset=["stock_id", "date", f"{component}_score"])


def load_base_frame(lstm_path: Path, feature_path: Path) -> pd.DataFrame:
    pred = pd.read_csv(lstm_path, encoding="utf-8-sig", dtype={"stock_id": str})
    required = {"stock_id", "date", "target_return", "pred_return"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"LSTM prediction file missing columns {sorted(missing)}: {lstm_path}")
    pred["stock_id"] = pred["stock_id"].astype(str).str.zfill(6)
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
    pred["target_return"] = pd.to_numeric(pred["target_return"], errors="coerce")
    pred["pred_return"] = pd.to_numeric(pred["pred_return"], errors="coerce")
    keep_cols = ["stock_id", "date", "target_return", "pred_return"]
    if "fold_id" in pred.columns:
        keep_cols.append("fold_id")
    base = pred[keep_cols].rename(columns={"pred_return": "lstm_score"}).dropna(
        subset=["stock_id", "date", "target_return", "lstm_score"]
    )

    features = load_feature_frame(feature_path)
    merge_cols = [
        "stock_id",
        "date",
        "volatility_5d",
        "volatility_20d",
        "turnover_rate",
        "turnover_ratio_10d",
        "amplitude_ratio_5d",
        "turnover_spike_5d",
        "crowding_reversal_risk_5d",
        "rel_strength_accel_5d_v2",
        "trend_persistence_score_10d_v2",
        "ret_5d",
        "mom_5d",
        "mom_10d",
        "rel_ret_5d",
    ]
    available = [column for column in merge_cols if column in features.columns]
    merged = base.merge(features[available].drop_duplicates(["stock_id", "date"]), on=["stock_id", "date"], how="left")
    return merged


def add_optional_enriched_features(base: pd.DataFrame, enriched_path: Path | None) -> pd.DataFrame:
    if enriched_path is None or not enriched_path.exists():
        return base
    enriched = pd.read_csv(enriched_path, encoding="utf-8-sig", dtype={"stock_id": str})
    enriched["stock_id"] = enriched["stock_id"].astype(str).str.zfill(6)
    enriched["date"] = pd.to_datetime(enriched["date"], errors="coerce")
    optional_cols = [
        "stock_id",
        "date",
        "reversal_risk_score",
        "overheat_score",
        "close_position_20d",
        "ret_3d_zscore_cross_section",
    ]
    available = [column for column in optional_cols if column in enriched.columns]
    if len(available) <= 2:
        return base
    out = base.merge(enriched[available].drop_duplicates(["stock_id", "date"]), on=["stock_id", "date"], how="left")
    return out


def add_factor_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    out = frame.copy()
    notes: dict[str, str] = {}
    momentum_candidates = [column for column in ["rel_strength_accel_5d_v2", "rel_ret_5d", "mom_5d", "ret_5d"] if column in out.columns]
    if momentum_candidates:
        ranks = []
        for column in momentum_candidates:
            values = pd.to_numeric(out[column], errors="coerce")
            ranks.append(values.groupby(out["date"]).rank(pct=True))
        out["momentum_score"] = pd.concat(ranks, axis=1).mean(axis=1)
        notes["momentum"] = " + ".join(momentum_candidates)
    if "reversal_risk_score" in out.columns:
        out["reversal_score"] = -pd.to_numeric(out["reversal_risk_score"], errors="coerce")
        notes["reversal"] = "-reversal_risk_score"
    elif "crowding_reversal_risk_5d" in out.columns:
        out["reversal_score"] = -pd.to_numeric(out["crowding_reversal_risk_5d"], errors="coerce")
        notes["reversal"] = "-crowding_reversal_risk_5d"
    return out, notes


def merge_component_scores(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    lstm_path = resolve_path(args.lstm_path)
    feature_path = resolve_path(args.feature_path)
    assert lstm_path is not None and feature_path is not None
    if not lstm_path.exists():
        raise FileNotFoundError(f"Missing required LSTM score file: {lstm_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_path}")

    base = load_base_frame(lstm_path, feature_path)
    base = add_optional_enriched_features(base, resolve_path(args.enriched_lstm_path))
    available: dict[str, str] = {"lstm": str(lstm_path)}
    skipped: dict[str, str] = {}

    for component, attr in [("lightgbm", "lightgbm_path"), ("xgboost", "xgboost_path")]:
        path = resolve_path(getattr(args, attr))
        if path is None or not path.exists():
            skipped[component] = f"missing file: {path}"
            continue
        source = read_prediction_source(path, component)
        if source is None:
            skipped[component] = f"missing file: {path}"
            continue
        base = base.merge(source, on=["stock_id", "date"], how="left")
        available[component] = str(path)

    base, factor_notes = add_factor_scores(base)
    for component, description in factor_notes.items():
        score_col = f"{component}_score"
        if score_col in base.columns and base[score_col].notna().any():
            available[component] = description
    for component in ["momentum", "reversal"]:
        if component not in available:
            skipped[component] = "no usable factor columns found"
    return base, available, skipped


def daily_rank_percentile(df: pd.DataFrame, score_col: str) -> pd.Series:
    values = pd.to_numeric(df[score_col], errors="coerce")
    return values.groupby(df["date"]).rank(pct=True)


def build_blend_prediction(
    base: pd.DataFrame,
    blend_name: str,
    requested_weights: dict[str, float],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    rank_cols: dict[str, pd.Series] = {}
    missing = []
    for component in requested_weights:
        score_col = f"{component}_score"
        if score_col not in base.columns or not base[score_col].notna().any():
            missing.append(component)
            continue
        rank_cols[component] = daily_rank_percentile(base, score_col)

    if not rank_cols:
        raise ValueError(f"No usable components for blend {blend_name}")
    used = list(rank_cols)
    total_weight = sum(float(requested_weights[component]) for component in used)
    if total_weight <= 1e-12:
        raise ValueError(f"Non-positive usable weight for blend {blend_name}")
    out = base.copy()
    blended = pd.Series(0.0, index=out.index, dtype=float)
    for component, ranks in rank_cols.items():
        blended += ranks.fillna(0.5) * (float(requested_weights[component]) / total_weight)
    out["pred_return"] = blended
    out["blend_name"] = blend_name
    return out, used, missing


def ndcg_at_k(true_relevance: np.ndarray, k: int = 5) -> float:
    if true_relevance.size == 0:
        return 0.0
    rel = true_relevance[:k]
    gains = np.power(2.0, rel) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(true_relevance)[::-1][:k]
    ideal_gains = np.power(2.0, ideal) - 1.0
    idcg = float(np.sum(ideal_gains * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 1e-12 else 0.0


def topk_quality_metrics(prediction_df: pd.DataFrame, top_k: int = 5) -> dict[str, float]:
    ndcg_values = []
    hit_values = []
    for _, day_df in prediction_df.groupby("date", sort=True):
        day = day_df.dropna(subset=["pred_return", "target_return"]).copy()
        if len(day) < top_k:
            continue
        day["true_rank_pct"] = day["target_return"].rank(pct=True)
        pred_top = day.sort_values(["pred_return", "stock_id"], ascending=[False, True]).head(top_k)
        true_top = set(day.sort_values(["target_return", "stock_id"], ascending=[False, True]).head(top_k)["stock_id"])
        pred_set = set(pred_top["stock_id"])
        hit_values.append(len(pred_set & true_top) / float(top_k))
        ordered_true_relevance = pred_top["true_rank_pct"].to_numpy(dtype=float)
        all_true_relevance = day.sort_values(["pred_return", "stock_id"], ascending=[False, True])["true_rank_pct"].to_numpy(dtype=float)
        ndcg_values.append(ndcg_at_k(all_true_relevance, k=top_k) if ordered_true_relevance.size else 0.0)
    return {
        "NDCG@5": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "HitRate@5": float(np.mean(hit_values)) if hit_values else 0.0,
    }


def summarize_rank_metrics(prediction_df: pd.DataFrame, top_k: int = 5) -> dict[str, float]:
    daily = build_daily_rank_ic(prediction_df, prediction_column="pred_return", target_column="target_return", top_k=top_k)
    fold = build_fold_rank_ic(daily)
    fold_rank_ic = pd.to_numeric(fold.get("rank_ic", pd.Series(dtype=float)), errors="coerce").dropna()
    fold_top5 = pd.to_numeric(fold.get("top5_mean_return", pd.Series(dtype=float)), errors="coerce").dropna()
    quality = topk_quality_metrics(prediction_df, top_k=top_k)
    return {
        "rank_ic_mean": float(fold_rank_ic.mean()) if not fold_rank_ic.empty else 0.0,
        "worst_fold_rank_ic": float(fold_rank_ic.min()) if not fold_rank_ic.empty else 0.0,
        "top5_return_mean": float(fold_top5.mean()) if not fold_top5.empty else 0.0,
        "top5_return_min_by_fold": float(fold_top5.min()) if not fold_top5.empty else 0.0,
        **quality,
    }


def build_backtest_args(config: dict[str, Any]) -> argparse.Namespace:
    defaults = build_default_inference_args(config)
    return argparse.Namespace(
        top_k=int(defaults["top_k"]),
        primary_candidate_size=int(defaults["primary_candidate_size"]),
        enable_risk_filters=int(defaults["enable_risk_filters"]),
        allow_cash_fallback=0,
        max_volatility_20d_pct=float(defaults["max_volatility_20d_pct"]),
        max_volatility_5d_pct=float(defaults["max_volatility_5d_pct"]),
        turnover_rate_lower_pct=float(defaults["turnover_rate_lower_pct"]),
        turnover_rate_upper_pct=float(defaults["turnover_rate_upper_pct"]),
        turnover_ratio_upper_pct=float(defaults["turnover_ratio_upper_pct"]),
        risk_penalty_weight=float(defaults["risk_penalty_weight"]),
        weighting_scheme=str(defaults["weighting_scheme"]),
        weight_blend_alpha=float(defaults["weight_blend_alpha"]),
        max_single_weight=float(defaults["max_single_weight"]),
        sort_strategy=str(defaults["sort_strategy"]),
        transaction_cost=float(defaults["transaction_cost"]),
        max_turnover=float(defaults["max_turnover"]),
        rerank_signal_column=None,
        rerank_signal_weight=0.0,
        secondary_candidate_size=None,
        secondary_screen_mode="none",
        secondary_screen_weight=0.0,
        local_tiebreak_start_rank=8,
        local_tiebreak_end_rank=15,
    )


def result_from_holdings(holdings_df: pd.DataFrame, path: Path) -> None:
    if holdings_df.empty:
        pd.DataFrame(columns=["stock_id", "weight"]).to_csv(path, index=False, encoding="utf-8")
        return
    latest = holdings_df["date"].max()
    result = (
        holdings_df[holdings_df["date"].eq(latest)]
        .sort_values(["executed_weight", "stock_id"], ascending=[False, True])
        .loc[:, ["stock_id", "executed_weight"]]
        .rename(columns={"executed_weight": "weight"})
    )
    result.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def evaluate_blend(
    base: pd.DataFrame,
    blend_name: str,
    requested_weights: dict[str, float],
    base_args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    profile_dir = ensure_dir(str(output_dir / blend_name))
    prediction_df, used, missing = build_blend_prediction(base, blend_name, requested_weights)
    prediction_df.to_csv(profile_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")

    config = make_config(base_args, overrides={"profile_name": blend_name})
    bt_summary, bt_daily, bt_holdings = run_backtest(prediction_df, config, prediction_source=blend_name)
    bt_summary.to_csv(profile_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    bt_daily.to_csv(profile_dir / "backtest_daily_results.csv", index=False, encoding="utf-8-sig")
    bt_holdings.to_csv(profile_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    result_from_holdings(bt_holdings, profile_dir / "result.csv")

    rank = summarize_rank_metrics(prediction_df, top_k=int(base_args.top_k))
    bt = bt_summary.iloc[0].to_dict()
    latest_date = prediction_df["date"].max()
    latest_selected = bt_holdings[bt_holdings["date"].eq(pd.to_datetime(latest_date).date().isoformat())].copy()
    single_slice_score = (
        float((latest_selected["executed_weight"] * latest_selected["target_return"]).sum())
        if not latest_selected.empty
        else 0.0
    )
    return {
        "blend_name": blend_name,
        "requested_components": json.dumps(requested_weights, ensure_ascii=False, sort_keys=True),
        "used_components": ",".join(used),
        "missing_components": ",".join(missing),
        **rank,
        "cost_after_return": float(bt.get("cumulative_return_after_cost", 0.0)),
        "Sharpe": float(bt.get("sharpe_after_cost", 0.0)),
        "max_drawdown": float(bt.get("max_drawdown_after_cost", 0.0)),
        "avg_turnover": float(bt.get("avg_turnover", 0.0)),
        "single_slice_score": single_slice_score,
        "max_single_contribution_ratio": float(bt.get("max_single_contribution_share", 0.0)),
        "result_source_count": len(used),
        "strict_blend": len(missing) == 0 and len(used) >= 2,
    }


def write_report(summary: pd.DataFrame, output_dir: Path, available: dict[str, str], skipped: dict[str, str]) -> None:
    baseline = summary[summary["blend_name"].eq("baseline_lstm_single")].iloc[0]
    candidates = summary[summary["blend_name"].ne("baseline_lstm_single")].copy()
    best_stability = candidates.sort_values(
        ["worst_fold_rank_ic", "rank_ic_mean", "cost_after_return"], ascending=[False, False, False]
    ).iloc[0]
    best_return = candidates.sort_values(
        ["cost_after_return", "Sharpe", "worst_fold_rank_ic"], ascending=[False, False, False]
    ).iloc[0]
    best_concentration = candidates.sort_values(
        ["max_single_contribution_ratio", "cost_after_return"], ascending=[True, False]
    ).iloc[0]

    viable = candidates[
        (
            (candidates["worst_fold_rank_ic"] > baseline["worst_fold_rank_ic"] + 1e-6)
            | (candidates["cost_after_return"] > baseline["cost_after_return"] + 0.02)
            | (candidates["Sharpe"] > baseline["Sharpe"] + 0.05)
        )
        & (candidates["max_drawdown"] >= baseline["max_drawdown"] - 0.02)
        & (candidates["avg_turnover"] <= baseline["avg_turnover"] + 0.05)
    ].copy()
    adopted = not viable.empty
    recommendation = (
        viable.sort_values(["worst_fold_rank_ic", "cost_after_return", "Sharpe"], ascending=[False, False, False]).iloc[0]
        if adopted
        else best_stability
    )

    lines = [
        "# Rank Blend Report",
        "",
        "All blends convert each component score to daily rank percentile before weighted averaging. Raw prediction values are not directly averaged.",
        "",
        "## Available Sources",
        "",
    ]
    for name, source in available.items():
        lines.append(f"- {name}: `{source}`")
    if skipped:
        lines.extend(["", "## Skipped Sources", ""])
        for name, reason in skipped.items():
            lines.append(f"- {name}: {reason}")

    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. 融合是否提高 worst fold: {'yes' if best_stability['worst_fold_rank_ic'] > baseline['worst_fold_rank_ic'] else 'no'}, "
            f"best `{best_stability['blend_name']}` worst_fold `{best_stability['worst_fold_rank_ic']:.6f}` vs baseline `{baseline['worst_fold_rank_ic']:.6f}`.",
            f"2. 融合是否降低单票集中度: {'yes' if best_concentration['max_single_contribution_ratio'] < baseline['max_single_contribution_ratio'] else 'no'}, "
            f"best `{best_concentration['blend_name']}` max_contrib `{best_concentration['max_single_contribution_ratio']:.6f}` vs baseline `{baseline['max_single_contribution_ratio']:.6f}`.",
            f"3. 融合是否提高收益或 Sharpe: {'yes' if (best_return['cost_after_return'] > baseline['cost_after_return'] or best_return['Sharpe'] > baseline['Sharpe']) else 'no'}, "
            f"best return `{best_return['blend_name']}` cost_after `{best_return['cost_after_return']:.6f}` Sharpe `{best_return['Sharpe']:.6f}`.",
            f"4. 是否建议替代 LSTM 单模型: {'yes' if adopted and recommendation['cost_after_return'] >= baseline['cost_after_return'] and recommendation['worst_fold_rank_ic'] >= baseline['worst_fold_rank_ic'] else 'no'}.",
            f"5. 如果不替代，是否适合作为 robust 配置: {'yes' if adopted else 'no'}, recommended candidate `{recommendation['blend_name']}`.",
            "",
            "## Summary Table",
            "",
            "| blend | used | missing | rank_ic | worst_fold | top5 | min_fold_top5 | ndcg5 | hit5 | cost_after | sharpe | max_dd | turnover | single_slice | max_contrib | strict |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['blend_name']} | {row['used_components']} | {row['missing_components']} | "
            f"{row['rank_ic_mean']:.6f} | {row['worst_fold_rank_ic']:.6f} | {row['top5_return_mean']:.6f} | "
            f"{row['top5_return_min_by_fold']:.6f} | {row['NDCG@5']:.6f} | {row['HitRate@5']:.6f} | "
            f"{row['cost_after_return']:.6f} | {row['Sharpe']:.6f} | {row['max_drawdown']:.6f} | "
            f"{row['avg_turnover']:.6f} | {row['single_slice_score']:.6f} | "
            f"{row['max_single_contribution_ratio']:.6f} | {str(bool(row['strict_blend'])).lower()} |"
        )
    lines.append("")
    (output_dir / "blend_report.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(str(resolve_path(args.output_dir)))
    base_config_path = resolve_path(args.base_config)
    assert base_config_path is not None
    base_config = load_submission_config(base_config_path)
    base_args = build_backtest_args(base_config)
    base, available, skipped = merge_component_scores(args)

    rows: list[dict[str, Any]] = []
    baseline_spec = {"lstm": 1.0}
    rows.append(evaluate_blend(base, "baseline_lstm_single", baseline_spec, base_args, output_dir))
    for blend_name, weights in BLEND_SPECS.items():
        print(f"[rank_blend] running {blend_name}")
        rows.append(evaluate_blend(base, blend_name, weights, base_args, output_dir))

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_path = output_dir / "blend_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary, output_dir, available, skipped)
    metadata = {
        "base_config": str(base_config_path),
        "available_sources": available,
        "skipped_sources": skipped,
        "blend_specs": BLEND_SPECS,
    }
    (output_dir / "rank_blend_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rank_blend] wrote {summary_path}")
    print(f"[rank_blend] wrote {output_dir / 'blend_report.md'}")
    print(summary[["blend_name", "used_components", "rank_ic_mean", "worst_fold_rank_ic", "cost_after_return", "Sharpe"]].to_string(index=False))


if __name__ == "__main__":
    main()
