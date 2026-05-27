from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import DEFAULT_MODEL_DIR, load_or_generate_predictions, make_config, run_backtest
from load_submission_config import build_default_inference_args, load_submission_config
from result_validator import validate_result_file
from utils import build_portfolio_weights, ensure_dir, select_top_candidates


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "weight_cap_search"
DEFAULT_BASE_CONFIG = ROOT_DIR / "app" / "model" / "default_submission_config.json"
SUMMARY_COLUMNS = [
    "cap",
    "single_slice_score",
    "top5_equal_return",
    "top5_weighted_return",
    "max_single_weight",
    "max_single_contribution_ratio",
    "cost_after_return",
    "sharpe",
    "max_drawdown",
    "avg_turnover",
    "pass_basic_validation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search max_single_weight caps without retraining the model.")
    parser.add_argument("--pred_path", "--prediction_path", dest="pred_path", required=True)
    parser.add_argument("--data_path", "--feature_path", dest="data_path", required=True)
    parser.add_argument("--base_config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--caps", nargs="+", type=float, default=[0.25, 0.20, 0.18, 0.16, 0.14])
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--include_uncapped_baseline", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def cap_weights_by_original_proportion(raw_weights: pd.Series, cap: float | None) -> pd.Series:
    weights = pd.to_numeric(raw_weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    if cap is None:
        return weights
    cap = float(cap)
    if cap <= 0.0:
        return pd.Series(0.0, index=weights.index, dtype=float)
    if cap >= 1.0 or weights.empty:
        return weights

    target_total = min(float(weights.sum()), 1.0)
    capped = weights.clip(upper=cap).astype(float)
    for _ in range(len(capped) + 2):
        shortfall = target_total - float(capped.sum())
        if shortfall <= 1e-12:
            break
        room_mask = capped < cap - 1e-12
        if not bool(room_mask.any()):
            break
        base = weights.where(room_mask, 0.0)
        if float(base.sum()) <= 1e-12:
            base = pd.Series(np.where(room_mask, 1.0, 0.0), index=weights.index, dtype=float)
        add = shortfall * base / float(base.sum())
        capped = (capped + add).clip(upper=cap)
    return capped.clip(lower=0.0)


def normalize_prediction_frame(prediction_df: pd.DataFrame) -> pd.DataFrame:
    out = prediction_df.copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["pred_return"] = pd.to_numeric(out["pred_return"], errors="coerce")
    out["target_return"] = pd.to_numeric(out["target_return"], errors="coerce")
    return out.dropna(subset=["stock_id", "date", "pred_return", "target_return"]).copy()


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
        weighting_scheme="pred",
        weight_blend_alpha=1.0,
        max_single_weight=defaults["max_single_weight"],
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


def raw_pred_weights(selected: pd.DataFrame, top_k: int) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=float)
    raw = pd.to_numeric(selected["pred_return"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if float(raw.sum()) <= 1e-12:
        raw = pd.Series(1.0, index=selected.index, dtype=float)
    invested_ratio = min(len(selected), top_k) / float(top_k)
    return invested_ratio * raw / float(raw.sum())


def analyze_single_slice(
    prediction_df: pd.DataFrame,
    config: dict[str, Any],
    cap: float | None,
    cap_dir: Path,
) -> tuple[dict[str, float], pd.DataFrame]:
    defaults = build_default_inference_args(config)
    latest_date = prediction_df["date"].max()
    latest_df = prediction_df[prediction_df["date"] == latest_date].copy()
    selected, _ = select_top_candidates(
        latest_df=latest_df,
        top_k=int(defaults["top_k"]),
        primary_candidate_size=int(defaults["primary_candidate_size"]),
        max_volatility_20d_pct=float(defaults["max_volatility_20d_pct"]),
        max_volatility_5d_pct=float(defaults["max_volatility_5d_pct"]),
        turnover_rate_lower_pct=float(defaults["turnover_rate_lower_pct"]),
        turnover_rate_upper_pct=float(defaults["turnover_rate_upper_pct"]),
        turnover_ratio_upper_pct=float(defaults["turnover_ratio_upper_pct"]),
        risk_penalty_weight=float(defaults["risk_penalty_weight"]),
        sort_strategy=str(defaults["sort_strategy"]),
        enable_risk_filters=bool(defaults["enable_risk_filters"]),
        allow_cash_fallback=False,
    )
    selected = selected.copy()
    selected["raw_pred_weight"] = raw_pred_weights(selected, int(defaults["top_k"]))
    selected["weight"] = cap_weights_by_original_proportion(selected["raw_pred_weight"], cap)
    selected["weighted_contribution"] = selected["weight"] * selected["target_return"]
    total_abs_contribution = float(selected["weighted_contribution"].abs().sum())
    selected["contribution_ratio"] = (
        0.0 if total_abs_contribution <= 1e-12 else selected["weighted_contribution"].abs() / total_abs_contribution
    )
    equal_return = float(selected["target_return"].mean()) if not selected.empty else 0.0
    weighted_return = float(selected["weighted_contribution"].sum()) if not selected.empty else 0.0
    result_df = selected[["stock_id", "weight"]].copy()
    result_path = cap_dir / "result.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8", lineterminator="\n")
    detail_path = cap_dir / "single_slice_detail.csv"
    selected[
        ["stock_id", "pred_return", "target_return", "raw_pred_weight", "weight", "weighted_contribution", "contribution_ratio"]
    ].to_csv(detail_path, index=False, encoding="utf-8-sig")
    validation_ok = True
    try:
        validate_result_file(result_path)
    except Exception:
        validation_ok = False

    return (
        {
            "single_slice_score": weighted_return,
            "top5_equal_return": equal_return,
            "top5_weighted_return": weighted_return,
            "slice_max_single_weight": float(selected["weight"].max()) if not selected.empty else 0.0,
            "slice_max_single_contribution_ratio": float(selected["contribution_ratio"].max()) if not selected.empty else 0.0,
            "pass_basic_validation": bool(validation_ok),
        },
        selected,
    )


def run_cap_experiment(
    *,
    prediction_df: pd.DataFrame,
    prediction_source: str,
    base_args: argparse.Namespace,
    base_config: dict[str, Any],
    cap: float | None,
    output_dir: Path,
) -> dict[str, Any]:
    label = "none" if cap is None else f"{cap:.2f}"
    cap_dir = ensure_dir(output_dir / f"cap_{label}")
    config = make_config(
        base_args,
        overrides={
            "profile_name": f"weight_cap_{label}",
            "weighting_scheme": "pred",
            "weight_blend_alpha": 1.0,
            "max_single_weight": 1.0 if cap is None else float(cap),
        },
    )
    summary_df, daily_df, holdings_df = run_backtest(prediction_df=prediction_df, config=config, prediction_source=prediction_source)
    daily_df.to_csv(cap_dir / "backtest_daily_results.csv", index=False, encoding="utf-8-sig")
    holdings_df.to_csv(cap_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(cap_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    slice_metrics, _ = analyze_single_slice(prediction_df, base_config, cap, cap_dir)
    summary_row = summary_df.iloc[0].to_dict()
    return {
        "cap": label,
        "single_slice_score": slice_metrics["single_slice_score"],
        "top5_equal_return": slice_metrics["top5_equal_return"],
        "top5_weighted_return": slice_metrics["top5_weighted_return"],
        "max_single_weight": max(
            float(slice_metrics["slice_max_single_weight"]),
            float(summary_row.get("max_single_weight_observed", 0.0)),
        ),
        "max_single_contribution_ratio": max(
            float(slice_metrics["slice_max_single_contribution_ratio"]),
            float(summary_row.get("max_single_contribution_share", 0.0)),
        ),
        "cost_after_return": float(summary_row.get("cumulative_return_after_cost", 0.0)),
        "sharpe": float(summary_row.get("sharpe_after_cost", 0.0)),
        "max_drawdown": float(summary_row.get("max_drawdown_after_cost", 0.0)),
        "avg_turnover": float(summary_row.get("avg_turnover", 0.0)),
        "pass_basic_validation": bool(slice_metrics["pass_basic_validation"]),
        "_avg_max_single_contribution_ratio": float(summary_row.get("avg_max_single_contribution_share", 0.0)),
    }


def render_report(summary: pd.DataFrame, output_dir: Path) -> str:
    comparable = summary[summary["cap"] != "none"].copy()
    baseline = summary[summary["cap"] == "none"].iloc[0] if (summary["cap"] == "none").any() else None
    best_return = comparable.sort_values("cost_after_return", ascending=False).iloc[0] if not comparable.empty else None
    best_concentration = comparable.sort_values("max_single_contribution_ratio", ascending=True).iloc[0] if not comparable.empty else None
    robust_pool = comparable[comparable["cap"].isin(["0.20", "0.18"])].copy()
    robust_pick = None
    if not robust_pool.empty:
        robust_pick = robust_pool.sort_values(
            ["pass_basic_validation", "max_single_contribution_ratio", "cost_after_return"],
            ascending=[False, True, False],
        ).iloc[0]

    lines = [
        "# Weight Cap Search Report",
        "",
        f"- output_dir: `{output_dir}`",
        f"- rows: `{len(summary)}`",
        "",
        "## Key Findings",
        "",
    ]
    if baseline is not None:
        lines.append(
            f"- Uncapped baseline cost_after_return `{baseline['cost_after_return']:.6f}`, "
            f"max_single_contribution_ratio `{baseline['max_single_contribution_ratio']:.6f}`."
        )
    if best_return is not None:
        loss = 0.0 if baseline is None else float(baseline["cost_after_return"] - best_return["cost_after_return"])
        lines.append(f"- 收益损失最小的 cap: `{best_return['cap']}`，相对 uncapped 收益差 `{loss:.6f}`。")
    if best_concentration is not None:
        reduction = 0.0 if baseline is None else float(baseline["max_single_contribution_ratio"] - best_concentration["max_single_contribution_ratio"])
        lines.append(f"- 集中度下降最明显的 cap: `{best_concentration['cap']}`，贡献占比下降 `{reduction:.6f}`。")
    if robust_pick is not None:
        lines.append(
            f"- robust 默认建议优先考虑 `{robust_pick['cap']}`；"
            f"cost_after_return `{robust_pick['cost_after_return']:.6f}`，"
            f"max_single_contribution_ratio `{robust_pick['max_single_contribution_ratio']:.6f}`。"
        )
    else:
        lines.append("- 0.18/0.20 不在本次 caps 内，无法直接给出 robust cap 建议。")

    lines.extend(
        [
            "",
            "说明：max_single_weight cap 会直接降低单票仓位上限；但 max_single_contribution_ratio 还取决于所选股票真实收益分布，"
            "因此它不一定随 cap 线性下降。如果贡献占比下降不明显，应继续做 pred/equal blend 或排序层诊断。",
            "",
            "## Summary Table",
            "",
            "| cap | single_slice_score | cost_after_return | sharpe | max_drawdown | avg_turnover | max_contrib_ratio | valid |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['cap']} | {row['single_slice_score']:.6f} | {row['cost_after_return']:.6f} | "
            f"{row['sharpe']:.6f} | {row['max_drawdown']:.6f} | {row['avg_turnover']:.6f} | "
            f"{row['max_single_contribution_ratio']:.6f} | {bool(row['pass_basic_validation'])} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- aggressive: 可以保留 pred 无 cap 作为冲分候选，但需要接受单票贡献更集中的风险。",
        ]
    )
    if robust_pick is not None:
        lines.append(f"- robust: 建议采用 max_single_weight=`{robust_pick['cap']}` 作为候选默认值。")
    else:
        lines.append("- robust: 暂无建议，需要先纳入 0.18/0.20 cap 重新比较。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(resolve_path(args.output_dir))
    pred_path = resolve_path(args.pred_path)
    data_path = resolve_path(args.data_path)
    base_config_path = resolve_path(args.base_config)
    base_config = load_submission_config(base_config_path)
    base_args = build_backtest_args(base_config)

    prediction_df, prediction_source = load_or_generate_predictions(
        prediction_path=pred_path,
        feature_path=data_path,
        model_dir=resolve_path(args.model_dir),
    )
    prediction_df = normalize_prediction_frame(prediction_df)

    caps: list[float | None] = []
    if args.include_uncapped_baseline:
        caps.append(None)
    caps.extend(float(cap) for cap in args.caps)

    rows = []
    for cap in caps:
        rows.append(
            run_cap_experiment(
                prediction_df=prediction_df,
                prediction_source=prediction_source,
                base_args=base_args,
                base_config=base_config,
                cap=cap,
                output_dir=output_dir,
            )
        )

    summary = pd.DataFrame(rows)
    public_summary = summary[SUMMARY_COLUMNS].copy()
    summary_path = output_dir / "weight_cap_summary.csv"
    public_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path = output_dir / "weight_cap_report.md"
    report_path.write_text(render_report(public_summary, output_dir), encoding="utf-8-sig")
    metadata = {
        "pred_path": str(pred_path),
        "data_path": str(data_path),
        "base_config": str(base_config_path),
        "prediction_source": prediction_source,
        "caps": ["none" if cap is None else cap for cap in caps],
    }
    (output_dir / "weight_cap_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[weight_cap_search] wrote {summary_path}")
    print(f"[weight_cap_search] wrote {report_path}")
    print(public_summary.to_string(index=False))


if __name__ == "__main__":
    main()
