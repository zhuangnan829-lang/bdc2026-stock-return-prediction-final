from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest import DEFAULT_MODEL_DIR, load_or_generate_predictions, make_config, run_backtest
from load_submission_config import build_default_inference_args, load_submission_config
from result_validator import validate_result_file
from utils import build_portfolio_weights, ensure_dir, select_top_candidates


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_BASE_CONFIG = ROOT_DIR / "app" / "model" / "default_submission_config.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "weight_blend_search"
SUMMARY_COLUMNS = [
    "alpha",
    "max_single_weight",
    "single_slice_score",
    "top5_equal_return",
    "top5_weighted_return",
    "cost_after_return",
    "sharpe",
    "max_drawdown",
    "avg_turnover",
    "max_single_contribution_ratio",
    "result_valid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search pred/equal blend weights without retraining the model.")
    parser.add_argument("--pred_path", "--prediction_path", dest="pred_path", required=True)
    parser.add_argument("--data_path", "--feature_path", dest="data_path", required=True)
    parser.add_argument("--base_config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--max_single_weights", nargs="+", default=["none", "0.20", "0.18"])
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def parse_cap(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "null", "nan", ""}:
        return None
    return float(text)


def cap_label(cap: float | None) -> str:
    return "none" if cap is None else f"{cap:.2f}"


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
        weighting_scheme="pred_equal_blend",
        weight_blend_alpha=1.0,
        max_single_weight=1.0,
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


def contribution_ratio(weights: pd.Series, returns: pd.Series) -> float:
    contribution = (weights.astype(float) * returns.astype(float)).abs()
    total = float(contribution.sum())
    if total <= 1e-12:
        return 0.0
    return float(contribution.max() / total)


def analyze_single_slice(
    prediction_df: pd.DataFrame,
    base_config: dict[str, Any],
    alpha: float,
    cap: float | None,
    combo_dir: Path,
) -> dict[str, Any]:
    defaults = build_default_inference_args(base_config)
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
    equal_weighted = build_portfolio_weights(
        selected,
        top_k=int(defaults["top_k"]),
        weighting_scheme="equal",
        max_single_weight=None,
        weight_blend_alpha=1.0,
    )
    weighted = build_portfolio_weights(
        selected,
        top_k=int(defaults["top_k"]),
        weighting_scheme="pred_equal_blend",
        max_single_weight=cap,
        weight_blend_alpha=alpha,
    )
    detail = weighted[["stock_id", "pred_return", "target_return", "weight"]].copy()
    detail["weighted_contribution"] = detail["weight"] * detail["target_return"]
    detail["contribution_ratio"] = 0.0
    denom = float(detail["weighted_contribution"].abs().sum())
    if denom > 1e-12:
        detail["contribution_ratio"] = detail["weighted_contribution"].abs() / denom
    detail.to_csv(combo_dir / "single_slice_detail.csv", index=False, encoding="utf-8-sig")

    result_path = combo_dir / "result.csv"
    detail[["stock_id", "weight"]].to_csv(result_path, index=False, encoding="utf-8", lineterminator="\n")
    result_valid = True
    try:
        validate_result_file(result_path)
    except Exception:
        result_valid = False

    top5_equal_return = float((equal_weighted["weight"] * equal_weighted["target_return"]).sum()) if not equal_weighted.empty else 0.0
    top5_weighted_return = float(detail["weighted_contribution"].sum()) if not detail.empty else 0.0
    return {
        "single_slice_score": top5_weighted_return,
        "top5_equal_return": top5_equal_return,
        "top5_weighted_return": top5_weighted_return,
        "slice_max_single_weight": float(detail["weight"].max()) if not detail.empty else 0.0,
        "slice_max_single_contribution_ratio": contribution_ratio(detail["weight"], detail["target_return"]) if not detail.empty else 0.0,
        "result_valid": bool(result_valid),
    }


def run_combo(
    *,
    prediction_df: pd.DataFrame,
    prediction_source: str,
    base_args: argparse.Namespace,
    base_config: dict[str, Any],
    alpha: float,
    cap: float | None,
    output_dir: Path,
) -> dict[str, Any]:
    label = f"alpha_{alpha:.2f}_cap_{cap_label(cap)}"
    combo_dir = ensure_dir(output_dir / label)
    config = make_config(
        base_args,
        overrides={
            "profile_name": label,
            "weighting_scheme": "pred_equal_blend",
            "weight_blend_alpha": float(alpha),
            "max_single_weight": 1.0 if cap is None else float(cap),
        },
    )
    summary_df, daily_df, holdings_df = run_backtest(prediction_df=prediction_df, config=config, prediction_source=prediction_source)
    daily_df.to_csv(combo_dir / "backtest_daily_results.csv", index=False, encoding="utf-8-sig")
    holdings_df.to_csv(combo_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(combo_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    slice_metrics = analyze_single_slice(prediction_df, base_config, alpha, cap, combo_dir)
    backtest_row = summary_df.iloc[0].to_dict()
    return {
        "alpha": float(alpha),
        "max_single_weight": cap_label(cap),
        "single_slice_score": float(slice_metrics["single_slice_score"]),
        "top5_equal_return": float(slice_metrics["top5_equal_return"]),
        "top5_weighted_return": float(slice_metrics["top5_weighted_return"]),
        "cost_after_return": float(backtest_row.get("cumulative_return_after_cost", 0.0)),
        "sharpe": float(backtest_row.get("sharpe_after_cost", 0.0)),
        "max_drawdown": float(backtest_row.get("max_drawdown_after_cost", 0.0)),
        "avg_turnover": float(backtest_row.get("avg_turnover", 0.0)),
        "max_single_contribution_ratio": max(
            float(slice_metrics["slice_max_single_contribution_ratio"]),
            float(backtest_row.get("max_single_contribution_share", 0.0)),
        ),
        "result_valid": bool(slice_metrics["result_valid"]),
    }


def pick_row(summary: pd.DataFrame, alpha: float, cap: str) -> pd.Series | None:
    matched = summary[(summary["alpha"].round(8) == round(float(alpha), 8)) & (summary["max_single_weight"] == cap)]
    if matched.empty:
        return None
    return matched.iloc[0]


def render_report(summary: pd.DataFrame, output_dir: Path) -> str:
    pred = pick_row(summary, 1.0, "none")
    equal = pick_row(summary, 0.0, "none")
    blend05 = pick_row(summary, 0.5, "none")
    blend05_cap18 = pick_row(summary, 0.5, "0.18")
    best_return = summary.sort_values(["cost_after_return", "sharpe"], ascending=[False, False]).iloc[0]
    best_robust = summary.sort_values(
        ["result_valid", "max_single_contribution_ratio", "avg_turnover", "cost_after_return"],
        ascending=[False, True, True, False],
    ).iloc[0]

    lines = [
        "# Weight Blend Search Report",
        "",
        f"- output_dir: `{output_dir}`",
        f"- grid_rows: `{len(summary)}`",
        "",
        "## Answers",
        "",
    ]
    if pred is not None and equal is not None:
        pred_better = float(pred["cost_after_return"]) > float(equal["cost_after_return"])
        lines.append(
            "- alpha=1.0 的 pred 权重是否真的优于等权？"
            + ("是" if pred_better else "否")
            + f"。pred cost_after_return `{pred['cost_after_return']:.6f}`，"
            + f"equal `{equal['cost_after_return']:.6f}`。"
        )
    else:
        lines.append("- alpha=1.0 与等权对比不可用，缺少 cap=none 的对应结果。")

    if blend05 is not None and pred is not None:
        return_loss = float(pred["cost_after_return"] - blend05["cost_after_return"])
        concentration_delta = float(pred["max_single_contribution_ratio"] - blend05["max_single_contribution_ratio"])
        balanced = return_loss <= 0.01 and concentration_delta > 0.0
        lines.append(
            f"- alpha=0.5 是否在收益和稳定性之间更均衡？{'是' if balanced else '否'}。"
            f"收益相对 pred 差 `{return_loss:.6f}`，"
            f"最大贡献占比下降 `{concentration_delta:.6f}`。"
        )
    else:
        lines.append("- alpha=0.5 的均衡性对比不可用。")

    if blend05_cap18 is not None:
        cap18_alpha_rows = summary[summary["max_single_weight"] == "0.18"].copy()
        alpha_irrelevant = (
            not cap18_alpha_rows.empty
            and cap18_alpha_rows["cost_after_return"].round(12).nunique() == 1
            and cap18_alpha_rows["single_slice_score"].round(12).nunique() == 1
        )
        lines.append(
            f"- 是否推荐 robust 使用 0.5*pred + 0.5*equal？{'不单独推荐' if alpha_irrelevant else '可以作为候选'}。cap=0.18 下 "
            f"cost_after_return `{blend05_cap18['cost_after_return']:.6f}`，"
            f"max_single_contribution_ratio `{blend05_cap18['max_single_contribution_ratio']:.6f}`。"
        )
        if alpha_irrelevant:
            lines.append("- 是否应该与 max_single_weight=0.18 联用？可以把 cap=0.18 作为 robust 候选；但本次五只股票均触及 cap，alpha 对结果几乎不起作用。")
        else:
            lines.append("- 是否应该与 max_single_weight=0.18 联用？若目标是降低仓位集中和换手，可以作为 robust 候选；若目标是最高收益，则不是最优。")
    else:
        lines.append("- cap=0.18 + alpha=0.5 不在结果中，无法评估联用。")

    lines.extend(
        [
            "",
            "## Picks",
            "",
            f"- Best return: alpha `{best_return['alpha']:.2f}`, cap `{best_return['max_single_weight']}`, "
            f"cost_after_return `{best_return['cost_after_return']:.6f}`.",
            f"- Lowest concentration robust candidate: alpha `{best_robust['alpha']:.2f}`, cap `{best_robust['max_single_weight']}`, "
            f"max_single_contribution_ratio `{best_robust['max_single_contribution_ratio']:.6f}`.",
            "",
            "## Summary Table",
            "",
            "| alpha | cap | single_slice | cost_after | sharpe | max_drawdown | avg_turnover | max_contrib | valid |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['alpha']:.2f} | {row['max_single_weight']} | {row['single_slice_score']:.6f} | "
            f"{row['cost_after_return']:.6f} | {row['sharpe']:.6f} | {row['max_drawdown']:.6f} | "
            f"{row['avg_turnover']:.6f} | {row['max_single_contribution_ratio']:.6f} | {bool(row['result_valid'])} |"
        )
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

    caps = [parse_cap(value) for value in args.max_single_weights]
    rows = []
    for alpha in args.alphas:
        for cap in caps:
            rows.append(
                run_combo(
                    prediction_df=prediction_df,
                    prediction_source=prediction_source,
                    base_args=base_args,
                    base_config=base_config,
                    alpha=float(alpha),
                    cap=cap,
                    output_dir=output_dir,
                )
            )

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary = summary.sort_values(["alpha", "max_single_weight"]).reset_index(drop=True)
    summary_path = output_dir / "weight_blend_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path = output_dir / "weight_blend_report.md"
    report_path.write_text(render_report(summary, output_dir), encoding="utf-8-sig")
    metadata = {
        "pred_path": str(pred_path),
        "data_path": str(data_path),
        "base_config": str(base_config_path),
        "prediction_source": prediction_source,
        "alphas": [float(value) for value in args.alphas],
        "max_single_weights": [cap_label(cap) for cap in caps],
    }
    (output_dir / "weight_blend_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[weight_blend_search] wrote {summary_path}")
    print(f"[weight_blend_search] wrote {report_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
