from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import load_prediction_frame, run_backtest
from config import ROOT_DIR
from evaluate_rank_stability import evaluate_one
from load_submission_config import build_default_inference_args, load_submission_config


DEFAULT_SEARCH_ROOT = ROOT_DIR / "app" / "model"
DEFAULT_FEATURE_PATHS = [
    ROOT_DIR / "app" / "model" / "offline_v4_medium_compare" / "temp" / "train_features.csv",
    ROOT_DIR / "app" / "temp" / "train_features.csv",
]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "feature_set_comparison"
TARGET_SPECS = [
    {
        "label": "v3_mini4_lstm_sl20",
        "feature_set": "base_alpha_v3_rs_crowding_mini4",
        "model": "lstm",
        "sequence_length": 20,
    },
    {
        "label": "v4_medium_lstm_sl20",
        "feature_set": "base_alpha_v4_medium",
        "model": "lstm",
        "sequence_length": 20,
    },
    {
        "label": "v4_medium_lstm_sl40",
        "feature_set": "base_alpha_v4_medium",
        "model": "lstm",
        "sequence_length": 40,
    },
    {
        "label": "v4_medium_lstm_sl60",
        "feature_set": "base_alpha_v4_medium",
        "model": "lstm",
        "sequence_length": 60,
    },
    {
        "label": "v4_medium_lightgbm",
        "feature_set": "base_alpha_v4_medium",
        "model": "lightgbm",
        "sequence_length": "",
    },
]
SUMMARY_COLUMNS = [
    "label",
    "status",
    "model",
    "feature_set",
    "sequence_length",
    "rank_ic_mean",
    "worst_fold_rank_ic",
    "top5_return_mean",
    "top5_return_min_by_fold",
    "cost_after_return",
    "sharpe",
    "max_drawdown",
    "avg_turnover",
    "single_slice_score",
    "negative_day_rank_ic_ratio",
    "adopt_rule_pass",
    "source_dir",
    "notes",
]
NEW_V4_FEATURES = [
    "ret_2d",
    "ret_7d",
    "ret_15d",
    "ret_20d",
    "amount_ratio_10d",
    "close_position_10d",
    "close_position_20d",
    "ret_1d_zscore_cross_section",
    "ret_3d_zscore_cross_section",
    "volume_spike_zscore",
    "turnover_spike_zscore",
    "overheat_score",
    "reversal_risk_score",
    "relative_to_market_5d",
    "relative_to_market_10d",
]


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def choose_feature_path(path_arg: str | None) -> Path:
    if path_arg:
        resolved = resolve_path(path_arg)
        if not resolved.exists():
            raise FileNotFoundError(f"feature_path not found: {resolved}")
        return resolved
    for path in DEFAULT_FEATURE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("No feature_path found. Run featurework first or pass --feature_path.")


def load_meta(model_dir: Path) -> dict[str, Any]:
    meta_path = model_dir / "model_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def meta_model(meta: dict[str, Any], path: Path) -> str:
    value = str(meta.get("model_family") or meta.get("backend") or "").lower()
    if "lstm" in value or (path / "lstm_model.pt").exists():
        return "lstm"
    if "lightgbm" in value or "lgbm" in value:
        return "lightgbm"
    if "xgboost" in value:
        return "xgboost"
    return value


def meta_feature_set(meta: dict[str, Any]) -> str:
    return str(meta.get("feature_set") or meta.get("training", {}).get("feature_set") or "")


def meta_sequence_length(meta: dict[str, Any]) -> Any:
    candidates = [
        meta.get("sequence_length"),
        meta.get("training", {}).get("sequence_length") if isinstance(meta.get("training"), dict) else None,
        meta.get("validation_scheme", {}).get("sequence_length") if isinstance(meta.get("validation_scheme"), dict) else None,
    ]
    for value in candidates:
        if value not in [None, ""]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
    return ""


def discover_runs(search_root: Path) -> list[dict[str, Any]]:
    runs = []
    for pred_path in sorted(search_root.rglob("walk_forward_predictions.csv")):
        model_dir = pred_path.parent
        meta = load_meta(model_dir)
        runs.append(
            {
                "model_dir": model_dir,
                "prediction_path": pred_path,
                "meta": meta,
                "feature_set": meta_feature_set(meta),
                "model": meta_model(meta, model_dir),
                "sequence_length": meta_sequence_length(meta),
                "mtime": pred_path.stat().st_mtime,
            }
        )
    return runs


def spec_match(run: dict[str, Any], spec: dict[str, Any]) -> bool:
    if run["feature_set"] != spec["feature_set"]:
        return False
    if spec["model"] not in str(run["model"]).lower():
        return False
    expected_sl = spec["sequence_length"]
    if expected_sl == "":
        return True
    try:
        return int(run["sequence_length"]) == int(expected_sl)
    except (TypeError, ValueError):
        return False


def choose_run(runs: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any] | None:
    matches = [run for run in runs if spec_match(run, spec)]
    if not matches:
        return None
    # Prefer the explicit offline comparison directory from Prompt 9, then latest artifact.
    matches.sort(
        key=lambda run: (
            "offline_v4_medium_compare" in str(run["model_dir"]),
            run["mtime"],
        ),
        reverse=True,
    )
    return matches[0]


def build_backtest_config() -> dict[str, Any]:
    cfg = load_submission_config()
    args = build_default_inference_args(cfg)
    return {
        "profile_name": "feature_set_comparison",
        "top_k": int(args["top_k"]),
        "primary_candidate_size": int(args["primary_candidate_size"]),
        "enable_risk_filters": bool(args["enable_risk_filters"]),
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": float(args["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(args["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(args["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(args["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(args["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(args["risk_penalty_weight"]),
        "weighting_scheme": str(args["weighting_scheme"]),
        "weight_blend_alpha": float(args.get("weight_blend_alpha", 1.0)),
        "max_single_weight": args.get("max_single_weight"),
        "sort_strategy": str(args["sort_strategy"]),
        "transaction_cost": float(args["transaction_cost"]),
        "max_turnover": float(args["max_turnover"]),
    }


def single_slice_score(backtest_daily: pd.DataFrame) -> float:
    if backtest_daily.empty or "net_return" not in backtest_daily.columns:
        return 0.0
    return float(pd.to_numeric(backtest_daily["net_return"], errors="coerce").fillna(0.0).iloc[-1])


def evaluate_run(run: dict[str, Any], spec: dict[str, Any], feature_path: Path, output_dir: Path) -> dict[str, Any]:
    pred_path = run["prediction_path"]
    model_dir = run["model_dir"]
    row, _, _ = evaluate_one(
        experiment_name=spec["label"],
        prediction_path=pred_path,
        fold_diagnostics_path=model_dir / "fold_diagnostics.csv" if (model_dir / "fold_diagnostics.csv").exists() else None,
        fold_daily_diagnostics_path=model_dir / "fold_daily_diagnostics.csv" if (model_dir / "fold_daily_diagnostics.csv").exists() else None,
    )
    prediction_df = load_prediction_frame(pred_path, feature_path)
    backtest_summary, backtest_daily, _ = run_backtest(
        prediction_df=prediction_df,
        config=build_backtest_config(),
        prediction_source=str(pred_path),
    )
    bt = backtest_summary.iloc[0] if not backtest_summary.empty else {}
    artifact_dir = output_dir / "artifacts" / spec["label"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    backtest_summary.to_csv(artifact_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    backtest_daily.to_csv(artifact_dir / "backtest_daily.csv", index=False, encoding="utf-8-sig")
    return {
        "label": spec["label"],
        "status": "ok",
        "model": spec["model"],
        "feature_set": spec["feature_set"],
        "sequence_length": spec["sequence_length"],
        "rank_ic_mean": float(row.get("rank_ic_mean", 0.0)),
        "worst_fold_rank_ic": float(row.get("worst_fold_rank_ic", 0.0)),
        "top5_return_mean": float(row.get("top5_return_mean", 0.0)),
        "top5_return_min_by_fold": float(row.get("top5_return_min_by_fold", 0.0)),
        "cost_after_return": float(bt.get("cumulative_return_after_cost", 0.0)),
        "sharpe": float(bt.get("sharpe_after_cost", 0.0)),
        "max_drawdown": float(bt.get("max_drawdown_after_cost", 0.0)),
        "avg_turnover": float(bt.get("avg_turnover", 0.0)),
        "single_slice_score": single_slice_score(backtest_daily),
        "negative_day_rank_ic_ratio": float(row.get("negative_day_rank_ic_ratio", 0.0)),
        "adopt_rule_pass": False,
        "source_dir": str(model_dir),
        "notes": "same_protocol_evaluated",
    }


def missing_row(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": spec["label"],
        "status": "missing",
        "model": spec["model"],
        "feature_set": spec["feature_set"],
        "sequence_length": spec["sequence_length"],
        "rank_ic_mean": 0.0,
        "worst_fold_rank_ic": 0.0,
        "top5_return_mean": 0.0,
        "top5_return_min_by_fold": 0.0,
        "cost_after_return": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "avg_turnover": 0.0,
        "single_slice_score": 0.0,
        "negative_day_rank_ic_ratio": 0.0,
        "adopt_rule_pass": False,
        "source_dir": "",
        "notes": "no_existing_artifact_found",
    }


def apply_adoption_rule(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    baseline = out[(out["label"].eq("v3_mini4_lstm_sl20")) & (out["status"].eq("ok"))]
    if baseline.empty:
        return out
    base = baseline.iloc[0]
    for idx, row in out.iterrows():
        if row["status"] != "ok" or row["feature_set"] != "base_alpha_v4_medium":
            continue
        not_worse = [
            row["top5_return_mean"] >= base["top5_return_mean"],
            row["worst_fold_rank_ic"] >= base["worst_fold_rank_ic"],
            row["sharpe"] >= base["sharpe"],
        ]
        drawdown_ok = row["max_drawdown"] >= base["max_drawdown"] - 0.01
        turnover_ok = row["avg_turnover"] <= base["avg_turnover"] + 0.03
        out.loc[idx, "adopt_rule_pass"] = bool(sum(not_worse) >= 2 and drawdown_ok and turnover_ok)
    return out


def analyze_new_feature_effects(pred_path: Path, feature_path: Path) -> pd.DataFrame:
    pred = pd.read_csv(pred_path, encoding="utf-8-sig", dtype={"stock_id": str})
    feat = pd.read_csv(feature_path, encoding="utf-8-sig", dtype={"stock_id": str})
    for df in [pred, feat]:
        df["stock_id"] = df["stock_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    available = [column for column in NEW_V4_FEATURES if column in feat.columns]
    if not available:
        return pd.DataFrame()
    merged = pred[["stock_id", "date", "target_return", "pred_return"]].merge(
        feat[["stock_id", "date", *available]].drop_duplicates(["stock_id", "date"]),
        on=["stock_id", "date"],
        how="left",
    )
    rows = []
    for column in available:
        data = merged[[column, "target_return", "pred_return"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(data) < 10 or data[column].nunique() <= 1:
            continue
        rows.append(
            {
                "feature": column,
                "spearman_to_target": float(data[column].corr(data["target_return"], method="spearman")),
                "spearman_to_prediction": float(data[column].corr(data["pred_return"], method="spearman")),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_to_target", key=lambda s: s.abs(), ascending=False)


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(summary_df: pd.DataFrame, feature_effects: pd.DataFrame, output_path: Path) -> None:
    ok = summary_df[summary_df["status"].eq("ok")].copy()
    baseline = ok[ok["label"].eq("v3_mini4_lstm_sl20")]
    v4_sl20 = ok[ok["label"].eq("v4_medium_lstm_sl20")]
    base = baseline.iloc[0] if not baseline.empty else None
    cand = v4_sl20.iloc[0] if not v4_sl20.empty else None
    if base is not None and cand is not None:
        top5_up = cand["top5_return_mean"] > base["top5_return_mean"]
        single_up = cand["single_slice_score"] > base["single_slice_score"]
        stable_down = cand["worst_fold_rank_ic"] < base["worst_fold_rank_ic"] or cand["negative_day_rank_ic_ratio"] > base["negative_day_rank_ic_ratio"] + 0.05
        overfit = cand["rank_ic_mean"] > base["rank_ic_mean"] and cand["cost_after_return"] < base["cost_after_return"]
        adopt = bool(cand["adopt_rule_pass"])
    else:
        top5_up = single_up = stable_down = overfit = adopt = False

    lines = [
        "# Feature Set Same-Protocol Comparison",
        "",
        "## Required Answers",
        "",
        f"1. v4_medium 是否真的提升 Top5 收益？{'是' if top5_up else '否'}。同协议 `top5_return_mean` 对比见下表。",
        f"2. 是否只提升单切片但降低 Walk-forward 稳定性？{'是' if (single_up and stable_down) else '否'}。",
        f"3. 是否存在明显过拟合？{'有迹象' if overfit else '暂未确认'}。判断依据是 RankIC 改善但 after-cost/Sharpe 转化不足。",
        "4. 哪些新增特征可能有效？见 `Potential Feature Signals`，按与真实收益的绝对 Spearman 相关排序。",
        f"5. 是否建议 v4_medium 进入下一轮主线或只作为候选？{'进入下一轮主线候选' if adopt else '只作为候选继续拆解，不进入主线'}。",
        "",
        "## Adoption Rule",
        "",
        "Only retain v4 if at least two of `top5_return_mean`, `worst_fold_rank_ic`, and `sharpe` are not worse than v3, without materially increasing drawdown or turnover.",
        "",
        "## Summary",
        "",
        "| label | status | rank_ic_mean | worst_fold | top5_mean | top5_min_fold | cost_after | sharpe | mdd | turnover | slice | neg_day_ic | adopt |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['label']} | {row['status']} | {fmt(row['rank_ic_mean'])} | {fmt(row['worst_fold_rank_ic'])} | "
            f"{fmt(row['top5_return_mean'])} | {fmt(row['top5_return_min_by_fold'])} | {fmt(row['cost_after_return'])} | "
            f"{fmt(row['sharpe'])} | {fmt(row['max_drawdown'])} | {fmt(row['avg_turnover'])} | "
            f"{fmt(row['single_slice_score'])} | {fmt(row['negative_day_rank_ic_ratio'])} | {bool(row['adopt_rule_pass'])} |"
        )
    lines.extend(["", "## Potential Feature Signals", ""])
    if feature_effects.empty:
        lines.append("- No v4 feature effect table could be computed.")
    else:
        lines.extend(
            [
                "| feature | spearman_to_target | spearman_to_prediction |",
                "|---|---:|---:|",
            ]
        )
        for _, row in feature_effects.head(12).iterrows():
            lines.append(
                f"| `{row['feature']}` | {fmt(row['spearman_to_target'])} | {fmt(row['spearman_to_prediction'])} |"
            )
    missing = summary_df[summary_df["status"].eq("missing")]
    if not missing.empty:
        lines.extend(["", "## Missing Artifacts", ""])
        for _, row in missing.iterrows():
            lines.append(f"- `{row['label']}`: {row['notes']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare feature sets using the same validation and backtest protocol.")
    parser.add_argument("--search_root", default=str(DEFAULT_SEARCH_ROOT))
    parser.add_argument("--feature_path", default="")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_root = resolve_path(args.search_root)
    feature_path = choose_feature_path(args.feature_path or None)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(search_root)
    rows = []
    selected_runs: dict[str, dict[str, Any]] = {}
    for spec in TARGET_SPECS:
        run = choose_run(runs, spec)
        if run is None:
            rows.append(missing_row(spec))
            continue
        selected_runs[spec["label"]] = run
        try:
            rows.append(evaluate_run(run, spec, feature_path, output_dir))
        except Exception as exc:
            failed = missing_row(spec)
            failed["status"] = "failed"
            failed["source_dir"] = str(run["model_dir"])
            failed["notes"] = f"{type(exc).__name__}: {exc}"
            rows.append(failed)

    summary_df = apply_adoption_rule(pd.DataFrame(rows)[SUMMARY_COLUMNS])
    summary_path = output_dir / "feature_set_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    feature_effects = pd.DataFrame()
    v4_run = selected_runs.get("v4_medium_lstm_sl20")
    if v4_run is not None:
        feature_effects = analyze_new_feature_effects(v4_run["prediction_path"], feature_path)
        if not feature_effects.empty:
            feature_effects.to_csv(output_dir / "v4_medium_feature_signal_stats.csv", index=False, encoding="utf-8-sig")

    report_path = output_dir / "feature_set_report.md"
    write_report(summary_df, feature_effects, report_path)
    print(summary_df.to_string(index=False))
    print(f"[feature_set_compare] feature_path={feature_path}")
    print(f"[feature_set_compare] wrote {summary_path}")
    print(f"[feature_set_compare] wrote {report_path}")


if __name__ == "__main__":
    main()
