from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from backtest import load_prediction_frame, run_backtest
from config import ROOT_DIR
from evaluate_rank_stability import evaluate_one
from load_submission_config import build_default_inference_args, load_submission_config


DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "sequence_length_search"
DEFAULT_BASE_CONFIG = ROOT_DIR / "app" / "model" / "default_submission_config.json"
FEATURE_SETS = ["base_alpha_v3_rs_crowding_mini4", "base_alpha_v4_medium"]
SEQUENCE_LENGTHS = [20, 40, 60]
SUMMARY_COLUMNS = [
    "experiment_id",
    "feature_set",
    "sequence_length",
    "status",
    "rank_ic_mean",
    "worst_fold_rank_ic",
    "top5_return_mean",
    "top5_return_min_by_fold",
    "cost_after_return",
    "Sharpe",
    "max_drawdown",
    "avg_turnover",
    "single_slice_score",
    "training_time",
    "train_sequence_rows_full",
    "sample_retention_ratio",
    "source_dir",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or summarize LSTM sl20/sl40/sl60 sequence-length candidates.")
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base_config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--feature_sets", nargs="+", default=FEATURE_SETS)
    parser.add_argument("--sequence_lengths", nargs="+", type=int, default=SEQUENCE_LENGTHS)
    parser.add_argument("--train_missing", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def feature_label(feature_set: str) -> str:
    if feature_set == "base_alpha_v3_rs_crowding_mini4":
        return "v3_mini4"
    if feature_set == "base_alpha_v4_medium":
        return "v4_medium"
    return feature_set.replace("base_alpha_", "").replace("/", "_")


def experiment_id(feature_set: str, sequence_length: int) -> str:
    return f"{feature_label(feature_set)}_lstm_sl{int(sequence_length)}"


def load_meta(model_dir: Path) -> dict[str, Any]:
    meta_path = model_dir / "model_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def meta_feature_set(meta: dict[str, Any]) -> str:
    return str(meta.get("feature_set") or meta.get("training", {}).get("feature_set") or "")


def meta_sequence_length(meta: dict[str, Any]) -> int | None:
    for value in [
        meta.get("sequence_length"),
        meta.get("training", {}).get("sequence_length") if isinstance(meta.get("training"), dict) else None,
        meta.get("validation_scheme", {}).get("sequence_length") if isinstance(meta.get("validation_scheme"), dict) else None,
    ]:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def discover_existing_runs(search_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for pred_path in sorted((ROOT_DIR / "app" / "model").rglob("walk_forward_predictions.csv")):
        model_dir = pred_path.parent
        meta = load_meta(model_dir)
        feature_set = meta_feature_set(meta)
        sequence_length = meta_sequence_length(meta)
        if not feature_set or sequence_length is None:
            path_text = str(model_dir).lower()
            if "v4_medium" in path_text:
                feature_set = "base_alpha_v4_medium"
            elif "v3" in path_text or "sl20" in path_text or "sequence_length_search" in path_text:
                feature_set = feature_set or "base_alpha_v3_rs_crowding_mini4"
            for sl in [20, 40, 60]:
                if f"sl{sl}" in path_text or model_dir.name == f"sl{sl}":
                    sequence_length = sl
                    break
        if not feature_set or sequence_length is None:
            continue
        runs.append(
            {
                "model_dir": model_dir,
                "prediction_path": pred_path,
                "feature_set": feature_set,
                "sequence_length": int(sequence_length),
                "mtime": pred_path.stat().st_mtime,
                "meta": meta,
            }
        )
    direct = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"
    if direct.exists():
        runs.append(
            {
                "model_dir": direct.parent,
                "prediction_path": direct,
                "feature_set": "base_alpha_v3_rs_crowding_mini4",
                "sequence_length": 20,
                "mtime": direct.stat().st_mtime,
                "meta": load_meta(direct.parent),
            }
        )
    return runs


def choose_existing_run(runs: list[dict[str, Any]], output_dir: Path, feature_set: str, sequence_length: int) -> dict[str, Any] | None:
    matches = [
        run
        for run in runs
        if run["feature_set"] == feature_set and int(run["sequence_length"]) == int(sequence_length)
    ]
    if not matches:
        return None
    expected_dir = output_dir / experiment_id(feature_set, sequence_length)
    legacy_dir = output_dir / f"sl{int(sequence_length)}"
    matches.sort(
        key=lambda run: (
            run["model_dir"] == expected_dir,
            run["model_dir"] == legacy_dir and feature_set == "base_alpha_v3_rs_crowding_mini4",
            "offline_v4_medium_compare" in str(run["model_dir"]),
            run["mtime"],
        ),
        reverse=True,
    )
    return matches[0]


def train_missing_run(args: argparse.Namespace, feature_set: str, sequence_length: int, model_dir: Path) -> tuple[Path, float]:
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT_DIR / "app" / "code" / "src" / "train_lstm.py"),
        "--feature_path",
        str(resolve_path(args.feature_path)),
        "--model_dir",
        str(model_dir),
        "--feature_set",
        feature_set,
        "--target_mode",
        "cross_section_rank",
        "--sequence_length",
        str(sequence_length),
        "--hidden_size",
        str(args.hidden_size),
        "--num_layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--learning_rate",
        str(args.learning_rate),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--experiment_remark",
        "sequence_length_search",
    ]
    started = time.perf_counter()
    subprocess.run(cmd, cwd=str(ROOT_DIR / "app" / "code"), check=True)
    return model_dir, float(time.perf_counter() - started)


def build_backtest_config(base_config_path: Path) -> dict[str, Any]:
    defaults = build_default_inference_args(load_submission_config(base_config_path))
    return {
        "profile_name": "sequence_length_search",
        "top_k": int(defaults["top_k"]),
        "primary_candidate_size": int(defaults["primary_candidate_size"]),
        "enable_risk_filters": bool(defaults["enable_risk_filters"]),
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": float(defaults["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(defaults["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(defaults["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(defaults["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(defaults["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(defaults["risk_penalty_weight"]),
        "weighting_scheme": str(defaults["weighting_scheme"]),
        "weight_blend_alpha": float(defaults["weight_blend_alpha"]),
        "max_single_weight": float(defaults["max_single_weight"]),
        "sort_strategy": str(defaults["sort_strategy"]),
        "transaction_cost": float(defaults["transaction_cost"]),
        "max_turnover": float(defaults["max_turnover"]),
    }


def single_slice_score(daily_df: pd.DataFrame) -> float:
    if daily_df.empty or "net_return" not in daily_df.columns:
        return 0.0
    return float(pd.to_numeric(daily_df["net_return"], errors="coerce").fillna(0.0).iloc[-1])


def evaluate_run(
    *,
    run: dict[str, Any],
    feature_set: str,
    sequence_length: int,
    feature_path: Path,
    output_dir: Path,
    base_config_path: Path,
    measured_training_time: float | None = None,
) -> dict[str, Any]:
    exp_id = experiment_id(feature_set, sequence_length)
    artifact_dir = output_dir / exp_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(run["model_dir"])
    pred_path = Path(run["prediction_path"])
    meta = load_meta(model_dir)
    row, _, _ = evaluate_one(
        experiment_name=exp_id,
        prediction_path=pred_path,
        fold_diagnostics_path=model_dir / "fold_diagnostics.csv" if (model_dir / "fold_diagnostics.csv").exists() else None,
        fold_daily_diagnostics_path=model_dir / "fold_daily_diagnostics.csv" if (model_dir / "fold_daily_diagnostics.csv").exists() else None,
        model="lstm",
        feature_set=feature_set,
        sequence_length=sequence_length,
    )
    prediction_df = load_prediction_frame(pred_path, feature_path)
    bt_summary, bt_daily, bt_holdings = run_backtest(
        prediction_df=prediction_df,
        config=build_backtest_config(base_config_path),
        prediction_source=str(pred_path),
    )
    bt_summary.to_csv(artifact_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    bt_daily.to_csv(artifact_dir / "backtest_daily_results.csv", index=False, encoding="utf-8-sig")
    bt_holdings.to_csv(artifact_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    bt = bt_summary.iloc[0] if not bt_summary.empty else {}
    train_rows_full = int(meta.get("train_rows_full", 0) or 0)
    sequence_rows = int(meta.get("train_sequence_rows_full", 0) or 0)
    training_time = measured_training_time
    if training_time is None:
        training_time = meta.get("training_time_seconds", 0.0)
    try:
        training_time_float = float(training_time or 0.0)
    except (TypeError, ValueError):
        training_time_float = 0.0
    return {
        "experiment_id": exp_id,
        "feature_set": feature_set,
        "sequence_length": int(sequence_length),
        "status": "ok",
        "rank_ic_mean": float(row.get("rank_ic_mean", 0.0)),
        "worst_fold_rank_ic": float(row.get("worst_fold_rank_ic", 0.0)),
        "top5_return_mean": float(row.get("top5_return_mean", 0.0)),
        "top5_return_min_by_fold": float(row.get("top5_return_min_by_fold", 0.0)),
        "cost_after_return": float(bt.get("cumulative_return_after_cost", 0.0)),
        "Sharpe": float(bt.get("sharpe_after_cost", 0.0)),
        "max_drawdown": float(bt.get("max_drawdown_after_cost", 0.0)),
        "avg_turnover": float(bt.get("avg_turnover", 0.0)),
        "single_slice_score": single_slice_score(bt_daily),
        "training_time": training_time_float,
        "train_sequence_rows_full": sequence_rows,
        "sample_retention_ratio": float(sequence_rows / train_rows_full) if train_rows_full else 0.0,
        "source_dir": str(model_dir),
        "notes": "trained_or_reused_existing_artifact" if training_time_float else "reused_existing_artifact_training_time_unavailable",
    }


def missing_row(feature_set: str, sequence_length: int, note: str) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id(feature_set, sequence_length),
        "feature_set": feature_set,
        "sequence_length": int(sequence_length),
        "status": "missing",
        "rank_ic_mean": 0.0,
        "worst_fold_rank_ic": 0.0,
        "top5_return_mean": 0.0,
        "top5_return_min_by_fold": 0.0,
        "cost_after_return": 0.0,
        "Sharpe": 0.0,
        "max_drawdown": 0.0,
        "avg_turnover": 0.0,
        "single_slice_score": 0.0,
        "training_time": 0.0,
        "train_sequence_rows_full": 0,
        "sample_retention_ratio": 0.0,
        "source_dir": "",
        "notes": note,
    }


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    ok = summary[summary["status"].eq("ok")].copy()
    base = ok[
        ok["feature_set"].eq("base_alpha_v3_rs_crowding_mini4") & ok["sequence_length"].eq(20)
    ]
    baseline = base.iloc[0] if not base.empty else None
    sl60_rows = ok[ok["sequence_length"].eq(60)].copy()
    if baseline is not None and not sl60_rows.empty:
        best_sl60 = sl60_rows.sort_values(["worst_fold_rank_ic", "Sharpe"], ascending=[False, False]).iloc[0]
        sl60_keep = (
            (best_sl60["worst_fold_rank_ic"] > baseline["worst_fold_rank_ic"] + 0.01)
            or (best_sl60["Sharpe"] > baseline["Sharpe"] + 0.1)
        ) and best_sl60["single_slice_score"] >= baseline["single_slice_score"] - 0.02
    else:
        best_sl60 = None
        sl60_keep = False

    lines = [
        "# LSTM Sequence Length Search Report",
        "",
        "Default sl20 is kept unchanged. sl40/sl60 are evaluated only as candidates.",
        "",
        "## Required Answers",
        "",
    ]
    if baseline is None:
        lines.append("1. sl40/sl60 是否真的提升稳定性: cannot judge because sl20 baseline is missing.")
    else:
        long_rows = ok[ok["sequence_length"].isin([40, 60])]
        stability_up = bool((long_rows["worst_fold_rank_ic"] > baseline["worst_fold_rank_ic"]).any()) if not long_rows.empty else False
        best_long = (
            long_rows.sort_values(["worst_fold_rank_ic", "Sharpe"], ascending=[False, False]).iloc[0]
            if not long_rows.empty
            else baseline
        )
        slice_down = bool((long_rows["single_slice_score"] < baseline["single_slice_score"]).any()) if not long_rows.empty else False
        sample_loss = bool((long_rows["sample_retention_ratio"] < baseline["sample_retention_ratio"]).any()) if not long_rows.empty else False
        lines.extend(
            [
                f"1. sl40/sl60 是否真的提升稳定性: {'yes' if stability_up else 'no'}, best long candidate `{best_long['experiment_id']}` worst_fold `{best_long['worst_fold_rank_ic']:.6f}` vs sl20 `{baseline['worst_fold_rank_ic']:.6f}`.",
                f"2. 是否牺牲单切片得分: {'yes' if slice_down else 'no'}, sl20 single_slice `{baseline['single_slice_score']:.6f}`.",
                f"3. 是否存在训练时间过长或样本减少问题: {'yes' if sample_loss else 'no'}, longer windows naturally reduce usable sequence rows; reused artifacts without recorded wall time show training_time=0.",
                f"4. 是否建议 sl60 进入融合候选而不是替换 sl20: {'yes' if sl60_keep else 'no'}, adoption rule result `{str(sl60_keep).lower()}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Adoption Rule",
            "",
            "Only keep sl60 as a candidate if worst_fold_rank_ic or Sharpe is clearly better than sl20 and single_slice_score does not drop materially.",
            "",
            "## Summary",
            "",
            "| experiment | feature_set | sl | status | rank_ic | worst_fold | top5 | min_fold_top5 | cost_after | sharpe | mdd | turnover | slice | train_time_s | seq_rows | retention |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['experiment_id']} | {row['feature_set']} | {int(row['sequence_length'])} | {row['status']} | "
            f"{fmt(row['rank_ic_mean'])} | {fmt(row['worst_fold_rank_ic'])} | {fmt(row['top5_return_mean'])} | "
            f"{fmt(row['top5_return_min_by_fold'])} | {fmt(row['cost_after_return'])} | {fmt(row['Sharpe'])} | "
            f"{fmt(row['max_drawdown'])} | {fmt(row['avg_turnover'])} | {fmt(row['single_slice_score'])} | "
            f"{fmt(row['training_time'])} | {int(row['train_sequence_rows_full'])} | {fmt(row['sample_retention_ratio'])} |"
        )
    missing = summary[summary["status"].ne("ok")]
    if not missing.empty:
        lines.extend(["", "## Missing / Skipped", ""])
        for _, row in missing.iterrows():
            lines.append(f"- `{row['experiment_id']}`: {row['notes']}")
    (output_dir / "sl20_sl40_sl60_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = resolve_path(args.feature_path)
    base_config_path = resolve_path(args.base_config)
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature_path: {feature_path}")

    rows: list[dict[str, Any]] = []
    runs = discover_existing_runs(output_dir)
    for feature_set in args.feature_sets:
        for sl in args.sequence_lengths:
            exp_id = experiment_id(feature_set, sl)
            target_dir = output_dir / exp_id
            run = None if args.force_train else choose_existing_run(runs, output_dir, feature_set, sl)
            measured_time = None
            if run is None and args.train_missing:
                print(f"[sequence_length_search] training {exp_id}")
                model_dir, measured_time = train_missing_run(args, feature_set, sl, target_dir)
                run = {
                    "model_dir": model_dir,
                    "prediction_path": model_dir / "walk_forward_predictions.csv",
                    "feature_set": feature_set,
                    "sequence_length": int(sl),
                    "mtime": time.time(),
                    "meta": load_meta(model_dir),
                }
            if run is None:
                rows.append(missing_row(feature_set, sl, "no_existing_artifact_found; rerun with --train_missing"))
                continue
            print(f"[sequence_length_search] evaluating {exp_id} from {run['model_dir']}")
            try:
                rows.append(
                    evaluate_run(
                        run=run,
                        feature_set=feature_set,
                        sequence_length=sl,
                        feature_path=feature_path,
                        output_dir=output_dir,
                        base_config_path=base_config_path,
                        measured_training_time=measured_time,
                    )
                )
            except Exception as exc:
                failed = missing_row(feature_set, sl, f"{type(exc).__name__}: {exc}")
                failed["status"] = "failed"
                failed["source_dir"] = str(run["model_dir"])
                rows.append(failed)

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_path = output_dir / "sl20_sl40_sl60_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary, output_dir)
    print(summary.to_string(index=False))
    print(f"[sequence_length_search] wrote {summary_path}")
    print(f"[sequence_length_search] wrote {output_dir / 'sl20_sl40_sl60_report.md'}")


if __name__ == "__main__":
    main()
