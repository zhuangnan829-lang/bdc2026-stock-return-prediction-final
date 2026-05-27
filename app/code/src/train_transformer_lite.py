from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import load_prediction_frame, run_backtest
from config import ROOT_DIR
from evaluate_rank_stability import evaluate_one
from load_submission_config import build_default_inference_args, load_submission_config
from lstm_utils import build_sequence_dataset, fit_sequence_scaler, transform_sequences
from stability_diagnostics import build_analysis_config, build_fold_diagnostics, merge_prediction_with_features, write_fold_prediction_exports
from train import (
    DEFAULT_NUM_FOLDS,
    DEFAULT_TARGET_MODE,
    DEFAULT_VALID_DATES,
    MODEL_LABEL_COLUMN,
    RAW_LABEL_COLUMN,
    SAMPLE_WEIGHT_COLUMN,
    SEED,
    add_training_target,
    build_walk_forward_folds,
    compute_fold_metrics,
    load_training_frame,
    resolve_feature_columns,
    summarise_metrics,
)
from transformer_lite_utils import (
    predict_transformer_lite,
    save_transformer_lite_checkpoint,
    train_transformer_lite_model,
)
from utils_seed import set_seed


DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "transformer_lite"
DEFAULT_BASE_CONFIG = ROOT_DIR / "app" / "model" / "default_submission_config.json"
DEFAULT_LSTM_PRED_PATH = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"
LEGACY_SL60_DIR = ROOT_DIR / "app" / "model" / "transformer_lite_sl60"
SUMMARY_COLUMNS = [
    "experiment_id",
    "model_family",
    "feature_set",
    "sequence_length",
    "status",
    "rank_ic_mean",
    "worst_fold_rank_ic",
    "top5_return_mean",
    "NDCG@5",
    "HitRate@5",
    "cost_after_return",
    "Sharpe",
    "max_drawdown",
    "avg_turnover",
    "single_slice_score",
    "training_time",
    "inference_time",
    "source_dir",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate Transformer-lite candidate branches.")
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base_config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--lstm_pred_path", default=str(DEFAULT_LSTM_PRED_PATH))
    parser.add_argument("--feature_set", default="base_alpha_v3_rs_crowding_mini4")
    parser.add_argument("--target_mode", default=DEFAULT_TARGET_MODE)
    parser.add_argument("--sequence_lengths", nargs="+", type=int, default=[20, 40, 60])
    parser.add_argument("--valid_dates", type=int, default=DEFAULT_VALID_DATES)
    parser.add_argument("--num_folds", type=int, default=DEFAULT_NUM_FOLDS)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, choices=[1, 2], default=1)
    parser.add_argument("--dim_feedforward", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train_missing", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--reuse_legacy_sl60", action="store_true", default=True)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def model_dir(output_dir: Path, sequence_length: int) -> Path:
    return output_dir / f"sl{int(sequence_length)}"


def artifacts_exist(path: Path) -> bool:
    return (
        (path / "model_meta.json").exists()
        and (path / "walk_forward_predictions.csv").exists()
        and (path / "walk_forward_metrics.csv").exists()
    )


def copy_legacy_sl60(target_dir: Path) -> bool:
    if not artifacts_exist(LEGACY_SL60_DIR):
        return False
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "model_meta.json",
        "walk_forward_predictions.csv",
        "walk_forward_metrics.csv",
        "fold_diagnostics.csv",
        "fold_daily_diagnostics.csv",
        "transformer_lite_model.pt",
    ]:
        src = LEGACY_SL60_DIR / name
        if src.exists():
            shutil.copy2(src, target_dir / name)
    return artifacts_exist(target_dir)


def build_fold_sequence_sets(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_columns: list[str], sequence_length: int) -> tuple:
    combined = (
        pd.concat([train_df, valid_df], ignore_index=True)
        .sort_values(["stock_id", "date"])
        .drop_duplicates(["stock_id", "date"], keep="last")
        .reset_index(drop=True)
    )
    train_dates = set(pd.to_datetime(train_df["date"]).tolist())
    valid_dates = set(pd.to_datetime(valid_df["date"]).tolist())
    train_bundle = build_sequence_dataset(
        combined,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        target_dates=train_dates,
        label_column=MODEL_LABEL_COLUMN,
        raw_label_column=RAW_LABEL_COLUMN,
        sample_weight_column=SAMPLE_WEIGHT_COLUMN,
    )
    valid_bundle = build_sequence_dataset(
        combined,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        target_dates=valid_dates,
        label_column=MODEL_LABEL_COLUMN,
        raw_label_column=RAW_LABEL_COLUMN,
        sample_weight_column=SAMPLE_WEIGHT_COLUMN,
    )
    return train_bundle, valid_bundle


def run_walk_forward(df: pd.DataFrame, feature_columns: list[str], args: argparse.Namespace, sequence_length: int) -> tuple[list[dict], pd.DataFrame, list[dict]]:
    fold_metrics: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    training_summaries: list[dict] = []
    for train_df, valid_df, fold_id in build_walk_forward_folds(df, args.valid_dates, args.num_folds):
        train_bundle, valid_bundle = build_fold_sequence_sets(train_df, valid_df, feature_columns, sequence_length)
        if len(train_bundle.x) == 0 or len(valid_bundle.x) == 0:
            raise ValueError(f"Fold {fold_id} has insufficient sequences for sl{sequence_length}")
        scaler_mean, scaler_std = fit_sequence_scaler(train_bundle.x)
        train_x = transform_sequences(train_bundle.x, scaler_mean, scaler_std)
        valid_x = transform_sequences(valid_bundle.x, scaler_mean, scaler_std)
        model, training_info = train_transformer_lite_model(
            train_x=train_x,
            train_y=train_bundle.y,
            train_weight=train_bundle.sample_weight,
            valid_x=valid_x,
            valid_y=valid_bundle.y,
            valid_weight=valid_bundle.sample_weight,
            input_size=len(feature_columns),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed + fold_id,
        )
        started = time.perf_counter()
        valid_pred = predict_transformer_lite(model, valid_x, batch_size=args.batch_size, device=next(model.parameters()).device)
        inference_time = float(time.perf_counter() - started)
        scored = valid_bundle.meta[["stock_id", "date", RAW_LABEL_COLUMN, MODEL_LABEL_COLUMN]].copy()
        scored["pred_return"] = valid_pred
        scored["fold_id"] = fold_id
        prediction_frames.append(scored)
        metrics = compute_fold_metrics(scored)
        metrics.update(
            {
                "fold_id": fold_id,
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "train_sequence_rows": int(len(train_x)),
                "valid_sequence_rows": int(len(valid_x)),
                "train_date_start": str(train_df["date"].min().date()),
                "train_date_end": str(train_df["date"].max().date()),
                "valid_date_start": str(valid_df["date"].min().date()),
                "valid_date_end": str(valid_df["date"].max().date()),
            }
        )
        fold_metrics.append(metrics)
        training_info.update(
            {
                "fold_id": fold_id,
                "train_sequence_rows": int(len(train_x)),
                "valid_sequence_rows": int(len(valid_x)),
                "inference_time_seconds": inference_time,
            }
        )
        training_summaries.append(training_info)
    return fold_metrics, pd.concat(prediction_frames, ignore_index=True), training_summaries


def fit_final(df: pd.DataFrame, feature_columns: list[str], args: argparse.Namespace, sequence_length: int) -> tuple:
    bundle = build_sequence_dataset(
        df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        target_dates=set(pd.to_datetime(df["date"]).tolist()),
        label_column=MODEL_LABEL_COLUMN,
        raw_label_column=RAW_LABEL_COLUMN,
        sample_weight_column=SAMPLE_WEIGHT_COLUMN,
    )
    if len(bundle.x) == 0:
        raise ValueError(f"No final sequences for sl{sequence_length}")
    scaler_mean, scaler_std = fit_sequence_scaler(bundle.x)
    train_x = transform_sequences(bundle.x, scaler_mean, scaler_std)
    model, training_info = train_transformer_lite_model(
        train_x=train_x,
        train_y=bundle.y,
        train_weight=bundle.sample_weight,
        valid_x=None,
        valid_y=None,
        valid_weight=None,
        input_size=len(feature_columns),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )
    return model, scaler_mean, scaler_std, bundle, training_info


def train_one(args: argparse.Namespace, sequence_length: int, out_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    set_seed(args.seed)
    df = load_training_frame(resolve_path(args.feature_path))
    df = add_training_target(df, args.target_mode)
    feature_columns = resolve_feature_columns(args.feature_set)
    fold_metrics, predictions, training_summaries = run_walk_forward(df, feature_columns, args, sequence_length)
    metric_summary = summarise_metrics(fold_metrics)
    diagnostic_prediction_df = merge_prediction_with_features(prediction_df=predictions, feature_df=df)
    fold_diagnostics, fold_daily = build_fold_diagnostics(
        prediction_df=diagnostic_prediction_df,
        config=build_analysis_config(profile_name="transformer_lite_walk_forward"),
    )
    final_model, scaler_mean, scaler_std, final_bundle, final_training_info = fit_final(df, feature_columns, args, sequence_length)
    model_path = out_dir / "transformer_lite_model.pt"
    save_transformer_lite_checkpoint(
        path=model_path,
        model=final_model,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    pd.DataFrame(fold_metrics).to_csv(out_dir / "walk_forward_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    write_fold_prediction_exports(prediction_df=predictions, output_dir=out_dir)
    fold_diagnostics.merge(
        pd.DataFrame(fold_metrics)[
            [
                "fold_id",
                "train_rows",
                "valid_rows",
                "train_sequence_rows",
                "valid_sequence_rows",
                "train_date_start",
                "train_date_end",
                "valid_date_start",
                "valid_date_end",
            ]
        ],
        on="fold_id",
        how="left",
    ).to_csv(out_dir / "fold_diagnostics.csv", index=False, encoding="utf-8-sig")
    fold_daily.to_csv(out_dir / "fold_daily_diagnostics.csv", index=False, encoding="utf-8-sig")
    training_time = float(time.perf_counter() - started)
    metadata = {
        "status": "trained",
        "backend": "torch_transformer_lite",
        "walk_forward_backend": "torch_transformer_lite",
        "candidate_status": "candidate_only_do_not_replace_mainline",
        "feature_path": str(resolve_path(args.feature_path)),
        "model_path": str(model_path),
        "feature_columns": feature_columns,
        "feature_set": args.feature_set,
        "raw_label_column": RAW_LABEL_COLUMN,
        "model_label_column": MODEL_LABEL_COLUMN,
        "target_mode": args.target_mode,
        "model_family": "transformer_lite",
        "seed": int(args.seed),
        "valid_dates": int(args.valid_dates),
        "num_folds": int(args.num_folds),
        "sequence_length": int(sequence_length),
        "d_model": int(args.d_model),
        "nhead": int(args.nhead),
        "num_layers": int(args.num_layers),
        "dim_feedforward": int(args.dim_feedforward),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "training_time_seconds": training_time,
        "inference_time_seconds": float(sum(item.get("inference_time_seconds", 0.0) for item in training_summaries)),
        "train_rows_full": int(len(df)),
        "train_sequence_rows_full": int(len(final_bundle.x)),
        "train_date_range_full": [str(df["date"].min().date()), str(df["date"].max().date())],
        "walk_forward_summary": metric_summary,
        "walk_forward_folds": fold_metrics,
        "walk_forward_training": training_summaries,
        "final_training": final_training_info,
    }
    (out_dir / "model_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def backtest_config(base_config_path: Path) -> dict[str, Any]:
    defaults = build_default_inference_args(load_submission_config(base_config_path))
    return {
        "profile_name": "transformer_lite",
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


def ndcg_at_k(relevance: np.ndarray, k: int = 5) -> float:
    rel = relevance[:k]
    if len(rel) == 0:
        return 0.0
    gains = np.power(2.0, rel) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(relevance)[::-1][:k]
    idcg = float(np.sum((np.power(2.0, ideal) - 1.0) * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 1e-12 else 0.0


def quality_metrics(prediction_df: pd.DataFrame, top_k: int = 5) -> dict[str, float]:
    ndcg_values = []
    hit_values = []
    for _, day_df in prediction_df.groupby("date"):
        day = day_df.dropna(subset=["pred_return", "target_return"]).copy()
        if len(day) < top_k:
            continue
        day["true_rank_pct"] = day["target_return"].rank(pct=True)
        pred_sorted = day.sort_values(["pred_return", "stock_id"], ascending=[False, True])
        pred_top = pred_sorted.head(top_k)
        true_top = set(day.sort_values(["target_return", "stock_id"], ascending=[False, True]).head(top_k)["stock_id"])
        hit_values.append(len(set(pred_top["stock_id"]) & true_top) / float(top_k))
        ndcg_values.append(ndcg_at_k(pred_sorted["true_rank_pct"].to_numpy(dtype=float), k=top_k))
    return {
        "NDCG@5": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "HitRate@5": float(np.mean(hit_values)) if hit_values else 0.0,
    }


def single_slice_score(daily_df: pd.DataFrame) -> float:
    if daily_df.empty or "net_return" not in daily_df.columns:
        return 0.0
    return float(pd.to_numeric(daily_df["net_return"], errors="coerce").fillna(0.0).iloc[-1])


def evaluate_artifact(
    *,
    experiment_id: str,
    model_family: str,
    model_path: Path,
    feature_path: Path,
    base_config_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    pred_path = model_path / "walk_forward_predictions.csv"
    row, _, _ = evaluate_one(
        experiment_name=experiment_id,
        prediction_path=pred_path,
        fold_diagnostics_path=model_path / "fold_diagnostics.csv" if (model_path / "fold_diagnostics.csv").exists() else None,
        fold_daily_diagnostics_path=model_path / "fold_daily_diagnostics.csv" if (model_path / "fold_daily_diagnostics.csv").exists() else None,
        model=model_family,
        feature_set=str(metadata.get("feature_set", "")),
        sequence_length=metadata.get("sequence_length", 20),
    )
    prediction_df = load_prediction_frame(pred_path, feature_path)
    bt_summary, bt_daily, bt_holdings = run_backtest(
        prediction_df=prediction_df,
        config={**backtest_config(base_config_path), "profile_name": experiment_id},
        prediction_source=str(pred_path),
    )
    out_bt_dir = model_path / "backtest_same_protocol"
    out_bt_dir.mkdir(parents=True, exist_ok=True)
    bt_summary.to_csv(out_bt_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    bt_daily.to_csv(out_bt_dir / "backtest_daily_results.csv", index=False, encoding="utf-8-sig")
    bt_holdings.to_csv(out_bt_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    bt = bt_summary.iloc[0]
    quality = quality_metrics(prediction_df)
    return {
        "experiment_id": experiment_id,
        "model_family": model_family,
        "feature_set": metadata.get("feature_set", ""),
        "sequence_length": int(metadata.get("sequence_length", 20) or 20),
        "status": "ok",
        "rank_ic_mean": float(row.get("rank_ic_mean", 0.0)),
        "worst_fold_rank_ic": float(row.get("worst_fold_rank_ic", 0.0)),
        "top5_return_mean": float(row.get("top5_return_mean", 0.0)),
        **quality,
        "cost_after_return": float(bt.get("cumulative_return_after_cost", 0.0)),
        "Sharpe": float(bt.get("sharpe_after_cost", 0.0)),
        "max_drawdown": float(bt.get("max_drawdown_after_cost", 0.0)),
        "avg_turnover": float(bt.get("avg_turnover", 0.0)),
        "single_slice_score": single_slice_score(bt_daily),
        "training_time": float(metadata.get("training_time_seconds", 0.0) or 0.0),
        "inference_time": float(metadata.get("inference_time_seconds", 0.0) or 0.0),
        "source_dir": str(model_path),
        "notes": str(metadata.get("candidate_status", "")),
    }


def lstm_baseline_row(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    pred_path = resolve_path(args.lstm_pred_path)
    lstm_dir = pred_path.parent
    meta_path = lstm_dir / "model_meta.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        metadata = {
            "feature_set": "base_alpha_v3_rs_crowding_mini4",
            "sequence_length": 20,
            "model_family": "lstm",
        }
    target = output_dir / "lstm_sl20_baseline"
    target.mkdir(parents=True, exist_ok=True)
    if pred_path.resolve() != (target / "walk_forward_predictions.csv").resolve():
        shutil.copy2(pred_path, target / "walk_forward_predictions.csv")
    for name in ["fold_diagnostics.csv", "fold_daily_diagnostics.csv"]:
        src = lstm_dir / name
        if src.exists():
            shutil.copy2(src, target / name)
    return evaluate_artifact(
        experiment_id="lstm_sl20_baseline",
        model_family="lstm",
        model_path=target,
        feature_path=resolve_path(args.feature_path),
        base_config_path=resolve_path(args.base_config),
        metadata=metadata,
    )


def missing_row(sequence_length: int, note: str) -> dict[str, Any]:
    return {
        "experiment_id": f"transformer_lite_sl{sequence_length}",
        "model_family": "transformer_lite",
        "feature_set": "",
        "sequence_length": int(sequence_length),
        "status": "missing",
        "rank_ic_mean": 0.0,
        "worst_fold_rank_ic": 0.0,
        "top5_return_mean": 0.0,
        "NDCG@5": 0.0,
        "HitRate@5": 0.0,
        "cost_after_return": 0.0,
        "Sharpe": 0.0,
        "max_drawdown": 0.0,
        "avg_turnover": 0.0,
        "single_slice_score": 0.0,
        "training_time": 0.0,
        "inference_time": 0.0,
        "source_dir": "",
        "notes": note,
    }


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    baseline = summary[summary["experiment_id"].eq("lstm_sl20_baseline")].iloc[0]
    candidates = summary[(summary["model_family"].eq("transformer_lite")) & (summary["status"].eq("ok"))].copy()
    if candidates.empty:
        best = baseline
        replace = False
        blend_candidate = False
    else:
        best = candidates.sort_values(["Sharpe", "worst_fold_rank_ic", "single_slice_score"], ascending=[False, False, False]).iloc[0]
        replace = bool(
            best["single_slice_score"] > baseline["single_slice_score"]
            and best["Sharpe"] > baseline["Sharpe"]
            and best["worst_fold_rank_ic"] > baseline["worst_fold_rank_ic"]
            and best["max_drawdown"] >= baseline["max_drawdown"]
            and best["avg_turnover"] <= baseline["avg_turnover"]
        )
        blend_candidate = bool(
            not replace
            and (
                best["single_slice_score"] > baseline["single_slice_score"]
                or best["Sharpe"] > baseline["Sharpe"]
                or best["worst_fold_rank_ic"] > baseline["worst_fold_rank_ic"]
            )
        )
    lines = [
        "# Transformer-lite Candidate Report",
        "",
        "Transformer-lite is evaluated as a candidate branch only. The LSTM sl20 mainline remains unchanged.",
        "",
        "## Required Answers",
        "",
        f"1. Transformer-lite 是否超过 LSTM sl20: {'yes' if replace else 'no'}, best `{best['experiment_id']}`.",
        f"2. 是否只是单次得分高但稳定性差: {'yes' if (best['single_slice_score'] > baseline['single_slice_score'] and best['worst_fold_rank_ic'] < baseline['worst_fold_rank_ic']) else 'no'}.",
        f"3. 是否适合进入 rank blend: {'yes' if blend_candidate else 'no'}.",
        f"4. 是否不建议替换主线: {'no' if replace else 'yes'}.",
        "",
        "## Adoption Rule",
        "",
        "Replacement is allowed only if single_slice_score, Sharpe, and worst_fold_rank_ic all beat LSTM sl20, while drawdown and turnover are not worse.",
        "",
        "## Summary",
        "",
        "| experiment | model | sl | status | rank_ic | worst_fold | top5 | ndcg5 | hit5 | cost_after | sharpe | mdd | turnover | slice | train_s | infer_s |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['experiment_id']} | {row['model_family']} | {int(row['sequence_length'])} | {row['status']} | "
            f"{fmt(row['rank_ic_mean'])} | {fmt(row['worst_fold_rank_ic'])} | {fmt(row['top5_return_mean'])} | "
            f"{fmt(row['NDCG@5'])} | {fmt(row['HitRate@5'])} | {fmt(row['cost_after_return'])} | "
            f"{fmt(row['Sharpe'])} | {fmt(row['max_drawdown'])} | {fmt(row['avg_turnover'])} | "
            f"{fmt(row['single_slice_score'])} | {fmt(row['training_time'])} | {fmt(row['inference_time'])} |"
        )
    missing = summary[summary["status"].ne("ok")]
    if not missing.empty:
        lines.extend(["", "## Missing / Skipped", ""])
        for _, row in missing.iterrows():
            lines.append(f"- `{row['experiment_id']}`: {row['notes']}")
    (output_dir / "transformer_lite_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = resolve_path(args.feature_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature_path: {feature_path}")
    rows = [lstm_baseline_row(args, output_dir)]

    for sl in args.sequence_lengths:
        out_dir = model_dir(output_dir, sl)
        if args.force_train and out_dir.exists():
            # Avoid destructive cleanup; train will overwrite known artifacts only.
            pass
        if not artifacts_exist(out_dir) and sl == 60 and args.reuse_legacy_sl60:
            copied = copy_legacy_sl60(out_dir)
            if copied:
                print(f"[transformer_lite] reused legacy sl60 artifact: {LEGACY_SL60_DIR}")
        if (args.force_train or not artifacts_exist(out_dir)) and args.train_missing:
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[transformer_lite] training sl{sl}")
            train_one(args, sl, out_dir)
        if not artifacts_exist(out_dir):
            rows.append(missing_row(sl, "no_existing_artifact_found; rerun with --train_missing"))
            continue
        metadata = json.loads((out_dir / "model_meta.json").read_text(encoding="utf-8"))
        rows.append(
            evaluate_artifact(
                experiment_id=f"transformer_lite_sl{sl}",
                model_family="transformer_lite",
                model_path=out_dir,
                feature_path=feature_path,
                base_config_path=resolve_path(args.base_config),
                metadata=metadata,
            )
        )

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_path = output_dir / "transformer_lite_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary, output_dir)
    print(summary.to_string(index=False))
    print(f"[transformer_lite] wrote {summary_path}")
    print(f"[transformer_lite] wrote {output_dir / 'transformer_lite_report.md'}")


if __name__ == "__main__":
    main()
