from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import load_prediction_frame, run_backtest
from config import ROOT_DIR
from load_submission_config import build_default_inference_args, load_submission_config
from lstm_utils import load_lstm_checkpoint, predict_sequences, transform_sequences
from train import (
    RAW_LABEL_COLUMN,
    add_training_target,
    build_walk_forward_folds,
    load_training_frame,
    resolve_feature_columns,
)
from train_lstm import build_fold_sequence_sets


SRC_DIR = ROOT_DIR / "app" / "code" / "src"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "lstm_snapshot_ensemble"
DEFAULT_MODEL_DIR = DEFAULT_OUTPUT_DIR / "model"
DEFAULT_FEATURE_SET = "base_alpha_v3_rs_crowding_mini4"
SUMMARY_COLUMNS = [
    "profile_name",
    "rank_ic_mean",
    "worst_fold_rank_ic",
    "top5_return_mean",
    "cost_after_return",
    "Sharpe",
    "max_drawdown",
    "avg_turnover",
    "single_slice_score",
    "negative_day_rank_ic_ratio",
    "notes",
]


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSTM top-k snapshot checkpoints with average-rank fusion.")
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--feature_set", default=DEFAULT_FEATURE_SET)
    parser.add_argument("--target_mode", default="cross_section_rank")
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--valid_dates", type=int, default=20)
    parser.add_argument("--num_folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--snapshot_top_k", type=int, default=3)
    parser.add_argument("--force_train", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def ensure_snapshot_model(args: argparse.Namespace, feature_path: Path, model_dir: Path) -> None:
    snapshot_root = model_dir / "snapshots"
    if snapshot_root.exists() and list(snapshot_root.glob("fold_*/snapshot_manifest.csv")) and not args.force_train:
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SRC_DIR / "train_lstm.py"),
        "--feature_path",
        str(feature_path),
        "--model_dir",
        str(model_dir),
        "--feature_set",
        args.feature_set,
        "--target_mode",
        args.target_mode,
        "--sequence_length",
        str(args.sequence_length),
        "--valid_dates",
        str(args.valid_dates),
        "--num_folds",
        str(args.num_folds),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch_size",
        "256",
        "--learning_rate",
        "0.001",
        "--hidden_size",
        "64",
        "--num_layers",
        "1",
        "--dropout",
        "0.0",
        "--seed",
        "2026",
        "--save_snapshot_checkpoints",
        "--snapshot_top_k",
        str(args.snapshot_top_k),
    ]
    run_cmd(cmd)


def load_training_data(feature_path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    df = load_training_frame(feature_path)
    df = add_training_target(df, args.target_mode)
    return df, resolve_feature_columns(args.feature_set)


def predict_checkpoint(
    checkpoint_path: Path,
    valid_bundle,
    fold_id: int,
    snapshot_label: str,
    snapshot_rank: int,
) -> pd.DataFrame:
    model, checkpoint = load_lstm_checkpoint(checkpoint_path)
    valid_x = transform_sequences(valid_bundle.x, checkpoint["scaler_mean"], checkpoint["scaler_std"])
    pred = predict_sequences(model, valid_x, batch_size=256, device=next(model.parameters()).device)
    scored = valid_bundle.meta[["stock_id", "date", RAW_LABEL_COLUMN]].copy()
    scored["pred_return"] = pred
    scored["fold_id"] = int(fold_id)
    scored["snapshot_label"] = snapshot_label
    scored["snapshot_rank"] = int(snapshot_rank)
    scored["checkpoint_path"] = str(checkpoint_path)
    return scored


def build_snapshot_predictions(args: argparse.Namespace, feature_path: Path, model_dir: Path, output_dir: Path) -> pd.DataFrame:
    df, feature_columns = load_training_data(feature_path, args)
    frames = []
    checkpoint_rows = []
    folds = build_walk_forward_folds(df, args.valid_dates, args.num_folds)
    for train_df, valid_df, fold_id in folds:
        _, valid_bundle = build_fold_sequence_sets(train_df, valid_df, feature_columns, args.sequence_length)
        fold_dir = model_dir / "snapshots" / f"fold_{fold_id}"
        manifest = pd.read_csv(fold_dir / "snapshot_manifest.csv", encoding="utf-8-sig")
        for _, row in manifest.iterrows():
            checkpoint_path = model_dir / Path(str(row["checkpoint_path"]))
            snapshot_rank = int(row["snapshot_rank"])
            label = f"snapshot_rank{snapshot_rank}"
            pred = predict_checkpoint(checkpoint_path, valid_bundle, fold_id, label, snapshot_rank)
            frames.append(pred)
            checkpoint_rows.append(
                {
                    "fold_id": int(fold_id),
                    "profile_name": label,
                    "snapshot_rank": snapshot_rank,
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "valid_loss": float(row["valid_loss"]),
                    "checkpoint_path": str(checkpoint_path),
                }
            )
        last_manifest_path = fold_dir / "last_checkpoint_manifest.csv"
        if last_manifest_path.exists():
            last_manifest = pd.read_csv(last_manifest_path, encoding="utf-8-sig")
            row = last_manifest.iloc[0]
            checkpoint_path = model_dir / Path(str(row["checkpoint_path"]))
            pred = predict_checkpoint(checkpoint_path, valid_bundle, fold_id, "last_epoch", 999)
            frames.append(pred)
            checkpoint_rows.append(
                {
                    "fold_id": int(fold_id),
                    "profile_name": "last_epoch",
                    "snapshot_rank": 999,
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "valid_loss": float(row["valid_loss"]),
                    "checkpoint_path": str(checkpoint_path),
                }
            )
    checkpoint_df = pd.DataFrame(checkpoint_rows)
    checkpoint_df.to_csv(output_dir / "checkpoint_metrics.csv", index=False, encoding="utf-8-sig")
    all_predictions = pd.concat(frames, ignore_index=True)
    all_predictions.to_csv(output_dir / "snapshot_checkpoint_predictions.csv", index=False, encoding="utf-8-sig")
    return all_predictions


def average_rank_ensemble(predictions: pd.DataFrame, ranks: list[int], profile_name: str) -> pd.DataFrame:
    subset = predictions[predictions["snapshot_rank"].isin(ranks)].copy()
    subset["rank_score"] = subset.groupby(["snapshot_label", "date"])["pred_return"].rank(pct=True)
    averaged = (
        subset.groupby(["stock_id", "date", "fold_id", RAW_LABEL_COLUMN], as_index=False)["rank_score"]
        .mean()
        .rename(columns={"rank_score": "pred_return"})
    )
    averaged["profile_name"] = profile_name
    return averaged


def single_profile(predictions: pd.DataFrame, snapshot_label: str) -> pd.DataFrame:
    out = predictions[predictions["snapshot_label"].eq(snapshot_label)].copy()
    out["profile_name"] = snapshot_label
    return out[["stock_id", "date", "fold_id", RAW_LABEL_COLUMN, "pred_return", "profile_name"]]


def rank_metrics(prediction_df: pd.DataFrame) -> dict[str, float]:
    rows = []
    for _, day_df in prediction_df.groupby("date", sort=True):
        day = day_df.dropna(subset=["pred_return", RAW_LABEL_COLUMN])
        if day.empty:
            continue
        valid = day[["pred_return", RAW_LABEL_COLUMN]].apply(pd.to_numeric, errors="coerce").dropna()
        rank_ic = np.nan
        if len(valid) > 1 and valid["pred_return"].nunique() > 1 and valid[RAW_LABEL_COLUMN].nunique() > 1:
            rank_ic = valid["pred_return"].corr(valid[RAW_LABEL_COLUMN], method="spearman")
        top5 = day.sort_values(["pred_return", "stock_id"], ascending=[False, True]).head(5)
        rows.append(
            {
                "date": day["date"].iloc[0],
                "fold_id": int(day["fold_id"].iloc[0]),
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "top5_return": float(top5[RAW_LABEL_COLUMN].mean()),
            }
        )
    daily = pd.DataFrame(rows)
    fold = daily.groupby("fold_id", as_index=False).agg(rank_ic=("rank_ic", "mean"), top5_return=("top5_return", "mean"))
    rank_ic = pd.to_numeric(daily["rank_ic"], errors="coerce").dropna()
    return {
        "rank_ic_mean": float(fold["rank_ic"].mean()),
        "worst_fold_rank_ic": float(fold["rank_ic"].min()),
        "top5_return_mean": float(daily["top5_return"].mean()),
        "negative_day_rank_ic_ratio": float((rank_ic < 0).mean()) if not rank_ic.empty else 0.0,
    }


def build_backtest_config(profile_name: str) -> dict[str, Any]:
    cfg = load_submission_config()
    args = build_default_inference_args(cfg)
    return {
        "profile_name": profile_name,
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


def write_result_from_backtest(backtest_daily: pd.DataFrame, output_path: Path) -> None:
    last = backtest_daily.sort_values("date").iloc[-1]
    ids = [item.strip().zfill(6) for item in str(last.get("selected_stock_ids", "")).split(",") if item.strip()]
    weights = [float(item) for item in str(last.get("selected_weights", "")).split(",") if item.strip()]
    pd.DataFrame({"stock_id": ids[: len(weights)], "weight": weights[: len(ids)]}).to_csv(output_path, index=False, encoding="utf-8-sig")


def evaluate_profile(profile_df: pd.DataFrame, feature_path: Path, output_dir: Path, profile_name: str) -> dict[str, Any]:
    profile_dir = output_dir / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    pred_path = profile_dir / "walk_forward_predictions.csv"
    out = profile_df.rename(columns={RAW_LABEL_COLUMN: "target_return"}).copy()
    out.to_csv(pred_path, index=False, encoding="utf-8-sig")
    metrics = rank_metrics(profile_df)
    prediction_df = load_prediction_frame(pred_path, feature_path)
    bt_summary, bt_daily, holdings = run_backtest(prediction_df, build_backtest_config(profile_name), prediction_source=str(pred_path))
    bt_summary.to_csv(profile_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    bt_daily.to_csv(profile_dir / "backtest_daily.csv", index=False, encoding="utf-8-sig")
    holdings.to_csv(profile_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    write_result_from_backtest(bt_daily, profile_dir / "result.csv")
    bt = bt_summary.iloc[0]
    return {
        "profile_name": profile_name,
        **metrics,
        "cost_after_return": float(bt["cumulative_return_after_cost"]),
        "Sharpe": float(bt["sharpe_after_cost"]),
        "max_drawdown": float(bt["max_drawdown_after_cost"]),
        "avg_turnover": float(bt["avg_turnover"]),
        "single_slice_score": float(pd.to_numeric(bt_daily["net_return"], errors="coerce").fillna(0.0).iloc[-1]),
        "notes": "",
    }


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    top1 = summary[summary["profile_name"].eq("snapshot_rank1")].iloc[0]
    last = summary[summary["profile_name"].eq("last_epoch")].iloc[0]
    ensemble = summary[summary["profile_name"].eq("snapshot_top3_average_rank")].iloc[0]
    stability_improved = ensemble["worst_fold_rank_ic"] >= top1["worst_fold_rank_ic"]
    return_ok = ensemble["top5_return_mean"] >= top1["top5_return_mean"] - 0.001
    recommend = bool(stability_improved and return_ok)
    lines = [
        "# LSTM Snapshot Ensemble Report",
        "",
        f"- last_epoch_worse_than_best_checkpoint: `{str(last['top5_return_mean'] < top1['top5_return_mean']).lower()}`",
        f"- top3_snapshot_improves_stability: `{str(stability_improved).lower()}`",
        f"- recommend_replace_single_checkpoint_for_robust: `{str(recommend).lower()}`",
        "",
        "## Key Deltas",
        "",
        f"- ensemble_vs_rank1_worst_fold_delta: `{ensemble['worst_fold_rank_ic'] - top1['worst_fold_rank_ic']:.6f}`",
        f"- ensemble_vs_rank1_top5_delta: `{ensemble['top5_return_mean'] - top1['top5_return_mean']:.6f}`",
        f"- ensemble_vs_rank1_cost_after_delta: `{ensemble['cost_after_return'] - top1['cost_after_return']:.6f}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    (output_dir / "snapshot_ensemble_report.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    feature_path = resolve_path(args.feature_path)
    output_dir = resolve_path(args.output_dir)
    model_dir = resolve_path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_snapshot_model(args, feature_path, model_dir)
    predictions = build_snapshot_predictions(args, feature_path, model_dir, output_dir)
    profiles = [
        single_profile(predictions, "last_epoch"),
        single_profile(predictions, "snapshot_rank1"),
        single_profile(predictions, "snapshot_rank2"),
        single_profile(predictions, "snapshot_rank3"),
        average_rank_ensemble(predictions, [1, 2, 3], "snapshot_top3_average_rank"),
    ]
    rows = [evaluate_profile(profile, feature_path, output_dir, str(profile["profile_name"].iloc[0])) for profile in profiles]
    summary = pd.DataFrame(rows)[SUMMARY_COLUMNS]
    summary.to_csv(output_dir / "snapshot_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    write_report(summary, output_dir)
    print(f"[snapshot_ensemble] wrote {output_dir / 'checkpoint_metrics.csv'}")
    print(f"[snapshot_ensemble] wrote {output_dir / 'snapshot_ensemble_summary.csv'}")
    print(f"[snapshot_ensemble] wrote {output_dir / 'snapshot_ensemble_report.md'}")


if __name__ == "__main__":
    main()
