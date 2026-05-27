import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import BEST_CONFIG, BEST_PROFILE_NAME, SELECTION_DEFAULTS, TRAINING_DEFAULTS
from experiment_utils import (
    resolve_training_output_dir,
    tee_run_log,
    write_experiment_config,
    write_training_metric_exports,
)
from lstm_utils import (
    build_sequence_dataset,
    fit_sequence_scaler,
    predict_sequences,
    save_lstm_checkpoint,
    train_lstm_model,
    transform_sequences,
)
from stability_diagnostics import (
    build_analysis_config,
    build_fold_diagnostics,
    write_fold_prediction_exports,
    merge_prediction_with_features,
)
from evaluate_rank_stability import append_experiment_rank_stability
from train import (
    DEFAULT_FEATURE_SET,
    DEFAULT_NUM_FOLDS,
    DEFAULT_TARGET_MODE,
    DEFAULT_VALID_DATES,
    FEATURE_SET_PRESETS,
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
from utils_seed import set_seed


DEFAULT_SEQUENCE_LENGTH = 10
DEFAULT_TOPK_TOP5_WEIGHT = 3.0
DEFAULT_TOPK_TOP10_WEIGHT = 1.8
DEFAULT_TOPK_RANK_PCT_FLOOR = 0.90
DEFAULT_TOPK_RANK_FLOOR_WEIGHT = 2.2
DEFAULT_SEED = SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM baseline training entrypoint.")
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--valid_dates", type=int, default=DEFAULT_VALID_DATES)
    parser.add_argument("--num_folds", type=int, default=DEFAULT_NUM_FOLDS)
    parser.add_argument(
        "--target_mode",
        choices=["raw_return", "cross_section_zscore", "cross_section_rank", "topk_weighted_rank"],
        default=DEFAULT_TARGET_MODE,
    )
    parser.add_argument(
        "--feature_set",
        choices=sorted(FEATURE_SET_PRESETS.keys()),
        default=DEFAULT_FEATURE_SET,
    )
    parser.add_argument("--sequence_length", type=int, choices=[10, 20, 30, 40, 60], default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--topk_top5_weight", type=float, default=DEFAULT_TOPK_TOP5_WEIGHT)
    parser.add_argument("--topk_top10_weight", type=float, default=DEFAULT_TOPK_TOP10_WEIGHT)
    parser.add_argument("--topk_rank_pct_floor", type=float, default=DEFAULT_TOPK_RANK_PCT_FLOOR)
    parser.add_argument("--topk_rank_floor_weight", type=float, default=DEFAULT_TOPK_RANK_FLOOR_WEIGHT)
    parser.add_argument("--topk_focus_k", type=int, default=0)
    parser.add_argument("--topk_gamma", type=float, default=0.0)
    parser.add_argument("--save_snapshot_checkpoints", action="store_true")
    parser.add_argument("--snapshot_top_k", type=int, default=3)
    parser.add_argument("--experiment_root", default=None)
    parser.add_argument("--experiment_id", default=None)
    parser.add_argument("--experiment_remark", default="exp")
    parser.add_argument("--sort_strategy", default=SELECTION_DEFAULTS.get("sort_strategy", "risk_adjusted"))
    parser.add_argument("--weighting_scheme", default=SELECTION_DEFAULTS.get("weighting_scheme", "pred"))
    return parser.parse_args()


def build_fold_sequence_sets(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
) -> tuple:
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


def run_walk_forward_lstm(
    df: pd.DataFrame,
    feature_columns: list[str],
    valid_dates: int,
    num_folds: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    base_seed: int,
    snapshot_dir: Path | None = None,
    snapshot_top_k: int = 0,
) -> tuple[list[dict], pd.DataFrame, list[dict]]:
    folds = build_walk_forward_folds(df, valid_dates, num_folds)
    fold_metrics: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    training_summaries: list[dict] = []

    for train_df, valid_df, fold_id in folds:
        train_bundle, valid_bundle = build_fold_sequence_sets(
            train_df=train_df,
            valid_df=valid_df,
            feature_columns=feature_columns,
            sequence_length=sequence_length,
        )
        if len(train_bundle.x) == 0 or len(valid_bundle.x) == 0:
            raise ValueError(
                f"Fold {fold_id} has insufficient sequence samples. "
                f"Try a shorter sequence_length or inspect feature continuity."
            )

        scaler_mean, scaler_std = fit_sequence_scaler(train_bundle.x)
        train_x = transform_sequences(train_bundle.x, scaler_mean, scaler_std)
        valid_x = transform_sequences(valid_bundle.x, scaler_mean, scaler_std)

        model, training_info = train_lstm_model(
            train_x=train_x,
            train_y=train_bundle.y,
            train_weight=train_bundle.sample_weight,
            valid_x=valid_x,
            valid_y=valid_bundle.y,
            valid_weight=valid_bundle.sample_weight,
            input_size=len(feature_columns),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            seed=base_seed + fold_id,
            snapshot_top_k=snapshot_top_k if snapshot_dir is not None else 0,
        )
        snapshot_infos = training_info.pop("snapshot_states", [])
        last_snapshot = training_info.pop("last_state", None)
        if snapshot_dir is not None and snapshot_infos:
            fold_snapshot_dir = snapshot_dir / f"fold_{fold_id}"
            fold_snapshot_dir.mkdir(parents=True, exist_ok=True)
            for rank, snapshot in enumerate(snapshot_infos, start=1):
                model.load_state_dict(snapshot["state_dict"])
                snapshot_path = fold_snapshot_dir / f"snapshot_rank{rank}_epoch{int(snapshot['epoch'])}.pt"
                save_lstm_checkpoint(
                    path=snapshot_path,
                    model=model,
                    feature_columns=feature_columns,
                    sequence_length=sequence_length,
                    scaler_mean=scaler_mean,
                    scaler_std=scaler_std,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            if last_snapshot is not None:
                model.load_state_dict(last_snapshot["state_dict"])
                save_lstm_checkpoint(
                    path=fold_snapshot_dir / f"last_epoch{int(last_snapshot['epoch'])}.pt",
                    model=model,
                    feature_columns=feature_columns,
                    sequence_length=sequence_length,
                    scaler_mean=scaler_mean,
                    scaler_std=scaler_std,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            pd.DataFrame(
                [
                    {
                        "fold_id": int(fold_id),
                        "snapshot_rank": int(rank),
                        "epoch": int(snapshot["epoch"]),
                        "train_loss": float(snapshot["train_loss"]),
                        "valid_loss": float(snapshot["valid_loss"]),
                        "checkpoint_path": str((fold_snapshot_dir / f"snapshot_rank{rank}_epoch{int(snapshot['epoch'])}.pt").relative_to(snapshot_dir.parent)),
                    }
                    for rank, snapshot in enumerate(snapshot_infos, start=1)
                ]
            ).to_csv(fold_snapshot_dir / "snapshot_manifest.csv", index=False, encoding="utf-8-sig")
            if last_snapshot is not None:
                pd.DataFrame(
                    [
                        {
                            "fold_id": int(fold_id),
                            "epoch": int(last_snapshot["epoch"]),
                            "train_loss": float(last_snapshot["train_loss"]),
                            "valid_loss": float(last_snapshot["valid_loss"]),
                            "checkpoint_path": str((fold_snapshot_dir / f"last_epoch{int(last_snapshot['epoch'])}.pt").relative_to(snapshot_dir.parent)),
                        }
                    ]
                ).to_csv(fold_snapshot_dir / "last_checkpoint_manifest.csv", index=False, encoding="utf-8-sig")
        valid_pred = predict_sequences(model, valid_x, batch_size=batch_size, device=next(model.parameters()).device)

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
            }
        )
        training_summaries.append(training_info)

    return fold_metrics, pd.concat(prediction_frames, ignore_index=True), training_summaries


def fit_final_lstm(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
) -> tuple:
    all_dates = set(pd.to_datetime(df["date"]).tolist())
    bundle = build_sequence_dataset(
        df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        target_dates=all_dates,
        label_column=MODEL_LABEL_COLUMN,
        raw_label_column=RAW_LABEL_COLUMN,
        sample_weight_column=SAMPLE_WEIGHT_COLUMN,
    )
    if len(bundle.x) == 0:
        raise ValueError("No final training sequences were built for LSTM")

    scaler_mean, scaler_std = fit_sequence_scaler(bundle.x)
    train_x = transform_sequences(bundle.x, scaler_mean, scaler_std)
    model, training_info = train_lstm_model(
        train_x=train_x,
        train_y=bundle.y,
        train_weight=bundle.sample_weight,
        valid_x=None,
        valid_y=None,
        valid_weight=None,
        input_size=len(feature_columns),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        seed=seed,
    )
    return model, scaler_mean, scaler_std, bundle, training_info


def main() -> None:
    args = parse_args()
    feature_path = Path(args.feature_path)
    model_dir, experiment_id = resolve_training_output_dir(
        requested_model_dir=Path(args.model_dir),
        experiment_root=args.experiment_root,
        experiment_id=args.experiment_id,
        model="lstm",
        feature=args.feature_set,
        sequence_length=args.sequence_length,
        sort_strategy=args.sort_strategy,
        weighting_scheme=args.weighting_scheme,
        remark=args.experiment_remark,
    )
    initial_experiment_config = {
        "experiment_id": experiment_id,
        "model_dir": str(model_dir),
        "feature_path": str(feature_path),
        "model_family": "lstm",
        "feature_set": args.feature_set,
        "sequence_length": int(args.sequence_length),
        "sort_strategy": args.sort_strategy,
        "weighting_scheme": args.weighting_scheme,
        "target_mode": args.target_mode,
        "valid_dates": int(args.valid_dates),
        "num_folds": int(args.num_folds),
        "seed": int(args.seed),
        "remark": args.experiment_remark,
        "lstm_params": {
            "hidden_size": int(args.hidden_size),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
        },
        "artifacts": {
            "config": "config.json",
            "metrics": "metrics.csv",
            "fold_results": "fold_results.csv",
            "backtest_summary": "backtest_summary.csv",
            "result": "result.csv",
            "run_log": "run.log",
            "figures": "figures/",
        },
    }
    if experiment_id:
        write_experiment_config(model_dir, initial_experiment_config)

    with tee_run_log(model_dir / "run.log"):
        run_training(
            args=args,
            feature_path=feature_path,
            model_dir=model_dir,
            experiment_id=experiment_id,
            initial_experiment_config=initial_experiment_config,
        )


def run_training(
    *,
    args: argparse.Namespace,
    feature_path: Path,
    model_dir: Path,
    experiment_id: str | None,
    initial_experiment_config: dict,
) -> None:

    training_started_at = time.perf_counter()
    seed = set_seed(args.seed)
    df = load_training_frame(feature_path)
    df = add_training_target(
        df,
        args.target_mode,
        topk_top5_weight=args.topk_top5_weight,
        topk_top10_weight=args.topk_top10_weight,
        topk_rank_pct_floor=args.topk_rank_pct_floor,
        topk_rank_floor_weight=args.topk_rank_floor_weight,
        topk_focus_k=args.topk_focus_k,
        topk_gamma=args.topk_gamma,
    )
    feature_columns = resolve_feature_columns(args.feature_set)

    fold_metrics, walk_forward_predictions, fold_training_summaries = run_walk_forward_lstm(
        df=df,
        feature_columns=feature_columns,
        valid_dates=args.valid_dates,
        num_folds=args.num_folds,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        base_seed=seed,
        snapshot_dir=(model_dir / "snapshots") if args.save_snapshot_checkpoints else None,
        snapshot_top_k=max(0, int(args.snapshot_top_k)),
    )
    metric_summary = summarise_metrics(fold_metrics)
    diagnostic_prediction_df = merge_prediction_with_features(
        prediction_df=walk_forward_predictions,
        feature_df=df,
    )
    fold_diagnostics_df, fold_daily_diagnostics_df = build_fold_diagnostics(
        prediction_df=diagnostic_prediction_df,
        config=build_analysis_config(profile_name="walk_forward_default"),
    )

    final_model, scaler_mean, scaler_std, final_bundle, final_training_info = fit_final_lstm(
        df=df,
        feature_columns=feature_columns,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=seed,
    )
    model_path = model_dir / "lstm_model.pt"
    save_lstm_checkpoint(
        path=model_path,
        model=final_model,
        feature_columns=feature_columns,
        sequence_length=args.sequence_length,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    pd.DataFrame(fold_metrics).to_csv(model_dir / "walk_forward_metrics.csv", index=False, encoding="utf-8-sig")
    if experiment_id:
        write_training_metric_exports(
            experiment_dir=model_dir,
            fold_metrics=fold_metrics,
            metric_summary=metric_summary,
        )
    walk_forward_predictions.to_csv(model_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    fold_prediction_paths = write_fold_prediction_exports(
        prediction_df=walk_forward_predictions,
        output_dir=model_dir,
    )
    fold_diagnostics_df.merge(
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
    ).to_csv(model_dir / "fold_diagnostics.csv", index=False, encoding="utf-8-sig")
    fold_daily_diagnostics_df.to_csv(
        model_dir / "fold_daily_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    stability_row = append_experiment_rank_stability(
        experiment_name=f"train_lstm_{args.feature_set}_sl{args.sequence_length}",
        prediction_path=model_dir / "walk_forward_predictions.csv",
        fold_diagnostics_path=model_dir / "fold_diagnostics.csv",
        fold_daily_diagnostics_path=model_dir / "fold_daily_diagnostics.csv",
        extra_fields={
            "model_dir": str(model_dir),
            "feature_set": args.feature_set,
            "target_mode": args.target_mode,
            "model_family": "lstm",
            "sequence_length": int(args.sequence_length),
            "seed": int(seed),
        },
    )

    training_time_seconds = float(time.perf_counter() - training_started_at)
    metadata = {
        "status": "trained",
        "experiment_id": experiment_id,
        "best_profile_name": BEST_PROFILE_NAME,
        "default_submission_profile": {
            "profile_name": BEST_CONFIG["profile_name"],
            "feature_set": BEST_CONFIG["training"]["feature_set"],
            "target_mode": BEST_CONFIG["training"]["target_mode"],
            "model_family": "lstm",
            "seed": int(BEST_CONFIG["training"].get("seed", SEED)),
            "sort_strategy": BEST_CONFIG["selection"]["sort_strategy"],
            "weighting_scheme": BEST_CONFIG["selection"]["weighting_scheme"],
            "top_k": int(BEST_CONFIG["selection"]["top_k"]),
            "primary_candidate_size": int(BEST_CONFIG["selection"]["primary_candidate_size"]),
            "max_turnover": float(BEST_CONFIG["execution"]["max_turnover"]),
            "transaction_cost": float(BEST_CONFIG["execution"]["transaction_cost"]),
        },
        "backend": "torch_lstm",
        "walk_forward_backend": "torch_lstm",
        "feature_path": str(feature_path),
        "model_path": str(model_path),
        "feature_columns": feature_columns,
        "feature_set": args.feature_set,
        "raw_label_column": RAW_LABEL_COLUMN,
        "model_label_column": MODEL_LABEL_COLUMN,
        "target_mode": args.target_mode,
        "sample_weight_mode": "topk_head_weighted" if args.target_mode == "topk_weighted_rank" else "uniform",
        "snapshot_checkpoints": {
            "enabled": bool(args.save_snapshot_checkpoints),
            "top_k": int(args.snapshot_top_k),
            "directory": "snapshots" if args.save_snapshot_checkpoints else "",
        },
        "topk_weight_config": {
            "top5_weight": float(args.topk_top5_weight),
            "top10_weight": float(args.topk_top10_weight),
            "rank_pct_floor": float(args.topk_rank_pct_floor),
            "rank_floor_weight": float(args.topk_rank_floor_weight),
            "focus_k": int(args.topk_focus_k),
            "gamma": float(args.topk_gamma),
        },
        "model_family": "lstm",
        "seed": seed,
        "valid_dates": args.valid_dates,
        "num_folds": args.num_folds,
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "training_time_seconds": training_time_seconds,
        "train_rows_full": int(len(df)),
        "train_sequence_rows_full": int(len(final_bundle.x)),
        "train_date_range_full": [
            str(df["date"].min().date()),
            str(df["date"].max().date()),
        ],
        "walk_forward_summary": metric_summary,
        "rank_stability_summary": stability_row,
        "walk_forward_folds": fold_metrics,
        "walk_forward_fold_prediction_files": [path.name for path in fold_prediction_paths],
        "walk_forward_diagnostics_profile": "walk_forward_default",
        "walk_forward_training": fold_training_summaries,
        "final_training": final_training_info,
    }
    (model_dir / "model_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if experiment_id:
        write_experiment_config(
            model_dir,
            {
                **initial_experiment_config,
                "status": "trained",
                "backend": "torch_lstm",
                "walk_forward_backend": "torch_lstm",
                "model_path": str(model_path),
                "feature_columns": feature_columns,
                "train_rows_full": int(len(df)),
                "train_sequence_rows_full": int(len(final_bundle.x)),
                "training_time_seconds": training_time_seconds,
                "train_date_range_full": metadata["train_date_range_full"],
                "walk_forward_summary": metric_summary,
                "rank_stability_summary": stability_row,
                "topk_weight_config": {
                    "top5_weight": float(args.topk_top5_weight),
                    "top10_weight": float(args.topk_top10_weight),
                    "rank_pct_floor": float(args.topk_rank_pct_floor),
                    "rank_floor_weight": float(args.topk_rank_floor_weight),
                    "focus_k": int(args.topk_focus_k),
                    "gamma": float(args.topk_gamma),
                },
            },
        )

    print("[train_lstm] backend=torch_lstm")
    if experiment_id:
        print(f"[train_lstm] experiment_id={experiment_id}")
        print(f"[train_lstm] experiment_dir={model_dir}")
    print(f"[train_lstm] feature_set={args.feature_set}")
    print(f"[train_lstm] target_mode={args.target_mode}")
    if args.target_mode == "topk_weighted_rank":
        print(
            "[train_lstm] topk_weight_config="
            f"top5={args.topk_top5_weight:.3f} "
            f"top10={args.topk_top10_weight:.3f} "
            f"rank_pct_floor={args.topk_rank_pct_floor:.3f} "
            f"rank_floor_weight={args.topk_rank_floor_weight:.3f} "
            f"focus_k={args.topk_focus_k} gamma={args.topk_gamma:.3f}"
        )
    print(f"[train_lstm] sequence_length={args.sequence_length}")
    print(f"[train_lstm] hidden_size={args.hidden_size} num_layers={args.num_layers}")
    print(f"[train_lstm] seed={seed}")
    print(f"[train_lstm] feature_count={len(feature_columns)}")
    print(f"[train_lstm] train_rows_full={len(df)} train_sequences_full={len(final_bundle.x)}")
    print(f"[train_lstm] training_time_seconds={training_time_seconds:.3f}")
    print(f"[train_lstm] walk_forward rank_ic_mean={metric_summary['rank_ic_mean']:.6f} top5_mean_return_mean={metric_summary['top5_mean_return_mean']:.6f}")
    print(f"[train_lstm] walk_forward rmse_mean={metric_summary['rmse_mean']:.6f} mae_mean={metric_summary['mae_mean']:.6f}")
    print(f"[train_lstm] wrote fold diagnostics to {model_dir / 'fold_diagnostics.csv'}")
    print(f"[train_lstm] wrote fold prediction exports to {model_dir}")
    print(f"[train_lstm] wrote model to {model_path}")
    print(f"[train_lstm] wrote metadata to {model_dir / 'model_meta.json'}")


if __name__ == "__main__":
    main()
