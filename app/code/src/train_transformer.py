import argparse
import json
from pathlib import Path

import pandas as pd

from config import BEST_CONFIG, BEST_PROFILE_NAME
from utils_seed import set_seed
from lstm_utils import (
    build_sequence_dataset,
    fit_sequence_scaler,
    predict_sequences,
    save_transformer_checkpoint,
    train_transformer_model,
    transform_sequences,
)
from stability_diagnostics import (
    build_analysis_config,
    build_fold_diagnostics,
    merge_prediction_with_features,
    write_fold_prediction_exports,
)
from train import (
    DEFAULT_FEATURE_SET,
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


DEFAULT_SEQUENCE_LENGTH = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight Transformer baseline training entrypoint.")
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--valid_dates", type=int, default=DEFAULT_VALID_DATES)
    parser.add_argument("--num_folds", type=int, default=DEFAULT_NUM_FOLDS)
    parser.add_argument(
        "--target_mode",
        choices=["raw_return", "cross_section_zscore", "cross_section_rank"],
        default=DEFAULT_TARGET_MODE,
    )
    parser.add_argument(
        "--feature_set",
        choices=["all", "base", "base_technical", "base_technical_risk", "base_technical_risk_alpha"],
        default=DEFAULT_FEATURE_SET,
    )
    parser.add_argument("--sequence_length", type=int, choices=[10, 20], default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
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


def run_walk_forward_transformer(
    df: pd.DataFrame,
    feature_columns: list[str],
    valid_dates: int,
    num_folds: int,
    sequence_length: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    base_seed: int,
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

        model, training_info = train_transformer_model(
            train_x=train_x,
            train_y=train_bundle.y,
            train_weight=train_bundle.sample_weight,
            valid_x=valid_x,
            valid_y=valid_bundle.y,
            valid_weight=valid_bundle.sample_weight,
            input_size=len(feature_columns),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            seed=base_seed + fold_id,
        )
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


def fit_final_transformer(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
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
        raise ValueError("No final training sequences were built for Transformer")

    scaler_mean, scaler_std = fit_sequence_scaler(bundle.x)
    train_x = transform_sequences(bundle.x, scaler_mean, scaler_std)
    model, training_info = train_transformer_model(
        train_x=train_x,
        train_y=bundle.y,
        train_weight=bundle.sample_weight,
        valid_x=None,
        valid_y=None,
        valid_weight=None,
        input_size=len(feature_columns),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
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
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    seed = set_seed(args.seed)
    df = load_training_frame(feature_path)
    df = add_training_target(df, args.target_mode)
    feature_columns = resolve_feature_columns(args.feature_set)

    fold_metrics, walk_forward_predictions, fold_training_summaries = run_walk_forward_transformer(
        df=df,
        feature_columns=feature_columns,
        valid_dates=args.valid_dates,
        num_folds=args.num_folds,
        sequence_length=args.sequence_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        base_seed=seed,
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

    final_model, scaler_mean, scaler_std, final_bundle, final_training_info = fit_final_transformer(
        df=df,
        feature_columns=feature_columns,
        sequence_length=args.sequence_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=seed,
    )
    model_path = model_dir / "transformer_model.pt"
    save_transformer_checkpoint(
        path=model_path,
        model=final_model,
        feature_columns=feature_columns,
        sequence_length=args.sequence_length,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )

    pd.DataFrame(fold_metrics).to_csv(model_dir / "walk_forward_metrics.csv", index=False, encoding="utf-8-sig")
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

    metadata = {
        "status": "trained",
        "best_profile_name": BEST_PROFILE_NAME,
        "default_submission_profile": {
            "profile_name": BEST_CONFIG["profile_name"],
            "feature_set": BEST_CONFIG["training"]["feature_set"],
            "target_mode": BEST_CONFIG["training"]["target_mode"],
            "model_family": "transformer",
            "seed": int(BEST_CONFIG["training"].get("seed", SEED)),
            "sort_strategy": BEST_CONFIG["selection"]["sort_strategy"],
            "weighting_scheme": BEST_CONFIG["selection"]["weighting_scheme"],
            "top_k": int(BEST_CONFIG["selection"]["top_k"]),
            "primary_candidate_size": int(BEST_CONFIG["selection"]["primary_candidate_size"]),
            "max_turnover": float(BEST_CONFIG["execution"]["max_turnover"]),
            "transaction_cost": float(BEST_CONFIG["execution"]["transaction_cost"]),
        },
        "backend": "torch_transformer",
        "walk_forward_backend": "torch_transformer",
        "feature_path": str(feature_path),
        "model_path": str(model_path),
        "feature_columns": feature_columns,
        "feature_set": args.feature_set,
        "raw_label_column": RAW_LABEL_COLUMN,
        "model_label_column": MODEL_LABEL_COLUMN,
        "target_mode": args.target_mode,
        "model_family": "transformer",
        "seed": seed,
        "valid_dates": args.valid_dates,
        "num_folds": args.num_folds,
        "sequence_length": args.sequence_length,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "train_rows_full": int(len(df)),
        "train_sequence_rows_full": int(len(final_bundle.x)),
        "train_date_range_full": [
            str(df["date"].min().date()),
            str(df["date"].max().date()),
        ],
        "walk_forward_summary": metric_summary,
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

    print("[train_transformer] backend=torch_transformer")
    print(f"[train_transformer] feature_set={args.feature_set}")
    print(f"[train_transformer] target_mode={args.target_mode}")
    print(f"[train_transformer] seed={seed}")
    print(f"[train_transformer] sequence_length={args.sequence_length}")
    print(f"[train_transformer] d_model={args.d_model} nhead={args.nhead} num_layers={args.num_layers}")
    print(f"[train_transformer] feature_count={len(feature_columns)}")
    print(f"[train_transformer] train_rows_full={len(df)} train_sequences_full={len(final_bundle.x)}")
    print(f"[train_transformer] walk_forward rank_ic_mean={metric_summary['rank_ic_mean']:.6f} top5_mean_return_mean={metric_summary['top5_mean_return_mean']:.6f}")
    print(f"[train_transformer] walk_forward rmse_mean={metric_summary['rmse_mean']:.6f} mae_mean={metric_summary['mae_mean']:.6f}")
    print(f"[train_transformer] wrote fold diagnostics to {model_dir / 'fold_diagnostics.csv'}")
    print(f"[train_transformer] wrote fold prediction exports to {model_dir}")
    print(f"[train_transformer] wrote model to {model_path}")
    print(f"[train_transformer] wrote metadata to {model_dir / 'model_meta.json'}")


if __name__ == "__main__":
    main()
