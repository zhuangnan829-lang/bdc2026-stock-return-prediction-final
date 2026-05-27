from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from config import ROOT_DIR
from load_submission_config import build_default_inference_args, load_submission_config
from lstm_utils import build_sequence_dataset, transform_sequences
from result_validator import validate_result_file
from transformer_lite_utils import load_transformer_lite_checkpoint, predict_transformer_lite
from utils import build_portfolio_weights, load_feature_frame, select_top_candidates


DEFAULT_MODEL_DIR = ROOT_DIR / "app" / "model" / "transformer_lite" / "sl60"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "predict_features.csv"
DEFAULT_HISTORY_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "app" / "model" / "transformer_lite" / "candidate_result.csv"
DEFAULT_SCORE_PATH = ROOT_DIR / "app" / "model" / "transformer_lite" / "candidate_scores.csv"
DEFAULT_BASE_CONFIG = ROOT_DIR / "app" / "model" / "default_submission_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with Transformer-lite candidate model.")
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--history_feature_path", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--base_config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--score_output_path", default=str(DEFAULT_SCORE_PATH))
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def load_metadata(model_dir: Path) -> dict:
    meta_path = model_dir / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    model_dir = resolve_path(args.model_dir)
    feature_path = resolve_path(args.feature_path)
    history_path = resolve_path(args.history_feature_path)
    output_path = resolve_path(args.output_path)
    score_output_path = resolve_path(args.score_output_path)
    config = build_default_inference_args(load_submission_config(resolve_path(args.base_config)))
    metadata = load_metadata(model_dir)
    model_path = Path(metadata["model_path"])
    if not model_path.is_absolute():
        model_path = ROOT_DIR / model_path
    feature_columns = metadata["feature_columns"]
    sequence_length = int(metadata["sequence_length"])
    batch_size = int(metadata.get("batch_size", 2048))

    target_df = load_feature_frame(feature_path)
    if history_path.exists():
        history_df = load_feature_frame(history_path)
        history_df = history_df[history_df["date"] < target_df["date"].min()].copy()
        context_df = pd.concat([history_df, target_df], ignore_index=True)
    else:
        context_df = target_df.copy()
    context_df = (
        context_df.sort_values(["stock_id", "date"])
        .drop_duplicates(["stock_id", "date"], keep="last")
        .reset_index(drop=True)
    )
    missing = [column for column in feature_columns if column not in context_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for Transformer-lite inference: {missing}")

    bundle = build_sequence_dataset(
        context_df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        target_dates=set(pd.to_datetime(target_df["date"]).tolist()),
        label_column=None,
        raw_label_column=None,
    )
    if len(bundle.x) == 0:
        raise ValueError("No Transformer-lite inference sequences were built.")

    model, checkpoint = load_transformer_lite_checkpoint(model_path)
    score_x = transform_sequences(bundle.x, checkpoint["scaler_mean"], checkpoint["scaler_std"])
    started = time.perf_counter()
    pred = predict_transformer_lite(model, score_x, batch_size=batch_size, device=next(model.parameters()).device)
    inference_time = time.perf_counter() - started
    scored = bundle.meta.copy()
    scored["pred_return"] = pred
    latest_date = scored["date"].max()
    latest_scores = scored[scored["date"].eq(latest_date)][["stock_id", "date", "pred_return"]]
    latest_df = target_df[target_df["date"].eq(latest_date)].merge(latest_scores, on=["stock_id", "date"], how="inner")
    if latest_df.empty:
        raise ValueError("No latest-date Transformer-lite scores found.")

    selected, diagnostics = select_top_candidates(
        latest_df=latest_df,
        top_k=int(config["top_k"]),
        primary_candidate_size=int(config["primary_candidate_size"]),
        max_volatility_20d_pct=float(config["max_volatility_20d_pct"]),
        max_volatility_5d_pct=float(config["max_volatility_5d_pct"]),
        turnover_rate_lower_pct=float(config["turnover_rate_lower_pct"]),
        turnover_rate_upper_pct=float(config["turnover_rate_upper_pct"]),
        turnover_ratio_upper_pct=float(config["turnover_ratio_upper_pct"]),
        risk_penalty_weight=float(config["risk_penalty_weight"]),
        sort_strategy=str(config["sort_strategy"]),
        enable_risk_filters=bool(config["enable_risk_filters"]),
        allow_cash_fallback=False,
    )
    weighted = build_portfolio_weights(
        selected,
        top_k=int(config["top_k"]),
        weighting_scheme=str(config["weighting_scheme"]),
        max_single_weight=float(config["max_single_weight"]),
        weight_blend_alpha=float(config["weight_blend_alpha"]),
    )
    result = weighted[["stock_id", "weight"]].sort_values(["weight", "stock_id"], ascending=[False, True]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")
    score_output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(score_output_path, index=False, encoding="utf-8-sig")
    validate_result_file(output_path)
    print(f"[predict_transformer_lite] model_dir={model_dir}")
    print(f"[predict_transformer_lite] latest_date={latest_date.date()} rows={len(result)} inference_time={inference_time:.6f}s")
    print(f"[predict_transformer_lite] diagnostics={diagnostics}")
    print(f"[predict_transformer_lite] wrote {output_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
