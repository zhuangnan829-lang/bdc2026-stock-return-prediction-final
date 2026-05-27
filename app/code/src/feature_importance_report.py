from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import resolve_metadata_artifact_path
from featurework import FEATURE_COLUMNS


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT_DIR / "app" / "model" / "baseline_lightgbm_same_protocol"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "model_meta.json"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_PREDICTION_PATH = DEFAULT_MODEL_DIR / "walk_forward_predictions.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_MODEL_DIR / "feature_importance.csv"
RAW_LABEL_COLUMN = "target_return"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LightGBM feature importance together with fold-level feature IC."
    )
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--metadata_path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--prediction_path", default=str(DEFAULT_PREDICTION_PATH))
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def resolve_path(path: str | Path, base_dir: Path = ROOT_DIR) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else base_dir / resolved


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_model_path(model_dir: Path, metadata: dict[str, Any], explicit_model_path: str | None) -> Path:
    if explicit_model_path:
        return resolve_path(explicit_model_path)
    if metadata.get("model_path"):
        return resolve_metadata_artifact_path(model_dir, metadata["model_path"])
    return model_dir / "baseline_model.pkl"


def resolve_feature_columns(metadata: dict[str, Any], feature_df: pd.DataFrame) -> list[str]:
    candidates = metadata.get("feature_columns") or FEATURE_COLUMNS
    return [column for column in candidates if column in feature_df.columns]


def load_fold_context(feature_path: Path, prediction_path: Path) -> pd.DataFrame:
    feature_df = pd.read_csv(feature_path, encoding="utf-8-sig", dtype={"stock_id": str})
    prediction_df = pd.read_csv(prediction_path, encoding="utf-8-sig", dtype={"stock_id": str})

    feature_df["stock_id"] = feature_df["stock_id"].astype(str).str.zfill(6)
    prediction_df["stock_id"] = prediction_df["stock_id"].astype(str).str.zfill(6)
    feature_df["date"] = pd.to_datetime(feature_df["date"])
    prediction_df["date"] = pd.to_datetime(prediction_df["date"])

    key_columns = ["stock_id", "date", "fold_id"]
    if RAW_LABEL_COLUMN in prediction_df.columns:
        key_columns.append(RAW_LABEL_COLUMN)

    merged = feature_df.merge(
        prediction_df[key_columns].drop_duplicates(["stock_id", "date"]),
        on=["stock_id", "date"],
        how="inner",
        suffixes=("", "_pred"),
    )
    if f"{RAW_LABEL_COLUMN}_pred" in merged.columns:
        merged[RAW_LABEL_COLUMN] = merged[f"{RAW_LABEL_COLUMN}_pred"]
        merged = merged.drop(columns=[f"{RAW_LABEL_COLUMN}_pred"])
    return merged


def compute_feature_ic_by_fold(
    fold_context: pd.DataFrame,
    feature_columns: list[str],
    label_column: str = RAW_LABEL_COLUMN,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    working = fold_context.dropna(subset=["fold_id", label_column]).copy()

    for feature in feature_columns:
        for fold_id, fold_df in working.groupby("fold_id", sort=True):
            day_ics = []
            for _, day_df in fold_df.groupby("date", sort=True):
                pair = day_df[[feature, label_column]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(pair) < 3 or pair[feature].nunique() <= 1 or pair[label_column].nunique() <= 1:
                    continue
                corr = pair[feature].corr(pair[label_column], method="spearman")
                if pd.notna(corr):
                    day_ics.append(float(corr))
            rows.append(
                {
                    "feature": feature,
                    "fold_id": int(fold_id),
                    "stage_ic": float(np.mean(day_ics)) if day_ics else 0.0,
                    "stage_ic_std": float(np.std(day_ics, ddof=0)) if day_ics else 0.0,
                    "stage_ic_positive_ratio": float(np.mean(np.array(day_ics) > 0.0)) if day_ics else 0.0,
                    "stage_ic_days": int(len(day_ics)),
                }
            )
    return pd.DataFrame(rows)


def extract_model_importance(model: object, feature_columns: list[str]) -> pd.DataFrame:
    split_importance = np.zeros(len(feature_columns), dtype=float)
    gain_importance = np.zeros(len(feature_columns), dtype=float)

    booster = getattr(model, "booster_", None)
    if booster is not None:
        split_importance = np.asarray(booster.feature_importance(importance_type="split"), dtype=float)
        gain_importance = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    elif hasattr(model, "feature_importances_"):
        split_importance = np.asarray(getattr(model, "feature_importances_"), dtype=float)
        gain_importance = split_importance.copy()

    split_importance = split_importance[: len(feature_columns)]
    gain_importance = gain_importance[: len(feature_columns)]
    if len(split_importance) < len(feature_columns):
        split_importance = np.pad(split_importance, (0, len(feature_columns) - len(split_importance)))
    if len(gain_importance) < len(feature_columns):
        gain_importance = np.pad(gain_importance, (0, len(feature_columns) - len(gain_importance)))

    out = pd.DataFrame(
        {
            "feature": feature_columns,
            "split_importance": split_importance,
            "gain_importance": gain_importance,
        }
    )
    total_gain = float(out["gain_importance"].sum())
    total_split = float(out["split_importance"].sum())
    out["gain_importance_pct"] = out["gain_importance"] / total_gain if total_gain > 0 else 0.0
    out["split_importance_pct"] = out["split_importance"] / total_split if total_split > 0 else 0.0
    out["importance_rank"] = out["gain_importance"].rank(method="min", ascending=False).astype(int)
    return out


def build_feature_importance_table(
    model_path: Path,
    feature_path: Path,
    prediction_path: Path,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    feature_df = pd.read_csv(feature_path, encoding="utf-8-sig", dtype={"stock_id": str}, nrows=5)
    feature_columns = resolve_feature_columns(metadata or {}, feature_df)
    if not feature_columns:
        raise ValueError("No feature columns were found in the feature file.")

    model = joblib.load(model_path)
    importance_df = extract_model_importance(model, feature_columns)

    if prediction_path.exists():
        fold_context = load_fold_context(feature_path, prediction_path)
        ic_df = compute_feature_ic_by_fold(fold_context, feature_columns)
        if not ic_df.empty:
            ic_wide = ic_df.pivot(index="feature", columns="fold_id", values="stage_ic").reset_index()
            ic_wide.columns = [
                "feature" if column == "feature" else f"fold_{int(column)}_stage_ic"
                for column in ic_wide.columns
            ]
            ic_summary = (
                ic_df.groupby("feature", as_index=False)
                .agg(
                    stage_ic_mean=("stage_ic", "mean"),
                    stage_ic_std=("stage_ic", lambda s: float(np.std(s, ddof=0))),
                    stage_ic_min=("stage_ic", "min"),
                    stage_ic_max=("stage_ic", "max"),
                )
            )
            ic_summary["stage_ic_range"] = ic_summary["stage_ic_max"] - ic_summary["stage_ic_min"]
            importance_df = importance_df.merge(ic_summary, on="feature", how="left").merge(
                ic_wide, on="feature", how="left"
            )

    return importance_df.sort_values(["gain_importance", "split_importance"], ascending=[False, False]).reset_index(
        drop=True
    )


def main() -> None:
    args = parse_args()
    model_dir = resolve_path(args.model_dir)
    metadata_path = resolve_path(args.metadata_path)
    metadata = load_metadata(metadata_path)
    model_path = resolve_model_path(model_dir, metadata, args.model_path)
    feature_path = resolve_path(args.feature_path)
    prediction_path = resolve_path(args.prediction_path)
    output_path = resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    importance_df = build_feature_importance_table(
        model_path=model_path,
        feature_path=feature_path,
        prediction_path=prediction_path,
        metadata=metadata,
    )
    importance_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[feature_importance_report] wrote {output_path}")


if __name__ == "__main__":
    main()
