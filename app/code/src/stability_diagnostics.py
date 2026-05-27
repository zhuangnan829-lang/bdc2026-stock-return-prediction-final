from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from config import BEST_CONFIG
from utils import build_portfolio_weights, select_top_candidates


RAW_LABEL_COLUMN = "target_return"


MERGE_FEATURE_COLUMNS = [
    "volatility_5d",
    "volatility_20d",
    "turnover_rate",
    "turnover_ratio_10d",
    "amplitude_ratio_5d",
    "turnover_spike_5d",
    "crowding_reversal_risk_5d",
    "rel_strength_accel_5d_v2",
    "trend_persistence_score_10d_v2",
]


def build_analysis_config(profile_name: str = "default_diagnostics", overrides: dict | None = None) -> dict:
    selection = BEST_CONFIG["selection"]
    risk = BEST_CONFIG["risk_filter_thresholds"]
    execution = BEST_CONFIG["execution"]
    config = {
        "profile_name": profile_name,
        "top_k": int(selection["top_k"]),
        "primary_candidate_size": int(selection["primary_candidate_size"]),
        "enable_risk_filters": bool(selection.get("enable_risk_filters", True)),
        "allow_cash_fallback": False,
        "max_volatility_20d_pct": float(risk["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(risk["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(risk["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(risk["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(risk["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(risk["risk_penalty_weight"]),
        "weighting_scheme": selection["weighting_scheme"],
        "sort_strategy": selection["sort_strategy"],
        "transaction_cost": float(execution["transaction_cost"]),
        "max_turnover": float(execution["max_turnover"]),
        "rerank_signal_column": None,
        "rerank_signal_weight": 0.0,
        "secondary_candidate_size": 0,
        "secondary_screen_mode": "none",
        "secondary_screen_weight": 0.0,
        "local_tiebreak_start_rank": 8,
        "local_tiebreak_end_rank": 15,
    }
    if overrides:
        config.update(overrides)
    return config


def merge_prediction_with_features(prediction_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    base = prediction_df.copy()
    base["stock_id"] = base["stock_id"].astype(str).str.zfill(6)
    base["date"] = pd.to_datetime(base["date"])

    features = feature_df.copy()
    features["stock_id"] = features["stock_id"].astype(str).str.zfill(6)
    features["date"] = pd.to_datetime(features["date"])

    merge_columns = ["stock_id", "date", *MERGE_FEATURE_COLUMNS]
    available_feature_columns = [
        column for column in merge_columns if column in features.columns and column not in base.columns
    ]
    if not available_feature_columns:
        return base

    merged = base.merge(
        features[["stock_id", "date", *available_feature_columns]].drop_duplicates(["stock_id", "date"]),
        on=["stock_id", "date"],
        how="left",
    )
    return merged


def load_prediction_artifact(prediction_path: Path, feature_path: Path) -> pd.DataFrame:
    prediction_df = pd.read_csv(prediction_path, dtype={"stock_id": str})
    feature_df = pd.read_csv(feature_path, dtype={"stock_id": str})
    return merge_prediction_with_features(prediction_df, feature_df)


def build_fold_prediction_exports(prediction_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    working = prediction_df.copy()
    working["stock_id"] = working["stock_id"].astype(str).str.zfill(6)
    working["date"] = pd.to_datetime(working["date"])
    working["y_true"] = pd.to_numeric(working[RAW_LABEL_COLUMN], errors="coerce")
    working["y_pred"] = pd.to_numeric(working["pred_return"], errors="coerce")
    working["rank_true"] = working.groupby("date")["y_true"].rank(method="average", ascending=False, pct=True)
    working["rank_pred"] = working.groupby("date")["y_pred"].rank(method="average", ascending=False, pct=True)
    working["trade_date"] = working["date"].dt.date.astype(str)

    export_columns = [
        "trade_date",
        "stock_id",
        "fold_id",
        "y_true",
        "y_pred",
        "rank_true",
        "rank_pred",
        RAW_LABEL_COLUMN,
        "pred_return",
    ]
    if "train_target" in working.columns:
        export_columns.append("train_target")

    export_frames: dict[int, pd.DataFrame] = {}
    for fold_id, fold_df in working.groupby("fold_id", sort=True):
        ordered = fold_df.sort_values(["trade_date", "rank_pred", "stock_id"]).reset_index(drop=True)
        export_frames[int(fold_id)] = ordered[export_columns].copy()
    return export_frames


def write_fold_prediction_exports(prediction_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for fold_id, fold_df in build_fold_prediction_exports(prediction_df).items():
        output_path = output_dir / f"fold_{int(fold_id)}_predictions.csv"
        fold_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        written_paths.append(output_path)
    return written_paths


def _safe_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").dropna().mean()) if df[column].notna().any() else 0.0


def _safe_std(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(series.std(ddof=0)) if len(series) > 1 else 0.0


def _compute_day_rank_ic(day_df: pd.DataFrame) -> float:
    if day_df["pred_return"].nunique() <= 1 or day_df[RAW_LABEL_COLUMN].nunique() <= 1:
        return 0.0
    corr = day_df["pred_return"].corr(day_df[RAW_LABEL_COLUMN], method="spearman")
    return float(corr) if pd.notna(corr) else 0.0


def _compute_rank_ic(valid_df: pd.DataFrame) -> float:
    per_day = []
    for _, day_df in valid_df.groupby("date"):
        if day_df["pred_return"].nunique() <= 1 or day_df[RAW_LABEL_COLUMN].nunique() <= 1:
            continue
        corr = day_df["pred_return"].corr(day_df[RAW_LABEL_COLUMN], method="spearman")
        if pd.notna(corr):
            per_day.append(float(corr))
    return float(np.mean(per_day)) if per_day else 0.0


def _compute_top5_mean_return(valid_df: pd.DataFrame) -> float:
    returns = []
    for _, day_df in valid_df.groupby("date"):
        top5 = day_df.sort_values("pred_return", ascending=False).head(5)
        if not top5.empty:
            returns.append(float(top5[RAW_LABEL_COLUMN].mean()))
    return float(np.mean(returns)) if returns else 0.0


def _compute_fold_metrics(valid_df: pd.DataFrame) -> dict:
    y_true = pd.to_numeric(valid_df[RAW_LABEL_COLUMN], errors="coerce")
    y_pred = pd.to_numeric(valid_df["pred_return"], errors="coerce")
    errors = y_true - y_pred
    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "rank_ic": _compute_rank_ic(valid_df),
        "top5_mean_return": _compute_top5_mean_return(valid_df),
    }


def build_fold_diagnostics(prediction_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = prediction_df.copy()
    working["stock_id"] = working["stock_id"].astype(str).str.zfill(6)
    working["date"] = pd.to_datetime(working["date"])
    working = working.sort_values(["fold_id", "date", "stock_id"]).reset_index(drop=True)

    fold_rows: list[dict] = []
    daily_rows: list[dict] = []

    for fold_id, fold_df in working.groupby("fold_id", sort=True):
        fold_metrics = _compute_fold_metrics(fold_df)

        for trade_date, day_df in fold_df.groupby("date", sort=True):
            selected, diagnostics = select_top_candidates(
                latest_df=day_df,
                top_k=int(config["top_k"]),
                primary_candidate_size=int(config["primary_candidate_size"]),
                max_volatility_20d_pct=float(config["max_volatility_20d_pct"]),
                max_volatility_5d_pct=float(config["max_volatility_5d_pct"]),
                turnover_rate_lower_pct=float(config["turnover_rate_lower_pct"]),
                turnover_rate_upper_pct=float(config["turnover_rate_upper_pct"]),
                turnover_ratio_upper_pct=float(config["turnover_ratio_upper_pct"]),
                risk_penalty_weight=float(config["risk_penalty_weight"]),
                sort_strategy=str(config["sort_strategy"]),
                rerank_signal_column=config.get("rerank_signal_column"),
                rerank_signal_weight=float(config.get("rerank_signal_weight", 0.0)),
                secondary_candidate_size=int(config.get("secondary_candidate_size", 0)) or None,
                secondary_screen_mode=str(config.get("secondary_screen_mode", "none")),
                secondary_screen_weight=float(config.get("secondary_screen_weight", 0.0)),
                local_tiebreak_start_rank=int(config.get("local_tiebreak_start_rank", 8)),
                local_tiebreak_end_rank=int(config.get("local_tiebreak_end_rank", 15)),
                enable_risk_filters=bool(config.get("enable_risk_filters", True)),
                allow_cash_fallback=bool(config.get("allow_cash_fallback", False)),
            )
            selected = build_portfolio_weights(
                selected,
                top_k=int(config["top_k"]),
                weighting_scheme=str(config["weighting_scheme"]),
            )

            weighted_selected_return = (
                float((selected["weight"] * selected[RAW_LABEL_COLUMN]).sum()) if not selected.empty else 0.0
            )
            daily_rows.append(
                {
                    "profile_name": config["profile_name"],
                    "fold_id": int(fold_id),
                    "date": trade_date.date().isoformat(),
                    "day_rank_ic": _compute_day_rank_ic(day_df),
                    "candidate_count": int(len(day_df)),
                    "prediction_mean": _safe_mean(day_df, "pred_return"),
                    "prediction_std": _safe_std(day_df, "pred_return"),
                    "target_mean": _safe_mean(day_df, RAW_LABEL_COLUMN),
                    "target_std": _safe_std(day_df, RAW_LABEL_COLUMN),
                    "avg_volatility_20d": _safe_mean(day_df, "volatility_20d"),
                    "avg_volatility_5d": _safe_mean(day_df, "volatility_5d"),
                    "avg_turnover_rate": _safe_mean(day_df, "turnover_rate"),
                    "avg_turnover_ratio_10d": _safe_mean(day_df, "turnover_ratio_10d"),
                    "avg_amplitude_ratio_5d": _safe_mean(day_df, "amplitude_ratio_5d"),
                    "selected_count": int(len(selected)),
                    "selected_pred_return_mean": _safe_mean(selected, "pred_return"),
                    "selected_target_return_mean": _safe_mean(selected, RAW_LABEL_COLUMN),
                    "selected_weighted_target_return": weighted_selected_return,
                    "selected_volatility_20d_mean": _safe_mean(selected, "volatility_20d"),
                    "selected_volatility_5d_mean": _safe_mean(selected, "volatility_5d"),
                    "selected_turnover_rate_mean": _safe_mean(selected, "turnover_rate"),
                    "filter_initial_candidates": int(diagnostics.get("initial_candidates", len(day_df))),
                    "filter_after_primary_screen": int(diagnostics.get("after_primary_screen", len(day_df))),
                    "filter_after_risk_filters": int(diagnostics.get("after_risk_filters", len(day_df))),
                    "filter_after_secondary_screen": int(diagnostics.get("after_secondary_screen", len(day_df))),
                    "filter_fallback_used": int(bool(diagnostics.get("fallback_used", False))),
                }
            )

        fold_daily_df = pd.DataFrame([row for row in daily_rows if row["fold_id"] == int(fold_id)])
        fold_rows.append(
            {
                "profile_name": config["profile_name"],
                "fold_id": int(fold_id),
                **fold_metrics,
                "validation_days": int(fold_df["date"].nunique()),
                "validation_rows": int(len(fold_df)),
                "prediction_mean": _safe_mean(fold_df, "pred_return"),
                "prediction_std": _safe_std(fold_df, "pred_return"),
                "target_mean": _safe_mean(fold_df, RAW_LABEL_COLUMN),
                "target_std": _safe_std(fold_df, RAW_LABEL_COLUMN),
                "avg_daily_pred_std": _safe_mean(fold_daily_df, "prediction_std"),
                "avg_daily_target_std": _safe_mean(fold_daily_df, "target_std"),
                "avg_day_rank_ic": _safe_mean(fold_daily_df, "day_rank_ic"),
                "negative_day_rank_ic_ratio": float(
                    (pd.to_numeric(fold_daily_df["day_rank_ic"], errors="coerce") < 0).mean()
                )
                if not fold_daily_df.empty
                else 0.0,
                "avg_volatility_20d": _safe_mean(fold_df, "volatility_20d"),
                "avg_volatility_5d": _safe_mean(fold_df, "volatility_5d"),
                "avg_turnover_rate": _safe_mean(fold_df, "turnover_rate"),
                "avg_turnover_ratio_10d": _safe_mean(fold_df, "turnover_ratio_10d"),
                "avg_amplitude_ratio_5d": _safe_mean(fold_df, "amplitude_ratio_5d"),
                "avg_initial_candidates": _safe_mean(fold_daily_df, "filter_initial_candidates"),
                "avg_after_primary_screen": _safe_mean(fold_daily_df, "filter_after_primary_screen"),
                "avg_after_risk_filters": _safe_mean(fold_daily_df, "filter_after_risk_filters"),
                "avg_after_secondary_screen": _safe_mean(fold_daily_df, "filter_after_secondary_screen"),
                "fallback_day_ratio": _safe_mean(fold_daily_df, "filter_fallback_used"),
                "avg_selected_count": _safe_mean(fold_daily_df, "selected_count"),
                "avg_selected_pred_return": _safe_mean(fold_daily_df, "selected_pred_return_mean"),
                "avg_selected_target_return": _safe_mean(fold_daily_df, "selected_target_return_mean"),
                "avg_selected_weighted_target_return": _safe_mean(
                    fold_daily_df, "selected_weighted_target_return"
                ),
                "avg_selected_volatility_20d": _safe_mean(fold_daily_df, "selected_volatility_20d_mean"),
                "avg_selected_volatility_5d": _safe_mean(fold_daily_df, "selected_volatility_5d_mean"),
                "avg_selected_turnover_rate": _safe_mean(fold_daily_df, "selected_turnover_rate_mean"),
            }
        )

    return pd.DataFrame(fold_rows), pd.DataFrame(daily_rows)
