import argparse
from itertools import combinations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from stability_diagnostics import (
    build_analysis_config,
    build_fold_diagnostics,
    merge_prediction_with_features,
    write_fold_prediction_exports,
)
from evaluate_rank_stability import append_experiment_rank_stability

from config import BEST_CONFIG, BEST_PROFILE_NAME, SELECTION_DEFAULTS, TRAINING_DEFAULTS
from experiment_utils import (
    resolve_training_output_dir,
    tee_run_log,
    write_experiment_config,
    write_training_metric_exports,
)
from featurework import BASE_ALPHA_V4_MEDIUM_COLUMNS
from utils_seed import DEFAULT_SEED, set_seed


FEATURE_COLUMNS = [
    "ret_1d",
    "ret_2d",
    "ret_3d",
    "ret_5d",
    "ret_7d",
    "ret_10d",
    "ret_15d",
    "ret_20d",
    "mom_5d",
    "mom_10d",
    "intraday_return",
    "range_pct",
    "close_to_high",
    "close_to_low",
    "close_to_ma_5d",
    "close_to_ma_10d",
    "close_to_ma_20d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_ratio_5_10",
    "volatility_ratio_10_20",
    "volume_change_1d",
    "volume_change_5d",
    "volume_ratio_5d",
    "volume_ratio_10d",
    "amount_change_1d",
    "amount_ratio_5d",
    "amount_ratio_10d",
    "turnover_mean_5d",
    "turnover_mean_10d",
    "turnover_mean_20d",
    "turnover_change_1d",
    "turnover_ratio_5d",
    "turnover_ratio_10d",
    "amplitude_mean_5d",
    "amplitude_mean_10d",
    "amplitude_ratio_5d",
    "amplitude_change_1d",
    "risk_adjusted_mom_5d",
    "risk_adjusted_mom_10d",
    "rank_ret_1d",
    "rank_ret_5d",
    "rank_mom_5d",
    "rank_close_to_ma_10d",
    "rank_turnover_rate",
    "rank_volume_change_1d",
    "rank_volatility_5d",
    "rank_volatility_20d",
    "rank_amount_ratio_5d",
    "rel_ret_1d",
    "rel_ret_5d",
    "rel_mom_10d",
    "rank_rel_ret_5d",
    "consecutive_up_days",
    "consecutive_down_days",
    "positive_day_ratio_10d",
    "negative_day_ratio_10d",
    "volatility_ratio_5_20",
    "volatility_state_change_5d",
    "volatility_regime_rank",
    "rel_hs300_mean_ret_1d",
    "rel_hs300_mean_ret_5d",
    "rel_hs300_mean_mom_10d",
    "rel_cs_mean_close_to_ma_10d",
    "distance_to_20d_high",
    "rebound_from_10d_low",
    "breakout_streak_20d",
    "drawdown_recovery_ratio_10d",
    "turnover_spike_5d",
    "rank_volume_change_5d",
    "volume_price_divergence_5d",
    "crowding_risk_5d",
    "volatility_switch_signal",
    "volatility_switch_rank",
    "rel_strength_accel_5d",
    "rel_strength_accel_5d_v2",
    "rel_strength_accel_5d_v3a",
    "rel_strength_accel_5d_v3b",
    "rel_strength_accel_5d_v3c",
    "trend_persistence_score_10d",
    "trend_persistence_score_10d_v2",
    "volatility_compression_breakout_20d",
    "crowding_reversal_risk_5d",
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

FEATURE_GROUPS = {
    "base": [
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "mom_5d",
        "mom_10d",
        "intraday_return",
        "range_pct",
        "close_to_high",
        "close_to_low",
        "volume_change_1d",
        "volume_change_5d",
        "volume_ratio_5d",
        "volume_ratio_10d",
        "amount_change_1d",
        "amount_ratio_5d",
    ],
    "technical": [
        "close_to_ma_5d",
        "close_to_ma_10d",
        "close_to_ma_20d",
        "turnover_mean_5d",
        "turnover_mean_10d",
        "turnover_mean_20d",
        "turnover_change_1d",
        "turnover_ratio_5d",
        "turnover_ratio_10d",
        "amplitude_mean_5d",
        "amplitude_mean_10d",
        "amplitude_ratio_5d",
        "amplitude_change_1d",
        "rank_ret_1d",
        "rank_ret_5d",
        "rank_mom_5d",
        "rank_close_to_ma_10d",
        "rank_turnover_rate",
        "rank_volume_change_1d",
        "rank_amount_ratio_5d",
    ],
    "risk": [
        "volatility_5d",
        "volatility_10d",
        "volatility_20d",
        "volatility_ratio_5_10",
        "volatility_ratio_10_20",
        "risk_adjusted_mom_5d",
        "risk_adjusted_mom_10d",
        "rank_volatility_5d",
        "rank_volatility_20d",
    ],
    "alpha": [
        "rel_ret_1d",
        "rel_ret_5d",
        "rel_mom_10d",
        "rank_rel_ret_5d",
        "consecutive_up_days",
        "consecutive_down_days",
        "positive_day_ratio_10d",
        "negative_day_ratio_10d",
        "volatility_ratio_5_20",
        "volatility_state_change_5d",
        "volatility_regime_rank",
    ],
    "alpha_v3": [
        "rel_hs300_mean_ret_1d",
        "rel_hs300_mean_ret_5d",
        "rel_hs300_mean_mom_10d",
        "rel_cs_mean_close_to_ma_10d",
        "distance_to_20d_high",
        "rebound_from_10d_low",
        "breakout_streak_20d",
        "drawdown_recovery_ratio_10d",
        "turnover_spike_5d",
        "rank_volume_change_5d",
        "volume_price_divergence_5d",
        "crowding_risk_5d",
        "volatility_switch_signal",
        "volatility_switch_rank",
    ],
    "alpha_v3_relative_strength": [
        "rel_hs300_mean_ret_1d",
        "rel_hs300_mean_ret_5d",
        "rel_hs300_mean_mom_10d",
        "rel_cs_mean_close_to_ma_10d",
    ],
    "alpha_v3_trend_persistence": [
        "distance_to_20d_high",
        "rebound_from_10d_low",
        "breakout_streak_20d",
        "drawdown_recovery_ratio_10d",
    ],
    "alpha_v3_crowding_risk": [
        "turnover_spike_5d",
        "rank_volume_change_5d",
        "volume_price_divergence_5d",
        "crowding_risk_5d",
        "volatility_switch_signal",
        "volatility_switch_rank",
    ],
    "alpha_v3_selected5": [
        "rel_hs300_mean_ret_5d",
        "distance_to_20d_high",
        "rebound_from_10d_low",
        "turnover_spike_5d",
        "volume_price_divergence_5d",
    ],
    "alpha_v3_rs_crowding": [
        "rel_hs300_mean_ret_1d",
        "rel_hs300_mean_ret_5d",
        "rel_hs300_mean_mom_10d",
        "rel_cs_mean_close_to_ma_10d",
        "turnover_spike_5d",
        "rank_volume_change_5d",
        "volume_price_divergence_5d",
        "crowding_risk_5d",
        "volatility_switch_signal",
        "volatility_switch_rank",
    ],
    "alpha_v3_rs_crowding_mini4": [
        "rel_hs300_mean_ret_5d",
        "rel_cs_mean_close_to_ma_10d",
        "turnover_spike_5d",
        "volume_price_divergence_5d",
    ],
    "alpha_v4_micro": [
        "rel_strength_accel_5d",
        "trend_persistence_score_10d",
        "volatility_compression_breakout_20d",
        "crowding_reversal_risk_5d",
    ],
    "alpha_v4_rewrite_v2": [
        "rel_strength_accel_5d_v2",
        "trend_persistence_score_10d_v2",
    ],
    "alpha_v4_rs_accel_v3": [
        "rel_strength_accel_5d_v3a",
        "rel_strength_accel_5d_v3b",
        "rel_strength_accel_5d_v3c",
    ],
}

FEATURE_SET_PRESETS = {
    "all": FEATURE_COLUMNS,
    "base": FEATURE_GROUPS["base"],
    "base_technical": FEATURE_GROUPS["base"] + FEATURE_GROUPS["technical"],
    "base_technical_risk": FEATURE_GROUPS["base"] + FEATURE_GROUPS["technical"] + FEATURE_GROUPS["risk"],
    "base_technical_risk_alpha": FEATURE_GROUPS["base"] + FEATURE_GROUPS["technical"] + FEATURE_GROUPS["risk"] + FEATURE_GROUPS["alpha"],
    "base_alpha_v3": FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3"],
    "base_technical_risk_alpha_v3": FEATURE_GROUPS["base"] + FEATURE_GROUPS["technical"] + FEATURE_GROUPS["risk"] + FEATURE_GROUPS["alpha"] + FEATURE_GROUPS["alpha_v3"],
    "base_alpha_v3_relative_strength": FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_relative_strength"],
    "base_alpha_v3_trend_persistence": FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_trend_persistence"],
    "base_alpha_v3_crowding_risk": FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_crowding_risk"],
    "base_alpha_v3_selected5": FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_selected5"],
    "base_alpha_v3_rs_crowding": FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding"],
    "base_alpha_v3_rs_crowding_mini4": FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"],
    "base_alpha_v4_micro": FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v4_micro"],
    "base_alpha_v4_medium": BASE_ALPHA_V4_MEDIUM_COLUMNS,
    "base_alpha_v3_rs_crowding_mini4_alpha_v4_micro": (
        FEATURE_GROUPS["base"]
        + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"]
        + FEATURE_GROUPS["alpha_v4_micro"]
    ),
    "base_alpha_v3_rs_crowding_mini4__rel_strength_accel_5d_v2": (
        FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"] + ["rel_strength_accel_5d_v2"]
    ),
    "base_alpha_v3_rs_crowding_mini4__trend_persistence_score_10d_v2": (
        FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"] + ["trend_persistence_score_10d_v2"]
    ),
    "base_alpha_v3_rs_crowding_mini4__rel_strength_accel_5d_v2__trend_persistence_score_10d_v2": (
        FEATURE_GROUPS["base"]
        + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"]
        + FEATURE_GROUPS["alpha_v4_rewrite_v2"]
    ),
    "base_alpha_v3_rs_crowding_mini4__rel_strength_accel_5d_v3a": (
        FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"] + ["rel_strength_accel_5d_v3a"]
    ),
    "base_alpha_v3_rs_crowding_mini4__rel_strength_accel_5d_v3b": (
        FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"] + ["rel_strength_accel_5d_v3b"]
    ),
    "base_alpha_v3_rs_crowding_mini4__rel_strength_accel_5d_v3c": (
        FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"] + ["rel_strength_accel_5d_v3c"]
    ),
    "base_alpha_v3_rs_crowding_mini4__drop_short_term_noise": [
        column
        for column in (FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"])
        if column not in {"ret_1d", "ret_3d", "intraday_return"}
    ],
    "base_alpha_v3_rs_crowding_mini4__drop_crowding_disturbance": [
        column
        for column in (FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"])
        if column not in {"turnover_spike_5d", "volume_price_divergence_5d"}
    ],
    "base_alpha_v3_rs_crowding_mini4__drop_ret_1d": [
        column
        for column in (FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"])
        if column not in {"ret_1d"}
    ],
    "base_alpha_v3_rs_crowding_mini4__drop_ret_3d": [
        column
        for column in (FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"])
        if column not in {"ret_3d"}
    ],
    "base_alpha_v3_rs_crowding_mini4__drop_intraday_return": [
        column
        for column in (FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"])
        if column not in {"intraday_return"}
    ],
    "base_alpha_v3_rs_crowding_mini4__drop_ret_1d__intraday_return": [
        column
        for column in (FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"])
        if column not in {"ret_1d", "intraday_return"}
    ],
    "base_alpha_v3_rs_crowding_mini4__drop_ret_1d__ret_3d": [
        column
        for column in (FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"])
        if column not in {"ret_1d", "ret_3d"}
    ],
    "base_alpha_v3_rs_crowding_mini4__drop_ret_3d__intraday_return": [
        column
        for column in (FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"])
        if column not in {"ret_3d", "intraday_return"}
    ],
}

ALPHA_V4_MICRO_COLUMNS = FEATURE_GROUPS["alpha_v4_micro"]
for column in ALPHA_V4_MICRO_COLUMNS:
    FEATURE_SET_PRESETS[f"base_alpha_v3_rs_crowding_mini4__{column}"] = (
        FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"] + [column]
    )

for left, right in combinations(ALPHA_V4_MICRO_COLUMNS, 2):
    FEATURE_SET_PRESETS[f"base_alpha_v3_rs_crowding_mini4__{left}__{right}"] = (
        FEATURE_GROUPS["base"] + FEATURE_GROUPS["alpha_v3_rs_crowding_mini4"] + [left, right]
    )

RAW_LABEL_COLUMN = "target_return"
MODEL_LABEL_COLUMN = "train_target"
SAMPLE_WEIGHT_COLUMN = "train_sample_weight"
SEED = int(TRAINING_DEFAULTS.get("seed", DEFAULT_SEED))
DEFAULT_VALID_DATES = int(TRAINING_DEFAULTS["valid_dates"])
DEFAULT_NUM_FOLDS = int(TRAINING_DEFAULTS["num_folds"])
DEFAULT_TARGET_MODE = TRAINING_DEFAULTS["target_mode"]
DEFAULT_FEATURE_SET = TRAINING_DEFAULTS["feature_set"]
DEFAULT_MODEL_FAMILY = TRAINING_DEFAULTS.get("model_family", "lightgbm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ranking-oriented baseline training entrypoint.")
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
    parser.add_argument(
        "--model_family",
        choices=["lightgbm", "xgboost", "auto", "sklearn_histgbr", "linear_regression"],
        default=DEFAULT_MODEL_FAMILY,
    )
    parser.add_argument("--topk_top5_weight", type=float, default=3.0)
    parser.add_argument("--topk_top10_weight", type=float, default=1.8)
    parser.add_argument("--topk_rank_pct_floor", type=float, default=0.90)
    parser.add_argument("--topk_rank_floor_weight", type=float, default=2.2)
    parser.add_argument("--topk_focus_k", type=int, default=0)
    parser.add_argument("--topk_gamma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--experiment_root", default=None)
    parser.add_argument("--experiment_id", default=None)
    parser.add_argument("--experiment_remark", default="exp")
    parser.add_argument("--sort_strategy", default=SELECTION_DEFAULTS.get("sort_strategy", "risk_adjusted"))
    parser.add_argument("--weighting_scheme", default=SELECTION_DEFAULTS.get("weighting_scheme", "pred"))
    return parser.parse_args()


def load_training_frame(feature_path: Path) -> pd.DataFrame:
    df = pd.read_csv(feature_path, encoding="utf-8-sig", dtype={"stock_id": str})
    df["stock_id"] = df["stock_id"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df[RAW_LABEL_COLUMN].notna()].copy()
    if df["date"].isna().any():
        bad_rows = int(df["date"].isna().sum())
        raise ValueError(f"Found {bad_rows} rows with invalid dates in {feature_path}")
    return df


def add_training_target(
    df: pd.DataFrame,
    target_mode: str,
    topk_top5_weight: float = 3.0,
    topk_top10_weight: float = 1.8,
    topk_rank_pct_floor: float = 0.90,
    topk_rank_floor_weight: float = 2.2,
    topk_focus_k: int = 0,
    topk_gamma: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    out[SAMPLE_WEIGHT_COLUMN] = 1.0
    if target_mode == "raw_return":
        out[MODEL_LABEL_COLUMN] = out[RAW_LABEL_COLUMN]
        return out

    if target_mode == "cross_section_rank":
        out[MODEL_LABEL_COLUMN] = out.groupby("date")[RAW_LABEL_COLUMN].rank(pct=True)
        return out

    if target_mode == "topk_weighted_rank":
        out[MODEL_LABEL_COLUMN] = out.groupby("date")[RAW_LABEL_COLUMN].rank(pct=True)
        desc_rank = out.groupby("date")[RAW_LABEL_COLUMN].rank(method="first", ascending=False)
        sample_weight = np.ones(len(out), dtype=np.float32)
        if int(topk_focus_k) > 0:
            focused_weight = 1.0 + float(topk_gamma)
            sample_weight = np.where(desc_rank <= int(topk_focus_k), focused_weight, sample_weight)
        else:
            sample_weight = np.where(desc_rank <= 5, float(topk_top5_weight), sample_weight)
            sample_weight = np.where(
                (desc_rank > 5) & (desc_rank <= 10),
                np.maximum(sample_weight, float(topk_top10_weight)),
                sample_weight,
            )
            sample_weight = np.where(
                out[MODEL_LABEL_COLUMN] >= float(topk_rank_pct_floor),
                np.maximum(sample_weight, float(topk_rank_floor_weight)),
                sample_weight,
            )
        out[SAMPLE_WEIGHT_COLUMN] = sample_weight.astype(np.float32)
        return out

    if target_mode == "cross_section_zscore":
        def _zscore(s: pd.Series) -> pd.Series:
            std = s.std(ddof=0)
            if pd.isna(std) or std < 1e-12:
                return pd.Series(np.zeros(len(s)), index=s.index)
            return (s - s.mean()) / std

        out[MODEL_LABEL_COLUMN] = out.groupby("date")[RAW_LABEL_COLUMN].transform(_zscore)
        return out

    raise ValueError(f"Unsupported target_mode: {target_mode}")


def build_model(model_family: str = "auto", seed: int = SEED):
    if model_family == "linear_regression":
        return LinearRegression(), "linear_regression"

    if model_family in {"lightgbm", "auto"}:
        try:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                objective="regression",
                learning_rate=0.03,
                n_estimators=400,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=80,
                random_state=seed,
                n_jobs=-1,
            )
            return model, "lightgbm"
        except Exception:
            if model_family == "lightgbm":
                raise

    if model_family in {"xgboost", "auto"}:
        try:
            from xgboost import XGBRegressor

            model = XGBRegressor(
                objective="reg:squarederror",
                learning_rate=0.03,
                n_estimators=400,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1,
            )
            return model, "xgboost"
        except Exception:
            if model_family == "xgboost":
                raise

    model = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=400,
        max_depth=6,
        min_samples_leaf=80,
        random_state=seed,
    )
    return model, "sklearn_histgbr"


def compute_rank_ic(valid_df: pd.DataFrame) -> float:
    per_day = []
    for _, day_df in valid_df.groupby("date"):
        if day_df["pred_return"].nunique() <= 1 or day_df[RAW_LABEL_COLUMN].nunique() <= 1:
            continue
        corr = day_df["pred_return"].corr(day_df[RAW_LABEL_COLUMN], method="spearman")
        if pd.notna(corr):
            per_day.append(float(corr))
    return float(np.mean(per_day)) if per_day else 0.0


def compute_top5_mean_return(valid_df: pd.DataFrame) -> float:
    returns = []
    for _, day_df in valid_df.groupby("date"):
        top5 = day_df.sort_values("pred_return", ascending=False).head(5)
        if not top5.empty:
            returns.append(float(top5[RAW_LABEL_COLUMN].mean()))
    return float(np.mean(returns)) if returns else 0.0


def compute_fold_metrics(valid_df: pd.DataFrame) -> dict:
    y_true = valid_df[RAW_LABEL_COLUMN].to_numpy()
    y_pred = valid_df["pred_return"].to_numpy()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rank_ic": compute_rank_ic(valid_df),
        "top5_mean_return": compute_top5_mean_return(valid_df),
    }


def build_walk_forward_folds(df: pd.DataFrame, valid_dates: int, num_folds: int) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    unique_dates = sorted(df["date"].drop_duplicates())
    required_dates = valid_dates * (num_folds + 1)
    if len(unique_dates) <= required_dates:
        raise ValueError(
            f"Not enough unique dates for walk-forward validation: total={len(unique_dates)}, "
            f"need more than {required_dates} for valid_dates={valid_dates}, num_folds={num_folds}"
        )

    folds = []
    start_index = len(unique_dates) - valid_dates * num_folds
    for fold_idx in range(num_folds):
        valid_start = start_index + fold_idx * valid_dates
        valid_end = valid_start + valid_dates
        valid_date_block = unique_dates[valid_start:valid_end]
        train_date_block = unique_dates[:valid_start]
        train_df = df[df["date"].isin(train_date_block)].copy()
        valid_df = df[df["date"].isin(valid_date_block)].copy()
        if train_df.empty or valid_df.empty:
            continue
        folds.append((train_df, valid_df, fold_idx + 1))
    if not folds:
        raise ValueError("Failed to construct any non-empty walk-forward folds")
    return folds


def resolve_feature_columns(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SET_PRESETS:
        raise ValueError(f"Unsupported feature_set: {feature_set}")
    return list(FEATURE_SET_PRESETS[feature_set])


def run_walk_forward(
    df: pd.DataFrame,
    feature_columns: list[str],
    valid_dates: int,
    num_folds: int,
    model_family: str = "auto",
    seed: int = SEED,
) -> tuple[list[dict], pd.DataFrame, str]:
    folds = build_walk_forward_folds(df, valid_dates, num_folds)
    fold_metrics = []
    prediction_frames = []
    backend_used = None

    for train_df, valid_df, fold_id in folds:
        model, backend = build_model(model_family, seed=seed + fold_id)
        backend_used = backend

        model.fit(train_df[feature_columns], train_df[MODEL_LABEL_COLUMN])
        valid_pred = model.predict(valid_df[feature_columns])

        scored = valid_df[["stock_id", "date", RAW_LABEL_COLUMN, MODEL_LABEL_COLUMN]].copy()
        scored["pred_return"] = valid_pred
        scored["fold_id"] = fold_id
        prediction_frames.append(scored)

        metrics = compute_fold_metrics(scored)
        metrics.update(
            {
                "fold_id": fold_id,
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "train_date_start": str(train_df["date"].min().date()),
                "train_date_end": str(train_df["date"].max().date()),
                "valid_date_start": str(valid_df["date"].min().date()),
                "valid_date_end": str(valid_df["date"].max().date()),
            }
        )
        fold_metrics.append(metrics)

    return fold_metrics, pd.concat(prediction_frames, ignore_index=True), backend_used or "unknown"


def summarise_metrics(fold_metrics: list[dict]) -> dict:
    metrics_df = pd.DataFrame(fold_metrics)
    return {
        "rmse_mean": float(metrics_df["rmse"].mean()),
        "mae_mean": float(metrics_df["mae"].mean()),
        "rank_ic_mean": float(metrics_df["rank_ic"].mean()),
        "top5_mean_return_mean": float(metrics_df["top5_mean_return"].mean()),
        "rmse_std": float(metrics_df["rmse"].std(ddof=0)),
        "mae_std": float(metrics_df["mae"].std(ddof=0)),
        "rank_ic_std": float(metrics_df["rank_ic"].std(ddof=0)),
        "top5_mean_return_std": float(metrics_df["top5_mean_return"].std(ddof=0)),
    }


def fit_final_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    model_family: str = "auto",
    seed: int = SEED,
) -> tuple[object, str]:
    model, backend = build_model(model_family, seed=seed)
    model.fit(df[feature_columns], df[MODEL_LABEL_COLUMN])
    return model, backend


def main() -> None:
    args = parse_args()
    feature_path = Path(args.feature_path)
    model_dir, experiment_id = resolve_training_output_dir(
        requested_model_dir=Path(args.model_dir),
        experiment_root=args.experiment_root,
        experiment_id=args.experiment_id,
        model=args.model_family,
        feature=args.feature_set,
        sequence_length=None,
        sort_strategy=args.sort_strategy,
        weighting_scheme=args.weighting_scheme,
        remark=args.experiment_remark,
    )
    initial_experiment_config = {
        "experiment_id": experiment_id,
        "model_dir": str(model_dir),
        "feature_path": str(feature_path),
        "model_family": args.model_family,
        "feature_set": args.feature_set,
        "sequence_length": None,
        "sort_strategy": args.sort_strategy,
        "weighting_scheme": args.weighting_scheme,
        "target_mode": args.target_mode,
        "valid_dates": int(args.valid_dates),
        "num_folds": int(args.num_folds),
        "seed": int(args.seed),
        "remark": args.experiment_remark,
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
        run_training(args=args, feature_path=feature_path, model_dir=model_dir, experiment_id=experiment_id, initial_experiment_config=initial_experiment_config)


def run_training(
    *,
    args: argparse.Namespace,
    feature_path: Path,
    model_dir: Path,
    experiment_id: str | None,
    initial_experiment_config: dict,
) -> None:

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

    fold_metrics, walk_forward_predictions, walk_forward_backend = run_walk_forward(
        df, feature_columns, args.valid_dates, args.num_folds, args.model_family, seed=seed
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

    final_model, final_backend = fit_final_model(df, feature_columns, args.model_family, seed=seed)
    model_path = model_dir / "baseline_model.pkl"
    joblib.dump(final_model, model_path)

    pd.DataFrame(fold_metrics).to_csv(
        model_dir / "walk_forward_metrics.csv", index=False, encoding="utf-8-sig"
    )
    if experiment_id:
        write_training_metric_exports(
            experiment_dir=model_dir,
            fold_metrics=fold_metrics,
            metric_summary=metric_summary,
        )
    walk_forward_predictions.to_csv(
        model_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig"
    )
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
        experiment_name=f"train_{args.feature_set}_{walk_forward_backend}",
        prediction_path=model_dir / "walk_forward_predictions.csv",
        fold_diagnostics_path=model_dir / "fold_diagnostics.csv",
        fold_daily_diagnostics_path=model_dir / "fold_daily_diagnostics.csv",
        extra_fields={
            "model_dir": str(model_dir),
            "feature_set": args.feature_set,
            "target_mode": args.target_mode,
            "model_family": final_backend,
            "seed": int(seed),
        },
    )

    metadata = {
        "status": "trained",
        "experiment_id": experiment_id,
        "best_profile_name": BEST_PROFILE_NAME,
        "default_submission_profile": {
            "profile_name": BEST_CONFIG["profile_name"],
            "feature_set": BEST_CONFIG["training"]["feature_set"],
            "target_mode": BEST_CONFIG["training"]["target_mode"],
            "model_family": BEST_CONFIG["training"]["model_family"],
            "seed": int(BEST_CONFIG["training"].get("seed", SEED)),
            "sort_strategy": BEST_CONFIG["selection"]["sort_strategy"],
            "weighting_scheme": BEST_CONFIG["selection"]["weighting_scheme"],
            "top_k": int(BEST_CONFIG["selection"]["top_k"]),
            "primary_candidate_size": int(BEST_CONFIG["selection"]["primary_candidate_size"]),
            "max_turnover": float(BEST_CONFIG["execution"]["max_turnover"]),
            "transaction_cost": float(BEST_CONFIG["execution"]["transaction_cost"]),
        },
        "backend": final_backend,
        "walk_forward_backend": walk_forward_backend,
        "feature_path": str(feature_path),
        "model_path": str(model_path),
        "feature_columns": feature_columns,
        "feature_set": args.feature_set,
        "requested_model_family": args.model_family,
        "raw_label_column": RAW_LABEL_COLUMN,
        "model_label_column": MODEL_LABEL_COLUMN,
        "target_mode": args.target_mode,
        "seed": seed,
        "valid_dates": args.valid_dates,
        "num_folds": args.num_folds,
        "train_rows_full": int(len(df)),
        "train_date_range_full": [
            str(df["date"].min().date()),
            str(df["date"].max().date()),
        ],
        "walk_forward_summary": metric_summary,
        "rank_stability_summary": stability_row,
        "walk_forward_folds": fold_metrics,
        "walk_forward_fold_prediction_files": [path.name for path in fold_prediction_paths],
        "walk_forward_diagnostics_profile": "walk_forward_default",
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
                "backend": final_backend,
                "walk_forward_backend": walk_forward_backend,
                "model_path": str(model_path),
                "feature_columns": feature_columns,
                "train_rows_full": int(len(df)),
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

    print(f"[train] backend={final_backend}")
    if experiment_id:
        print(f"[train] experiment_id={experiment_id}")
        print(f"[train] experiment_dir={model_dir}")
    print(f"[train] requested_model_family={args.model_family}")
    print(f"[train] feature_set={args.feature_set}")
    print(f"[train] target_mode={args.target_mode}")
    print(f"[train] seed={seed}")
    print(f"[train] feature_count={len(feature_columns)}")
    print(f"[train] train_rows_full={len(df)}")
    print(
        f"[train] train_date_range_full={df['date'].min().date()}~{df['date'].max().date()}"
    )
    print(
        f"[train] walk_forward rank_ic_mean={metric_summary['rank_ic_mean']:.6f} "
        f"top5_mean_return_mean={metric_summary['top5_mean_return_mean']:.6f}"
    )
    print(
        f"[train] walk_forward rmse_mean={metric_summary['rmse_mean']:.6f} "
        f"mae_mean={metric_summary['mae_mean']:.6f}"
    )
    print(f"[train] wrote fold diagnostics to {model_dir / 'fold_diagnostics.csv'}")
    print(f"[train] wrote fold prediction exports to {model_dir}")
    print(f"[train] wrote model to {model_path}")
    print(f"[train] wrote metadata to {model_dir / 'model_meta.json'}")


if __name__ == "__main__":
    main()
