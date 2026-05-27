from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT_DIR


DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "market_regime_analysis"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "daily_market_regimes.csv"


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else 0.0


def _direction_consistency(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    signs = np.sign(numeric)
    return float(abs(signs.mean()))


def build_daily_market_regimes(
    feature_df: pd.DataFrame,
    volatility_column: str = "volatility_20d",
    return_column: str = "ret_1d",
    rolling_window: int = 20,
) -> pd.DataFrame:
    required = {"date", volatility_column, return_column}
    missing = sorted(required - set(feature_df.columns))
    if missing:
        raise ValueError(f"Feature data is missing required columns for regime split: {missing}")

    working = feature_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    working = working.dropna(subset=["date"])
    daily = (
        working.groupby("date", as_index=False)
        .agg(
            market_volatility_20d=(volatility_column, _safe_mean),
            market_return_1d=(return_column, _safe_mean),
            direction_consistency=(return_column, _direction_consistency),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily["market_return_20d"] = daily["market_return_1d"].rolling(rolling_window, min_periods=5).sum()
    daily["trend_strength"] = daily["market_return_20d"].abs()
    daily["trend_direction"] = np.where(daily["market_return_20d"] >= 0, "up", "down")

    volatility_threshold = float(daily["market_volatility_20d"].median())
    trend_threshold = float(daily["trend_strength"].median())
    consistency_threshold = float(daily["direction_consistency"].median())
    daily["volatility_threshold"] = volatility_threshold
    daily["trend_strength_threshold"] = trend_threshold
    daily["direction_consistency_threshold"] = consistency_threshold

    daily["volatility_regime"] = np.where(
        daily["market_volatility_20d"] >= volatility_threshold,
        "high_volatility",
        "low_volatility",
    )
    daily["trend_regime"] = np.where(
        (daily["trend_strength"] >= trend_threshold)
        & (daily["direction_consistency"] >= consistency_threshold),
        "trend",
        "range",
    )
    daily["primary_regime"] = daily["volatility_regime"] + "_" + daily["trend_regime"]
    daily["is_low_volatility"] = daily["volatility_regime"].eq("low_volatility").astype(int)
    daily["is_high_volatility"] = daily["volatility_regime"].eq("high_volatility").astype(int)
    daily["is_trend"] = daily["trend_regime"].eq("trend").astype(int)
    daily["is_range"] = daily["trend_regime"].eq("range").astype(int)
    daily["is_high_volatility_range"] = daily["primary_regime"].eq("high_volatility_range").astype(int)
    daily["is_low_volatility_trend"] = daily["primary_regime"].eq("low_volatility_trend").astype(int)
    return daily


def load_and_split_market_regimes(
    feature_path: str | Path = DEFAULT_FEATURE_PATH,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    feature_df = pd.read_csv(resolve_path(feature_path), encoding="utf-8-sig", dtype={"stock_id": str})
    daily = build_daily_market_regimes(feature_df)
    if output_path is not None:
        output_resolved = resolve_path(output_path)
        output_resolved.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(output_resolved, index=False, encoding="utf-8-sig")
    return daily


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split simple, explainable daily market regimes.")
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = resolve_path(args.output_path)
    if args.output_dir:
        output_path = resolve_path(args.output_dir) / "daily_market_regimes.csv"
    daily = load_and_split_market_regimes(feature_path=args.feature_path, output_path=output_path)
    counts = daily["primary_regime"].value_counts().sort_index().rename_axis("regime").reset_index(name="days")
    print(counts.to_string(index=False))
    print(f"[market_regime_split] wrote {output_path}")


if __name__ == "__main__":
    main()
