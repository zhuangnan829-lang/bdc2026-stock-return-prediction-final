import argparse
import sys
from pathlib import Path

import pandas as pd
import pytest


SRC_ROOT = Path(__file__).resolve().parents[1] / "app" / "code" / "src"
sys.path.insert(0, str(SRC_ROOT))

from backtest import calculate_max_drawdown, make_config, run_backtest


def base_args() -> argparse.Namespace:
    return argparse.Namespace(
        top_k=1,
        primary_candidate_size=2,
        enable_risk_filters=0,
        allow_cash_fallback=0,
        max_volatility_20d_pct=1.0,
        max_volatility_5d_pct=1.0,
        turnover_rate_lower_pct=0.0,
        turnover_rate_upper_pct=1.0,
        turnover_ratio_upper_pct=1.0,
        risk_penalty_weight=0.0,
        weighting_scheme="equal",
        weight_blend_alpha=1.0,
        max_single_weight=1.0,
        sort_strategy="pure_prediction",
        transaction_cost=0.01,
        max_turnover=1.0,
        rerank_signal_column=None,
        rerank_signal_weight=0.0,
        secondary_candidate_size=None,
        secondary_screen_mode="none",
        secondary_screen_weight=0.0,
        local_tiebreak_start_rank=8,
        local_tiebreak_end_rank=15,
    )


def test_calculate_max_drawdown() -> None:
    values = pd.Series([1.0, 1.2, 0.9, 1.1])

    assert calculate_max_drawdown(values) == pytest.approx(-0.25)


def test_run_backtest_tiny_sample_applies_cost_and_net_value() -> None:
    prediction_df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "stock_id": "000001",
                "pred_return": 0.9,
                "target_return": 0.10,
                "volatility_5d": 0.01,
                "volatility_20d": 0.02,
                "turnover_rate": 1.0,
                "turnover_ratio_10d": 0.1,
                "amplitude_ratio_5d": 0.1,
                "turnover_spike_5d": 1.0,
                "crowding_reversal_risk_5d": 0.0,
                "rel_strength_accel_5d_v2": 0.0,
                "trend_persistence_score_10d_v2": 0.0,
            },
            {
                "date": "2026-01-02",
                "stock_id": "000001",
                "pred_return": 0.9,
                "target_return": -0.20,
                "volatility_5d": 0.01,
                "volatility_20d": 0.02,
                "turnover_rate": 1.0,
                "turnover_ratio_10d": 0.1,
                "amplitude_ratio_5d": 0.1,
                "turnover_spike_5d": 1.0,
                "crowding_reversal_risk_5d": 0.0,
                "rel_strength_accel_5d_v2": 0.0,
                "trend_persistence_score_10d_v2": 0.0,
            },
        ]
    )
    config = make_config(base_args(), overrides={"profile_name": "tiny_basic"})

    summary_df, daily_df, holdings_df = run_backtest(
        prediction_df=prediction_df,
        config=config,
        prediction_source="unit_test",
    )

    assert len(daily_df) == 2
    assert len(holdings_df) == 2
    assert daily_df.loc[0, "gross_return"] == pytest.approx(0.10)
    assert daily_df.loc[0, "transaction_cost"] == pytest.approx(0.01)
    assert daily_df.loc[0, "net_return"] == pytest.approx(0.09)
    assert daily_df.loc[1, "turnover"] == pytest.approx(0.0)
    assert daily_df.loc[1, "net_value_after_cost"] == pytest.approx(1.09 * 0.8)
    assert summary_df.loc[0, "cumulative_return_after_cost"] == pytest.approx(-0.128)
    assert summary_df.loc[0, "max_drawdown_after_cost"] == pytest.approx(-0.20)
