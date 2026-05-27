import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "app" / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from featurework import BASE_ALPHA_V4_MEDIUM_COLUMNS, FEATURE_COLUMNS  # noqa: E402
from train import resolve_feature_columns  # noqa: E402


def test_base_alpha_v4_medium_feature_count_and_registration() -> None:
    columns = resolve_feature_columns("base_alpha_v4_medium")

    assert columns == BASE_ALPHA_V4_MEDIUM_COLUMNS
    assert 40 <= len(columns) <= 60
    assert len(columns) == len(set(columns))
    assert set(columns).issubset(FEATURE_COLUMNS)


def test_base_alpha_v4_medium_covers_required_signal_families() -> None:
    columns = set(resolve_feature_columns("base_alpha_v4_medium"))

    required_by_family = {
        "multi_window_momentum": {"ret_3d", "ret_5d", "ret_10d", "mom_5d", "mom_10d"},
        "volatility": {"volatility_5d", "volatility_10d", "volatility_20d"},
        "volume_amount": {"volume_ratio_5d", "volume_ratio_10d", "amount_ratio_5d"},
        "price_position": {"close_to_ma_10d", "distance_to_20d_high", "rebound_from_10d_low"},
        "relative_strength": {"rel_ret_5d", "rank_rel_ret_5d", "rel_hs300_mean_ret_5d"},
        "crowding": {"turnover_spike_5d", "volume_price_divergence_5d", "crowding_risk_5d"},
        "reversal_protection": {"crowding_reversal_risk_5d", "volatility_compression_breakout_20d"},
    }

    for family_columns in required_by_family.values():
        assert family_columns.issubset(columns)
