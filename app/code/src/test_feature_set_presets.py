from __future__ import annotations

from featurework import BASE_ALPHA_V4_MEDIUM_COLUMNS, FEATURE_COLUMNS
from train import FEATURE_SET_PRESETS, resolve_feature_columns


def test_base_alpha_v4_medium_preset_is_registered() -> None:
    columns = resolve_feature_columns("base_alpha_v4_medium")
    assert columns == BASE_ALPHA_V4_MEDIUM_COLUMNS
    assert len(columns) == len(set(columns))


def test_base_alpha_v4_medium_columns_are_generated_features() -> None:
    missing = sorted(set(BASE_ALPHA_V4_MEDIUM_COLUMNS) - set(FEATURE_COLUMNS))
    assert missing == []


def test_base_alpha_v4_medium_has_no_target_leakage_columns() -> None:
    leakage_columns = {"future_open_1", "future_open_5", "target_return", "train_target"}
    assert sorted(leakage_columns & set(BASE_ALPHA_V4_MEDIUM_COLUMNS)) == []


def test_existing_mainline_preset_is_unchanged_subset() -> None:
    mainline = FEATURE_SET_PRESETS["base_alpha_v3_rs_crowding_mini4"]
    expected_tail = [
        "rel_hs300_mean_ret_5d",
        "rel_cs_mean_close_to_ma_10d",
        "turnover_spike_5d",
        "volume_price_divergence_5d",
    ]
    assert mainline[-4:] == expected_tail
