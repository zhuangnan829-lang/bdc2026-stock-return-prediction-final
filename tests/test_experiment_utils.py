import sys
from datetime import datetime
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "app" / "code" / "src"
sys.path.insert(0, str(SRC_ROOT))

from experiment_utils import (
    REQUIRED_EXPERIMENT_FILES,
    build_experiment_id,
    init_experiment_dir,
    resolve_training_output_dir,
)


def test_build_experiment_id_uses_standard_parts() -> None:
    experiment_id = build_experiment_id(
        model="lstm",
        feature="base_alpha_v3_rs_crowding_mini4",
        sequence_length=10,
        sort_strategy="risk_adjusted",
        weighting_scheme="pred",
        remark="Smoke Run",
        now=datetime(2026, 5, 22, 23, 1, 2),
    )

    assert experiment_id == (
        "20260522230102_lstm_base-alpha-v3-rs-crowding-mini4_"
        "sl10_risk-adjusted_pred_smoke-run"
    )


def test_init_experiment_dir_creates_required_artifacts(tmp_path: Path) -> None:
    experiment_dir = init_experiment_dir(tmp_path / "20260522_lstm_feature_sl10_sort_weight_remark")

    for filename in REQUIRED_EXPERIMENT_FILES:
        assert (experiment_dir / filename).exists()
    assert (experiment_dir / "figures").is_dir()


def test_resolve_training_output_dir_creates_unique_experiment(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiments"
    first_dir, first_id = resolve_training_output_dir(
        requested_model_dir=tmp_path / "model",
        experiment_root=experiment_root,
        experiment_id="20260522_lstm_feature_sl10_sort_weight_remark",
        model="lstm",
        feature="feature",
        sequence_length=10,
        sort_strategy="sort",
        weighting_scheme="weight",
        remark="remark",
    )
    second_dir, second_id = resolve_training_output_dir(
        requested_model_dir=tmp_path / "model",
        experiment_root=experiment_root,
        experiment_id="20260522_lstm_feature_sl10_sort_weight_remark",
        model="lstm",
        feature="feature",
        sequence_length=10,
        sort_strategy="sort",
        weighting_scheme="weight",
        remark="remark",
    )

    assert first_id == first_dir.name
    assert second_id == second_dir.name
    assert first_dir.name == "20260522_lstm_feature_sl10_sort_weight_remark"
    assert second_dir.name == "20260522_lstm_feature_sl10_sort_weight_remark_v2"
