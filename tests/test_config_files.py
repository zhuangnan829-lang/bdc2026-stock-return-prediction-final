import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "app" / "code" / "src"
sys.path.insert(0, str(SRC_ROOT))

from compare_config_consistency import compare_config_consistency
from load_submission_config import build_default_inference_args, load_submission_config


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_submission_config_loads_required_defaults() -> None:
    config = load_submission_config(ROOT_DIR / "app" / "model" / "default_submission_config.json")
    defaults = build_default_inference_args(config)

    assert config["profile_name"]
    assert defaults["top_k"] == 5
    assert defaults["primary_candidate_size"] > 0
    assert defaults["sort_strategy"] in {"pure_prediction", "risk_adjusted"}
    assert defaults["weighting_scheme"] in {"equal", "pred", "risk_adjusted", "pred_equal_blend"}
    assert 0 <= defaults["weight_blend_alpha"] <= 1
    assert 0 <= defaults["max_turnover"] <= 1
    assert config["seed"] == 2026


def test_config_files_are_consistent() -> None:
    ok, checks = compare_config_consistency(
        default_config_path=ROOT_DIR / "app" / "model" / "default_submission_config.json",
        best_config_path=ROOT_DIR / "app" / "model" / "best_config.json",
        model_meta_path=ROOT_DIR / "app" / "model" / "model_meta.json",
    )

    assert ok, [check for check in checks if not check["ok"]]
