import json
from pathlib import Path

from load_submission_config import (
    DEFAULT_SUBMISSION_CONFIG_PATH,
    build_default_inference_args as build_submission_inference_args,
    load_submission_config,
)


ROOT_DIR = Path(__file__).resolve().parents[3]
BEST_CONFIG_PATH = ROOT_DIR / "app" / "model" / "best_config.json"


def load_best_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else BEST_CONFIG_PATH
    if not path.is_absolute():
        path = ROOT_DIR / path
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_metadata_artifact_path(model_dir: str | Path, stored_path: str | Path) -> Path:
    model_dir_path = Path(model_dir)
    artifact_path = Path(stored_path)
    if artifact_path.is_absolute() and artifact_path.exists():
        return artifact_path
    if not artifact_path.is_absolute():
        model_dir_relative_candidate = model_dir_path / artifact_path
        if model_dir_relative_candidate.exists():
            return model_dir_relative_candidate
    if not artifact_path.is_absolute():
        cwd_candidate = Path.cwd() / artifact_path
        if cwd_candidate.exists():
            return cwd_candidate
    model_dir_candidate = model_dir_path / artifact_path.name
    if model_dir_candidate.exists():
        return model_dir_candidate
    if artifact_path.is_absolute():
        return model_dir_candidate
    return Path.cwd() / artifact_path


BEST_CONFIG = load_best_config()

BEST_PROFILE_NAME = BEST_CONFIG["profile_name"]

TRAINING_DEFAULTS = BEST_CONFIG["training"]
SELECTION_DEFAULTS = BEST_CONFIG["selection"]
RISK_FILTER_DEFAULTS = BEST_CONFIG["risk_filter_thresholds"]
EXECUTION_DEFAULTS = BEST_CONFIG["execution"]


def build_default_inference_args() -> dict:
    return build_submission_inference_args(load_submission_config(DEFAULT_SUBMISSION_CONFIG_PATH))
