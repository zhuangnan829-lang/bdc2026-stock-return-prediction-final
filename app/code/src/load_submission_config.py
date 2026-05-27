import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from utils_seed import DEFAULT_SEED


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SUBMISSION_CONFIG_PATH = ROOT_DIR / "app" / "model" / "default_submission_config.json"

REQUIRED_UNIFIED_FIELDS = [
    "model_name",
    "feature_set",
    "sequence_length",
    "sort_strategy",
    "weight_strategy",
    "top_k",
    "candidate_size",
    "risk_penalty_weight",
    "max_turnover",
    "transaction_cost",
    "max_single_weight",
]


def load_json(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"default_submission_config.json missing required section: {key}")
    return value


def build_unified_submission_config(config: dict[str, Any]) -> dict[str, Any]:
    validation = _require_mapping(config, "validation_scheme")
    selection = _require_mapping(config, "selection_logic")
    risk = _require_mapping(config, "risk_filter_thresholds")
    execution = _require_mapping(config, "execution_logic")

    unified = {
        "model_name": config.get("model_name", config.get("model_family")),
        "feature_set": config.get("feature_set"),
        "sequence_length": validation.get("sequence_length", config.get("sequence_length")),
        "sort_strategy": selection.get("sort_strategy"),
        "weight_strategy": selection.get("weight_strategy", selection.get("weighting_scheme")),
        "top_k": selection.get("top_k"),
        "candidate_size": selection.get("candidate_size", selection.get("primary_candidate_size")),
        "risk_penalty_weight": risk.get("risk_penalty_weight"),
        "max_turnover": execution.get("max_turnover"),
        "transaction_cost": execution.get("transaction_cost"),
        "max_single_weight": selection.get("max_single_weight"),
    }
    missing = [
        key
        for key in REQUIRED_UNIFIED_FIELDS
        if key != "max_single_weight" and unified.get(key) is None
    ]
    if missing:
        raise ValueError(
            "default_submission_config.json missing required field(s): "
            + ", ".join(missing)
        )
    return unified


def validate_submission_config(config: dict[str, Any]) -> None:
    build_unified_submission_config(config)


def load_submission_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_json(config_path or DEFAULT_SUBMISSION_CONFIG_PATH)
    unified = build_unified_submission_config(config)
    merged = dict(config)
    merged.update(unified)
    return merged


def build_default_inference_args(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_submission_config()
    validate_submission_config(cfg)
    selection = cfg["selection_logic"]
    risk = cfg["risk_filter_thresholds"]
    execution = cfg["execution_logic"]
    return {
        "top_k": int(selection["top_k"]),
        "primary_candidate_size": int(selection["primary_candidate_size"]),
        "enable_risk_filters": bool(selection.get("enable_risk_filters", True)),
        "sort_strategy": str(selection["sort_strategy"]),
        "weighting_scheme": str(selection["weighting_scheme"]),
        "weight_blend_alpha": float(selection.get("weight_blend_alpha", 1.0)),
        "max_single_weight": None if selection.get("max_single_weight") is None else float(selection.get("max_single_weight")),
        "rerank_signal_column": selection.get("rerank_signal_column") or "",
        "rerank_signal_weight": float(selection.get("rerank_signal_weight", 0.0)),
        "regime_rerank_enabled": bool(cfg.get("regime_rerank", {}).get("enabled", False)),
        "regime_rerank_flag": str(cfg.get("regime_rerank", {}).get("regime_flag", "")),
        "regime_rerank_signal": str(cfg.get("regime_rerank", {}).get("signal", "")),
        "regime_rerank_weight": float(cfg.get("regime_rerank", {}).get("weight", 0.0)),
        "max_volatility_20d_pct": float(risk["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(risk["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(risk["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(risk["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(risk["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(risk["risk_penalty_weight"]),
        "use_previous_result_when_available": bool(execution.get("use_previous_result_when_available", False)),
        "auto_use_previous_result": int(bool(execution.get("use_previous_result_when_available", False))),
        "max_turnover": float(execution["max_turnover"]),
        "transaction_cost": float(execution["transaction_cost"]),
    }


def build_training_defaults(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_submission_config()
    validate_submission_config(cfg)
    validation = cfg["validation_scheme"]
    return {
        "feature_set": cfg["feature_set"],
        "target_mode": cfg["target_mode"],
        "model_family": cfg["model_family"],
        "seed": int(cfg.get("seed", DEFAULT_SEED)),
        "valid_dates": int(validation["valid_dates"]),
        "num_folds": int(validation["num_folds"]),
        "sequence_length": int(validation["sequence_length"]),
    }


def build_best_config_from_submission(
    submission_config: dict[str, Any],
    best_template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    best = dict(best_template or {})
    training = dict(best.get("training", {}))
    training.update(build_training_defaults(submission_config))

    best.update(
        {
            "profile_name": submission_config["profile_name"],
            "status": "frozen_best_config",
            "training": training,
            "selection": dict(submission_config["selection_logic"]),
            "risk_filter_thresholds": dict(submission_config["risk_filter_thresholds"]),
            "execution": dict(submission_config["execution_logic"]),
            "ablation_conclusion": dict(submission_config.get("ablation_conclusion", {})),
        }
    )
    return best


def build_cli_payload(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_submission_config()
    return {
        "profile_name": cfg["profile_name"],
        "config_path": str(DEFAULT_SUBMISSION_CONFIG_PATH),
        "training_defaults": build_training_defaults(cfg),
        "inference_args": build_default_inference_args(cfg),
    }


def _shell_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def render_shell_defaults(payload: dict[str, Any]) -> str:
    mapping = {
        "TOP_K": payload["inference_args"]["top_k"],
        "PRIMARY_CANDIDATE_SIZE": payload["inference_args"]["primary_candidate_size"],
        "MAX_VOLATILITY_20D_PCT": payload["inference_args"]["max_volatility_20d_pct"],
        "MAX_VOLATILITY_5D_PCT": payload["inference_args"]["max_volatility_5d_pct"],
        "TURNOVER_RATE_LOWER_PCT": payload["inference_args"]["turnover_rate_lower_pct"],
        "TURNOVER_RATE_UPPER_PCT": payload["inference_args"]["turnover_rate_upper_pct"],
        "TURNOVER_RATIO_UPPER_PCT": payload["inference_args"]["turnover_ratio_upper_pct"],
        "RISK_PENALTY_WEIGHT": payload["inference_args"]["risk_penalty_weight"],
        "SORT_STRATEGY": payload["inference_args"]["sort_strategy"],
        "WEIGHTING_SCHEME": payload["inference_args"]["weighting_scheme"],
        "WEIGHT_BLEND_ALPHA": payload["inference_args"]["weight_blend_alpha"],
        "MAX_SINGLE_WEIGHT": payload["inference_args"]["max_single_weight"],
        "RERANK_SIGNAL_COLUMN": payload["inference_args"]["rerank_signal_column"],
        "RERANK_SIGNAL_WEIGHT": payload["inference_args"]["rerank_signal_weight"],
        "REGIME_RERANK_ENABLED": int(bool(payload["inference_args"]["regime_rerank_enabled"])),
        "REGIME_RERANK_FLAG": payload["inference_args"]["regime_rerank_flag"],
        "REGIME_RERANK_SIGNAL": payload["inference_args"]["regime_rerank_signal"],
        "REGIME_RERANK_WEIGHT": payload["inference_args"]["regime_rerank_weight"],
        "MAX_TURNOVER": payload["inference_args"]["max_turnover"],
        "TRANSACTION_COST": payload["inference_args"]["transaction_cost"],
        "AUTO_USE_PREVIOUS_RESULT": payload["inference_args"]["auto_use_previous_result"],
    }
    lines = []
    for key, value in mapping.items():
        default = shlex.quote(_shell_value(value))
        lines.append(f'{key}="${{{key}:-{default}}}"')
    return "\n".join(lines)


def render_powershell_defaults(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load formal submission config from the authoritative JSON file.")
    parser.add_argument("--config_path", default=str(DEFAULT_SUBMISSION_CONFIG_PATH))
    parser.add_argument("--format", choices=["json", "shell", "powershell"], default="json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_submission_config(args.config_path)
    payload = build_cli_payload(config)
    if args.format == "shell":
        print(render_shell_defaults(payload))
    elif args.format == "powershell":
        print(render_powershell_defaults(payload))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
