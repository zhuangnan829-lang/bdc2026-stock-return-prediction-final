import argparse
import json
import sys
from pathlib import Path

from compare_config_consistency import compare_config_consistency as run_config_consistency_check
from config import BEST_CONFIG_PATH, resolve_metadata_artifact_path
from result_validator import validate_result_file


REQUIRED_PATHS = [
    "app/model/best_config.json",
    "app/model/default_submission_config.json",
    "app/model/model_meta.json",
    "app/output/result.csv",
    "app/train.sh",
    "app/test.sh",
    "app/init.sh",
    "app/readme.md",
]

MODEL_ARTIFACT_CANDIDATES = [
    "app/model/baseline_model.pkl",
    "app/model/lstm_model.pt",
    "app/model/transformer_model.pt",
    "app/model/submission_artifacts/lstm_model.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-submit checks for the competition package.")
    parser.add_argument("--root_dir", default=".")
    parser.add_argument("--result_path", default="app/output/result.csv")
    return parser.parse_args()


def ensure_required_files(root_dir: Path) -> list[str]:
    missing = []
    for rel_path in REQUIRED_PATHS:
        if not (root_dir / rel_path).exists():
            missing.append(rel_path)
    if not any((root_dir / rel_path).exists() for rel_path in MODEL_ARTIFACT_CANDIDATES):
        missing.append("one_of:" + ",".join(MODEL_ARTIFACT_CANDIDATES))
    return missing


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_model_artifact_resolution(root_dir: Path) -> Path:
    model_meta = load_json(root_dir / "app/model/model_meta.json")
    resolved = resolve_metadata_artifact_path(root_dir / "app/model", model_meta["model_path"])
    if not resolved.exists():
        raise FileNotFoundError(f"Resolved model artifact does not exist: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    result_path = root_dir / args.result_path

    missing = ensure_required_files(root_dir)
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    result_summary = validate_result_file(result_path)
    consistency_ok, consistency_checks = run_config_consistency_check(
        default_config_path=root_dir / "app/model/default_submission_config.json",
        best_config_path=root_dir / "app/model/best_config.json",
        model_meta_path=root_dir / "app/model/model_meta.json",
    )
    resolved_model_path = validate_model_artifact_resolution(root_dir)
    if not consistency_ok:
        failed_checks = [check["name"] for check in consistency_checks if not check["ok"]]
        raise ValueError(f"Configuration consistency checks failed: {failed_checks}")

    print(f"[pre_submit_check] root_dir={root_dir}")
    print(f"[pre_submit_check] best_config_path={BEST_CONFIG_PATH}")
    print(f"[pre_submit_check] required_files_ok={len(REQUIRED_PATHS)}")
    print(
        f"[pre_submit_check] result_ok rows={result_summary['rows']} "
        f"weight_sum={result_summary['weight_sum']:.6f} encoding={result_summary['encoding']}"
    )
    print(f"[pre_submit_check] config_consistency_ok={len(consistency_checks)}")
    print(f"[pre_submit_check] resolved_model_path={resolved_model_path}")
    print("[pre_submit_check] all checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[pre_submit_check][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
