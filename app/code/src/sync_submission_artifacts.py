import argparse
import json
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "app" / "model"
SUBMISSION_ARTIFACTS_DIR = MODEL_DIR / "submission_artifacts"
MODEL_META_PATH = MODEL_DIR / "model_meta.json"
BEST_CONFIG_PATH = MODEL_DIR / "best_config.json"
DEFAULT_SUBMISSION_CONFIG_PATH = MODEL_DIR / "default_submission_config.json"
SNAPSHOT_PATH = MODEL_DIR / "final_submission_snapshot.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect formal submission artifacts into a minimal directory.")
    parser.add_argument("--model_meta_path", default=str(MODEL_META_PATH))
    parser.add_argument("--best_config_path", default=str(BEST_CONFIG_PATH))
    parser.add_argument("--default_submission_config_path", default=str(DEFAULT_SUBMISSION_CONFIG_PATH))
    parser.add_argument("--snapshot_path", default=str(SNAPSHOT_PATH))
    parser.add_argument("--artifacts_dir", default=str(SUBMISSION_ARTIFACTS_DIR))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_stored_path(base_dir: Path, stored_path: str | Path) -> Path:
    candidate = Path(stored_path)
    if candidate.is_absolute():
        return candidate
    if (ROOT_DIR / candidate).exists():
        return ROOT_DIR / candidate
    return base_dir / candidate


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if src.resolve() == dst.resolve():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_manifest(meta: dict, copied_files: list[str]) -> dict:
    return {
        "status": "submission_artifacts_ready",
        "profile_name": meta.get("best_profile_name") or meta.get("default_submission_profile", {}).get("profile_name", ""),
        "model_family": meta.get("model_family", ""),
        "feature_set": meta.get("feature_set", ""),
        "model_path": meta.get("model_path", ""),
        "copied_files": copied_files,
    }


def sync_submission_artifacts(
    model_meta_path: Path,
    best_config_path: Path,
    default_submission_config_path: Path,
    snapshot_path: Path,
    artifacts_dir: Path,
) -> dict:
    meta = load_json(model_meta_path)
    model_src = resolve_stored_path(model_meta_path.parent, meta["model_path"])
    model_dst = artifacts_dir / Path(meta["model_path"]).name

    copied_files: list[str] = []
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if copy_if_exists(model_src, model_dst):
        copied_files.append(model_dst.relative_to(ROOT_DIR).as_posix())

    for src_path in (best_config_path, default_submission_config_path, snapshot_path):
        dst_path = artifacts_dir / src_path.name
        if copy_if_exists(src_path, dst_path):
            copied_files.append(dst_path.relative_to(ROOT_DIR).as_posix())

    normalized_meta = dict(meta)
    normalized_meta["model_path"] = f"submission_artifacts/{model_dst.name}"
    if "feature_path" in normalized_meta:
        normalized_meta["feature_path"] = "app/temp/train_features.csv"
    write_json(artifacts_dir / "model_meta.json", normalized_meta)
    copied_files.append((artifacts_dir / "model_meta.json").relative_to(ROOT_DIR).as_posix())

    manifest = build_manifest(normalized_meta, copied_files)
    write_json(artifacts_dir / "manifest.json", manifest)

    readme_path = artifacts_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Submission Artifacts",
                "",
                "This directory contains the minimal frozen artifacts needed by the formal submission path.",
                "",
                "Included files:",
                *[f"- `{path}`" for path in copied_files],
                "",
                "The root `app/model/model_meta.json` remains the live metadata entrypoint.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "artifacts_dir": str(artifacts_dir),
        "model_src": str(model_src),
        "model_dst": str(model_dst),
        "copied_files": copied_files,
    }


def main() -> None:
    args = parse_args()
    result = sync_submission_artifacts(
        model_meta_path=Path(args.model_meta_path),
        best_config_path=Path(args.best_config_path),
        default_submission_config_path=Path(args.default_submission_config_path),
        snapshot_path=Path(args.snapshot_path),
        artifacts_dir=Path(args.artifacts_dir),
    )
    print(f"[sync_submission_artifacts] artifacts_dir={result['artifacts_dir']}")
    print(f"[sync_submission_artifacts] model_src={result['model_src']}")
    print(f"[sync_submission_artifacts] model_dst={result['model_dst']}")
    print(f"[sync_submission_artifacts] copied_files={len(result['copied_files'])}")


if __name__ == "__main__":
    main()
