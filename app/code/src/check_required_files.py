import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


class RequiredFilesError(RuntimeError):
    """Raised when an entrypoint is missing required local files."""


def _display_path(app_root: Path, path: Path) -> str:
    try:
        return Path("app", path.relative_to(app_root)).as_posix()
    except ValueError:
        return path.as_posix()


def _fix_hint(display_path: str) -> str:
    hints = {
        "app/data/test.csv": "Put the competition test data at app/data/test.csv, then rerun app/test.sh or app/test.ps1.",
        "app/data/train.csv": "Put the training data at app/data/train.csv, or keep a valid app/temp/train_features.csv history file.",
        "app/model/default_submission_config.json": "Restore it from app/model/submission_artifacts/default_submission_config.json or rerun the freeze/sync step.",
        "app/model/best_config.json": "Restore it from app/model/submission_artifacts/best_config.json or rerun python app/code/src/sync_submission_config.py.",
        "app/model/model_meta.json": "Restore it from app/model/submission_artifacts/model_meta.json or rerun the freeze/sync step.",
        "app/model/submission_artifacts": "Run python app/code/src/sync_submission_artifacts.py after training/freezing the model.",
        "app/model/submission_artifacts/manifest.json": "Run python app/code/src/sync_submission_artifacts.py to rebuild the frozen artifact manifest.",
        "app/model/submission_artifacts/model_meta.json": "Run python app/code/src/sync_submission_artifacts.py to copy the frozen metadata.",
        "app/model/submission_artifacts/default_submission_config.json": "Run python app/code/src/sync_submission_artifacts.py to copy the frozen config.",
    }
    return hints.get(display_path, f"Restore or regenerate {display_path}, then rerun the command.")


def _missing_message(display_path: str) -> str:
    return f"[ERROR] Missing {display_path}\n        Fix: {_fix_hint(display_path)}"


def _read_json(path: Path, display_path: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RequiredFilesError(
            f"[ERROR] Invalid JSON {display_path}\n"
            f"        Fix: repair the JSON syntax or restore the file from a known-good snapshot.\n"
            f"        Detail: {exc}"
        ) from exc


def _append_missing(missing: list[str], app_root: Path, path: Path, *, expect_dir: bool = False) -> None:
    exists = path.is_dir() if expect_dir else path.is_file()
    if not exists:
        missing.append(_missing_message(_display_path(app_root, path)))


def _resolve_model_artifact(app_root: Path, model_meta_path: Path) -> Path | None:
    if not model_meta_path.is_file():
        return None
    meta = _read_json(model_meta_path, _display_path(app_root, model_meta_path))
    stored_model_path = meta.get("model_path")
    if not stored_model_path:
        raise RequiredFilesError(
            f"[ERROR] Invalid {_display_path(app_root, model_meta_path)}\n"
            "        Fix: add the model_path field or regenerate model_meta.json with the freeze/sync step."
        )
    model_dir = app_root / "model"
    model_path = Path(stored_model_path)
    candidates = []
    if model_path.is_absolute():
        candidates.append(model_path)
    else:
        candidates.extend([model_dir / model_path, model_dir / model_path.name, Path.cwd() / model_path])
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _require_one_of(app_root: Path, paths: Iterable[Path], missing: list[str], reason: str) -> None:
    path_list = list(paths)
    if any(path.is_file() for path in path_list):
        return
    display_paths = ", ".join(_display_path(app_root, path) for path in path_list)
    missing.append(
        f"[ERROR] Missing one of: {display_paths}\n"
        f"        Fix: {reason}"
    )


def check_required_files(app_root: str | Path, command: str) -> None:
    app_root = Path(app_root).resolve()
    missing: list[str] = []

    _append_missing(missing, app_root, app_root / "model" / "default_submission_config.json")
    _append_missing(missing, app_root, app_root / "model" / "best_config.json")

    if command == "train":
        _append_missing(missing, app_root, app_root / "data" / "train.csv")

    if command in {"predict", "freeze"}:
        _append_missing(missing, app_root, app_root / "data" / "test.csv")
        _append_missing(missing, app_root, app_root / "model" / "model_meta.json")
        _append_missing(missing, app_root, app_root / "model" / "submission_artifacts", expect_dir=True)
        _append_missing(missing, app_root, app_root / "model" / "submission_artifacts" / "manifest.json")
        _append_missing(missing, app_root, app_root / "model" / "submission_artifacts" / "model_meta.json")
        _append_missing(
            missing,
            app_root,
            app_root / "model" / "submission_artifacts" / "default_submission_config.json",
        )
        _require_one_of(
            app_root,
            [app_root / "temp" / "train_features.csv", app_root / "data" / "train.csv"],
            missing,
            "Keep app/temp/train_features.csv or put training data at app/data/train.csv so LSTM history can be built.",
        )

        model_artifact = _resolve_model_artifact(app_root, app_root / "model" / "model_meta.json")
        if model_artifact is not None and not model_artifact.is_file():
            display_path = _display_path(app_root, model_artifact)
            missing.append(
                f"[ERROR] Missing {display_path}\n"
                "        Fix: copy the trained model artifact back, or rerun "
                "python app/code/src/sync_submission_artifacts.py after training."
            )

    if missing:
        raise RequiredFilesError("\n".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check required files before running project entrypoints.")
    parser.add_argument("command", choices=["train", "predict", "freeze"])
    parser.add_argument("--app-root", default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        check_required_files(args.app_root, args.command)
    except RequiredFilesError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(f"[check_required_files] {args.command} required files OK")


if __name__ == "__main__":
    main()
