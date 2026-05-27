import pytest

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT_DIR / "app" / "code" / "src"
sys.path.insert(0, str(SRC_ROOT))

from check_required_files import RequiredFilesError, check_required_files


def write_json(path: Path, payload: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def make_minimal_app(tmp_path: Path) -> Path:
    app_root = tmp_path / "app"
    write_json(app_root / "model" / "default_submission_config.json")
    write_json(app_root / "model" / "best_config.json")
    write_json(app_root / "model" / "model_meta.json", '{"model_path": "submission_artifacts/lstm_model.pt"}')
    write_json(app_root / "model" / "submission_artifacts" / "manifest.json")
    write_json(app_root / "model" / "submission_artifacts" / "model_meta.json")
    write_json(app_root / "model" / "submission_artifacts" / "default_submission_config.json")
    (app_root / "model" / "submission_artifacts" / "lstm_model.pt").write_bytes(b"model")
    (app_root / "data").mkdir(parents=True, exist_ok=True)
    (app_root / "data" / "train.csv").write_text("stock_id,date\n", encoding="utf-8")
    (app_root / "data" / "test.csv").write_text("stock_id,date\n", encoding="utf-8")
    return app_root


def test_predict_reports_missing_test_csv_without_traceback(tmp_path: Path) -> None:
    app_root = make_minimal_app(tmp_path)
    (app_root / "data" / "test.csv").unlink()

    with pytest.raises(RequiredFilesError) as exc_info:
        check_required_files(app_root, "predict")

    message = str(exc_info.value)
    assert "[ERROR] Missing app/data/test.csv" in message
    assert "Fix:" in message
    assert "Traceback" not in message


def test_predict_reports_missing_submission_artifacts(tmp_path: Path) -> None:
    app_root = make_minimal_app(tmp_path)
    (app_root / "model" / "submission_artifacts" / "lstm_model.pt").unlink()

    with pytest.raises(RequiredFilesError) as exc_info:
        check_required_files(app_root, "predict")

    assert "[ERROR] Missing app/model/submission_artifacts/lstm_model.pt" in str(exc_info.value)


def test_train_reports_missing_train_csv(tmp_path: Path) -> None:
    app_root = make_minimal_app(tmp_path)
    (app_root / "data" / "train.csv").unlink()

    with pytest.raises(RequiredFilesError) as exc_info:
        check_required_files(app_root, "train")

    assert "[ERROR] Missing app/data/train.csv" in str(exc_info.value)
