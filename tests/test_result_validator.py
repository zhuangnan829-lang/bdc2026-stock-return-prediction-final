import sys
from pathlib import Path

import pytest


SRC_ROOT = Path(__file__).resolve().parents[1] / "app" / "code" / "src"
sys.path.insert(0, str(SRC_ROOT))

from result_validator import validate_result_file


def write_result(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_validate_result_accepts_valid_file(tmp_path: Path) -> None:
    result_path = write_result(
        tmp_path / "result.csv",
        [
            "stock_id,weight",
            "000001,0.2",
            "600000,0.3",
            "300001,0.5",
        ],
    )

    summary = validate_result_file(result_path)

    assert summary["rows"] == 3
    assert summary["weight_sum"] == pytest.approx(1.0)
    assert summary["stock_ids"] == ["000001", "600000", "300001"]


def test_validate_result_rejects_weight_sum_above_one(tmp_path: Path) -> None:
    result_path = write_result(
        tmp_path / "result.csv",
        [
            "stock_id,weight",
            "000001,0.8",
            "000002,0.4",
        ],
    )

    with pytest.raises(ValueError, match="Weight sum must be <= 1"):
        validate_result_file(result_path)


def test_validate_result_rejects_bad_columns(tmp_path: Path) -> None:
    result_path = write_result(
        tmp_path / "result.csv",
        [
            "code,weight",
            "000001,0.5",
        ],
    )

    with pytest.raises(ValueError, match="exactly two columns"):
        validate_result_file(result_path)


def test_validate_result_rejects_duplicate_stock_ids(tmp_path: Path) -> None:
    result_path = write_result(
        tmp_path / "result.csv",
        [
            "stock_id,weight",
            "000001,0.2",
            "000001,0.2",
        ],
    )

    with pytest.raises(ValueError, match="unique"):
        validate_result_file(result_path)
