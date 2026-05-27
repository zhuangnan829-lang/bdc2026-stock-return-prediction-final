import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "app" / "code" / "src"
sys.path.insert(0, str(SRC_ROOT))

from compare_local_docker_result import compare_results, read_result, write_report


def write_result(path: Path, newline: str = "\n") -> Path:
    path.write_text(
        newline.join(
            [
                "stock_id,weight",
                "300316,0.6",
                "600183,0.4",
            ]
        )
        + newline,
        encoding="utf-8",
    )
    return path


def test_compare_results_accepts_identical_files(tmp_path: Path) -> None:
    local = read_result(write_result(tmp_path / "local.csv"))
    docker = read_result(write_result(tmp_path / "docker.csv"))

    checks = compare_results(local, docker)

    assert checks["passed"]
    assert checks["md5_match"]
    assert local["weight_sum"] == docker["weight_sum"]


def test_compare_results_rejects_line_ending_md5_mismatch(tmp_path: Path) -> None:
    local = read_result(write_result(tmp_path / "local.csv", newline="\r\n"))
    docker = read_result(write_result(tmp_path / "docker.csv", newline="\n"))

    checks = compare_results(local, docker)

    assert not checks["passed"]
    assert checks["stock_ids_match"]
    assert checks["weights_match"]
    assert checks["weight_sums_match"]
    assert not checks["md5_match"]


def test_write_report_records_required_md5_fields(tmp_path: Path) -> None:
    local = read_result(write_result(tmp_path / "local.csv"))
    docker = read_result(write_result(tmp_path / "docker.csv"))
    checks = compare_results(local, docker)
    report_path = tmp_path / "docker_consistency_check.md"

    write_report(report_path, local, docker, checks, error=None)

    report = report_path.read_text(encoding="utf-8")
    assert "status: `PASS`" in report
    assert "local_result_md5" in report
    assert "docker_result_md5" in report
