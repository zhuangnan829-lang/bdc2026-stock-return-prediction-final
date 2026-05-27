import argparse
import csv
import hashlib
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT_DIR / "app" / "code" / "src"
DEFAULT_LOCAL_RESULT_PATH = ROOT_DIR / "app" / "output" / "result.csv"
DEFAULT_DOCKER_RESULT_PATH = ROOT_DIR / "test" / "output" / "result.csv"
DEFAULT_REPORT_PATH = ROOT_DIR / "app" / "model" / "docker_consistency_check.md"
DEFAULT_IMAGE_NAME = "bdc2026:latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local and Docker submission entries, then compare result.csv "
            "stock ids, weights, weight sums, and raw MD5 checksums."
        )
    )
    parser.add_argument("--local_result_path", default=str(DEFAULT_LOCAL_RESULT_PATH))
    parser.add_argument("--docker_result_path", default=str(DEFAULT_DOCKER_RESULT_PATH))
    parser.add_argument("--report_path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--image_name", default=DEFAULT_IMAGE_NAME)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--skip_local_run", action="store_true", help="Only compare the existing local result file.")
    parser.add_argument("--skip_docker_run", action="store_true", help="Only compare the existing Docker result file.")
    parser.add_argument("--build_docker_image", action="store_true", help="Build the Docker image before running it.")
    parser.add_argument("--skip_docker_build", action="store_true", help="Deprecated: Docker build is skipped by default.")
    parser.add_argument("--timeout_seconds", type=int, default=900)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT_DIR / candidate


def run_command(command: list[str], timeout_seconds: int, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    output = completed.stdout.strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}\n{output}"
        )
    return output


def assert_result_updated(path: Path, started_at: float) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Result was not generated: {path}")
    if path.stat().st_mtime + 2 < started_at:
        raise RuntimeError(f"Result file was not updated by the latest run: {path}")


def run_local_submission(local_result_path: Path, python_bin: str, timeout_seconds: int) -> list[str]:
    env = os.environ.copy()
    env["PYTHON_BIN"] = python_bin
    outputs = []
    outputs.append(run_command([python_bin, str(SRC_ROOT / "sync_submission_config.py")], timeout_seconds, env))
    outputs.append(run_command([python_bin, str(SRC_ROOT / "compare_config_consistency.py")], timeout_seconds, env))

    started_at = time.time()
    if platform.system().lower().startswith("win"):
        outputs.append(
            run_command(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(ROOT_DIR / "app" / "test.ps1"),
                ],
                timeout_seconds,
                env,
            )
        )
    else:
        outputs.append(run_command(["bash", str(ROOT_DIR / "app" / "run_submission.sh")], timeout_seconds, env))

    assert_result_updated(local_result_path, started_at)
    return outputs


def run_docker_submission(
    docker_result_path: Path,
    image_name: str,
    build_docker_image: bool,
    timeout_seconds: int,
) -> list[str]:
    docker_result_path.parent.mkdir(parents=True, exist_ok=True)

    outputs = []
    if build_docker_image:
        outputs.append(run_command(["docker", "build", "-t", image_name, "."], timeout_seconds))

    volume_args = [
        "-v",
        f"{(ROOT_DIR / 'app' / 'data').resolve()}:/app/data",
        "-v",
        f"{docker_result_path.parent.resolve()}:/app/output",
        "-v",
        f"{(ROOT_DIR / 'app' / 'temp').resolve()}:/app/temp",
    ]
    command = [
        "docker",
        "run",
        "--rm",
        "-e",
        "RUN_TRAIN=0",
        *volume_args,
        image_name,
        "/bin/bash",
        "/app/data/run.sh",
    ]
    started_at = time.time()
    outputs.append(run_command(command, timeout_seconds))

    assert_result_updated(docker_result_path, started_at)
    return outputs


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["stock_id", "weight"]:
            raise ValueError(f"{path} must have columns stock_id,weight")
        rows = []
        for row in reader:
            rows.append(
                {
                    "stock_id": str(row["stock_id"]).zfill(6),
                    "weight": str(row["weight"]),
                    "weight_decimal": Decimal(str(row["weight"])),
                }
            )

    return {
        "path": path,
        "md5": file_md5(path),
        "rows": rows,
        "stock_ids": [row["stock_id"] for row in rows],
        "weights": [row["weight"] for row in rows],
        "weight_sum": sum((row["weight_decimal"] for row in rows), Decimal("0")),
    }


def compare_results(local: dict, docker: dict) -> dict:
    local_pairs = [(row["stock_id"], row["weight"]) for row in local["rows"]]
    docker_pairs = [(row["stock_id"], row["weight"]) for row in docker["rows"]]
    checks = {
        "stock_ids_match": local["stock_ids"] == docker["stock_ids"],
        "weights_match": local["weights"] == docker["weights"],
        "weight_sums_match": local["weight_sum"] == docker["weight_sum"],
        "md5_match": local["md5"] == docker["md5"],
        "rows_match": local_pairs == docker_pairs,
    }
    checks["passed"] = all(checks.values())
    return checks


def markdown_bool(value: bool) -> str:
    return "PASS" if value else "FAIL"


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def write_report(report_path: Path, local: dict | None, docker: dict | None, checks: dict, error: str | None) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    status = "PASS" if checks.get("passed") and error is None else "FAIL"

    lines = [
        "# Docker Consistency Check",
        "",
        f"- generated_at_utc: `{generated_at}`",
        f"- status: `{status}`",
        "",
    ]
    if error:
        lines.extend(["## Error", "", "```text", error.strip(), "```", ""])

    if local and docker:
        lines.extend(
            [
                "## Result Summary",
                "",
                "| item | local | docker | match |",
                "|---|---:|---:|---|",
                f"| path | `{display_path(local['path'])}` | `{display_path(docker['path'])}` |  |",
                f"| rows | {len(local['rows'])} | {len(docker['rows'])} | {markdown_bool(len(local['rows']) == len(docker['rows']))} |",
                f"| weight_sum | {local['weight_sum']} | {docker['weight_sum']} | {markdown_bool(checks['weight_sums_match'])} |",
                f"| result_md5 | `{local['md5']}` | `{docker['md5']}` | {markdown_bool(checks['md5_match'])} |",
                "",
                f"- local_result_md5: `{local['md5']}`",
                f"- docker_result_md5: `{docker['md5']}`",
                "",
                "## Row Comparison",
                "",
                "| rank | local_stock_id | local_weight | docker_stock_id | docker_weight | match |",
                "|---:|---|---:|---|---:|---|",
            ]
        )
        max_rows = max(len(local["rows"]), len(docker["rows"]))
        for index in range(max_rows):
            local_row = local["rows"][index] if index < len(local["rows"]) else {"stock_id": "", "weight": ""}
            docker_row = docker["rows"][index] if index < len(docker["rows"]) else {"stock_id": "", "weight": ""}
            row_match = local_row == docker_row
            lines.append(
                "| "
                f"{index + 1} | {local_row['stock_id']} | {local_row['weight']} | "
                f"{docker_row['stock_id']} | {docker_row['weight']} | {markdown_bool(row_match)} |"
            )
        lines.extend(
            [
                "",
                "## Checks",
                "",
                f"- stock_ids_match: `{checks['stock_ids_match']}`",
                f"- weights_match: `{checks['weights_match']}`",
                f"- weight_sums_match: `{checks['weight_sums_match']}`",
                f"- md5_match: `{checks['md5_match']}`",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    local_result_path = resolve_path(args.local_result_path)
    docker_result_path = resolve_path(args.docker_result_path)
    report_path = resolve_path(args.report_path)

    local = None
    docker = None
    checks = {"passed": False}
    error = None

    try:
        if not args.skip_local_run:
            run_local_submission(local_result_path, args.python_bin, args.timeout_seconds)
        if not args.skip_docker_run:
            run_docker_submission(
                docker_result_path=docker_result_path,
                image_name=args.image_name,
                build_docker_image=args.build_docker_image and not args.skip_docker_build,
                timeout_seconds=args.timeout_seconds,
            )

        local = read_result(local_result_path)
        docker = read_result(docker_result_path)
        checks = compare_results(local, docker)
        if not checks["passed"]:
            raise ValueError("Local and Docker result.csv files are not identical.")
    except Exception as exc:
        error = str(exc)
    finally:
        write_report(report_path, local, docker, checks, error)

    print(f"[compare_local_docker_result] report={report_path}")
    if error:
        print(f"[compare_local_docker_result][ERROR] {error}", file=sys.stderr)
        sys.exit(1)
    print("[compare_local_docker_result] local and Docker results match.")


if __name__ == "__main__":
    main()
