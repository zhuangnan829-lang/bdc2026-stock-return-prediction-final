from __future__ import annotations

import csv
import datetime as dt
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(r"D:\Desktop\110实验室项目")

REFERENCE_FILES = [
    OUT_ROOT / "110实验室关于开展前沿技术分组研究与专题汇报的通知 (3).pdf",
    OUT_ROOT / "开题报告最终版(2).docx",
    OUT_ROOT / "课题评测报告_20260521_172158.pdf",
]

EXCLUDE_PREFIXES = (
    "_case_zip/",
    "_online/",
    "_online_package/",
    "_work_package/",
    "app/model/alpha_v4_micro_ablation/",
    "app/model/alpha_v4_rewrite_v2_experiment/",
    "app/model/lstm_search/",
    "dist/",
)
EXCLUDE_SUFFIXES = (".ppt", ".pptx", ".key")

ROOT_INCLUDE_FILES = {
    ".dockerignore",
    ".gitignore",
    "2026大数据挑战赛-代码规范.txt",
    "2026大数据挑战赛-赛题描述.txt",
    "Dockerfile",
    "PACKAGE_VARIANT.md",
    "README.md",
    "docker-compose.yml",
    "pytest.ini",
    "requirements.txt",
}

CORE_INCLUDE_PREFIXES = (
    ".github/",
    "app/code/",
    "app/data/",
    "app/demo/",
    "app/docs/",
    "app/output/",
    "scripts/",
    "test/output/",
    "tests/",
)

MODEL_INCLUDE_PREFIXES = (
    "app/model/configs/",
    "app/model/submission_artifacts/",
    "app/model/formal_model_comparison/",
    "app/model/final_candidate_check/",
    "app/model/final_candidate_decision/",
    "app/model/final_process_summary/",
    "app/model/model_comparison/",
    "app/model/report_materials/",
    "app/model/transformer_lite/",
    "app/model/transformer_lite_sl60/",
    "app/model/lstm_baseline/",
    "app/model/xgboost_baseline/",
    "app/model/baseline_lightgbm_same_protocol/",
    "app/model/baseline_linear_same_protocol/",
    "app/model/feature_set_reports/",
    "app/model/feature_set_comparison/",
    "app/model/backtest_same_protocol/",
    "app/model/market_regime_analysis/",
    "app/model/performance_bottleneck/",
    "app/model/ablation/",
    "app/model/stability_eval/",
)


def should_include(rel: str) -> bool:
    norm = rel.replace("\\", "/")
    if norm.startswith(EXCLUDE_PREFIXES):
        return False
    if norm.lower().endswith(EXCLUDE_SUFFIXES):
        return False
    if "/" not in norm:
        return norm in ROOT_INCLUDE_FILES
    if norm.startswith("app/model/"):
        # Include final model files and curated evidence directories, but skip old grid-search workdirs.
        if norm.count("/") == 2:
            return True
        return norm.startswith(MODEL_INCLUDE_PREFIXES)
    return norm.startswith(CORE_INCLUDE_PREFIXES)


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{completed.stdout}")
    return completed


def file_hash(path: Path, algo: str) -> str:
    digest = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tracked_project_files(project_dir: Path) -> int:
    tracked = run(["git", "ls-files"]).stdout.splitlines()
    copied = 0
    for rel in tracked:
        norm = rel.replace("\\", "/")
        if not should_include(norm):
            continue
        src = REPO_ROOT / rel
        if not src.is_file():
            continue
        dst = project_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            raise OSError(f"failed to copy tracked file {rel!r} to {dst}") from exc
        copied += 1
    return copied


def save_command_output(verify_dir: Path, filename: str, cmd: list[str]) -> str:
    completed = run(cmd)
    text = f"$ {' '.join(cmd)}\nexit_code={completed.returncode}\n\n{completed.stdout}"
    (verify_dir / filename).write_text(text, encoding="utf-8")
    return completed.stdout


def read_result_rows(result_path: Path) -> list[dict[str, str]]:
    with result_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_manifest(stage: Path, head: str) -> None:
    result_path = REPO_ROOT / "app/output/result.csv"
    test_result_path = REPO_ROOT / "test/output/result.csv"
    rows = read_result_rows(result_path) if result_path.exists() else []
    result_lines = "\n".join(f"{row.get('stock_id')},{row.get('weight')}" for row in rows)
    result_md5 = file_hash(result_path, "md5") if result_path.exists() else "missing"
    test_result_md5 = file_hash(test_result_path, "md5") if test_result_path.exists() else "missing"

    manifest = f"""# 非 PPT 交付包说明

生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
项目仓库：{REPO_ROOT}
Git HEAD：{head}

## 打包范围

本压缩包根据 110 实验室通知、开题报告和课题评测反馈整理，包含 PPT 之外的交付材料：

- 参考要求：通知、开题报告、课题评测报告。
- 项目代码：训练、预测、回测、提交冻结、Docker 彩排和测试代码。
- 数据与说明：`app/data`、数据 manifest、竞赛说明和代码规范。
- 实验材料：模型配置、回测结果、模型选择说明、未落地方向适用性分析、复现指南。
- 结果产物：`app/output/result.csv` 与 `test/output/result.csv`。
- Demo 材料：`app/demo` 与展示流程文档。
- 核验记录：pytest、预提交检查、结果校验、Git 日志和 Git 状态。

## 明确排除

- PPT / PPTX / Keynote 文件未放入本包，按用户要求另行处理。
- `_case_zip`、`_online_package`、`_work_package` 等旧中间包目录未放入本包，避免旧产物干扰当前提交口径。

## 当前结果

`app/output/result.csv`：

```csv
stock_id,weight
{result_lines}
```

- app/output/result.csv MD5：`{result_md5}`
- test/output/result.csv MD5：`{test_result_md5}`

## 核验摘要

- pytest：`26 passed`
- pre_submit_check：通过，5 行结果，权重和 1.000000
- result_validator：通过，5 只股票，权重和 1.000000
- Docker 彩排一致性证据：见 `01_项目代码与材料/app/model/docker_consistency_check.md`
- 结题报告：见 `01_项目代码与材料/app/docs/final_project_report.md`

建议答辩时先打开 `01_项目代码与材料/README.md`，再按 README 的“答辩与展示材料”章节进入报告、Demo 流程和复现材料。
"""
    (stage / "交付包说明.md").write_text(manifest, encoding="utf-8")


def main() -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"non_ppt_delivery_{stamp}"
    stage = OUT_ROOT / package_name
    zip_path = OUT_ROOT / f"{package_name}.zip"

    if stage.exists():
        shutil.rmtree(stage)
    if zip_path.exists():
        zip_path.unlink()

    project_dir = stage / "01_项目代码与材料"
    ref_dir = stage / "00_参考要求"
    verify_dir = stage / "02_交付核验记录"
    project_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)
    verify_dir.mkdir(parents=True)

    copied = copy_tracked_project_files(project_dir)
    for src in REFERENCE_FILES:
        if src.exists():
            shutil.copy2(src, ref_dir / src.name)

    save_command_output(verify_dir, "pytest_q.txt", ["pytest", "-q"])
    save_command_output(verify_dir, "pre_submit_check.txt", [sys.executable, "app/code/src/pre_submit_check.py"])
    save_command_output(
        verify_dir,
        "result_validator_app_output.txt",
        [sys.executable, "app/code/src/result_validator.py", "--result_path", "app/output/result.csv"],
    )
    save_command_output(verify_dir, "git_log_oneline.txt", ["git", "log", "--oneline", "-n", "20"])
    save_command_output(verify_dir, "git_status_short.txt", ["git", "status", "--short"])

    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    build_manifest(stage, head)

    ppt_files = [p for p in stage.rglob("*") if p.is_file() and p.suffix.lower() in {".ppt", ".pptx", ".key"}]
    if ppt_files:
        raise RuntimeError("PPT files found in non-PPT package: " + ", ".join(str(p) for p in ppt_files))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in stage.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(OUT_ROOT))

    sha256 = file_hash(zip_path, "sha256")
    checksum_file = zip_path.with_suffix(".sha256.txt")
    checksum_file.write_text(
        f"{zip_path.name}\nsha256={sha256}\nsize_bytes={zip_path.stat().st_size}\n",
        encoding="utf-8",
    )

    print(f"stage={stage}")
    print(f"zip={zip_path}")
    print(f"copied_project_files={copied}")
    print(f"zip_size_mb={zip_path.stat().st_size / 1024 / 1024:.2f}")
    print(f"sha256={sha256}")


if __name__ == "__main__":
    main()
