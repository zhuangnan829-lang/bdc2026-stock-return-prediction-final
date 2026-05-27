from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE_DIR = ROOT_DIR / "app/model/final_submission_package/THU-BDC2026-hv-rerank-final_20260525_233344"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "app/model/dual_submission_packages"
DEFAULT_STABLE_RESULT = ROOT_DIR / "app/model/hv_rerank_submission_candidate/result_hv_rerank.csv"
DEFAULT_AGGRESSIVE_RESULT = ROOT_DIR / "app/model/aggressive_score_submission_candidate/result_aggressive_score.csv"
DEFAULT_CASE_ROOT = (
    ROOT_DIR
    / "app/model/external_case_zip/from_desktop_20260526/THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0"
)


VARIANTS = {
    "stable_engineering_submission": {
        "result_arg": "stable_result",
        "zip_slug": "stable-engineering",
        "decision": "Use this when the target is stable engineering submission.",
        "description": "HV rerank / sl20 stable engineering version.",
    },
    "aggressive_score_submission": {
        "result_arg": "aggressive_result",
        "zip_slug": "aggressive-score",
        "decision": "Use this when the target is visible single-slice score chasing.",
        "description": "Aggressive score version using full-weight single-slice candidate.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stable and aggressive final submission packages.")
    parser.add_argument("--template_dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--stable_result", default=str(DEFAULT_STABLE_RESULT))
    parser.add_argument("--aggressive_result", default=str(DEFAULT_AGGRESSIVE_RESULT))
    parser.add_argument("--case_root", default=str(DEFAULT_CASE_ROOT))
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_result(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str})
    df["stock_id"] = df["stock_id"].astype(str).str.zfill(6)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    return df


def run_step(name: str, command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "name": name,
        "command": " ".join(command),
        "returncode": int(completed.returncode),
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "stdout_tail": completed.stdout[-3000:],
        "stderr_tail": completed.stderr[-3000:],
    }


def write_variant_note(package_dir: Path, variant: str, result_path: Path, case_score: float | None) -> None:
    spec = VARIANTS[variant]
    result = read_result(result_path)
    manifest = {
        "variant": variant,
        "description": spec["description"],
        "decision": spec["decision"],
        "result_source": str(result_path),
        "result_sha256": sha256_file(result_path),
        "stocks": result["stock_id"].tolist(),
        "weights": [float(x) for x in result["weight"].tolist()],
        "weight_sum": float(result["weight"].sum()),
        "case_slice_score": case_score,
        "default_config_changed": False,
    }
    note_lines = [
        f"# {variant}",
        "",
        spec["description"],
        "",
        f"- decision: {spec['decision']}",
        f"- result source: `{result_path}`",
        f"- result sha256: `{manifest['result_sha256']}`",
        f"- weight sum: `{manifest['weight_sum']:.6f}`",
    ]
    if case_score is not None:
        note_lines.append(f"- visible case-slice score: `{case_score:.6f}`")
    note_lines += [
        "",
        "| stock_id | weight |",
        "|---|---:|",
    ]
    for _, row in result.iterrows():
        note_lines.append(f"| `{row['stock_id']}` | {float(row['weight']):.6f} |")
    note_lines += [
        "",
        "This package does not change default_submission_config.json; the package variant is controlled by app/output/result.csv.",
    ]
    (package_dir / "PACKAGE_VARIANT.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    variant_json = package_dir / "app/model/package_variant.json"
    variant_json.parent.mkdir(parents=True, exist_ok=True)
    variant_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score_result_with_case(result_path: Path, case_root: Path, score_dir: Path) -> tuple[float | None, list[dict[str, Any]]]:
    score_dir.mkdir(parents=True, exist_ok=True)
    step = run_step(
        "single_slice_score_recheck",
        [
            sys.executable,
            "app/code/src/compare_with_case_score.py",
            "--our_result_path",
            str(result_path),
            "--case_result_path",
            str(case_root / "output/result.csv"),
            "--case_test_path",
            str(case_root / "data/test.csv"),
            "--case_best_score_path",
            str(case_root / "model/60_158+39/final_score.txt"),
            "--output_dir",
            str(score_dir),
        ],
        ROOT_DIR,
    )
    score = None
    summary_path = score_dir / "latest_score_compare.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        row = summary[summary["metric"].eq("current_output_score")]
        if not row.empty:
            score = float(row.iloc[0]["our_score"])
    return score, [step]


def build_one_variant(
    *,
    variant: str,
    template_dir: Path,
    result_path: Path,
    output_root: Path,
    case_root: Path,
    timestamp: str,
) -> dict[str, Any]:
    spec = VARIANTS[variant]
    package_name = f"THU-BDC2026-{spec['zip_slug']}_{timestamp}"
    package_dir = output_root / package_name
    if package_dir.exists():
        shutil.rmtree(package_dir)
    shutil.copytree(template_dir, package_dir)

    target_result = package_dir / "app/output/result.csv"
    target_result.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result_path, target_result)

    score, score_steps = score_result_with_case(target_result, case_root, output_root / f"{variant}_score_recheck")
    write_variant_note(package_dir, variant, result_path, score)

    steps = []
    steps.append(
        run_step(
            "result_validator",
            [sys.executable, "app/code/src/result_validator.py", "--result_path", str(target_result)],
            ROOT_DIR,
        )
    )
    steps.append(
        run_step(
            "pre_submit_check",
            [sys.executable, "app/code/src/pre_submit_check.py", "--root_dir", str(package_dir), "--result_path", "app/output/result.csv"],
            ROOT_DIR,
        )
    )
    steps.extend(score_steps)

    zip_base = output_root / package_name
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=package_dir))
    return {
        "variant": variant,
        "package_name": package_name,
        "package_dir": str(package_dir),
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "zip_size": int(zip_path.stat().st_size),
        "result_path": str(result_path),
        "package_result_path": str(target_result),
        "case_slice_score": score,
        "steps": steps,
        "all_checks_passed": all(step["status"] == "PASS" for step in steps),
    }


def write_report(output_root: Path, rows: list[dict[str, Any]]) -> None:
    summary_rows = []
    for row in rows:
        summary_rows.append(
            {
                "variant": row["variant"],
                "package_name": row["package_name"],
                "zip_path": row["zip_path"],
                "zip_size": row["zip_size"],
                "zip_sha256": row["zip_sha256"],
                "case_slice_score": row["case_slice_score"],
                "all_checks_passed": row["all_checks_passed"],
                "result_path": row["result_path"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "dual_submission_package_summary.csv", index=False, encoding="utf-8-sig")

    step_rows = []
    for row in rows:
        for step in row["steps"]:
            step_rows.append({"variant": row["variant"], **step})
    pd.DataFrame(step_rows).to_csv(output_root / "dual_submission_package_validation_steps.csv", index=False, encoding="utf-8-sig")

    stable = summary[summary["variant"].eq("stable_engineering_submission")].iloc[0]
    aggressive = summary[summary["variant"].eq("aggressive_score_submission")].iloc[0]
    lines = [
        "# Dual Submission Package Report",
        "",
        "本报告生成两个最终提交包：稳定工程版和单切片冲分版。两个包互相独立，均已写入 `PACKAGE_VARIANT.md`。",
        "",
        "## Recommendation",
        "",
        "- 稳定策略/工程可靠性优先：提交 `stable_engineering_submission`。",
        "- 比赛冲分/可见单切片优先：提交 `aggressive_score_submission`。",
        "- 当前单切片分数最高的是 aggressive 包；稳定工程包保留 HV rerank/sl20 主线。",
        "",
        "## Packages",
        "",
        "| variant | case_score | checks | zip | sha256 |",
        "|---|---:|---|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['variant']}` | {float(row['case_slice_score']):.6f} | "
            f"`{'PASS' if row['all_checks_passed'] else 'FAIL'}` | `{row['zip_path']}` | `{row['zip_sha256']}` |"
        )
    lines += [
        "",
        "## Stable Engineering Submission",
        "",
        f"- package: `{stable['zip_path']}`",
        "- uses HV rerank/sl20 result: `300316,600115,600183,600584,688396`",
        "- purpose: lower regret, preserve validated engineering default.",
        "",
        "## Aggressive Score Submission",
        "",
        f"- package: `{aggressive['zip_path']}`",
        "- uses aggressive score result: `000792,600233,601669,600930,002463`",
        "- purpose: maximize visible single-slice score.",
        "",
        "## Important",
        "",
        "- `app/output/result.csv` in the working tree currently remains the aggressive score version.",
        "- The stable package has its own HV rerank `app/output/result.csv` inside the zip.",
        "- No default config was synchronized during this packaging step.",
    ]
    (output_root / "dual_submission_package_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    template_dir = resolve(args.template_dir)
    output_root = resolve(args.output_root)
    stable_result = resolve(args.stable_result)
    aggressive_result = resolve(args.aggressive_result)
    case_root = resolve(args.case_root)

    for path in [template_dir, stable_result, aggressive_result, case_root / "data/test.csv"]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, spec in VARIANTS.items():
        result = stable_result if spec["result_arg"] == "stable_result" else aggressive_result
        rows.append(
            build_one_variant(
                variant=variant,
                template_dir=template_dir,
                result_path=result,
                output_root=output_root,
                case_root=case_root,
                timestamp=args.timestamp,
            )
        )
    write_report(output_root, rows)
    for row in rows:
        print(
            f"[dual_submission] {row['variant']} checks={'PASS' if row['all_checks_passed'] else 'FAIL'} "
            f"score={row['case_slice_score']:.6f} zip={row['zip_path']}"
        )
    print(f"[dual_submission] report={output_root / 'dual_submission_package_report.md'}")


if __name__ == "__main__":
    main()
