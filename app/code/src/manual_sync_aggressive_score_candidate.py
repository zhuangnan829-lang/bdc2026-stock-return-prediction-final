import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CANDIDATE_RESULT = (
    ROOT_DIR
    / "app/model/aggressive_score_submission_candidate/result_aggressive_score.csv"
)
DEFAULT_OUTPUT_RESULT = ROOT_DIR / "app/output/result.csv"
DEFAULT_REPORT_DIR = ROOT_DIR / "app/model/aggressive_score_submission_candidate"
BACKUP_ROOT = ROOT_DIR / "app/model/manual_switch_backups"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually sync aggressive score candidate to app/output/result.csv with backup and validation."
    )
    parser.add_argument("--candidate_result", default=str(DEFAULT_CANDIDATE_RESULT))
    parser.add_argument("--output_result", default=str(DEFAULT_OUTPUT_RESULT))
    parser.add_argument("--report_dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument(
        "--restore_backup",
        help="Restore app/output/result.csv from a backup directory created by this script, then exit.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def run_step(name: str, command: list[str], cwd: Path) -> dict:
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
        "ok": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def read_result(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"stock_id": str})


def render_result_table(df: pd.DataFrame) -> list[str]:
    lines = [
        "| stock_id | weight |",
        "|---|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(f"| `{str(row['stock_id']).zfill(6)}` | {float(row['weight']):.6f} |")
    return lines


def parse_score_summary(score_dir: Path) -> dict:
    path = score_dir / "latest_score_compare.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        metric = str(row["metric"])
        out[metric] = {
            "our_score": float(row["our_score"]),
            "case_score": float(row["case_score"]),
            "diff": float(row["diff_our_minus_case"]),
            "judgement": str(row["judgement"]),
        }
    return out


def restore_backup(backup_dir: Path, output_result: Path) -> None:
    backup_result = backup_dir / "app_output_result.csv"
    if not backup_result.exists():
        raise FileNotFoundError(f"Backup result not found: {backup_result}")
    output_result.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_result, output_result)
    print(f"[manual_sync_aggressive] restored {output_result} from {backup_result}")


def write_report(
    report_path: Path,
    backup_dir: Path,
    candidate_result: Path,
    output_result: Path,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    before_sha: str,
    after_sha: str,
    steps: list[dict],
    score_summary: dict,
) -> None:
    all_ok = all(step["ok"] for step in steps)
    lines = [
        "# Manual Aggressive Score Sync Report",
        "",
        "本报告记录一次人工确认版 aggressive score 临时同步。脚本已备份当前 HV rerank 默认结果，再把 aggressive score 候选写入 `app/output/result.csv`。",
        "",
        "## Decision",
        "",
        f"- sync status: `{'active' if all_ok else 'active_with_failed_checks'}`",
        "- default config sync: `not changed`",
        "- result sync: `app/output/result.csv has been replaced by aggressive score candidate`",
        "- rollback: copy the backed-up result file back to `app/output/result.csv`, or run the restore command below.",
        "",
        "## Files",
        "",
        f"- backup directory: `{backup_dir}`",
        f"- candidate result: `{candidate_result}`",
        f"- output result: `{output_result}`",
        f"- backup result sha256: `{before_sha}`",
        f"- current output sha256: `{after_sha}`",
        "",
        "## Rollback Command",
        "",
        "```powershell",
        f"python app/code/src/manual_sync_aggressive_score_candidate.py --restore_backup \"{backup_dir}\"",
        "```",
        "",
        "## Before Sync: HV Rerank Result",
        "",
    ]
    lines.extend(render_result_table(before_df))
    lines.extend(["", "## After Sync: Aggressive Score Result", ""])
    lines.extend(render_result_table(after_df))

    if score_summary:
        current = score_summary.get("current_output_score", {})
        best = score_summary.get("recorded_best_score", {})
        lines.extend(
            [
                "",
                "## Single-Slice Score Recheck",
                "",
                f"- aggressive output score: `{current.get('our_score', float('nan')):.6f}`",
                f"- case zip current score: `{current.get('case_score', float('nan')):.6f}`",
                f"- diff vs case current: `{current.get('diff', float('nan')):+.6f}`",
                f"- case zip best score: `{best.get('case_score', float('nan')):.6f}`",
                f"- diff vs case best: `{best.get('diff', float('nan')):+.6f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Validation Steps",
            "",
            "| step | status | returncode | command |",
            "|---|---|---:|---|",
        ]
    )
    for step in steps:
        status = "PASS" if step["ok"] else "FAIL"
        lines.append(f"| `{step['name']}` | `{status}` | {step['returncode']} | `{step['command']}` |")

    failed = [step for step in steps if not step["ok"]]
    if failed:
        lines.extend(["", "## Failed Step Output", ""])
        for step in failed:
            lines.extend(
                [
                    f"### {step['name']}",
                    "",
                    "stdout:",
                    "```text",
                    step["stdout"][-4000:],
                    "```",
                    "stderr:",
                    "```text",
                    step["stderr"][-4000:],
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "",
            "## Manual Decision Note",
            "",
            "- 若目标是比赛冲分，可以继续使用当前 `app/output/result.csv`。",
            "- 若目标是稳定策略，应执行上面的 rollback command 回到 HV rerank 默认结果。",
            "- 本脚本没有修改 `app/model/default_submission_config.json`、`best_config.json` 或 `model_meta.json`。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate_result = Path(args.candidate_result).resolve()
    output_result = Path(args.output_result).resolve()
    report_dir = Path(args.report_dir).resolve()

    if args.restore_backup:
        restore_backup(Path(args.restore_backup).resolve(), output_result)
        return

    if not candidate_result.exists():
        raise FileNotFoundError(f"Missing aggressive score candidate result: {candidate_result}")
    if not output_result.exists():
        raise FileNotFoundError(f"Missing current output result: {output_result}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"aggressive_score_manual_sync_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    before_df = read_result(output_result)
    before_sha = sha256_file(output_result)

    backup_targets = {
        "app_output_result.csv": output_result,
        "default_submission_config.json": ROOT_DIR / "app/model/default_submission_config.json",
        "best_config.json": ROOT_DIR / "app/model/best_config.json",
        "model_meta.json": ROOT_DIR / "app/model/model_meta.json",
        "final_submission_snapshot.md": ROOT_DIR / "app/model/final_submission_snapshot.md",
    }
    for backup_name, source in backup_targets.items():
        copy_if_exists(source, backup_dir / backup_name)

    shutil.copy2(candidate_result, output_result)
    after_df = read_result(output_result)
    after_sha = sha256_file(output_result)

    score_dir = report_dir / "case_score_recheck"
    score_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    steps = [
        run_step(
            "result_validator",
            [python, "app/code/src/result_validator.py", "--result_path", "app/output/result.csv"],
            ROOT_DIR,
        ),
        run_step(
            "pre_submit_check",
            [python, "app/code/src/pre_submit_check.py", "--root_dir", ".", "--result_path", "app/output/result.csv"],
            ROOT_DIR,
        ),
        run_step(
            "single_slice_score_recheck",
            [
                python,
                "app/code/src/compare_with_case_score.py",
                "--our_result_path",
                "app/output/result.csv",
                "--output_dir",
                str(score_dir),
            ],
            ROOT_DIR,
        ),
    ]
    score_summary = parse_score_summary(score_dir)

    manifest = {
        "timestamp": timestamp,
        "backup_dir": str(backup_dir),
        "candidate_result": str(candidate_result),
        "output_result": str(output_result),
        "backup_result_sha256": before_sha,
        "current_output_sha256": after_sha,
        "steps": steps,
        "score_summary": score_summary,
        "restore_command": f'python app/code/src/manual_sync_aggressive_score_candidate.py --restore_backup "{backup_dir}"',
    }
    (backup_dir / "manual_sync_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = report_dir / "manual_aggressive_score_sync_report.md"
    write_report(
        report_path=report_path,
        backup_dir=backup_dir,
        candidate_result=candidate_result,
        output_result=output_result,
        before_df=before_df,
        after_df=after_df,
        before_sha=before_sha,
        after_sha=after_sha,
        steps=steps,
        score_summary=score_summary,
    )

    print(f"[manual_sync_aggressive] backup_dir={backup_dir}")
    print(f"[manual_sync_aggressive] output_result={output_result}")
    print(f"[manual_sync_aggressive] report={report_path}")
    for step in steps:
        status = "PASS" if step["ok"] else "FAIL"
        print(f"[manual_sync_aggressive] {step['name']}={status}")

    if not all(step["ok"] for step in steps):
        sys.exit(1)


if __name__ == "__main__":
    main()
