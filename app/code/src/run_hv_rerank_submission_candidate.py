from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from load_submission_config import build_default_inference_args, load_submission_config


ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = ROOT_DIR / "app"
SRC_DIR = APP_DIR / "code" / "src"
MODEL_DIR = APP_DIR / "model"
TEMP_DIR = APP_DIR / "temp"
DATA_DIR = APP_DIR / "data"
OUTPUT_DIR = APP_DIR / "output"

DEFAULT_CONFIG_PATH = MODEL_DIR / "configs" / "submission_hv_rerank_candidate.json"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "hv_rerank_submission_candidate"
DEFAULT_CURRENT_RESULT = OUTPUT_DIR / "result.csv"


@dataclass
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    elapsed_seconds: float
    stdout_path: Path
    stderr_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def run_command(name: str, command: list[str], log_dir: Path, timeout: int) -> CommandResult:
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    start = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        return CommandResult(
            name=name,
            command=command,
            returncode=completed.returncode,
            elapsed_seconds=time.time() - start,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    except Exception as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(str(exc), encoding="utf-8")
        return CommandResult(
            name=name,
            command=command,
            returncode=124 if isinstance(exc, subprocess.TimeoutExpired) else 1,
            elapsed_seconds=time.time() - start,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )


def read_result(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["stock_id", "weight"])
    df = pd.read_csv(path, dtype={"stock_id": str})
    df["stock_id"] = df["stock_id"].astype(str).str.zfill(6)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    return df[["stock_id", "weight"]]


def compare_results(current_path: Path, candidate_path: Path, output_path: Path) -> pd.DataFrame:
    current = read_result(current_path).rename(columns={"weight": "current_weight"})
    candidate = read_result(candidate_path).rename(columns={"weight": "candidate_weight"})
    merged = current.merge(candidate, on="stock_id", how="outer")
    merged["current_weight"] = merged["current_weight"].fillna(0.0)
    merged["candidate_weight"] = merged["candidate_weight"].fillna(0.0)
    merged["weight_delta"] = merged["candidate_weight"] - merged["current_weight"]
    merged["abs_weight_delta"] = merged["weight_delta"].abs()
    merged["action"] = "kept"
    merged.loc[(merged["current_weight"] <= 0) & (merged["candidate_weight"] > 0), "action"] = "added"
    merged.loc[(merged["current_weight"] > 0) & (merged["candidate_weight"] <= 0), "action"] = "dropped"
    merged = merged.sort_values(["action", "abs_weight_delta", "stock_id"], ascending=[True, False, True])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return merged


def fmt(value: Any, digits: int = 6) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        if isinstance(value, str):
            return value
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def command_error(result: CommandResult) -> str:
    if result.ok:
        return ""
    for path in (result.stderr_path, result.stdout_path):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in reversed(text.splitlines()):
            if line.strip():
                return line.strip()
    return f"returncode={result.returncode}"


def render_table(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> list[str]:
    if df.empty:
        return ["无记录。"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.head(max_rows).iterrows():
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    return lines


def build_report(
    *,
    config: dict[str, Any],
    defaults: dict[str, Any],
    command_results: list[CommandResult],
    current_result_path: Path,
    candidate_result_path: Path,
    comparison_path: Path,
    comparison: pd.DataFrame,
    report_path: Path,
) -> str:
    current = read_result(current_result_path)
    candidate = read_result(candidate_result_path)
    shared = set(current["stock_id"]).intersection(set(candidate["stock_id"]))
    added = set(candidate["stock_id"]) - set(current["stock_id"])
    dropped = set(current["stock_id"]) - set(candidate["stock_id"])
    switch_turnover = float(comparison["abs_weight_delta"].sum()) if not comparison.empty else 0.0

    lines = [
        "# HV Rerank Submission Candidate Report",
        "",
        "本报告只生成候选建议，不覆盖 `app/output/result.csv`，也不覆盖 `app/model/default_submission_config.json`。",
        "",
        "## 候选配置",
        "",
        f"- config: `{DEFAULT_CONFIG_PATH}`",
        f"- profile: `{config.get('profile_name')}`",
        f"- result: `{candidate_result_path}`",
        f"- comparison: `{comparison_path}`",
        f"- top_k: `{defaults['top_k']}`",
        f"- candidate_size: `{defaults['primary_candidate_size']}`",
        f"- weighting_scheme: `{defaults['weighting_scheme']}`",
        f"- max_single_weight: `{defaults['max_single_weight']}`",
        f"- regime_rerank: enabled=`{defaults['regime_rerank_enabled']}`, flag=`{defaults['regime_rerank_flag']}`, signal=`{defaults['regime_rerank_signal']}`, weight=`{defaults['regime_rerank_weight']}`",
        "",
        "## 执行状态",
        "",
        "| step | status | seconds | error |",
        "|---|---|---:|---|",
    ]
    for result in command_results:
        lines.append(
            f"| {result.name} | {'PASS' if result.ok else 'FAIL'} | "
            f"{result.elapsed_seconds:.1f} | {command_error(result)} |"
        )

    lines.extend(
        [
            "",
        "## 与当前正式结果对比",
            "",
            f"- 当前正式结果: `{current_result_path}`",
            f"- 当前行数: `{len(current)}`, 当前权重和: `{fmt(current['weight'].sum() if not current.empty else 0.0)}`",
            f"- 候选行数: `{len(candidate)}`, 候选权重和: `{fmt(candidate['weight'].sum() if not candidate.empty else 0.0)}`",
            f"- 共同股票数: `{len(shared)}`",
            f"- 新增股票: `{', '.join(sorted(added)) if added else 'none'}`",
            f"- 移除股票: `{', '.join(sorted(dropped)) if dropped else 'none'}`",
            f"- 相对当前结果的切换换手: `{fmt(switch_turnover)}`",
            "",
        ]
    )
    lines.extend(render_table(comparison, ["stock_id", "current_weight", "candidate_weight", "weight_delta", "action"], 12))

    lines.extend(
        [
            "",
            "## 判断",
            "",
            "- 手动确认前，继续把当前 LSTM sl20 默认结果作为权威提交结果。",
            "- 这份 HV rerank 结果可以作为独立候选进入人工复核。",
            "- 只有当换入/换出的股票可接受，并且与完整验证结论一致时，才考虑升级为正式默认配置。",
            "",
        ]
    )
    return "\n".join(lines)


def build_test_lstm_command(config_path: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    config = load_submission_config(config_path)
    defaults = build_default_inference_args(config)
    cmd = [
        sys.executable,
        str(SRC_DIR / "test_lstm.py"),
        "--feature_path",
        str(TEMP_DIR / "predict_features.csv"),
        "--model_dir",
        str(MODEL_DIR),
        "--output_path",
        str(output_dir / "result_hv_rerank.csv"),
        "--history_feature_path",
        str(TEMP_DIR / "train_features.csv"),
        "--score_output_path",
        str(output_dir / "predict_scores_hv_rerank.csv"),
        "--debug_candidates_path",
        str(output_dir / "debug_candidates_hv_rerank.csv"),
        "--top_k",
        str(defaults["top_k"]),
        "--primary_candidate_size",
        str(defaults["primary_candidate_size"]),
        "--max_volatility_20d_pct",
        str(defaults["max_volatility_20d_pct"]),
        "--max_volatility_5d_pct",
        str(defaults["max_volatility_5d_pct"]),
        "--turnover_rate_lower_pct",
        str(defaults["turnover_rate_lower_pct"]),
        "--turnover_rate_upper_pct",
        str(defaults["turnover_rate_upper_pct"]),
        "--turnover_ratio_upper_pct",
        str(defaults["turnover_ratio_upper_pct"]),
        "--risk_penalty_weight",
        str(defaults["risk_penalty_weight"]),
        "--sort_strategy",
        str(defaults["sort_strategy"]),
        "--weighting_scheme",
        str(defaults["weighting_scheme"]),
        "--weight_blend_alpha",
        str(defaults["weight_blend_alpha"]),
        "--max_single_weight",
        str(defaults["max_single_weight"]),
        "--max_turnover",
        str(defaults["max_turnover"]),
    ]
    if defaults.get("rerank_signal_column"):
        cmd.extend(["--rerank_signal_column", str(defaults["rerank_signal_column"])])
        cmd.extend(["--rerank_signal_weight", str(defaults["rerank_signal_weight"])])
    if defaults.get("regime_rerank_enabled"):
        cmd.extend(["--regime_rerank_enabled"])
        cmd.extend(["--regime_rerank_flag", str(defaults["regime_rerank_flag"])])
        cmd.extend(["--regime_rerank_signal", str(defaults["regime_rerank_signal"])])
        cmd.extend(["--regime_rerank_weight", str(defaults["regime_rerank_weight"])])
    return config, defaults, cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and validate the HV rerank submission candidate.")
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--current_result_path", default=str(DEFAULT_CURRENT_RESULT))
    parser.add_argument("--timeout_seconds", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config_path)
    output_dir = resolve_path(args.output_dir)
    current_result_path = resolve_path(args.current_result_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    command_results: list[CommandResult] = []
    command_results.append(
        run_command(
            "featurework_predict",
            [
                sys.executable,
                str(SRC_DIR / "featurework.py"),
                "--mode",
                "predict",
                "--data_dir",
                str(DATA_DIR),
                "--temp_dir",
                str(TEMP_DIR),
            ],
            log_dir,
            args.timeout_seconds,
        )
    )
    if not (TEMP_DIR / "train_features.csv").exists():
        command_results.append(
            run_command(
                "featurework_train",
                [
                    sys.executable,
                    str(SRC_DIR / "featurework.py"),
                    "--mode",
                    "train",
                    "--data_dir",
                    str(DATA_DIR),
                    "--temp_dir",
                    str(TEMP_DIR),
                ],
                log_dir,
                args.timeout_seconds,
            )
        )

    config, defaults, test_cmd = build_test_lstm_command(config_path, output_dir)
    if all(result.ok for result in command_results):
        command_results.append(run_command("test_lstm_hv_rerank", test_cmd, log_dir, args.timeout_seconds))

    candidate_result_path = output_dir / "result_hv_rerank.csv"
    if candidate_result_path.exists():
        command_results.append(
            run_command(
                "result_validator_hv_rerank",
                [sys.executable, str(SRC_DIR / "result_validator.py"), "--result_path", str(candidate_result_path)],
                log_dir,
                args.timeout_seconds,
            )
        )
        command_results.append(
            run_command(
                "pre_submit_check_hv_rerank_result",
                [
                    sys.executable,
                    str(SRC_DIR / "pre_submit_check.py"),
                    "--root_dir",
                    str(ROOT_DIR),
                    "--result_path",
                    str(candidate_result_path.relative_to(ROOT_DIR)),
                ],
                log_dir,
                args.timeout_seconds,
            )
        )

    comparison_path = output_dir / "result_hv_rerank_vs_current.csv"
    comparison = compare_results(current_result_path, candidate_result_path, comparison_path)
    report_path = output_dir / "hv_rerank_submission_candidate_report.md"
    report = build_report(
        config=config,
        defaults=defaults,
        command_results=command_results,
        current_result_path=current_result_path,
        candidate_result_path=candidate_result_path,
        comparison_path=comparison_path,
        comparison=comparison,
        report_path=report_path,
    )
    report_path.write_text(report, encoding="utf-8")

    failed = [result for result in command_results if not result.ok]
    print(f"[hv_rerank_candidate] wrote {candidate_result_path}")
    print(f"[hv_rerank_candidate] wrote {comparison_path}")
    print(f"[hv_rerank_candidate] wrote {report_path}")
    print(f"[hv_rerank_candidate] failed_steps={len(failed)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
