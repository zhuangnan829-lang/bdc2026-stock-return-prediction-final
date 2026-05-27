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


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "app" / "code" / "src"
MODEL_DIR = ROOT_DIR / "app" / "model"

DEFAULT_REPORT_PATH = MODEL_DIR / "full_optimization_validation_report.md"
DEFAULT_LOG_DIR = MODEL_DIR / "full_optimization_validation_logs"
DEFAULT_PRED_PATH = MODEL_DIR / "walk_forward_predictions.csv"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_RESULT_PATH = ROOT_DIR / "app" / "output" / "result.csv"


@dataclass
class PipelineStep:
    name: str
    script: str
    args: list[str]


@dataclass
class StepResult:
    name: str
    command: list[str]
    returncode: int
    elapsed_seconds: float
    stdout_path: Path
    stderr_path: Path
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def pipeline_steps(pred_path: Path, feature_path: Path, result_path: Path) -> list[PipelineStep]:
    return [
        PipelineStep("compare_config_consistency", "compare_config_consistency.py", []),
        PipelineStep(
            "analyze_performance_bottleneck",
            "analyze_performance_bottleneck.py",
            [
                "--pred_path",
                str(pred_path),
                "--data_path",
                str(feature_path),
                "--result_path",
                str(result_path),
            ],
        ),
        PipelineStep(
            "search_weight_cap",
            "search_weight_cap.py",
            ["--pred_path", str(pred_path), "--data_path", str(feature_path)],
        ),
        PipelineStep(
            "search_weight_blend",
            "search_weight_blend.py",
            ["--pred_path", str(pred_path), "--data_path", str(feature_path)],
        ),
        PipelineStep("turnover_stress_test", "turnover_stress_test.py", []),
        PipelineStep("evaluate_rank_stability", "evaluate_rank_stability.py", []),
        PipelineStep("diagnose_misranked_samples", "diagnose_misranked_samples.py", []),
        PipelineStep("evaluate_by_market_regime", "evaluate_by_market_regime.py", []),
        PipelineStep("regime_rerank_switch", "regime_rerank_switch.py", []),
        PipelineStep("rank_blend", "rank_blend.py", []),
        PipelineStep("build_experiment_leaderboard", "build_experiment_leaderboard.py", []),
        PipelineStep("select_final_submission_config", "select_final_submission_config.py", []),
        PipelineStep("result_validator", "result_validator.py", ["--result_path", str(result_path)]),
        PipelineStep("pre_submit_check", "pre_submit_check.py", ["--root_dir", str(ROOT_DIR)]),
    ]


def run_step(step: PipelineStep, log_dir: Path, timeout: int) -> StepResult:
    command = [sys.executable, str(SRC_DIR / step.script), *step.args]
    stdout_path = log_dir / f"{step.name}.stdout.log"
    stderr_path = log_dir / f"{step.name}.stderr.log"
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
        return StepResult(
            name=step.name,
            command=command,
            returncode=completed.returncode,
            elapsed_seconds=time.time() - start,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error="" if completed.returncode == 0 else last_nonempty_line(completed.stderr) or last_nonempty_line(completed.stdout),
        )
    except Exception as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(str(exc), encoding="utf-8")
        return StepResult(
            name=step.name,
            command=command,
            returncode=124 if isinstance(exc, subprocess.TimeoutExpired) else 1,
            elapsed_seconds=time.time() - start,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error=str(exc),
        )


def last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def num(row: pd.Series | dict[str, Any] | None, column: str) -> float | None:
    if row is None:
        return None
    value = row.get(column) if hasattr(row, "get") else None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 6) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        if isinstance(value, str):
            return value
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def first_row(df: pd.DataFrame, label_column: str | None = None, label_value: str | None = None) -> pd.Series | None:
    if df.empty:
        return None
    if label_column and label_value and label_column in df.columns:
        matched = df[df[label_column].astype(str) == label_value]
        if not matched.empty:
            return matched.iloc[0]
    return df.iloc[0]


def sort_best(df: pd.DataFrame, column: str, ascending: bool = False) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    ranked = df.copy()
    ranked["_sort_metric"] = pd.to_numeric(ranked[column], errors="coerce")
    ranked = ranked.dropna(subset=["_sort_metric"]).sort_values("_sort_metric", ascending=ascending)
    return ranked.drop(columns=["_sort_metric"])


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 5) -> list[str]:
    if df.empty:
        return ["无可用记录。"]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in df.head(max_rows).iterrows():
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    return lines


def result_validation_summary(result_path: Path) -> str:
    if not result_path.exists():
        return "result.csv 不存在。"
    df = read_csv(result_path)
    if df.empty:
        return "result.csv 为空或无法读取。"
    weight_sum = pd.to_numeric(df.get("weight"), errors="coerce").sum() if "weight" in df.columns else float("nan")
    return f"rows={len(df)}, weight_sum={fmt(weight_sum)}, stocks={', '.join(df.get('stock_id', pd.Series(dtype=str)).astype(str).tolist())}"


def build_final_judgement(leaderboard: pd.DataFrame, regime_rerank: pd.DataFrame | None = None) -> dict[str, str]:
    if leaderboard.empty:
        return {
            "replace_lstm_sl20": "否。排行榜不可用，按保守规则保留 LSTM sl20 主线。",
            "final_recommendation": "保留 LSTM sl20 主线；等待完整排行榜和验证结果后再考虑替换。",
            "risk": "缺少统一排行榜，无法证明新模型在 walk-forward、回测、单切片三方面同时胜出。",
        }

    sl20_mask = (
        leaderboard["candidate_label"].astype(str).str.contains("sl20|baseline_cross_section_rank|lstm", case=False, na=False)
        | leaderboard["source_file"].astype(str).str.contains("sl20|baseline_cross_section_rank|walk_forward", case=False, na=False)
    )
    sl20_rows = leaderboard[sl20_mask]
    sl20 = sl20_rows.iloc[0] if not sl20_rows.empty else leaderboard.iloc[0]
    eligible = leaderboard[leaderboard.get("decision", "") == "adopt"]
    best = eligible.iloc[0] if not eligible.empty else leaderboard.iloc[0]

    best_has_all = all(
        num(best, column) is not None
        for column in ["top5_mean_return", "cumulative_return_after_cost", "single_slice_score"]
    )
    better_than_sl20 = False
    if best_has_all:
        checks = []
        for column in ["top5_mean_return", "cumulative_return_after_cost", "single_slice_score"]:
            best_value = num(best, column)
            sl20_value = num(sl20, column)
            if best_value is not None and sl20_value is not None:
                checks.append(best_value > sl20_value)
        better_than_sl20 = bool(checks) and all(checks)

    robust_candidates = leaderboard[
        leaderboard["candidate_label"].astype(str).str.contains("blend|mt050|robust|rank_blend", case=False, na=False)
        & (leaderboard.get("decision", "") == "adopt")
    ].copy()
    robust_note = "未发现可明确提升风险侧指标的 robust/rank blend 候选。"
    if not robust_candidates.empty:
        labels = robust_candidates["candidate_label"].astype(str).str.lower()
        robust_candidates["_profile_priority"] = 3
        robust_candidates.loc[labels.str.contains("blend_0.5") & labels.str.contains("cap0.20"), "_profile_priority"] = 0
        robust_candidates.loc[labels.str.contains("blend_0.5"), "_profile_priority"] = robust_candidates.loc[
            labels.str.contains("blend_0.5"), "_profile_priority"
        ].clip(upper=1)
        robust_candidates.loc[labels.str.contains("cap0.20"), "_profile_priority"] = robust_candidates.loc[
            labels.str.contains("cap0.20"), "_profile_priority"
        ].clip(upper=2)
        candidate = robust_candidates.sort_values(
            ["_profile_priority", "robust_rank", "stable_alpha_score"],
            ascending=[True, True, False],
            na_position="last",
        ).iloc[0]
        robust_note = (
            f"`{candidate['candidate_label']}` 可进入候选观察："
            f"max_drawdown={fmt(candidate.get('max_drawdown_after_cost'))}, "
            f"avg_turnover={fmt(candidate.get('avg_turnover'))}, "
            f"return={fmt(candidate.get('cumulative_return_after_cost'))}。"
        )

    hv_note = ""
    if regime_rerank is not None and not regime_rerank.empty and "profile_name" in regime_rerank.columns:
        hv_rows = regime_rerank[regime_rerank["profile_name"].astype(str) == "hv_close_position_20d_m005"]
        if not hv_rows.empty:
            hv = hv_rows.iloc[0]
            hv_note = (
                f" `hv_close_position_20d_m005` 可作为高波动轻重排候选："
                f"delta_return={fmt(hv.get('delta_cost_after_return'))}, "
                f"delta_sharpe={fmt(hv.get('delta_sharpe'))}, "
                f"delta_max_drawdown={fmt(hv.get('delta_max_drawdown'))}, "
                f"delta_high_vol={fmt(hv.get('delta_high_volatility_selected_top5_return'))}。"
            )

    if better_than_sl20:
        replace = "是，但仅限完整复核后。当前 best candidate 在 walk-forward、回测、单切片三方面均优于 sl20。"
        final = f"可考虑以 `{best['candidate_label']}` 作为替换候选，同时保留 aggressive/robust 双配置复核。"
    else:
        replace = "否。未证明任何方案在 walk-forward、回测、单切片三方面同时优于 sl20。"
        final = "保留 LSTM sl20 主线；aggressive/robust 配置只作为最终提交目标不同的候选，不直接替换主线。"

    return {
        "replace_lstm_sl20": replace,
        "final_recommendation": final + " " + robust_note + hv_note,
        "risk": "主要风险仍是单票贡献集中、换手偏高、部分 TopK/新模型 worst fold 或 fold 内 Top5 最差收益不稳定。",
    }


def render_report(
    results: list[StepResult],
    report_path: Path,
    result_path: Path,
    log_dir: Path,
) -> str:
    default_config = read_json(MODEL_DIR / "default_submission_config.json")
    model_meta = read_json(MODEL_DIR / "model_meta.json")
    leaderboard = read_csv(MODEL_DIR / "experiment_leaderboard.csv")
    weight_cap = read_csv(MODEL_DIR / "weight_cap_search" / "weight_cap_summary.csv")
    weight_blend = read_csv(MODEL_DIR / "weight_blend_search" / "weight_blend_summary.csv")
    turnover = read_csv(MODEL_DIR / "turnover_stress_test" / "turnover_stress_summary.csv")
    stability = read_csv(MODEL_DIR / "stability_eval" / "stability_summary.csv")
    bottleneck = read_csv(MODEL_DIR / "performance_bottleneck" / "performance_bottleneck_summary.csv")
    regime = read_csv(MODEL_DIR / "market_regime_analysis" / "market_regime_performance.csv")
    regime_rerank = read_csv(MODEL_DIR / "regime_rerank_switch" / "regime_rerank_switch_summary.csv")
    rank_blend = read_csv(MODEL_DIR / "rank_blend" / "blend_summary.csv")

    failed = [result for result in results if not result.ok]
    judgement = build_final_judgement(leaderboard, regime_rerank)

    top_aggressive = sort_best(
        leaderboard[leaderboard.get("decision", "") == "adopt"] if not leaderboard.empty else leaderboard,
        "aggressive_rank",
        ascending=True,
    )
    top_robust = sort_best(
        leaderboard[leaderboard.get("decision", "") == "adopt"] if not leaderboard.empty else leaderboard,
        "robust_rank",
        ascending=True,
    )
    risky = leaderboard[leaderboard.get("decision", "") == "reject"] if not leaderboard.empty else pd.DataFrame()

    current_row = first_row(read_csv(MODEL_DIR / "default_profile_backtest" / "backtest_summary.csv"))
    walk_forward = model_meta.get("walk_forward_summary", {})

    lines = [
        "# Full Optimization Validation Report",
        "",
        f"Report path: `{report_path}`",
        f"Log directory: `{log_dir}`",
        "",
        "## Pipeline Status",
        "",
        "| step | status | seconds | error |",
        "|---|---|---:|---|",
    ]
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        lines.append(f"| {result.name} | {status} | {result.elapsed_seconds:.1f} | {result.error or ''} |")

    lines.extend(["", "## Failed Steps", ""])
    if failed:
        for result in failed:
            lines.append(f"- `{result.name}` failed with returncode={result.returncode}. See `{result.stderr_path}`.")
    else:
        lines.append("- 无失败步骤。")

    lines.extend(
        [
            "",
            "## 1. 当前主线表现",
            "",
            f"- profile: `{default_config.get('profile_name', 'unknown')}`",
            f"- feature_set: `{default_config.get('feature_set', 'unknown')}`",
            f"- model_family: `{default_config.get('model_family', 'unknown')}`",
            f"- result validation: {result_validation_summary(result_path)}",
            f"- walk-forward rank_ic_mean: `{fmt(walk_forward.get('rank_ic_mean'))}`",
            f"- walk-forward top5_mean_return_mean: `{fmt(walk_forward.get('top5_mean_return_mean'))}`",
            f"- backtest return: `{fmt(num(current_row, 'cumulative_return_after_cost'))}`",
            f"- backtest sharpe: `{fmt(num(current_row, 'sharpe_after_cost'))}`",
            f"- backtest max_drawdown: `{fmt(num(current_row, 'max_drawdown_after_cost'))}`",
            f"- backtest avg_turnover: `{fmt(num(current_row, 'avg_turnover'))}`",
            "",
            "## 2. 收益瓶颈判断",
            "",
        ]
    )
    lines.extend(md_table(bottleneck, list(bottleneck.columns[:8]) if not bottleneck.empty else [], 5) if not bottleneck.empty else ["性能瓶颈 summary 未生成或不可读。"])

    lines.extend(["", "## 3. 组合层优化结果", ""])
    if not weight_cap.empty:
        lines.append("Weight cap top rows:")
        lines.extend(md_table(sort_best(weight_cap, "cost_after_return"), ["cap", "single_slice_score", "cost_after_return", "sharpe", "max_drawdown", "avg_turnover"], 6))
    if not weight_blend.empty:
        lines.append("")
        lines.append("Weight blend top rows:")
        lines.extend(md_table(sort_best(weight_blend, "cost_after_return"), ["alpha", "max_single_weight", "single_slice_score", "cost_after_return", "sharpe", "max_drawdown", "avg_turnover"], 6))
    if not turnover.empty:
        lines.append("")
        lines.append("Turnover stress robust rows:")
        lines.extend(md_table(sort_best(turnover, "robust_score"), ["profile_name", "cost_after_return", "sharpe", "max_drawdown", "avg_turnover", "robust_score"], 6))

    lines.extend(["", "## 4. 稳定性优化结果", ""])
    lines.extend(md_table(sort_best(stability, "stability_score"), ["model", "feature_set", "sequence_length", "rank_ic_mean", "worst_fold_rank_ic", "top5_return_mean", "stability_score"], 8) if not stability.empty else ["稳定性 summary 未生成或不可读。"])

    lines.extend(["", "## 5. 标签/特征/融合实验结果", ""])
    if not rank_blend.empty:
        lines.append("Rank blend:")
        lines.extend(md_table(rank_blend, list(rank_blend.columns[:8]), 8))
    if not regime_rerank.empty:
        lines.append("")
        lines.append("Regime rerank:")
        lines.extend(
            md_table(
                regime_rerank,
                [
                    "profile_name",
                    "cost_after_return",
                    "sharpe",
                    "max_drawdown",
                    "avg_turnover",
                    "delta_cost_after_return",
                    "delta_sharpe",
                    "delta_high_volatility_selected_top5_return",
                ],
                8,
            )
        )
    if not regime.empty:
        lines.append("")
        lines.append("Market regime:")
        lines.extend(md_table(regime, list(regime.columns[:8]), 8))
    if not leaderboard.empty:
        lines.append("")
        lines.append("Leaderboard top rows:")
        lines.extend(md_table(leaderboard, ["rank", "candidate_label", "stable_alpha_score", "decision", "top5_mean_return", "cumulative_return_after_cost", "sharpe_after_cost", "risk_flags"], 8))

    lines.extend(["", "## 6. aggressive 候选", ""])
    lines.extend(md_table(top_aggressive, ["aggressive_rank", "candidate_label", "stable_alpha_score", "decision", "top5_mean_return", "cumulative_return_after_cost", "sharpe_after_cost", "risk_flags"], 10))

    lines.extend(["", "## 7. robust 候选", ""])
    lines.extend(md_table(top_robust, ["robust_rank", "candidate_label", "stable_alpha_score", "decision", "max_drawdown_after_cost", "avg_turnover", "cumulative_return_after_cost", "risk_flags"], 10))

    lines.extend(
        [
            "",
            "## 8. 是否建议替换 LSTM sl20",
            "",
            judgement["replace_lstm_sl20"],
            "",
            "## 9. 最终推荐",
            "",
            judgement["final_recommendation"],
            "",
            "## 10. 仍然存在的风险",
            "",
            f"- {judgement['risk']}",
        ]
    )
    if not risky.empty:
        lines.append("- 高风险候选如下：")
        lines.extend(md_table(risky, ["candidate_label", "stable_alpha_score", "risk_flags"], 10))
    lines.append("- 若 aggressive 与 robust 结论冲突，应保留双配置，并在最终提交前按比赛冲分或稳定策略目标手动选择。")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full optimization validation pipeline.")
    parser.add_argument("--pred_path", default=str(DEFAULT_PRED_PATH))
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--result_path", default=str(DEFAULT_RESULT_PATH))
    parser.add_argument("--report_path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--log_dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--timeout_seconds", type=int, default=900)
    parser.add_argument("--skip_steps", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = resolve_path(args.pred_path)
    feature_path = resolve_path(args.feature_path)
    result_path = resolve_path(args.result_path)
    report_path = resolve_path(args.report_path)
    log_dir = resolve_path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[StepResult] = []
    skip_steps = set(args.skip_steps)
    for step in pipeline_steps(pred_path, feature_path, result_path):
        if step.name in skip_steps:
            stdout_path = log_dir / f"{step.name}.stdout.log"
            stderr_path = log_dir / f"{step.name}.stderr.log"
            stdout_path.write_text("skipped\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            results.append(
                StepResult(
                    name=step.name,
                    command=[],
                    returncode=0,
                    elapsed_seconds=0.0,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    error="skipped",
                )
            )
            continue
        print(f"[full_validation] running {step.name} ...")
        result = run_step(step, log_dir=log_dir, timeout=args.timeout_seconds)
        print(
            f"[full_validation] {step.name} "
            f"{'PASS' if result.ok else 'FAIL'} in {result.elapsed_seconds:.1f}s"
        )
        results.append(result)

    report = render_report(results, report_path=report_path, result_path=result_path, log_dir=log_dir)
    report_path.write_text(report, encoding="utf-8")
    failed_count = sum(not result.ok for result in results)
    print(f"[full_validation] wrote {report_path}")
    print(f"[full_validation] failed_steps={failed_count}")


if __name__ == "__main__":
    main()
