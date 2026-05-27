from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from result_validator import validate_result_file


ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = ROOT_DIR / "app"
MODEL_DIR = APP_DIR / "model"
CONFIG_DIR = MODEL_DIR / "configs"
OUTPUT_DIR = MODEL_DIR / "final_candidate_decision"

DEFAULT_PATHS = {
    "default_config": MODEL_DIR / "default_submission_config.json",
    "aggressive_config": CONFIG_DIR / "submission_aggressive.json",
    "robust_config": CONFIG_DIR / "submission_robust.json",
    "hv_config": CONFIG_DIR / "submission_hv_rerank_candidate.json",
    "current_result": APP_DIR / "output" / "result.csv",
    "hv_result": MODEL_DIR / "hv_rerank_submission_candidate" / "result_hv_rerank.csv",
    "hv_compare": MODEL_DIR / "hv_rerank_submission_candidate" / "result_hv_rerank_vs_current.csv",
    "full_report": MODEL_DIR / "full_optimization_validation_report.md",
    "final_selection_report": CONFIG_DIR / "final_config_selection_report.md",
    "regime_rerank_summary": MODEL_DIR / "regime_rerank_switch" / "regime_rerank_switch_summary.csv",
    "turnover_summary": MODEL_DIR / "turnover_stress_test" / "turnover_stress_summary.csv",
    "output_csv": OUTPUT_DIR / "final_candidate_decision_table.csv",
    "output_md": OUTPUT_DIR / "final_candidate_decision_report.md",
}


@dataclass
class CandidateDecision:
    candidate: str
    profile_name: str
    config_path: Path
    result_path: Path | None
    role: str
    submit_decision: str
    reason: str
    cannot_submit_reason: str
    evidence: str
    sync_default_config: str
    manual_action: str


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def fmt(value: Any, digits: int = 6) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        if isinstance(value, str):
            return value
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def profile_summary(config: dict[str, Any]) -> str:
    if not config:
        return "n/a"
    selection = config.get("selection_logic", {})
    execution = config.get("execution_logic", {})
    rerank = config.get("regime_rerank", {})
    parts = [
        f"sl={config.get('validation_scheme', {}).get('sequence_length')}",
        f"cs={selection.get('primary_candidate_size')}",
        f"sort={selection.get('sort_strategy')}",
        f"weight={selection.get('weighting_scheme')}",
        f"cap={selection.get('max_single_weight')}",
        f"mt={execution.get('max_turnover')}",
    ]
    if selection.get("weight_blend_alpha") is not None:
        parts.append(f"blend_alpha={selection.get('weight_blend_alpha')}")
    if rerank.get("enabled"):
        parts.append(
            f"rerank={rerank.get('regime_flag')}/{rerank.get('signal')}:{rerank.get('weight')}"
        )
    return ", ".join(parts)


def validate_status(path: Path | None) -> str:
    if path is None:
        return "无独立提交文件"
    if not path.exists():
        return "提交文件不存在"
    try:
        summary = validate_result_file(path)
    except Exception as exc:
        return f"未通过 validator: {exc}"
    return f"validator 通过, rows={summary['rows']}, weight_sum={summary['weight_sum']:.6f}"


def row_by(df: pd.DataFrame, column: str, value: str) -> pd.Series | None:
    if df.empty or column not in df.columns:
        return None
    matched = df[df[column].astype(str) == value]
    return None if matched.empty else matched.iloc[0]


def result_delta_summary(compare_path: Path) -> str:
    df = read_csv(compare_path)
    if df.empty:
        return "未找到 HV 与当前结果对比。"
    added = df[df.get("action", "") == "added"]["stock_id"].astype(str).tolist()
    dropped = df[df.get("action", "") == "dropped"]["stock_id"].astype(str).tolist()
    switch_turnover = pd.to_numeric(df.get("abs_weight_delta"), errors="coerce").fillna(0.0).sum()
    return (
        f"相对当前结果保留 {int((df.get('action', '') == 'kept').sum())} 只，"
        f"新增 {','.join(added) if added else 'none'}，"
        f"移除 {','.join(dropped) if dropped else 'none'}，"
        f"切换换手 {switch_turnover:.6f}。"
    )


def build_decisions(paths: dict[str, Path]) -> list[CandidateDecision]:
    default_config = read_json(paths["default_config"])
    aggressive_config = read_json(paths["aggressive_config"])
    robust_config = read_json(paths["robust_config"])
    hv_config = read_json(paths["hv_config"])
    regime_rerank = read_csv(paths["regime_rerank_summary"])
    turnover = read_csv(paths["turnover_summary"])

    hv_row = row_by(regime_rerank, "profile_name", "hv_close_position_20d_m005")
    robust_row = row_by(turnover, "profile_name", "mt050_tc0010_blend_0.5_cap0.20")
    hv_is_default = default_config.get("profile_name") == hv_config.get("profile_name")
    hv_result_is_current = (
        paths["current_result"].exists()
        and paths["hv_result"].exists()
        and paths["current_result"].read_text(encoding="utf-8") == paths["hv_result"].read_text(encoding="utf-8")
    )
    hv_evidence = "HV rerank 提交级文件已生成。"
    if hv_row is not None:
        hv_evidence = (
            f"regime_rerank: delta_return={fmt(hv_row.get('delta_cost_after_return'))}, "
            f"delta_sharpe={fmt(hv_row.get('delta_sharpe'))}, "
            f"delta_max_drawdown={fmt(hv_row.get('delta_max_drawdown'))}, "
            f"delta_high_vol={fmt(hv_row.get('delta_high_volatility_selected_top5_return'))}。"
        )
    robust_evidence = "低换手 robust 配置用于稳定场景。"
    if robust_row is not None:
        robust_evidence = (
            f"turnover_stress: return={fmt(robust_row.get('cost_after_return'))}, "
            f"sharpe={fmt(robust_row.get('sharpe'))}, "
            f"max_drawdown={fmt(robust_row.get('max_drawdown'))}, "
            f"avg_turnover={fmt(robust_row.get('avg_turnover'))}。"
        )

    return [
        CandidateDecision(
            candidate="current_default_sl20",
            profile_name=default_config.get("profile_name", "n/a"),
            config_path=paths["default_config"],
            result_path=paths["current_result"],
            role="当前权威默认提交" if not hv_is_default else "当前权威默认提交，已同步 HV rerank",
            submit_decision="建议保留为默认提交",
            reason=(
                "已按人工确认切换为 HV rerank；它仍使用 LSTM sl20 主线，只增加高波动阶段轻重排。"
                if hv_is_default
                else "没有任何新方案在 walk-forward、回测、单切片三方面同时完整击败 sl20；默认结果已是正式提交入口产物。"
            ),
            cannot_submit_reason="无。当前就是默认正式结果。",
            evidence=f"{validate_status(paths['current_result'])}；{profile_summary(default_config)}。",
            sync_default_config="不需要",
            manual_action="若不想承担候选切换风险，直接保留当前 result.csv。",
        ),
        CandidateDecision(
            candidate="aggressive",
            profile_name=aggressive_config.get("profile_name", "n/a"),
            config_path=paths["aggressive_config"],
            result_path=None,
            role="比赛冲分配置建议",
            submit_decision="不建议直接提交",
            reason="它追求收益和单切片，保留 pred 权重、宽候选池和较高换手，适合冲分思路。",
            cannot_submit_reason="当前没有作为独立正式候选重新生成 result 并通过提交级复核；且高换手/集中风险更高。",
            evidence=f"{profile_summary(aggressive_config)}；最终选择器将其定位为冲分配置。",
            sync_default_config="只有人工明确选择比赛冲分时才同步",
            manual_action="若要选它，先单独生成 aggressive result，再跑 validator、pre-submit 和与当前结果对比。",
        ),
        CandidateDecision(
            candidate="robust",
            profile_name=robust_config.get("profile_name", "n/a"),
            config_path=paths["robust_config"],
            result_path=None,
            role="稳定/低换手配置建议",
            submit_decision="不建议作为本轮默认提交",
            reason="它降低换手、单票集中和纯 pred 依赖，适合高波动震荡或保守展示。",
            cannot_submit_reason="收益弹性明显低于 aggressive/默认主线，且当前未作为独立正式候选生成提交文件。",
            evidence=f"{robust_evidence} {profile_summary(robust_config)}。",
            sync_default_config="只有目标切换为稳定策略时才同步",
            manual_action="若目标变成低回撤/低换手，先生成 robust result，再做提交级验证。",
        ),
        CandidateDecision(
            candidate="hv_rerank",
            profile_name=hv_config.get("profile_name", "n/a"),
            config_path=paths["hv_config"],
            result_path=paths["hv_result"],
            role="最值得人工确认的增强候选",
            submit_decision="已同步为当前默认提交" if hv_is_default and hv_result_is_current else "可作为人工确认后的增强提交候选",
            reason="它不替换 sl20 模型，只在高波动 regime 触发 close_position_20d 轻微重排；本轮最新数据确实触发高波动状态。",
            cannot_submit_reason=(
                "无。已经人工确认并同步为当前默认。"
                if hv_is_default and hv_result_is_current
                else "不能自动提交，因为它改变了 1 只股票，且默认配置尚未同步；需要人工确认换入/换出是否接受。"
            ),
            evidence=f"{validate_status(paths['hv_result'])}；{hv_evidence} {result_delta_summary(paths['hv_compare'])}",
            sync_default_config="已同步" if hv_is_default and hv_result_is_current else "若决定提交它，需要手动同步默认配置/提交结果",
            manual_action=(
                "当前默认 result.csv 已经是 HV rerank 结果；后续只需做最终打包前检查。"
                if hv_is_default and hv_result_is_current
                else "人工确认 600115 换入、601877 换出后，再决定是否把 HV rerank 升级为正式默认。"
            ),
        ),
    ]


def decisions_to_frame(decisions: list[CandidateDecision]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate": item.candidate,
                "profile_name": item.profile_name,
                "role": item.role,
                "submit_decision": item.submit_decision,
                "reason": item.reason,
                "cannot_submit_reason": item.cannot_submit_reason,
                "evidence": item.evidence,
                "sync_default_config": item.sync_default_config,
                "manual_action": item.manual_action,
                "config_path": str(item.config_path),
                "result_path": "" if item.result_path is None else str(item.result_path),
            }
            for item in decisions
        ]
    )


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row.get(column, "")).replace("\n", " ").replace("|", "/") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_report(df: pd.DataFrame, paths: dict[str, Path]) -> str:
    hv_synced = bool(
        (
            df["candidate"].eq("hv_rerank")
            & df["submit_decision"].astype(str).str.contains("已同步", na=False)
        ).any()
    )
    conclusion = [
        "- 默认建议：当前默认结果已经可以作为权威提交。",
        "- HV rerank 已按人工确认同步为默认提交。"
        if hv_synced
        else "- 最值得人工确认的增强候选：HV rerank。",
        "- aggressive 只适合比赛冲分目标，当前不直接提交。",
        "- robust 只适合稳定/低换手目标，当前不作为默认提交。",
        "- 是否同步默认配置：已同步 HV rerank。"
        if hv_synced
        else "- 是否同步默认配置：暂不自动同步；只有人工确认选择 HV rerank、aggressive 或 robust 后才同步。",
    ]
    lines = [
        "# Final Candidate Decision Report",
        "",
        "本报告用于人工确认前的最终候选决策；不会覆盖 `app/output/result.csv`，也不会覆盖 `app/model/default_submission_config.json`。",
        "",
        "## 结论",
        "",
        *conclusion,
        "",
        "## 最终候选决策表",
        "",
    ]
    lines.extend(
        markdown_table(
            df,
            [
                "candidate",
                "role",
                "submit_decision",
                "reason",
                "cannot_submit_reason",
                "sync_default_config",
            ],
        )
    )
    lines.extend(["", "## 证据表", ""])
    lines.extend(markdown_table(df, ["candidate", "evidence", "manual_action"]))
    lines.extend(
        [
            "",
            "## 文件",
            "",
            f"- decision csv: `{paths['output_csv']}`",
            f"- current result: `{paths['current_result']}`",
            f"- HV rerank result: `{paths['hv_result']}`",
            f"- HV rerank comparison: `{paths['hv_compare']}`",
            f"- full validation report: `{paths['full_report']}`",
            f"- final config selection report: `{paths['final_selection_report']}`",
            "",
            "## 手动确认提示",
            "",
            "如果最终选择当前默认结果：不需要同步配置。",
            "",
            (
                "HV rerank 已经同步为当前默认结果；当前正式 `result.csv` 包含 `600115`，不再包含 `601877`。"
                if hv_synced
                else "如果最终选择 HV rerank：请人工确认 `600115` 换入、`601877` 换出后，再手动同步默认配置和提交结果。"
            ),
            "",
            "如果最终选择 aggressive 或 robust：请先生成对应独立提交文件并通过 validator / pre-submit，再决定是否同步默认配置。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final candidate decision table before manual confirmation.")
    for name, path in DEFAULT_PATHS.items():
        parser.add_argument(f"--{name}", default=str(path))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {name: resolve_path(getattr(args, name)) for name in DEFAULT_PATHS}
    paths["output_csv"].parent.mkdir(parents=True, exist_ok=True)
    paths["output_md"].parent.mkdir(parents=True, exist_ok=True)

    decisions = build_decisions(paths)
    df = decisions_to_frame(decisions)
    df.to_csv(paths["output_csv"], index=False, encoding="utf-8-sig")
    paths["output_md"].write_text(render_report(df, paths), encoding="utf-8")
    print(f"[final_candidate_decision] wrote {paths['output_csv']}")
    print(f"[final_candidate_decision] wrote {paths['output_md']}")
    print("[final_candidate_decision] default_submission_config.json was not modified")
    print("[final_candidate_decision] app/output/result.csv was not modified")


if __name__ == "__main__":
    main()
