import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "app" / "model"
CONFIG_DIR = MODEL_DIR / "configs"

DEFAULT_PATHS = {
    "leaderboard": MODEL_DIR / "experiment_leaderboard.csv",
    "turnover": MODEL_DIR / "turnover_stress_test" / "turnover_stress_summary.csv",
    "weight_cap": MODEL_DIR / "weight_cap_search" / "weight_cap_summary.csv",
    "weight_blend": MODEL_DIR / "weight_blend_search" / "weight_blend_summary.csv",
    "stability": MODEL_DIR / "stability_eval" / "stability_summary.csv",
    "regime_rerank": MODEL_DIR / "regime_rerank_switch" / "regime_rerank_switch_summary.csv",
    "aggressive_config": CONFIG_DIR / "submission_aggressive.json",
    "robust_config": CONFIG_DIR / "submission_robust.json",
    "hv_rerank_config": CONFIG_DIR / "submission_hv_rerank_candidate.json",
    "output": CONFIG_DIR / "final_config_selection_report.md",
}


def _resolve(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def first_valid(df: pd.DataFrame, column: str, default: Any = None) -> Any:
    if column not in df.columns or df.empty:
        return default
    valid = df[column].dropna()
    return valid.iloc[0] if not valid.empty else default


def row_by(df: pd.DataFrame, **filters: Any) -> pd.Series | None:
    if df.empty:
        return None
    mask = pd.Series([True] * len(df), index=df.index)
    for column, expected in filters.items():
        if column not in df.columns:
            return None
        values = df[column].astype(str)
        mask &= values == str(expected)
    matched = df[mask]
    if matched.empty:
        return None
    return matched.iloc[0]


def best_row(df: pd.DataFrame, column: str, ascending: bool = False) -> pd.Series | None:
    if df.empty or column not in df.columns:
        return None
    ranked = df.assign(_metric=numeric(df, column)).dropna(subset=["_metric"])
    if ranked.empty:
        return None
    return ranked.sort_values("_metric", ascending=ascending).iloc[0]


def profile_summary(config: dict[str, Any]) -> str:
    selection = config["selection_logic"]
    execution = config["execution_logic"]
    risk = config["risk_filter_thresholds"]
    summary = (
        f"`{config['profile_name']}`: model={config['model_family']}, "
        f"feature_set={config['feature_set']}, sl={config['validation_scheme']['sequence_length']}, "
        f"cs={selection['primary_candidate_size']}, sort={selection['sort_strategy']}, "
        f"weight={selection['weighting_scheme']}, blend_alpha={selection.get('weight_blend_alpha', 1.0)}, "
        f"cap={selection.get('max_single_weight')}, rp={risk['risk_penalty_weight']}, "
        f"max_turnover={execution['max_turnover']}, tc={execution['transaction_cost']}"
    )
    rerank = config.get("regime_rerank") or {}
    if rerank.get("enabled"):
        summary += (
            f", regime_rerank={rerank.get('regime_flag')}/"
            f"{rerank.get('signal')}:{rerank.get('weight')}"
        )
    return summary


def collect_evidence(
    leaderboard: pd.DataFrame,
    turnover: pd.DataFrame,
    weight_cap: pd.DataFrame,
    weight_blend: pd.DataFrame,
    stability: pd.DataFrame,
    regime_rerank: pd.DataFrame,
) -> dict[str, Any]:
    cap20 = row_by(weight_cap, cap="0.20")
    cap18 = row_by(weight_cap, cap="0.18")
    capnone = row_by(weight_cap, cap="none")
    blend05_cap20 = row_by(weight_blend, alpha="0.5", max_single_weight="0.20")
    turnover_blend05_cap20 = row_by(
        turnover,
        profile_name="mt050_tc0010_blend_0.5_cap0.20",
    )
    turnover_pred_cap20 = row_by(
        turnover,
        profile_name="mt050_tc0010_pred_cap0.20",
    )
    top_return = best_row(leaderboard, "cumulative_return_after_cost")
    top_composite = best_row(leaderboard, "composite_score")
    top_stability = best_row(stability, "stability_score")
    hv_rerank = row_by(regime_rerank, profile_name="hv_close_position_20d_m005")
    rerank_baseline = row_by(regime_rerank, profile_name="baseline")
    top_rerank = best_row(regime_rerank, "cost_after_return")
    target_stability = stability[
        (stability.get("feature_set", pd.Series(dtype=str)).astype(str) == "base_alpha_v3_rs_crowding_mini4")
        & (numeric(stability, "sequence_length") == 20)
        & (stability.get("target_mode", pd.Series(dtype=str)).astype(str).isin(["cross_section_rank", ""]))
    ]
    return {
        "cap20": cap20,
        "cap18": cap18,
        "capnone": capnone,
        "blend05_cap20": blend05_cap20,
        "turnover_blend05_cap20": turnover_blend05_cap20,
        "turnover_pred_cap20": turnover_pred_cap20,
        "top_return": top_return,
        "top_composite": top_composite,
        "top_stability": top_stability,
        "hv_rerank": hv_rerank,
        "rerank_baseline": rerank_baseline,
        "top_rerank": top_rerank,
        "target_stability": target_stability.iloc[0] if not target_stability.empty else None,
    }


def render_row(label: str, row: pd.Series | None, columns: list[str]) -> str:
    if row is None:
        return f"- {label}: n/a"
    values = ", ".join(f"{column}={fmt(row.get(column))}" for column in columns)
    return f"- {label}: {values}"


def render_report(
    aggressive_config: dict[str, Any],
    robust_config: dict[str, Any],
    hv_rerank_config: dict[str, Any],
    evidence: dict[str, Any],
    paths: dict[str, Path],
) -> str:
    aggressive_name = aggressive_config["profile_name"]
    robust_name = robust_config["profile_name"]
    hv_rerank_name = hv_rerank_config.get("profile_name", "hv_rerank_candidate_unavailable")
    lines = [
        "# Final Submission Config Selection Report",
        "",
        "This report is advisory only. It does not overwrite `app/model/default_submission_config.json`.",
        "",
        "## 1. Aggressive 配置是谁，为什么",
        "",
        f"- 建议 aggressive 配置: {profile_summary(aggressive_config)}",
        "- 原因: 该配置使用 `pred` 权重、`primary_candidate_size=180` 宽候选池、`max_turnover=1.0`，目标是单切片分数和累计收益。",
        render_row(
            "weight_cap cap=0.20",
            evidence["cap20"],
            [
                "single_slice_score",
                "cost_after_return",
                "sharpe",
                "max_drawdown",
                "avg_turnover",
                "max_single_weight",
                "max_single_contribution_ratio",
            ],
        ),
        render_row(
            "weight_cap cap=none",
            evidence["capnone"],
            [
                "single_slice_score",
                "cost_after_return",
                "max_drawdown",
                "avg_turnover",
                "max_single_contribution_ratio",
            ],
        ),
        render_row(
            "leaderboard top cumulative_return_after_cost",
            evidence["top_return"],
            [
                "candidate_label",
                "cumulative_return_after_cost",
                "sharpe_after_cost",
                "max_drawdown_after_cost",
                "avg_turnover",
                "slice_score",
            ],
        ),
        "",
        "## 2. Robust 配置是谁，为什么",
        "",
        f"- 建议 robust 配置: {profile_summary(robust_config)}",
        "- 原因: 该配置使用 `blend_0.5` 风格的 pred/equal 混合、`max_single_weight=0.20`、`max_turnover=0.50`，更适合高波动震荡阶段。",
        render_row(
            "turnover stress mt050_tc0010_blend_0.5_cap0.20",
            evidence["turnover_blend05_cap20"],
            [
                "cost_after_return",
                "sharpe",
                "max_drawdown",
                "avg_turnover",
                "robust_score",
                "win_rate",
            ],
        ),
        render_row(
            "weight_blend alpha=0.5 cap=0.20",
            evidence["blend05_cap20"],
            [
                "single_slice_score",
                "cost_after_return",
                "sharpe",
                "max_drawdown",
                "avg_turnover",
                "max_single_contribution_ratio",
            ],
        ),
        render_row(
            "weight_cap cap=0.18",
            evidence["cap18"],
            [
                "single_slice_score",
                "cost_after_return",
                "sharpe",
                "max_drawdown",
                "avg_turnover",
                "max_single_contribution_ratio",
            ],
        ),
        "",
        "## 2b. HV rerank 候选是谁，为什么",
        "",
        f"- 建议 HV rerank 候选: {profile_summary(hv_rerank_config) if hv_rerank_config else 'n/a'}",
        "- 原因: 该配置不替换 LSTM sl20 主线，只在 `is_high_volatility=1` 时对 `close_position_20d` 施加 -0.05 轻微重排惩罚，用于处理高波动阶段的误排样本。",
        render_row(
            "regime_rerank baseline",
            evidence["rerank_baseline"],
            [
                "cost_after_return",
                "sharpe",
                "max_drawdown",
                "avg_turnover",
                "selected_top5_return_mean",
                "high_volatility_selected_top5_return",
                "poor_false_positives",
            ],
        ),
        render_row(
            "regime_rerank hv_close_position_20d_m005",
            evidence["hv_rerank"],
            [
                "cost_after_return",
                "sharpe",
                "max_drawdown",
                "avg_turnover",
                "selected_top5_return_mean",
                "high_volatility_selected_top5_return",
                "delta_cost_after_return",
                "delta_sharpe",
                "delta_max_drawdown",
                "delta_poor_false_positives",
            ],
        ),
        render_row(
            "regime_rerank best cost_after_return",
            evidence["top_rerank"],
            [
                "profile_name",
                "cost_after_return",
                "sharpe",
                "max_drawdown",
                "avg_turnover",
                "delta_cost_after_return",
            ],
        ),
        "",
        "## 3. 最终默认建议用哪个",
        "",
        f"- 默认建议: 暂时保留当前 LSTM sl20 默认主线；新增 `{hv_rerank_name}` 作为优先复核候选。",
        "- 理由: aggressive/robust 仍然服务于不同提交目标；HV rerank 在现有回测中同时改善收益、Sharpe、高波动 Top5 和误报数量，但它是后处理重排，需要再经过完整提交文件验证后才能同步默认配置。",
        "- 注意: 本脚本不会自动同步默认配置，需要人工确认后再决定是否执行同步。",
        "",
        "## 4. 如果目标是比赛冲分，选哪个",
        "",
        f"- 选择 `{aggressive_name}`。",
        "- 采用证据: cap=0.20 保留最高单切片/累计收益附近的表现，`pred` 权重和 full turnover 不主动压制冲分能力。",
        "- 放弃 robust 的原因: robust 主动限制换手并混合权重，适合稳健但会牺牲一部分收益弹性。",
        f"- 可加测 `{hv_rerank_name}`: 它保留 pred 权重和宽候选池，同时只在高波动阶段轻微重排；若最终提交文件验证通过，可作为冲分增强候选。",
        "",
        "## 5. 如果目标是稳定策略，选哪个",
        "",
        f"- 选择 `{robust_name}`。",
        "- 采用证据: turnover stress 中 `mt050_tc0010_blend_0.5_cap0.20` 在低换手约束下保留较高 robust_score；cap=0.20 控制单票权重，blend_0.5 降低纯 pred 权重依赖。",
        "- 放弃 aggressive 的原因: aggressive 接受更高换手和集中度，遇到高波动震荡阶段时执行和回撤风险更高。",
        f"- 若目标是“不明显降收益的稳定增强”，优先观察 `{hv_rerank_name}`；若目标是强制低换手，则仍选 robust。",
        "",
        "## 6. 采用/放弃每个配置的证据",
        "",
        f"### `{aggressive_name}`",
        "",
        "- 采用: 宽候选池 `cs180`、`pred` 权重、`max_turnover=1.0` 与冲分目标一致。",
        "- 采用: weight_cap 中 cap=0.20 的 `single_slice_score` 和 `cost_after_return` 保持强势。",
        "- 放弃作为稳健默认: `avg_turnover` 和 `max_single_contribution_ratio` 偏高，风险暴露更集中。",
        "",
        f"### `{robust_name}`",
        "",
        "- 采用: `blend_0.5`、`max_single_weight=0.20`、`max_turnover=0.50` 同时处理权重集中和换手压力。",
        "- 采用: turnover stress 的低换手配置给出可接受收益与 Sharpe，适合震荡阶段。",
        "- 放弃作为比赛冲分默认: 累计收益上限低于 aggressive 路线，可能错过单切片最优权重。",
        "",
        f"### `{hv_rerank_name}`",
        "",
        "- 采用: `hv_close_position_20d_m005` 相对 baseline 提高 `cost_after_return`、`sharpe`，并小幅改善 `max_drawdown` 和高波动阶段 Top5 收益。",
        "- 采用: 仅在高波动 regime 触发，避免把防守惩罚扩散到全部市场阶段。",
        "- 放弃直接同步默认: 这是重排后处理候选，还需要重新生成提交文件并通过 `result_validator.py`、`pre_submit_check.py` 和完整验证流水线。",
        "",
        "## Stability 参考",
        "",
        render_row(
            "target stability row",
            evidence["target_stability"],
            [
                "model",
                "feature_set",
                "sequence_length",
                "rank_ic_mean",
                "worst_fold_rank_ic",
                "top5_return_mean",
                "stability_score",
            ],
        ),
        render_row(
            "best stability_score row",
            evidence["top_stability"],
            [
                "model",
                "feature_set",
                "sequence_length",
                "rank_ic_mean",
                "worst_fold_rank_ic",
                "top5_return_mean",
                "stability_score",
            ],
        ),
        "",
        "## Input Files",
        "",
        f"- experiment_leaderboard.csv: `{paths['leaderboard']}`",
        f"- turnover_stress_summary.csv: `{paths['turnover']}`",
        f"- weight_cap_summary.csv: `{paths['weight_cap']}`",
        f"- weight_blend_summary.csv: `{paths['weight_blend']}`",
        f"- stability_summary.csv: `{paths['stability']}`",
        f"- regime_rerank_switch_summary.csv: `{paths['regime_rerank']}`",
        f"- aggressive config: `{paths['aggressive_config']}`",
        f"- robust config: `{paths['robust_config']}`",
        f"- hv rerank config: `{paths['hv_rerank_config']}`",
        "",
        "## Manual Confirmation Required",
        "",
        "请手动确认是否将建议配置同步到 `app/model/default_submission_config.json`；本报告和脚本不会自动覆盖默认配置。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select aggressive/robust submission configs and write an advisory report."
    )
    for name, path in DEFAULT_PATHS.items():
        parser.add_argument(f"--{name}", default=str(path))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {name: _resolve(getattr(args, name)) for name in DEFAULT_PATHS}

    leaderboard = load_csv(paths["leaderboard"])
    turnover = load_csv(paths["turnover"])
    weight_cap = load_csv(paths["weight_cap"])
    weight_blend = load_csv(paths["weight_blend"])
    stability = load_csv(paths["stability"])
    regime_rerank = load_optional_csv(paths["regime_rerank"])
    aggressive_config = load_json(paths["aggressive_config"])
    robust_config = load_json(paths["robust_config"])
    hv_rerank_config = load_optional_json(paths["hv_rerank_config"])

    evidence = collect_evidence(
        leaderboard=leaderboard,
        turnover=turnover,
        weight_cap=weight_cap,
        weight_blend=weight_blend,
        stability=stability,
        regime_rerank=regime_rerank,
    )
    report = render_report(
        aggressive_config=aggressive_config,
        robust_config=robust_config,
        hv_rerank_config=hv_rerank_config,
        evidence=evidence,
        paths=paths,
    )
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(report, encoding="utf-8")
    print(f"[select_final_submission_config] wrote {paths['output']}")
    print("[select_final_submission_config] default_submission_config.json was not modified")


if __name__ == "__main__":
    main()
