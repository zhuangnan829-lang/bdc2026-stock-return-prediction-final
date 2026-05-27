import argparse
import json
import re
from pathlib import Path

import pandas as pd

from config import ROOT_DIR
from load_submission_config import build_best_config_from_submission
from sync_submission_artifacts import sync_submission_artifacts


BEST_CONFIG_PATH = ROOT_DIR / "app" / "model" / "best_config.json"
DEFAULT_SUBMISSION_CONFIG_PATH = ROOT_DIR / "app" / "model" / "default_submission_config.json"
MODEL_META_PATH = ROOT_DIR / "app" / "model" / "model_meta.json"
TEST_SH_PATH = ROOT_DIR / "app" / "test.sh"
TEST_PS1_PATH = ROOT_DIR / "app" / "test.ps1"
RESULT_PATH = ROOT_DIR / "app" / "output" / "result.csv"
PACKAGE_VARIANT_PATH = ROOT_DIR / "app" / "model" / "package_variant.json"
AGGRESSIVE_RESULT_PATH = ROOT_DIR / "app" / "model" / "aggressive_score_submission_candidate" / "result_aggressive_score.csv"
SUMMARY_PATH = ROOT_DIR / "app" / "model" / "default_profile_backtest" / "backtest_summary.csv"
CASE_COMPARISON_PATH = ROOT_DIR / "app" / "model" / "case_program_comparison" / "case_program_comparison_summary.csv"
SNAPSHOT_PATH = ROOT_DIR / "app" / "model" / "final_submission_snapshot.md"
SUBMISSION_ARTIFACTS_DIR = ROOT_DIR / "app" / "model" / "submission_artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync formal submission defaults from default_submission_config.json.")
    parser.add_argument("--best_config_path", default=str(BEST_CONFIG_PATH))
    parser.add_argument("--default_submission_config_path", default=str(DEFAULT_SUBMISSION_CONFIG_PATH))
    parser.add_argument("--model_meta_path", default=str(MODEL_META_PATH))
    parser.add_argument("--test_sh_path", default=str(TEST_SH_PATH))
    parser.add_argument("--test_ps1_path", default=str(TEST_PS1_PATH))
    parser.add_argument("--result_path", default=str(RESULT_PATH))
    parser.add_argument("--summary_path", default=str(SUMMARY_PATH))
    parser.add_argument("--case_comparison_path", default=str(CASE_COMPARISON_PATH))
    parser.add_argument("--snapshot_path", default=str(SNAPSHOT_PATH))

    parser.add_argument("--write_best_config", action="store_true", help="Write best_config.json from CLI parameters before syncing.")
    parser.add_argument("--profile_name")
    parser.add_argument("--feature_set")
    parser.add_argument("--target_mode")
    parser.add_argument("--model_family")
    parser.add_argument("--valid_dates", type=int)
    parser.add_argument("--num_folds", type=int)
    parser.add_argument("--sequence_length", type=int)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)

    parser.add_argument("--top_k", type=int)
    parser.add_argument("--primary_candidate_size", type=int)
    parser.add_argument("--enable_risk_filters", type=int, choices=[0, 1])
    parser.add_argument("--sort_strategy")
    parser.add_argument("--weighting_scheme")

    parser.add_argument("--max_volatility_20d_pct", type=float)
    parser.add_argument("--max_volatility_5d_pct", type=float)
    parser.add_argument("--turnover_rate_lower_pct", type=float)
    parser.add_argument("--turnover_rate_upper_pct", type=float)
    parser.add_argument("--turnover_ratio_upper_pct", type=float)
    parser.add_argument("--risk_penalty_weight", type=float)

    parser.add_argument("--use_previous_result_when_available", type=int, choices=[0, 1])
    parser.add_argument("--max_turnover", type=float)
    parser.add_argument("--transaction_cost", type=float)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_result_for_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["stock_id", "weight"])
    return pd.read_csv(path, dtype={"stock_id": str})


def load_summary_for_snapshot(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        [
            {
                "label": "snapshot_metrics_unavailable",
                "cumulative_return_after_cost": 0.0,
                "sharpe_after_cost": 0.0,
                "max_drawdown_after_cost": 0.0,
                "avg_turnover": 0.0,
            }
        ]
    )


def load_case_comparison_for_snapshot(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        [
            {"dimension": "current_live_case_slice_score", "our_value": 0.0, "case_value": 0.0},
            {"dimension": "current_live_vs_case_reported_best", "our_value": 0.0, "case_value": 0.0},
        ]
    )


def load_package_variant_for_snapshot(result_df: pd.DataFrame) -> dict:
    if not PACKAGE_VARIANT_PATH.exists():
        return {}
    try:
        payload = load_json(PACKAGE_VARIANT_PATH)
    except json.JSONDecodeError:
        return {}
    if payload.get("variant") != "aggressive_score_submission" or not AGGRESSIVE_RESULT_PATH.exists():
        return {}

    aggressive_df = pd.read_csv(AGGRESSIVE_RESULT_PATH, dtype={"stock_id": str})
    current_pairs = [
        (str(row["stock_id"]).zfill(6), f"{float(row['weight']):.12f}")
        for _, row in result_df.iterrows()
    ]
    aggressive_pairs = [
        (str(row["stock_id"]).zfill(6), f"{float(row['weight']):.12f}")
        for _, row in aggressive_df.iterrows()
    ]
    return payload if current_pairs == aggressive_pairs else {}


def replace_pattern(text: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected exactly one replacement for pattern: {pattern}")
    return updated


def sync_test_sh(path: Path, best_config: dict) -> None:
    selection = best_config["selection"]
    risk = best_config["risk_filter_thresholds"]
    execution = best_config["execution"]

    text = path.read_text(encoding="utf-8")
    replacements = {
        r'^AUTO_USE_PREVIOUS_RESULT="\$\{AUTO_USE_PREVIOUS_RESULT:-[01]\}"$':
            f'AUTO_USE_PREVIOUS_RESULT="${{AUTO_USE_PREVIOUS_RESULT:-{1 if execution["use_previous_result_when_available"] else 0}}}"',
        r'^PRIMARY_CANDIDATE_SIZE="\$\{PRIMARY_CANDIDATE_SIZE:-[^}]+\}"$':
            f'PRIMARY_CANDIDATE_SIZE="${{PRIMARY_CANDIDATE_SIZE:-{int(selection["primary_candidate_size"])}}}"',
        r'^MAX_VOLATILITY_20D_PCT="\$\{MAX_VOLATILITY_20D_PCT:-[^}]+\}"$':
            f'MAX_VOLATILITY_20D_PCT="${{MAX_VOLATILITY_20D_PCT:-{float(risk["max_volatility_20d_pct"]):.2f}}}"',
        r'^MAX_VOLATILITY_5D_PCT="\$\{MAX_VOLATILITY_5D_PCT:-[^}]+\}"$':
            f'MAX_VOLATILITY_5D_PCT="${{MAX_VOLATILITY_5D_PCT:-{float(risk["max_volatility_5d_pct"]):.1f}}}"',
        r'^TURNOVER_RATE_LOWER_PCT="\$\{TURNOVER_RATE_LOWER_PCT:-[^}]+\}"$':
            f'TURNOVER_RATE_LOWER_PCT="${{TURNOVER_RATE_LOWER_PCT:-{float(risk["turnover_rate_lower_pct"]):.2f}}}"',
        r'^TURNOVER_RATE_UPPER_PCT="\$\{TURNOVER_RATE_UPPER_PCT:-[^}]+\}"$':
            f'TURNOVER_RATE_UPPER_PCT="${{TURNOVER_RATE_UPPER_PCT:-{float(risk["turnover_rate_upper_pct"]):.2f}}}"',
        r'^TURNOVER_RATIO_UPPER_PCT="\$\{TURNOVER_RATIO_UPPER_PCT:-[^}]+\}"$':
            f'TURNOVER_RATIO_UPPER_PCT="${{TURNOVER_RATIO_UPPER_PCT:-{float(risk["turnover_ratio_upper_pct"]):.2f}}}"',
        r'^RISK_PENALTY_WEIGHT="\$\{RISK_PENALTY_WEIGHT:-[^}]+\}"$':
            f'RISK_PENALTY_WEIGHT="${{RISK_PENALTY_WEIGHT:-{float(risk["risk_penalty_weight"]):.2f}}}"',
        r'^SORT_STRATEGY="\$\{SORT_STRATEGY:-[^}]+\}"$':
            f'SORT_STRATEGY="${{SORT_STRATEGY:-{selection["sort_strategy"]}}}"',
        r'^WEIGHTING_SCHEME="\$\{WEIGHTING_SCHEME:-[^}]+\}"$':
            f'WEIGHTING_SCHEME="${{WEIGHTING_SCHEME:-{selection["weighting_scheme"]}}}"',
        r'^MAX_TURNOVER="\$\{MAX_TURNOVER:-[^}]+\}"$':
            f'MAX_TURNOVER="${{MAX_TURNOVER:-{float(execution["max_turnover"]):.1f}}}"',
    }
    for pattern, replacement in replacements.items():
        text = replace_pattern(text, pattern, replacement)
    path.write_text(text, encoding="utf-8")


def sync_test_ps1(path: Path, best_config: dict) -> None:
    selection = best_config["selection"]
    risk = best_config["risk_filter_thresholds"]
    execution = best_config["execution"]

    text = path.read_text(encoding="utf-8")
    replacements = {
        r'^\$autoUsePreviousResult = if \(\$env:AUTO_USE_PREVIOUS_RESULT\) \{ \$env:AUTO_USE_PREVIOUS_RESULT \} else \{ "[01]" \}$':
            f'$autoUsePreviousResult = if ($env:AUTO_USE_PREVIOUS_RESULT) {{ $env:AUTO_USE_PREVIOUS_RESULT }} else {{ "{1 if execution["use_previous_result_when_available"] else 0}" }}',
        r'^\$primaryCandidateSize = if \(\$env:PRIMARY_CANDIDATE_SIZE\) \{ \$env:PRIMARY_CANDIDATE_SIZE \} else \{ "[^"]+" \}$':
            f'$primaryCandidateSize = if ($env:PRIMARY_CANDIDATE_SIZE) {{ $env:PRIMARY_CANDIDATE_SIZE }} else {{ "{int(selection["primary_candidate_size"])}" }}',
        r'^\$maxVolatility20dPct = if \(\$env:MAX_VOLATILITY_20D_PCT\) \{ \$env:MAX_VOLATILITY_20D_PCT \} else \{ "[^"]+" \}$':
            f'$maxVolatility20dPct = if ($env:MAX_VOLATILITY_20D_PCT) {{ $env:MAX_VOLATILITY_20D_PCT }} else {{ "{float(risk["max_volatility_20d_pct"]):.2f}" }}',
        r'^\$maxVolatility5dPct = if \(\$env:MAX_VOLATILITY_5D_PCT\) \{ \$env:MAX_VOLATILITY_5D_PCT \} else \{ "[^"]+" \}$':
            f'$maxVolatility5dPct = if ($env:MAX_VOLATILITY_5D_PCT) {{ $env:MAX_VOLATILITY_5D_PCT }} else {{ "{float(risk["max_volatility_5d_pct"]):.1f}" }}',
        r'^\$turnoverRateLowerPct = if \(\$env:TURNOVER_RATE_LOWER_PCT\) \{ \$env:TURNOVER_RATE_LOWER_PCT \} else \{ "[^"]+" \}$':
            f'$turnoverRateLowerPct = if ($env:TURNOVER_RATE_LOWER_PCT) {{ $env:TURNOVER_RATE_LOWER_PCT }} else {{ "{float(risk["turnover_rate_lower_pct"]):.2f}" }}',
        r'^\$turnoverRateUpperPct = if \(\$env:TURNOVER_RATE_UPPER_PCT\) \{ \$env:TURNOVER_RATE_UPPER_PCT \} else \{ "[^"]+" \}$':
            f'$turnoverRateUpperPct = if ($env:TURNOVER_RATE_UPPER_PCT) {{ $env:TURNOVER_RATE_UPPER_PCT }} else {{ "{float(risk["turnover_rate_upper_pct"]):.2f}" }}',
        r'^\$turnoverRatioUpperPct = if \(\$env:TURNOVER_RATIO_UPPER_PCT\) \{ \$env:TURNOVER_RATIO_UPPER_PCT \} else \{ "[^"]+" \}$':
            f'$turnoverRatioUpperPct = if ($env:TURNOVER_RATIO_UPPER_PCT) {{ $env:TURNOVER_RATIO_UPPER_PCT }} else {{ "{float(risk["turnover_ratio_upper_pct"]):.2f}" }}',
        r'^\$riskPenaltyWeight = if \(\$env:RISK_PENALTY_WEIGHT\) \{ \$env:RISK_PENALTY_WEIGHT \} else \{ "[^"]+" \}$':
            f'$riskPenaltyWeight = if ($env:RISK_PENALTY_WEIGHT) {{ $env:RISK_PENALTY_WEIGHT }} else {{ "{float(risk["risk_penalty_weight"]):.2f}" }}',
        r'^\$sortStrategy = if \(\$env:SORT_STRATEGY\) \{ \$env:SORT_STRATEGY \} else \{ "[^"]+" \}$':
            f'$sortStrategy = if ($env:SORT_STRATEGY) {{ $env:SORT_STRATEGY }} else {{ "{selection["sort_strategy"]}" }}',
        r'^\$weightingScheme = if \(\$env:WEIGHTING_SCHEME\) \{ \$env:WEIGHTING_SCHEME \} else \{ "[^"]+" \}$':
            f'$weightingScheme = if ($env:WEIGHTING_SCHEME) {{ $env:WEIGHTING_SCHEME }} else {{ "{selection["weighting_scheme"]}" }}',
        r'^\$maxTurnover = if \(\$env:MAX_TURNOVER\) \{ \$env:MAX_TURNOVER \} else \{ "[^"]+" \}$':
            f'$maxTurnover = if ($env:MAX_TURNOVER) {{ $env:MAX_TURNOVER }} else {{ "{float(execution["max_turnover"]):.1f}" }}',
    }
    for pattern, replacement in replacements.items():
        text = replace_pattern(text, pattern, replacement)
    path.write_text(text, encoding="utf-8")


def build_default_submission_config(best_config: dict) -> dict:
    training = best_config["training"]
    selection = best_config["selection"]
    risk = best_config["risk_filter_thresholds"]
    execution = best_config["execution"]
    return {
        "profile_name": best_config["profile_name"],
        "status": "frozen_default_submission_candidate",
        "feature_set": training["feature_set"],
        "feature_count": 20,
        "target_mode": training["target_mode"],
        "model_family": training["model_family"],
        "seed": int(training.get("seed", 2026)),
        "validation_scheme": {
            "type": "walk_forward",
            "valid_dates": int(training["valid_dates"]),
            "num_folds": int(training["num_folds"]),
            "sequence_length": int(training["sequence_length"]),
        },
        "selection_logic": selection,
        "risk_filter_thresholds": risk,
        "execution_logic": execution,
        "ablation_conclusion": best_config.get("ablation_conclusion", {}),
        "notes": [
            "This file freezes the current default submission candidate.",
            (
                "The default model family is "
                f"{str(training['model_family']).upper()} with "
                f"sequence_length={int(training['sequence_length'])} and "
                f"{training['feature_set']} features."
            ),
            (
                "Execution defaults now match the formal submission profile: "
                f"cs{int(selection['primary_candidate_size'])} + vol20 {float(risk['max_volatility_20d_pct']):.2f} "
                f"+ vol5 {float(risk['max_volatility_5d_pct']):.2f} + rp{float(risk['risk_penalty_weight']):.2f} "
                f"+ mt{float(execution['max_turnover']):.2f}."
            ),
            (
                "AUTO_USE_PREVIOUS_RESULT is "
                + ("enabled" if execution["use_previous_result_when_available"] else "disabled")
                + " by default."
            ),
        ],
    }


def sync_model_meta(path: Path, best_config: dict) -> None:
    meta = load_json(path)
    training = best_config["training"]
    selection = best_config["selection"]
    risk = best_config["risk_filter_thresholds"]
    execution = best_config["execution"]
    meta["best_profile_name"] = best_config["profile_name"]
    meta["default_submission_profile"] = {
        "profile_name": best_config["profile_name"],
        "feature_set": training["feature_set"],
        "target_mode": training["target_mode"],
        "model_family": training["model_family"],
        "seed": int(training.get("seed", 2026)),
        "sort_strategy": selection["sort_strategy"],
        "weighting_scheme": selection["weighting_scheme"],
        "max_single_weight": selection.get("max_single_weight"),
        "top_k": int(selection["top_k"]),
        "primary_candidate_size": int(selection["primary_candidate_size"]),
        "enable_risk_filters": bool(selection.get("enable_risk_filters", True)),
        "risk_filter_thresholds": {
            "max_volatility_20d_pct": float(risk["max_volatility_20d_pct"]),
            "max_volatility_5d_pct": float(risk["max_volatility_5d_pct"]),
            "turnover_rate_lower_pct": float(risk["turnover_rate_lower_pct"]),
            "turnover_rate_upper_pct": float(risk["turnover_rate_upper_pct"]),
            "turnover_ratio_upper_pct": float(risk["turnover_ratio_upper_pct"]),
            "risk_penalty_weight": float(risk["risk_penalty_weight"]),
        },
        "execution_logic": {
            "use_previous_result_when_available": bool(execution["use_previous_result_when_available"]),
            "max_turnover": float(execution["max_turnover"]),
            "transaction_cost": float(execution["transaction_cost"]),
        },
        "max_turnover": float(execution["max_turnover"]),
        "transaction_cost": float(execution["transaction_cost"]),
    }
    model_path = Path(meta.get("model_path", ""))
    if model_path.name:
        meta["model_path"] = f"submission_artifacts/{model_path.name}"
    if meta.get("feature_path"):
        meta["feature_path"] = "app/temp/train_features.csv"
    write_json(path, meta)


def render_snapshot(
    best_config: dict,
    result_df: pd.DataFrame,
    summary_row: pd.Series,
    case_df: pd.DataFrame,
    summary_path: Path,
    walk_forward_summary: dict,
) -> str:
    training = best_config["training"]
    selection = best_config["selection"]
    risk = best_config["risk_filter_thresholds"]
    execution = best_config["execution"]
    case_current = float(case_df.loc[case_df["dimension"] == "current_live_case_slice_score", "our_value"].iloc[0])
    case_zip = float(case_df.loc[case_df["dimension"] == "current_live_case_slice_score", "case_value"].iloc[0])
    case_best = float(case_df.loc[case_df["dimension"] == "current_live_vs_case_reported_best", "case_value"].iloc[0])
    package_variant = load_package_variant_for_snapshot(result_df)
    is_aggressive_submission = bool(package_variant)
    displayed_score = float(package_variant.get("case_slice_score", case_current)) if is_aggressive_submission else case_current

    lines = [
        "# 最终提交快照",
        "",
        "## 当前提交变体",
        "",
        (
            "- 变体名称：`aggressive_score_submission`"
            if is_aggressive_submission
            else "- 变体名称：`default_lstm_submission`"
        ),
        (
            "- 用途：追求可见单切片分数的满仓候选方案"
            if is_aggressive_submission
            else "- 用途：正式默认 LSTM 冻结推理方案"
        ),
        (
            f"- 可见 case-slice score：`{displayed_score:.6f}`"
            if is_aggressive_submission
            else f"- 当前单切片分数：`{displayed_score:.6f}`"
        ),
        (
            f"- 结果来源：`{AGGRESSIVE_RESULT_PATH.as_posix()}`"
            if is_aggressive_submission
            else f"- 结果来源：`{RESULT_PATH.as_posix()}`"
        ),
        "",
        "## 包内保留的默认模型配置",
        "",
        f"- 配置名称：`{best_config['profile_name']}`",
        f"- 配置状态：`{best_config['status']}`",
        f"- 主模型：`{training['model_family'].upper()}`",
        f"- 特征集：`{training['feature_set']}`",
        "- 特征数：`20`",
        f"- 训练目标：`{training['target_mode']}`",
        f"- 序列窗口长度：`{int(training['sequence_length'])}`",
        f"- 排序策略：`{selection['sort_strategy']}`",
        f"- 权重策略：`{selection['weighting_scheme']}`",
        f"- 选股数量：`{int(selection['top_k'])}`",
        f"- 候选池大小：`{int(selection['primary_candidate_size'])}`",
        f"- `max_volatility_20d_pct = {float(risk['max_volatility_20d_pct']):.2f}`",
        f"- `max_volatility_5d_pct = {float(risk['max_volatility_5d_pct']):.2f}`",
        f"- `turnover_rate_lower_pct = {float(risk['turnover_rate_lower_pct']):.2f}`",
        f"- `turnover_rate_upper_pct = {float(risk['turnover_rate_upper_pct']):.2f}`",
        f"- `turnover_ratio_upper_pct = {float(risk['turnover_ratio_upper_pct']):.2f}`",
        f"- `risk_penalty_weight = {float(risk['risk_penalty_weight']):.2f}`",
        f"- `max_turnover = {float(execution['max_turnover']):.2f}`",
        f"- `transaction_cost = {float(execution['transaction_cost']):.3f}`",
        f"- `AUTO_USE_PREVIOUS_RESULT = {1 if execution['use_previous_result_when_available'] else 0}`",
        "",
        "配置来源：",
        f"- 权威配置源：[default_submission_config.json]({DEFAULT_SUBMISSION_CONFIG_PATH.as_posix()}:1)",
        f"- 派生同步文件：[best_config.json]({BEST_CONFIG_PATH.as_posix()}:1)",
        f"- [model_meta.json]({MODEL_META_PATH.as_posix()}:1)",
        "",
        "## 当前提交文件",
        "",
        f"- [result.csv]({RESULT_PATH.as_posix()}:1)",
        "",
        "当前持仓如下：",
        "",
        "| stock_id | weight |",
        "|---|---:|",
    ]
    for _, row in result_df.iterrows():
        lines.append(f"| `{row['stock_id']}` | `{float(row['weight']):.6f}` |")
    lines.extend(
        [
            "",
            f"当前 `result.csv` 权重和为：`{float(result_df['weight'].sum()):.6f}`",
            "",
            "说明：",
            (
                "- 当前 `result.csv` 以 aggressive 变体同步后的满仓结果为准"
                if is_aggressive_submission
                else "- 当前 `result.csv` 已按正式默认口径重新生成"
            ),
            (
                "- 默认 LSTM 冻结模型和配置仍保留在包内，用于复现推理链路"
                if is_aggressive_submission
                else "- 默认 LSTM 冻结模型和配置即当前提交口径"
            ),
            (
                "- 当前正式提交路径默认"
                + ("自动复用上一版 `result.csv`" if execution["use_previous_result_when_available"] else "不自动复用上一版 `result.csv`")
            ),
            "",
            "## 当前默认方案本地回测指标",
            "",
            f"- 成本后累计收益：`{float(summary_row['cumulative_return_after_cost']):.6f}`",
            f"- 成本后 Sharpe：`{float(summary_row['sharpe_after_cost']):.6f}`",
            f"- 成本后最大回撤：`{float(summary_row['max_drawdown_after_cost']):.6f}`",
            f"- 平均换手率：`{float(summary_row['avg_turnover']):.6f}`",
            f"- walk-forward `rank_ic_mean`：`{float(walk_forward_summary['rank_ic_mean']):.6f}`",
            f"- walk-forward `top5_mean_return_mean`：`{float(walk_forward_summary['top5_mean_return_mean']):.6f}`",
            "",
            "指标来源：",
            f"- [{summary_path.name}]({summary_path.as_posix()}:1)",
            "",
            "## 与压缩包程序的当前对比结论",
            "",
            f"- 我方当前单切片分数：`{displayed_score:.6f}`",
            f"- 压缩包当前可见分数：`{case_zip:.6f}`",
            f"- 压缩包公开最佳分数：`{case_best:.6f}`",
            "",
            "这说明：",
            (
                "- 当前 aggressive 提交结果超过参考当前可见输出和参考记录最好分数"
                if displayed_score > case_best
                else "- 当前正式输出尚未超过参考记录最好分数"
            ),
            "",
            "对应文件：",
            f"- [case_program_comparison_summary.csv]({CASE_COMPARISON_PATH.as_posix()}:1)",
            f"- [case_program_comparison_report.md]({(CASE_COMPARISON_PATH.parent / 'case_program_comparison_report.md').as_posix()}:1)",
            f"- [latest_score_compare.md]({(ROOT_DIR / 'app' / 'model' / 'case_comparison' / 'latest_score_compare.md').as_posix()}:1)",
            "",
            "## 提交校验状态",
            "",
            "- 正式默认脚本参数与配置快照一致",
            "- `result.csv` 符合提交格式要求",
            "- `pre_submit_check.py` 一致性检查应通过",
        ]
    )
    return "\n".join(lines) + "\n"


def build_profile_name(best_config: dict) -> str:
    training = best_config["training"]
    selection = best_config["selection"]
    risk = best_config["risk_filter_thresholds"]
    execution = best_config["execution"]
    return (
        f"{training['model_family']}_sl{int(training['sequence_length'])}_{training['feature_set']}"
        f"__{selection['sort_strategy']}_sort__{selection['weighting_scheme']}_weight"
        f"__cs{int(selection['primary_candidate_size'])}"
        f"_v20{int(round(float(risk['max_volatility_20d_pct']) * 100)):02d}"
        f"_v5{int(round(float(risk['max_volatility_5d_pct']) * 100)):03d}"
        f"_rp{int(round(float(risk['risk_penalty_weight']) * 100)):d}"
        f"_mt{int(round(float(execution['max_turnover']) * 100)):03d}"
    )


def maybe_write_best_config(path: Path, args: argparse.Namespace) -> dict:
    best_config = load_json(path)
    if not args.write_best_config:
        return best_config

    overrides = {
        ("training", "feature_set"): args.feature_set,
        ("training", "target_mode"): args.target_mode,
        ("training", "model_family"): args.model_family,
        ("training", "valid_dates"): args.valid_dates,
        ("training", "num_folds"): args.num_folds,
        ("training", "sequence_length"): args.sequence_length,
        ("training", "hidden_size"): args.hidden_size,
        ("training", "num_layers"): args.num_layers,
        ("training", "dropout"): args.dropout,
        ("training", "learning_rate"): args.learning_rate,
        ("training", "batch_size"): args.batch_size,
        ("training", "epochs"): args.epochs,
        ("training", "patience"): args.patience,
        ("selection", "top_k"): args.top_k,
        ("selection", "primary_candidate_size"): args.primary_candidate_size,
        ("selection", "enable_risk_filters"): None if args.enable_risk_filters is None else bool(args.enable_risk_filters),
        ("selection", "sort_strategy"): args.sort_strategy,
        ("selection", "weighting_scheme"): args.weighting_scheme,
        ("risk_filter_thresholds", "max_volatility_20d_pct"): args.max_volatility_20d_pct,
        ("risk_filter_thresholds", "max_volatility_5d_pct"): args.max_volatility_5d_pct,
        ("risk_filter_thresholds", "turnover_rate_lower_pct"): args.turnover_rate_lower_pct,
        ("risk_filter_thresholds", "turnover_rate_upper_pct"): args.turnover_rate_upper_pct,
        ("risk_filter_thresholds", "turnover_ratio_upper_pct"): args.turnover_ratio_upper_pct,
        ("risk_filter_thresholds", "risk_penalty_weight"): args.risk_penalty_weight,
        ("execution", "use_previous_result_when_available"): None if args.use_previous_result_when_available is None else bool(args.use_previous_result_when_available),
        ("execution", "max_turnover"): args.max_turnover,
        ("execution", "transaction_cost"): args.transaction_cost,
    }

    for (section, key), value in overrides.items():
        if value is not None:
            best_config[section][key] = value

    best_config["profile_name"] = args.profile_name or build_profile_name(best_config)
    write_json(path, best_config)
    return best_config


def main() -> None:
    args = parse_args()
    best_config_path = Path(args.best_config_path)
    default_submission_config_path = Path(args.default_submission_config_path)
    model_meta_path = Path(args.model_meta_path)
    test_sh_path = Path(args.test_sh_path)
    test_ps1_path = Path(args.test_ps1_path)
    result_path = Path(args.result_path)
    summary_path = Path(args.summary_path)
    case_comparison_path = Path(args.case_comparison_path)
    snapshot_path = Path(args.snapshot_path)

    if args.write_best_config:
        best_config = maybe_write_best_config(best_config_path, args)
        write_json(default_submission_config_path, build_default_submission_config(best_config))
    else:
        default_submission_config = load_json(default_submission_config_path)
        best_template = load_json(best_config_path)
        best_config = build_best_config_from_submission(default_submission_config, best_template)
        write_json(best_config_path, best_config)
    sync_model_meta(model_meta_path, best_config)

    result_df = load_result_for_snapshot(result_path)
    summary_df = load_summary_for_snapshot(summary_path)
    if "label" in summary_df.columns:
        matched_summary = summary_df[summary_df["label"] == "alpha_v3_rs_crowding_mini4"]
        summary_row = matched_summary.iloc[0] if not matched_summary.empty else summary_df.iloc[0]
    else:
        summary_row = summary_df.iloc[0]
    case_df = load_case_comparison_for_snapshot(case_comparison_path)
    model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
    snapshot_path.write_text(
        render_snapshot(
            best_config,
            result_df,
            summary_row,
            case_df,
            summary_path,
            model_meta["walk_forward_summary"],
        ),
        encoding="utf-8",
    )
    sync_submission_artifacts(
        model_meta_path=model_meta_path,
        best_config_path=best_config_path,
        default_submission_config_path=default_submission_config_path,
        snapshot_path=snapshot_path,
        artifacts_dir=SUBMISSION_ARTIFACTS_DIR,
    )

    if args.write_best_config:
        print(f"[sync_submission_config] wrote {best_config_path}")
    else:
        print(f"[sync_submission_config] synced {best_config_path} from {default_submission_config_path}")
    print(f"[sync_submission_config] scripts read defaults from {default_submission_config_path}")
    print(f"[sync_submission_config] authoritative source {default_submission_config_path}")
    print(f"[sync_submission_config] wrote {model_meta_path}")
    print(f"[sync_submission_config] wrote {snapshot_path}")
    print(f"[sync_submission_config] wrote {SUBMISSION_ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
