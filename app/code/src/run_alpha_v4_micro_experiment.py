import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from config import BEST_CONFIG


ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = ROOT_DIR / "app"
SRC_DIR = APP_DIR / "code" / "src"
DATA_DIR = APP_DIR / "data"
TEMP_DIR = APP_DIR / "temp"
MODEL_DIR = APP_DIR / "model"
CASE_DIR = ROOT_DIR / "_case_zip" / "THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0"

OUTPUT_DIR = MODEL_DIR / "alpha_v4_micro_experiment"
CANDIDATE_MODEL_DIR = MODEL_DIR / "alpha_v4_micro_candidate"
CANDIDATE_RESULT_PATH = APP_DIR / "output" / "result_alpha_v4_micro.csv"

BASELINE_LABEL = "current_mini4_default"
CANDIDATE_LABEL = "mini4_plus_alpha_v4_micro"
CANDIDATE_FEATURE_SET = "base_alpha_v3_rs_crowding_mini4_alpha_v4_micro"


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=ROOT_DIR)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score_result_against_case_test(result_path: Path) -> float:
    test_df = pd.read_csv(CASE_DIR / "data" / "test.csv")
    code_col = test_df.columns[0]
    open_col = test_df.columns[2]

    result_df = pd.read_csv(result_path, dtype={"stock_id": str})
    result_df["stock_id"] = result_df["stock_id"].astype(str).str.zfill(6)

    filtered = test_df[test_df[code_col].astype(str).str.zfill(6).isin(result_df["stock_id"])].copy()
    filtered[code_col] = filtered[code_col].astype(str).str.zfill(6)
    filtered = filtered.groupby(code_col).tail(5)
    returns = filtered.groupby(code_col).apply(
        lambda g: (g.iloc[-1][open_col] - g.iloc[0][open_col]) / g.iloc[0][open_col]
    ).reset_index()
    returns.columns = ["stock_id", "ret"]
    merged = returns.merge(result_df, on="stock_id", how="inner")
    return float((merged["ret"] * merged["weight"]).sum())


def build_backtest_cmd(prediction_path: Path, feature_path: Path, model_dir: Path, output_dir: Path) -> list[str]:
    selection = BEST_CONFIG["selection"]
    risk = BEST_CONFIG["risk_filter_thresholds"]
    execution = BEST_CONFIG["execution"]
    return [
        sys.executable,
        str(SRC_DIR / "backtest.py"),
        "--prediction_path",
        str(prediction_path),
        "--feature_path",
        str(feature_path),
        "--model_dir",
        str(model_dir),
        "--output_dir",
        str(output_dir),
        "--top_k",
        str(selection["top_k"]),
        "--primary_candidate_size",
        str(selection["primary_candidate_size"]),
        "--enable_risk_filters",
        "1",
        "--allow_cash_fallback",
        "0",
        "--max_volatility_20d_pct",
        str(risk["max_volatility_20d_pct"]),
        "--max_volatility_5d_pct",
        str(risk["max_volatility_5d_pct"]),
        "--turnover_rate_lower_pct",
        str(risk["turnover_rate_lower_pct"]),
        "--turnover_rate_upper_pct",
        str(risk["turnover_rate_upper_pct"]),
        "--turnover_ratio_upper_pct",
        str(risk["turnover_ratio_upper_pct"]),
        "--risk_penalty_weight",
        str(risk["risk_penalty_weight"]),
        "--weighting_scheme",
        str(selection["weighting_scheme"]),
        "--sort_strategy",
        str(selection["sort_strategy"]),
        "--transaction_cost",
        str(execution["transaction_cost"]),
        "--max_turnover",
        str(execution["max_turnover"]),
    ]


def should_retain_candidate(delta_slice: float, delta_cum: float, delta_sharpe: float, delta_mdd: float) -> tuple[bool, str]:
    checks = [
        ("case_slice_score_up", delta_slice > 0.0),
        ("local_cumulative_not_worse", delta_cum >= -0.02),
        ("local_sharpe_not_worse", delta_sharpe >= -0.20),
        ("drawdown_not_materially_worse", delta_mdd >= -0.01),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        return False, " / ".join(failed)
    return True, "all_gate_checks_passed"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            sys.executable,
            str(SRC_DIR / "featurework.py"),
            "--mode",
            "train",
            "--data_dir",
            str(DATA_DIR),
            "--temp_dir",
            str(TEMP_DIR),
        ]
    )
    run_cmd(
        [
            sys.executable,
            str(SRC_DIR / "featurework.py"),
            "--mode",
            "predict",
            "--data_dir",
            str(DATA_DIR),
            "--temp_dir",
            str(TEMP_DIR),
        ]
    )

    run_cmd(
        [
            sys.executable,
            str(SRC_DIR / "train_lstm.py"),
            "--feature_path",
            str(TEMP_DIR / "train_features.csv"),
            "--model_dir",
            str(CANDIDATE_MODEL_DIR),
            "--feature_set",
            CANDIDATE_FEATURE_SET,
            "--sequence_length",
            str(BEST_CONFIG["training"]["sequence_length"]),
            "--epochs",
            str(BEST_CONFIG["training"]["epochs"]),
            "--patience",
            str(BEST_CONFIG["training"]["patience"]),
            "--batch_size",
            str(BEST_CONFIG["training"]["batch_size"]),
            "--learning_rate",
            str(BEST_CONFIG["training"]["learning_rate"]),
            "--hidden_size",
            str(BEST_CONFIG["training"]["hidden_size"]),
            "--num_layers",
            str(BEST_CONFIG["training"]["num_layers"]),
            "--dropout",
            str(BEST_CONFIG["training"]["dropout"]),
            "--valid_dates",
            str(BEST_CONFIG["training"]["valid_dates"]),
            "--num_folds",
            str(BEST_CONFIG["training"]["num_folds"]),
            "--target_mode",
            str(BEST_CONFIG["training"]["target_mode"]),
        ]
    )

    selection = BEST_CONFIG["selection"]
    risk = BEST_CONFIG["risk_filter_thresholds"]
    execution = BEST_CONFIG["execution"]
    run_cmd(
        [
            sys.executable,
            str(SRC_DIR / "test_lstm.py"),
            "--feature_path",
            str(TEMP_DIR / "predict_features.csv"),
            "--model_dir",
            str(CANDIDATE_MODEL_DIR),
            "--output_path",
            str(CANDIDATE_RESULT_PATH),
            "--top_k",
            str(selection["top_k"]),
            "--primary_candidate_size",
            str(selection["primary_candidate_size"]),
            "--max_volatility_20d_pct",
            str(risk["max_volatility_20d_pct"]),
            "--max_volatility_5d_pct",
            str(risk["max_volatility_5d_pct"]),
            "--turnover_rate_lower_pct",
            str(risk["turnover_rate_lower_pct"]),
            "--turnover_rate_upper_pct",
            str(risk["turnover_rate_upper_pct"]),
            "--turnover_ratio_upper_pct",
            str(risk["turnover_ratio_upper_pct"]),
            "--risk_penalty_weight",
            str(risk["risk_penalty_weight"]),
            "--sort_strategy",
            str(selection["sort_strategy"]),
            "--weighting_scheme",
            str(selection["weighting_scheme"]),
            "--max_turnover",
            str(execution["max_turnover"]),
            "--previous_result_path",
            str(APP_DIR / "output" / "result.csv"),
            "--score_output_path",
            str(CANDIDATE_MODEL_DIR / "predict_scores.csv"),
        ]
    )

    candidate_backtest_dir = CANDIDATE_MODEL_DIR / "backtest"
    run_cmd(
        build_backtest_cmd(
            prediction_path=CANDIDATE_MODEL_DIR / "walk_forward_predictions.csv",
            feature_path=TEMP_DIR / "train_features.csv",
            model_dir=CANDIDATE_MODEL_DIR,
            output_dir=candidate_backtest_dir,
        )
    )

    baseline_meta = load_json(MODEL_DIR / "model_meta.json")
    baseline_bt = pd.read_csv(MODEL_DIR / "alpha_v3_rs_crowding_mini4" / "backtest" / "backtest_summary.csv").iloc[0]
    baseline_slice_score = score_result_against_case_test(APP_DIR / "output" / "result.csv")

    candidate_meta = load_json(CANDIDATE_MODEL_DIR / "model_meta.json")
    candidate_bt = pd.read_csv(candidate_backtest_dir / "backtest_summary.csv").iloc[0]
    candidate_slice_score = score_result_against_case_test(CANDIDATE_RESULT_PATH)

    summary = pd.DataFrame(
        [
            {
                "label": BASELINE_LABEL,
                "feature_set": baseline_meta["feature_set"],
                "feature_count": len(baseline_meta["feature_columns"]),
                "rank_ic_mean": baseline_meta["walk_forward_summary"]["rank_ic_mean"],
                "top5_mean_return_mean": baseline_meta["walk_forward_summary"]["top5_mean_return_mean"],
                "cumulative_return_after_cost": baseline_bt["cumulative_return_after_cost"],
                "sharpe_after_cost": baseline_bt["sharpe_after_cost"],
                "max_drawdown_after_cost": baseline_bt["max_drawdown_after_cost"],
                "avg_turnover": baseline_bt["avg_turnover"],
                "score_self_case_slice": baseline_slice_score,
            },
            {
                "label": CANDIDATE_LABEL,
                "feature_set": candidate_meta["feature_set"],
                "feature_count": len(candidate_meta["feature_columns"]),
                "rank_ic_mean": candidate_meta["walk_forward_summary"]["rank_ic_mean"],
                "top5_mean_return_mean": candidate_meta["walk_forward_summary"]["top5_mean_return_mean"],
                "cumulative_return_after_cost": candidate_bt["cumulative_return_after_cost"],
                "sharpe_after_cost": candidate_bt["sharpe_after_cost"],
                "max_drawdown_after_cost": candidate_bt["max_drawdown_after_cost"],
                "avg_turnover": candidate_bt["avg_turnover"],
                "score_self_case_slice": candidate_slice_score,
            },
        ]
    )

    baseline_row = summary.iloc[0]
    candidate_row = summary.iloc[1]
    delta_slice = float(candidate_row["score_self_case_slice"] - baseline_row["score_self_case_slice"])
    delta_cum = float(candidate_row["cumulative_return_after_cost"] - baseline_row["cumulative_return_after_cost"])
    delta_sharpe = float(candidate_row["sharpe_after_cost"] - baseline_row["sharpe_after_cost"])
    delta_mdd = float(candidate_row["max_drawdown_after_cost"] - baseline_row["max_drawdown_after_cost"])
    retain_candidate, retain_reason = should_retain_candidate(delta_slice, delta_cum, delta_sharpe, delta_mdd)

    decision = pd.DataFrame(
        [
            {
                "retain_candidate": int(retain_candidate),
                "retain_reason": retain_reason,
                "delta_case_slice_score": delta_slice,
                "delta_cumulative_return_after_cost": delta_cum,
                "delta_sharpe_after_cost": delta_sharpe,
                "delta_max_drawdown_after_cost": delta_mdd,
            }
        ]
    )

    summary_path = OUTPUT_DIR / "alpha_v4_micro_summary.csv"
    decision_path = OUTPUT_DIR / "alpha_v4_micro_decision.csv"
    report_path = OUTPUT_DIR / "alpha_v4_micro_report.md"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    decision.to_csv(decision_path, index=False, encoding="utf-8-sig")

    lines = [
        "# alpha_v4_micro 实验报告",
        "",
        "## 实验目的",
        "",
        "- 在当前 mini4 默认方案上追加一组小特征 `alpha_v4_micro`；",
        "- 同时比较压缩包单切片分数与本地多期回测表现；",
        "- 只保留“切片分数上升且本地收益不坏”的组合。",
        "",
        "## 特征定义",
        "",
        "- `rel_strength_accel_5d`：相对沪深 300 强弱加速度",
        "- `trend_persistence_score_10d`：趋势持续性综合分数",
        "- `volatility_compression_breakout_20d`：低波动压缩下的突破状态",
        "- `crowding_reversal_risk_5d`：拥挤交易后的反转风险",
        "",
        "## 对比结果",
        "",
        "| 方案 | 特征集 | 特征数 | rank_ic_mean | top5_mean_return_mean | cumulative_return_after_cost | sharpe_after_cost | max_drawdown_after_cost | avg_turnover | case_slice_score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['feature_set']} | {int(row['feature_count'])} | "
            f"{row['rank_ic_mean']:.6f} | {row['top5_mean_return_mean']:.6f} | "
            f"{row['cumulative_return_after_cost']:.6f} | {row['sharpe_after_cost']:.6f} | "
            f"{row['max_drawdown_after_cost']:.6f} | {row['avg_turnover']:.6f} | {row['score_self_case_slice']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 关键差值",
            "",
            f"- 切片分数变化：`{delta_slice:.6f}`",
            f"- 成本后累计收益变化：`{delta_cum:.6f}`",
            f"- 成本后夏普变化：`{delta_sharpe:.6f}`",
            f"- 成本后最大回撤变化：`{delta_mdd:.6f}`",
            "",
            "## 保留判断",
            "",
            f"- retain_candidate: `{retain_candidate}`",
            f"- retain_reason: `{retain_reason}`",
        ]
    )
    if retain_candidate:
        lines.extend(
            [
                "",
                "结论：",
                "",
                "- 这组 alpha_v4_micro 满足“切片分数上升且本地收益不坏”的门槛。",
                "- 候选方案建议进入下一轮正式默认版审查，但当前脚本不会自动切换默认配置。",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "结论：",
                "",
                "- 这组 alpha_v4_micro 没有同时满足双目标门槛。",
                "- 当前正式默认方案保持不变，候选结果仅保留为实验记录。",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[alpha_v4_micro] wrote {summary_path}")
    print(f"[alpha_v4_micro] wrote {decision_path}")
    print(f"[alpha_v4_micro] wrote {report_path}")
    print(f"[alpha_v4_micro] retain_candidate={retain_candidate} reason={retain_reason}")


if __name__ == "__main__":
    main()
