import json
import subprocess
import sys
from itertools import combinations
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

OUTPUT_DIR = MODEL_DIR / "alpha_v4_micro_ablation"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

BASELINE_LABEL = "current_mini4_default"
BASELINE_FEATURE_SET = "base_alpha_v3_rs_crowding_mini4"
ALPHA_V4_COLUMNS = [
    "rel_strength_accel_5d",
    "trend_persistence_score_10d",
    "volatility_compression_breakout_20d",
    "crowding_reversal_risk_5d",
]


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


def build_test_cmd(model_dir: Path, output_path: Path) -> list[str]:
    selection = BEST_CONFIG["selection"]
    risk = BEST_CONFIG["risk_filter_thresholds"]
    execution = BEST_CONFIG["execution"]
    return [
        sys.executable,
        str(SRC_DIR / "test_lstm.py"),
        "--feature_path",
        str(TEMP_DIR / "predict_features.csv"),
        "--model_dir",
        str(model_dir),
        "--output_path",
        str(output_path),
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
    ]


def build_train_cmd(feature_set: str, model_dir: Path) -> list[str]:
    training = BEST_CONFIG["training"]
    return [
        sys.executable,
        str(SRC_DIR / "train_lstm.py"),
        "--feature_path",
        str(TEMP_DIR / "train_features.csv"),
        "--model_dir",
        str(model_dir),
        "--feature_set",
        feature_set,
        "--sequence_length",
        str(training["sequence_length"]),
        "--epochs",
        str(training["epochs"]),
        "--patience",
        str(training["patience"]),
        "--batch_size",
        str(training["batch_size"]),
        "--learning_rate",
        str(training["learning_rate"]),
        "--hidden_size",
        str(training["hidden_size"]),
        "--num_layers",
        str(training["num_layers"]),
        "--dropout",
        str(training["dropout"]),
        "--valid_dates",
        str(training["valid_dates"]),
        "--num_folds",
        str(training["num_folds"]),
        "--target_mode",
        str(training["target_mode"]),
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


def feature_alias(columns: tuple[str, ...]) -> str:
    short_map = {
        "rel_strength_accel_5d": "rs_accel",
        "trend_persistence_score_10d": "trend_persist",
        "volatility_compression_breakout_20d": "vol_compress_breakout",
        "crowding_reversal_risk_5d": "crowding_reversal",
    }
    return "__".join(short_map[col] for col in columns)


def build_experiments() -> list[dict]:
    experiments: list[dict] = [
        {
            "label": BASELINE_LABEL,
            "feature_set": BASELINE_FEATURE_SET,
            "columns": (),
            "skip_train": True,
            "notes": "当前正式默认 mini4 基线。",
            "model_dir": MODEL_DIR,
            "result_path": APP_DIR / "output" / "result.csv",
            "backtest_summary_path": MODEL_DIR / "alpha_v3_rs_crowding_mini4" / "backtest" / "backtest_summary.csv",
        }
    ]

    for size in [1, 2]:
        for cols in combinations(ALPHA_V4_COLUMNS, size):
            alias = feature_alias(cols)
            feature_set = "base_alpha_v3_rs_crowding_mini4__" + "__".join(cols)
            model_dir = MODELS_DIR / alias
            result_path = RESULTS_DIR / f"{alias}.csv"
            experiments.append(
                {
                    "label": alias,
                    "feature_set": feature_set,
                    "columns": cols,
                    "skip_train": False,
                    "notes": f"mini4 + {' + '.join(cols)}",
                    "model_dir": model_dir,
                    "result_path": result_path,
                    "backtest_summary_path": model_dir / "backtest" / "backtest_summary.csv",
                }
            )

    experiments.append(
        {
            "label": "alpha_v4_micro_full",
            "feature_set": "base_alpha_v3_rs_crowding_mini4_alpha_v4_micro",
            "columns": tuple(ALPHA_V4_COLUMNS),
            "skip_train": False,
            "notes": "mini4 + alpha_v4_micro 全组合",
            "model_dir": MODELS_DIR / "alpha_v4_micro_full",
            "result_path": RESULTS_DIR / "alpha_v4_micro_full.csv",
            "backtest_summary_path": MODELS_DIR / "alpha_v4_micro_full" / "backtest" / "backtest_summary.csv",
        }
    )
    return experiments


def collect_row(exp: dict, baseline_row: dict | None = None) -> dict:
    meta = load_json(exp["model_dir"] / "model_meta.json") if (exp["model_dir"] / "model_meta.json").exists() else {}
    bt = pd.read_csv(exp["backtest_summary_path"]).iloc[0] if exp["backtest_summary_path"].exists() else None

    row = {
        "label": exp["label"],
        "feature_set": meta.get("feature_set", exp["feature_set"]),
        "feature_count": len(meta.get("feature_columns", [])) if meta else "",
        "added_columns": "|".join(exp["columns"]) if exp["columns"] else "",
        "rank_ic_mean": float(meta["walk_forward_summary"]["rank_ic_mean"]) if meta else "",
        "top5_mean_return_mean": float(meta["walk_forward_summary"]["top5_mean_return_mean"]) if meta else "",
        "cumulative_return_after_cost": float(bt["cumulative_return_after_cost"]) if bt is not None else "",
        "sharpe_after_cost": float(bt["sharpe_after_cost"]) if bt is not None else "",
        "max_drawdown_after_cost": float(bt["max_drawdown_after_cost"]) if bt is not None else "",
        "avg_turnover": float(bt["avg_turnover"]) if bt is not None else "",
        "score_self_case_slice": score_result_against_case_test(exp["result_path"]) if exp["result_path"].exists() else "",
        "notes": exp["notes"],
    }

    if baseline_row is None:
        row["delta_case_slice_score"] = 0.0
        row["delta_cumulative_return_after_cost"] = 0.0
        row["delta_sharpe_after_cost"] = 0.0
        row["delta_max_drawdown_after_cost"] = 0.0
        row["retain_candidate"] = 0
        row["retain_reason"] = "baseline"
        return row

    row["delta_case_slice_score"] = row["score_self_case_slice"] - baseline_row["score_self_case_slice"]
    row["delta_cumulative_return_after_cost"] = row["cumulative_return_after_cost"] - baseline_row["cumulative_return_after_cost"]
    row["delta_sharpe_after_cost"] = row["sharpe_after_cost"] - baseline_row["sharpe_after_cost"]
    row["delta_max_drawdown_after_cost"] = row["max_drawdown_after_cost"] - baseline_row["max_drawdown_after_cost"]
    keep, reason = should_retain_candidate(
        row["delta_case_slice_score"],
        row["delta_cumulative_return_after_cost"],
        row["delta_sharpe_after_cost"],
        row["delta_max_drawdown_after_cost"],
    )
    row["retain_candidate"] = int(keep)
    row["retain_reason"] = reason
    return row


def fmt(value) -> str:
    if value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(summary_df: pd.DataFrame, report_path: Path) -> None:
    baseline = summary_df[summary_df["label"] == BASELINE_LABEL].iloc[0]
    candidates = summary_df[summary_df["label"] != BASELINE_LABEL].copy()
    retained = candidates[candidates["retain_candidate"] == 1].copy()
    ranked_slice = candidates.sort_values(
        ["delta_case_slice_score", "cumulative_return_after_cost"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranked_retained = retained.sort_values(
        ["delta_case_slice_score", "cumulative_return_after_cost", "sharpe_after_cost"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    lines = [
        "# alpha_v4_micro 子特征细粒度消融报告",
        "",
        "## 实验目标",
        "",
        "- 在当前 mini4 默认方案上，拆开 `alpha_v4_micro` 的 4 个子特征；",
        "- 系统测试单特征与两两组合；",
        "- 只保留“压缩包单切片分数提升，且本地收益不坏”的候选。",
        "",
        "## 基线",
        "",
        f"- 基线方案：`{baseline['feature_set']}`",
        f"- 基线切片分数：`{baseline['score_self_case_slice']:.6f}`",
        f"- 基线成本后累计收益：`{baseline['cumulative_return_after_cost']:.6f}`",
        f"- 基线夏普：`{baseline['sharpe_after_cost']:.6f}`",
        f"- 基线最大回撤：`{baseline['max_drawdown_after_cost']:.6f}`",
        "",
        "## 关键结论",
        "",
    ]

    if ranked_retained.empty:
        lines.extend(
            [
                "- 这轮单特征/两两组合里，没有组合同时满足保留门槛。",
                "- 说明 `alpha_v4_micro` 的切片提升信号是真存在的，但当前写法仍偏激进。",
            ]
        )
    else:
        best = ranked_retained.iloc[0]
        lines.extend(
            [
                f"- 最优保留组合：`{best['label']}`",
                f"- 切片分数提升：`{best['delta_case_slice_score']:.6f}`",
                f"- 成本后累计收益变化：`{best['delta_cumulative_return_after_cost']:.6f}`",
                f"- 夏普变化：`{best['delta_sharpe_after_cost']:.6f}`",
                f"- 最大回撤变化：`{best['delta_max_drawdown_after_cost']:.6f}`",
            ]
        )
    if not ranked_slice.empty:
        top_slice = ranked_slice.iloc[0]
        lines.extend(
            [
                f"- 切片分数提升最大的组合：`{top_slice['label']}`，提升 ` {top_slice['delta_case_slice_score']:.6f}`。",
                f"- 但它的本地累计收益变化为 `{top_slice['delta_cumulative_return_after_cost']:.6f}`，需要结合保留门槛一起看。",
            ]
        )

    lines.extend(
        [
            "",
            "## 完整对比表",
            "",
            "| 方案 | 新增特征 | slice分数 | Δslice | 累计收益 | Δ累计收益 | 夏普 | Δ夏普 | 最大回撤 | Δ回撤 | 保留 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['label']} | {fmt(row['added_columns'])} | {fmt(row['score_self_case_slice'])} | "
            f"{fmt(row['delta_case_slice_score'])} | {fmt(row['cumulative_return_after_cost'])} | "
            f"{fmt(row['delta_cumulative_return_after_cost'])} | {fmt(row['sharpe_after_cost'])} | "
            f"{fmt(row['delta_sharpe_after_cost'])} | {fmt(row['max_drawdown_after_cost'])} | "
            f"{fmt(row['delta_max_drawdown_after_cost'])} | {row['retain_reason']} |"
        )

    lines.extend(
        [
            "",
            "## 建议",
            "",
            "- 下一步优先围绕通过门槛的组合继续做更小范围微调；",
            "- 如果没有组合通过，就从切片提升明显但本地损伤较小的那 1 到 2 组继续改写特征定义，而不是直接保留现版本。",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments()

    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "train", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])
    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "predict", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])

    for exp in experiments:
        if exp["skip_train"]:
            continue
        exp["model_dir"].mkdir(parents=True, exist_ok=True)
        run_cmd(build_train_cmd(exp["feature_set"], exp["model_dir"]))
        run_cmd(build_test_cmd(exp["model_dir"], exp["result_path"]))
        run_cmd(
            build_backtest_cmd(
                prediction_path=exp["model_dir"] / "walk_forward_predictions.csv",
                feature_path=TEMP_DIR / "train_features.csv",
                model_dir=exp["model_dir"],
                output_dir=exp["model_dir"] / "backtest",
            )
        )

    baseline_exp = experiments[0]
    baseline_row = collect_row(baseline_exp, baseline_row=None)
    rows = [baseline_row]
    for exp in experiments[1:]:
        rows.append(collect_row(exp, baseline_row=baseline_row))

    summary_df = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "alpha_v4_micro_ablation_summary.csv"
    retained_path = OUTPUT_DIR / "alpha_v4_micro_ablation_retained.csv"
    report_path = OUTPUT_DIR / "alpha_v4_micro_ablation_report.md"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary_df[summary_df["retain_candidate"] == 1].to_csv(retained_path, index=False, encoding="utf-8-sig")
    write_report(summary_df, report_path)

    print(f"[alpha_v4_micro_ablation] wrote {summary_path}")
    print(f"[alpha_v4_micro_ablation] wrote {retained_path}")
    print(f"[alpha_v4_micro_ablation] wrote {report_path}")


if __name__ == "__main__":
    main()
