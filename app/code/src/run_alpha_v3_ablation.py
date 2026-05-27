import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = ROOT_DIR / "app"
SRC_DIR = APP_DIR / "code" / "src"
DATA_DIR = APP_DIR / "data"
TEMP_DIR = APP_DIR / "temp"
MODEL_DIR = APP_DIR / "model"
OUTPUT_DIR = MODEL_DIR / "alpha_v3_ablation"
CASE_DIR = ROOT_DIR / "_case_zip" / "THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0"

EXPERIMENTS = [
    {
        "label": "current_refined_default",
        "feature_set": "base",
        "model_dir": MODEL_DIR,
        "result_path": APP_DIR / "output" / "result.csv",
        "backtest_summary_path": MODEL_DIR / "current_refined_backtest" / "backtest_summary.csv",
        "skip_train": True,
        "notes": "当前正式默认方案基线。",
    },
    {
        "label": "alpha_v3_relative_strength",
        "feature_set": "base_alpha_v3_relative_strength",
        "model_dir": MODEL_DIR / "alpha_v3_ablation_relative_strength",
        "result_path": APP_DIR / "output" / "result_alpha_v3_relative_strength.csv",
        "backtest_summary_path": MODEL_DIR / "alpha_v3_ablation_relative_strength" / "backtest" / "backtest_summary.csv",
        "skip_train": False,
        "notes": "只测试相对强弱子特征。",
    },
    {
        "label": "alpha_v3_trend_persistence",
        "feature_set": "base_alpha_v3_trend_persistence",
        "model_dir": MODEL_DIR / "alpha_v3_ablation_trend_persistence",
        "result_path": APP_DIR / "output" / "result_alpha_v3_trend_persistence.csv",
        "backtest_summary_path": MODEL_DIR / "alpha_v3_ablation_trend_persistence" / "backtest" / "backtest_summary.csv",
        "skip_train": False,
        "notes": "只测试趋势持续子特征。",
    },
    {
        "label": "alpha_v3_crowding_risk",
        "feature_set": "base_alpha_v3_crowding_risk",
        "model_dir": MODEL_DIR / "alpha_v3_ablation_crowding_risk",
        "result_path": APP_DIR / "output" / "result_alpha_v3_crowding_risk.csv",
        "backtest_summary_path": MODEL_DIR / "alpha_v3_ablation_crowding_risk" / "backtest" / "backtest_summary.csv",
        "skip_train": False,
        "notes": "只测试拥挤风险子特征。",
    },
    {
        "label": "alpha_v3_selected5",
        "feature_set": "base_alpha_v3_selected5",
        "model_dir": MODEL_DIR / "alpha_v3_ablation_selected5",
        "result_path": APP_DIR / "output" / "result_alpha_v3_selected5.csv",
        "backtest_summary_path": MODEL_DIR / "alpha_v3_ablation_selected5" / "backtest" / "backtest_summary.csv",
        "skip_train": False,
        "notes": "优先押注的 5 个 alpha v3 子特征组合。",
    },
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
    if "stock_id" in result_df.columns:
        result_df = result_df.rename(columns={"stock_id": "code", "weight": "weight"})
    else:
        result_df.columns = ["code", "weight"]
    result_df["code"] = result_df["code"].astype(str).str.zfill(6)

    filtered = test_df[test_df[code_col].astype(str).str.zfill(6).isin(result_df["code"])].copy()
    filtered[code_col] = filtered[code_col].astype(str).str.zfill(6)
    filtered = filtered.groupby(code_col).tail(5)
    returns = filtered.groupby(code_col).apply(
        lambda g: (g.iloc[-1][open_col] - g.iloc[0][open_col]) / g.iloc[0][open_col]
    ).reset_index()
    returns.columns = ["code", "ret"]

    merged = returns.merge(result_df, on="code", how="inner")
    return float((merged["ret"] * merged["weight"]).sum())


def collect_row(exp: dict) -> dict:
    row = {
        "label": exp["label"],
        "feature_set": "",
        "feature_count": "",
        "rank_ic_mean": "",
        "top5_mean_return_mean": "",
        "cumulative_return_after_cost": "",
        "sharpe_after_cost": "",
        "max_drawdown_after_cost": "",
        "avg_turnover": "",
        "score_self_case_slice": "",
        "notes": exp["notes"],
    }

    if exp["model_dir"].exists() and (exp["model_dir"] / "model_meta.json").exists():
        meta = load_json(exp["model_dir"] / "model_meta.json")
        row["feature_set"] = meta.get("feature_set", "")
        row["feature_count"] = len(meta.get("feature_columns", []))
        row["rank_ic_mean"] = float(meta["walk_forward_summary"]["rank_ic_mean"])
        row["top5_mean_return_mean"] = float(meta["walk_forward_summary"]["top5_mean_return_mean"])

    if exp["backtest_summary_path"].exists():
        summary = pd.read_csv(exp["backtest_summary_path"]).iloc[0]
        row["cumulative_return_after_cost"] = float(summary["cumulative_return_after_cost"])
        row["sharpe_after_cost"] = float(summary["sharpe_after_cost"])
        row["max_drawdown_after_cost"] = float(summary["max_drawdown_after_cost"])
        row["avg_turnover"] = float(summary["avg_turnover"])

    if exp["result_path"].exists():
        row["score_self_case_slice"] = score_result_against_case_test(exp["result_path"])

    return row


def fmt(value) -> str:
    if value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Alpha v3 子特征细粒度消融报告",
        "",
        "## 目标",
        "",
        "- 找出既能抬高压缩包单切片自测分数，又不明显打坏本地累计收益的小组合。",
        "- 所有实验统一使用当前 refined 默认执行参数，不借用压缩包建模逻辑。",
        "",
        "## 实验结论",
        "",
    ]

    baseline = df[df["label"] == "current_refined_default"].iloc[0]
    ranked_by_score = df[df["label"] != "current_refined_default"].sort_values(
        ["score_self_case_slice", "cumulative_return_after_cost"],
        ascending=[False, False],
    )
    ranked_by_backtest = df[df["label"] != "current_refined_default"].sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost"],
        ascending=[False, False],
    )
    best_score = ranked_by_score.iloc[0]
    best_backtest = ranked_by_backtest.iloc[0]

    lines.extend(
        [
            f"- 当前 refined 默认方案本地成本后累计收益最高，仍为 `{baseline['cumulative_return_after_cost']:.6f}`。",
            f"- 单切片自测分数最好的子特征组合是 `{best_score['label']}`，分数为 `{best_score['score_self_case_slice']:.6f}`。",
            f"- 本地多期回测表现最好的子特征组合是 `{best_backtest['label']}`，成本后累计收益为 `{best_backtest['cumulative_return_after_cost']:.6f}`。",
            "- 如果某个子特征组合只改善单切片分数，却明显牺牲本地累计收益或回撤控制，就不应替换当前默认方案。",
            "",
            "## 正式对比表",
            "",
            "| 方案 | 特征集 | 特征数 | rank_ic_mean | top5_mean_return_mean | cumulative_return_after_cost | sharpe_after_cost | max_drawdown_after_cost | avg_turnover | 压缩包单切片自测分数 | 说明 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for _, row in df.iterrows():
        lines.append(
            f"| {row['label']} | {fmt(row['feature_set'])} | {fmt(row['feature_count'])} | "
            f"{fmt(row['rank_ic_mean'])} | {fmt(row['top5_mean_return_mean'])} | "
            f"{fmt(row['cumulative_return_after_cost'])} | {fmt(row['sharpe_after_cost'])} | "
            f"{fmt(row['max_drawdown_after_cost'])} | {fmt(row['avg_turnover'])} | "
            f"{fmt(row['score_self_case_slice'])} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## 建议",
            "",
            "- 优先保留那些 `压缩包单切片自测分数` 明显改善，同时 `cumulative_return_after_cost` 没有大幅掉队的子特征组合。",
            "- 下一步不要再整包堆 alpha，而是从这轮最有希望的 1 到 2 组子特征里再做更小范围组合试验。",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "train", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])
    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "predict", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])

    for exp in EXPERIMENTS:
        if exp["skip_train"]:
            continue
        exp["model_dir"].mkdir(parents=True, exist_ok=True)
        backtest_dir = exp["model_dir"] / "backtest"

        run_cmd(
            [
                sys.executable,
                str(SRC_DIR / "train_lstm.py"),
                "--feature_path",
                str(TEMP_DIR / "train_features.csv"),
                "--model_dir",
                str(exp["model_dir"]),
                "--feature_set",
                exp["feature_set"],
                "--sequence_length",
                "10",
                "--epochs",
                "8",
                "--patience",
                "2",
            ]
        )
        run_cmd(
            [
                sys.executable,
                str(SRC_DIR / "test_lstm.py"),
                "--feature_path",
                str(TEMP_DIR / "predict_features.csv"),
                "--model_dir",
                str(exp["model_dir"]),
                "--output_path",
                str(exp["result_path"]),
                "--previous_result_path",
                str(APP_DIR / "output" / "result.csv"),
            ]
        )
        run_cmd(
            [
                sys.executable,
                str(SRC_DIR / "backtest.py"),
                "--prediction_path",
                str(exp["model_dir"] / "walk_forward_predictions.csv"),
                "--feature_path",
                str(TEMP_DIR / "train_features.csv"),
                "--model_dir",
                str(exp["model_dir"]),
                "--output_dir",
                str(backtest_dir),
            ]
        )

    rows = [collect_row(exp) for exp in EXPERIMENTS]
    summary_df = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "alpha_v3_ablation_summary.csv"
    report_path = OUTPUT_DIR / "alpha_v3_ablation_report.md"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary_df, report_path)

    print(f"[alpha_v3_ablation] wrote {summary_path}")
    print(f"[alpha_v3_ablation] wrote {report_path}")


if __name__ == "__main__":
    main()
