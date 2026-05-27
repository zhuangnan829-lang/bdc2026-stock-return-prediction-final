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
OUTPUT_DIR = MODEL_DIR / "alpha_rs_crowding_experiment"
CASE_DIR = ROOT_DIR / "_case_zip" / "THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0"

EXPERIMENT = {
    "label": "alpha_v3_rs_crowding",
    "feature_set": "base_alpha_v3_rs_crowding",
    "model_dir": MODEL_DIR / "alpha_v3_rs_crowding",
    "result_path": APP_DIR / "output" / "result_alpha_v3_rs_crowding.csv",
    "backtest_summary_path": MODEL_DIR / "alpha_v3_rs_crowding" / "backtest" / "backtest_summary.csv",
}


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=ROOT_DIR)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score_result_against_case_test(result_path: Path) -> float:
    test_df = pd.read_csv(CASE_DIR / "data" / "test.csv")
    code_col = test_df.columns[0]
    open_col = test_df.columns[2]

    result_df = pd.read_csv(result_path, dtype={"stock_id": str})
    result_df = result_df.rename(columns={"stock_id": "code", "weight": "weight"})
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT["model_dir"].mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "train", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])
    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "predict", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])
    run_cmd(
        [
            sys.executable,
            str(SRC_DIR / "train_lstm.py"),
            "--feature_path",
            str(TEMP_DIR / "train_features.csv"),
            "--model_dir",
            str(EXPERIMENT["model_dir"]),
            "--feature_set",
            EXPERIMENT["feature_set"],
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
            str(EXPERIMENT["model_dir"]),
            "--output_path",
            str(EXPERIMENT["result_path"]),
            "--previous_result_path",
            str(APP_DIR / "output" / "result.csv"),
        ]
    )
    run_cmd(
        [
            sys.executable,
            str(SRC_DIR / "backtest.py"),
            "--prediction_path",
            str(EXPERIMENT["model_dir"] / "walk_forward_predictions.csv"),
            "--feature_path",
            str(TEMP_DIR / "train_features.csv"),
            "--model_dir",
            str(EXPERIMENT["model_dir"]),
            "--output_dir",
            str(EXPERIMENT["model_dir"] / "backtest"),
        ]
    )

    baseline_meta = load_json(MODEL_DIR / "model_meta.json")
    baseline_bt = pd.read_csv(MODEL_DIR / "current_refined_backtest" / "backtest_summary.csv").iloc[0]
    exp_meta = load_json(EXPERIMENT["model_dir"] / "model_meta.json")
    exp_bt = pd.read_csv(EXPERIMENT["backtest_summary_path"]).iloc[0]

    summary = pd.DataFrame(
        [
            {
                "label": "current_refined_default",
                "feature_set": baseline_meta["feature_set"],
                "feature_count": len(baseline_meta["feature_columns"]),
                "rank_ic_mean": baseline_meta["walk_forward_summary"]["rank_ic_mean"],
                "top5_mean_return_mean": baseline_meta["walk_forward_summary"]["top5_mean_return_mean"],
                "cumulative_return_after_cost": baseline_bt["cumulative_return_after_cost"],
                "sharpe_after_cost": baseline_bt["sharpe_after_cost"],
                "max_drawdown_after_cost": baseline_bt["max_drawdown_after_cost"],
                "avg_turnover": baseline_bt["avg_turnover"],
                "score_self_case_slice": score_result_against_case_test(APP_DIR / "output" / "result.csv"),
            },
            {
                "label": EXPERIMENT["label"],
                "feature_set": exp_meta["feature_set"],
                "feature_count": len(exp_meta["feature_columns"]),
                "rank_ic_mean": exp_meta["walk_forward_summary"]["rank_ic_mean"],
                "top5_mean_return_mean": exp_meta["walk_forward_summary"]["top5_mean_return_mean"],
                "cumulative_return_after_cost": exp_bt["cumulative_return_after_cost"],
                "sharpe_after_cost": exp_bt["sharpe_after_cost"],
                "max_drawdown_after_cost": exp_bt["max_drawdown_after_cost"],
                "avg_turnover": exp_bt["avg_turnover"],
                "score_self_case_slice": score_result_against_case_test(EXPERIMENT["result_path"]),
            },
        ]
    )

    summary_path = OUTPUT_DIR / "alpha_rs_crowding_summary.csv"
    report_path = OUTPUT_DIR / "alpha_rs_crowding_report.md"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    lines = [
        "# Relative Strength + Crowding Risk 联合实验",
        "",
        "| 方案 | 特征集 | 特征数 | rank_ic_mean | top5_mean_return_mean | cumulative_return_after_cost | sharpe_after_cost | max_drawdown_after_cost | avg_turnover | 压缩包单切片自测分数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['feature_set']} | {int(row['feature_count'])} | "
            f"{row['rank_ic_mean']:.6f} | {row['top5_mean_return_mean']:.6f} | "
            f"{row['cumulative_return_after_cost']:.6f} | {row['sharpe_after_cost']:.6f} | "
            f"{row['max_drawdown_after_cost']:.6f} | {row['avg_turnover']:.6f} | {row['score_self_case_slice']:.6f} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[alpha_rs_crowding] wrote {summary_path}")
    print(f"[alpha_rs_crowding] wrote {report_path}")


if __name__ == "__main__":
    main()
