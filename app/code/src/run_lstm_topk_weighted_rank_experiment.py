import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from config import BEST_CONFIG


ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = ROOT_DIR / "app"
SRC_DIR = APP_DIR / "code" / "src"
TEMP_DIR = APP_DIR / "temp"
MODEL_DIR = APP_DIR / "model"

OUTPUT_DIR = MODEL_DIR / "lstm_topk_weighted_rank_experiment"
CANDIDATE_MODEL_DIR = OUTPUT_DIR / "model_topk_weighted_rank"
BACKTEST_DIR = OUTPUT_DIR / "backtest"


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=ROOT_DIR)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_backtest_cmd(prediction_path: Path, feature_path: Path, model_dir: Path, output_dir: Path) -> list[str]:
    s = BEST_CONFIG["selection"]
    r = BEST_CONFIG["risk_filter_thresholds"]
    e = BEST_CONFIG["execution"]
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
        str(s["top_k"]),
        "--primary_candidate_size",
        str(s["primary_candidate_size"]),
        "--enable_risk_filters",
        "1",
        "--allow_cash_fallback",
        "0",
        "--max_volatility_20d_pct",
        str(r["max_volatility_20d_pct"]),
        "--max_volatility_5d_pct",
        str(r["max_volatility_5d_pct"]),
        "--turnover_rate_lower_pct",
        str(r["turnover_rate_lower_pct"]),
        "--turnover_rate_upper_pct",
        str(r["turnover_rate_upper_pct"]),
        "--turnover_ratio_upper_pct",
        str(r["turnover_ratio_upper_pct"]),
        "--risk_penalty_weight",
        str(r["risk_penalty_weight"]),
        "--weighting_scheme",
        str(s["weighting_scheme"]),
        "--sort_strategy",
        str(s["sort_strategy"]),
        "--transaction_cost",
        str(e["transaction_cost"]),
        "--max_turnover",
        str(e["max_turnover"]),
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    training = BEST_CONFIG["training"]
    feature_set = training["feature_set"]
    sequence_length = int(training["sequence_length"])

    run_cmd([
        sys.executable,
        str(SRC_DIR / "train_lstm.py"),
        "--feature_path",
        str(TEMP_DIR / "train_features.csv"),
        "--model_dir",
        str(CANDIDATE_MODEL_DIR),
        "--valid_dates",
        str(training["valid_dates"]),
        "--num_folds",
        str(training["num_folds"]),
        "--target_mode",
        "topk_weighted_rank",
        "--feature_set",
        feature_set,
        "--sequence_length",
        str(sequence_length),
        "--hidden_size",
        str(training["hidden_size"]),
        "--num_layers",
        str(training["num_layers"]),
        "--dropout",
        str(training["dropout"]),
        "--learning_rate",
        str(training["learning_rate"]),
        "--batch_size",
        str(training["batch_size"]),
        "--epochs",
        str(training["epochs"]),
        "--patience",
        str(training["patience"]),
    ])

    run_cmd(build_backtest_cmd(
        prediction_path=CANDIDATE_MODEL_DIR / "walk_forward_predictions.csv",
        feature_path=TEMP_DIR / "train_features.csv",
        model_dir=CANDIDATE_MODEL_DIR,
        output_dir=BACKTEST_DIR,
    ))

    baseline_meta = load_json(MODEL_DIR / "model_meta.json")
    baseline_bt = pd.read_csv(MODEL_DIR / "alpha_v3_rs_crowding_mini4" / "backtest" / "backtest_summary.csv").iloc[0]
    candidate_meta = load_json(CANDIDATE_MODEL_DIR / "model_meta.json")
    candidate_bt = pd.read_csv(BACKTEST_DIR / "backtest_summary.csv").iloc[0]

    rows = [
        {
            "label": "baseline_cross_section_rank",
            "target_mode": baseline_meta.get("target_mode", ""),
            "sample_weight_mode": baseline_meta.get("sample_weight_mode", "uniform"),
            "rank_ic_mean": float(baseline_meta["walk_forward_summary"]["rank_ic_mean"]),
            "top5_mean_return_mean": float(baseline_meta["walk_forward_summary"]["top5_mean_return_mean"]),
            "cumulative_return_after_cost": float(baseline_bt["cumulative_return_after_cost"]),
            "sharpe_after_cost": float(baseline_bt["sharpe_after_cost"]),
            "max_drawdown_after_cost": float(baseline_bt["max_drawdown_after_cost"]),
            "avg_turnover": float(baseline_bt["avg_turnover"]),
        },
        {
            "label": "candidate_topk_weighted_rank",
            "target_mode": candidate_meta.get("target_mode", ""),
            "sample_weight_mode": candidate_meta.get("sample_weight_mode", "uniform"),
            "rank_ic_mean": float(candidate_meta["walk_forward_summary"]["rank_ic_mean"]),
            "top5_mean_return_mean": float(candidate_meta["walk_forward_summary"]["top5_mean_return_mean"]),
            "cumulative_return_after_cost": float(candidate_bt["cumulative_return_after_cost"]),
            "sharpe_after_cost": float(candidate_bt["sharpe_after_cost"]),
            "max_drawdown_after_cost": float(candidate_bt["max_drawdown_after_cost"]),
            "avg_turnover": float(candidate_bt["avg_turnover"]),
        },
    ]
    summary_df = pd.DataFrame(rows)
    summary_df["delta_rank_ic_mean"] = summary_df["rank_ic_mean"] - summary_df.loc[0, "rank_ic_mean"]
    summary_df["delta_top5_mean_return_mean"] = summary_df["top5_mean_return_mean"] - summary_df.loc[0, "top5_mean_return_mean"]
    summary_df["delta_cumulative_return_after_cost"] = summary_df["cumulative_return_after_cost"] - summary_df.loc[0, "cumulative_return_after_cost"]
    summary_df["delta_sharpe_after_cost"] = summary_df["sharpe_after_cost"] - summary_df.loc[0, "sharpe_after_cost"]
    summary_df["delta_max_drawdown_after_cost"] = summary_df["max_drawdown_after_cost"] - summary_df.loc[0, "max_drawdown_after_cost"]

    summary_path = OUTPUT_DIR / "lstm_topk_weighted_rank_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    report_lines = [
        "# LSTM TopK Weighted Rank 实验",
        "",
        "- 训练目标从 `cross_section_rank` 升级为 `topk_weighted_rank`。",
        "- 标签仍为横截面排序标签，但训练损失对当日收益排名前 5 和前 10 的样本给予更高权重。",
        "",
        "| 方案 | target_mode | sample_weight_mode | rank_ic_mean | top5_mean_return_mean | cumulative_return_after_cost | sharpe_after_cost | max_drawdown_after_cost | avg_turnover |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        report_lines.append(
            f"| {row['label']} | {row['target_mode']} | {row['sample_weight_mode']} | "
            f"{row['rank_ic_mean']:.6f} | {row['top5_mean_return_mean']:.6f} | "
            f"{row['cumulative_return_after_cost']:.6f} | {row['sharpe_after_cost']:.6f} | "
            f"{row['max_drawdown_after_cost']:.6f} | {row['avg_turnover']:.6f} |"
        )
    report_lines.extend([
        "",
        "## 变化",
        "",
        f"- rank_ic_mean 变化：`{summary_df.loc[1, 'delta_rank_ic_mean']:.6f}`",
        f"- top5_mean_return_mean 变化：`{summary_df.loc[1, 'delta_top5_mean_return_mean']:.6f}`",
        f"- 成本后累计收益变化：`{summary_df.loc[1, 'delta_cumulative_return_after_cost']:.6f}`",
        f"- 成本后夏普变化：`{summary_df.loc[1, 'delta_sharpe_after_cost']:.6f}`",
        f"- 最大回撤变化：`{summary_df.loc[1, 'delta_max_drawdown_after_cost']:.6f}`",
        "",
    ])
    report_path = OUTPUT_DIR / "lstm_topk_weighted_rank_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[lstm_topk_weighted_rank] wrote {summary_path}")
    print(f"[lstm_topk_weighted_rank] wrote {report_path}")


if __name__ == "__main__":
    main()
