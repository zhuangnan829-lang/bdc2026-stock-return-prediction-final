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

OUTPUT_DIR = MODEL_DIR / "lstm_topk_weight_grid_experiment"
MODELS_DIR = OUTPUT_DIR / "models"

TOP5_GRID = [1.4, 1.6, 1.8]
TOP10_GRID = [1.1, 1.2, 1.3]


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


def train_candidate(model_dir: Path, top5_weight: float, top10_weight: float) -> None:
    training = BEST_CONFIG["training"]
    rank_floor_weight = max(top10_weight, 1.2)
    run_cmd(
        [
            sys.executable,
            str(SRC_DIR / "train_lstm.py"),
            "--feature_path",
            str(TEMP_DIR / "train_features.csv"),
            "--model_dir",
            str(model_dir),
            "--valid_dates",
            str(training["valid_dates"]),
            "--num_folds",
            str(training["num_folds"]),
            "--target_mode",
            "topk_weighted_rank",
            "--feature_set",
            str(training["feature_set"]),
            "--sequence_length",
            str(training["sequence_length"]),
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
            "--topk_top5_weight",
            str(top5_weight),
            "--topk_top10_weight",
            str(top10_weight),
            "--topk_rank_pct_floor",
            "0.90",
            "--topk_rank_floor_weight",
            str(rank_floor_weight),
        ]
    )


def candidate_finished(model_dir: Path) -> bool:
    return (
        (model_dir / "model_meta.json").exists()
        and (model_dir / "walk_forward_predictions.csv").exists()
        and (model_dir / "backtest" / "backtest_summary.csv").exists()
    )


def collect_row(label: str, model_dir: Path, backtest_dir: Path) -> dict:
    meta = load_json(model_dir / "model_meta.json")
    bt = pd.read_csv(backtest_dir / "backtest_summary.csv").iloc[0]
    weight_cfg = meta.get("topk_weight_config", {})
    return {
        "label": label,
        "target_mode": meta.get("target_mode", ""),
        "sample_weight_mode": meta.get("sample_weight_mode", "uniform"),
        "top5_weight": float(weight_cfg.get("top5_weight", 1.0)),
        "top10_weight": float(weight_cfg.get("top10_weight", 1.0)),
        "rank_pct_floor": float(weight_cfg.get("rank_pct_floor", 0.0)),
        "rank_floor_weight": float(weight_cfg.get("rank_floor_weight", 1.0)),
        "rank_ic_mean": float(meta["walk_forward_summary"]["rank_ic_mean"]),
        "top5_mean_return_mean": float(meta["walk_forward_summary"]["top5_mean_return_mean"]),
        "cumulative_return_after_cost": float(bt["cumulative_return_after_cost"]),
        "sharpe_after_cost": float(bt["sharpe_after_cost"]),
        "max_drawdown_after_cost": float(bt["max_drawdown_after_cost"]),
        "avg_turnover": float(bt["avg_turnover"]),
    }


def write_report(summary_df: pd.DataFrame, output_dir: Path) -> None:
    best_row = summary_df.sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost", "top5_mean_return_mean"],
        ascending=[False, False, False],
    ).iloc[0]
    lines = [
        "# LSTM 温和头部加权网格实验",
        "",
        "- 目标：把训练目标往 Top5 排序靠，但避免上一版过强加权伤害本地成本后收益。",
        "- 固定主干：当前默认 LSTM + `base_alpha_v3_rs_crowding_mini4` + 现有执行层参数。",
        "- 变化项：仅搜索 `top5_weight` 与 `top10_weight`。",
        "",
        f"- 最优候选：`{best_row['label']}`",
        f"- 最优成本后累计收益：`{best_row['cumulative_return_after_cost']:.6f}`",
        f"- 最优成本后夏普：`{best_row['sharpe_after_cost']:.6f}`",
        "",
        "| label | top5_weight | top10_weight | rank_ic_mean | top5_mean_return_mean | cumulative_return_after_cost | sharpe_after_cost | max_drawdown_after_cost | avg_turnover |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['label']} | {row['top5_weight']:.2f} | {row['top10_weight']:.2f} | "
            f"{row['rank_ic_mean']:.6f} | {row['top5_mean_return_mean']:.6f} | "
            f"{row['cumulative_return_after_cost']:.6f} | {row['sharpe_after_cost']:.6f} | "
            f"{row['max_drawdown_after_cost']:.6f} | {row['avg_turnover']:.6f} |"
        )
    (output_dir / "lstm_topk_weight_grid_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    baseline_model_dir = MODEL_DIR
    baseline_backtest_dir = OUTPUT_DIR / "baseline_backtest"
    baseline_backtest_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        build_backtest_cmd(
            prediction_path=baseline_model_dir / "walk_forward_predictions.csv",
            feature_path=TEMP_DIR / "train_features.csv",
            model_dir=baseline_model_dir,
            output_dir=baseline_backtest_dir,
        )
    )
    rows.append(collect_row("baseline_cross_section_rank", baseline_model_dir, baseline_backtest_dir))

    for top5_weight in TOP5_GRID:
        for top10_weight in TOP10_GRID:
            label = f"top5_{top5_weight:.1f}_top10_{top10_weight:.1f}".replace(".", "")
            candidate_model_dir = MODELS_DIR / label
            candidate_backtest_dir = candidate_model_dir / "backtest"
            candidate_model_dir.mkdir(parents=True, exist_ok=True)
            candidate_backtest_dir.mkdir(parents=True, exist_ok=True)

            if not candidate_finished(candidate_model_dir):
                train_candidate(candidate_model_dir, top5_weight=top5_weight, top10_weight=top10_weight)
                run_cmd(
                    build_backtest_cmd(
                        prediction_path=candidate_model_dir / "walk_forward_predictions.csv",
                        feature_path=TEMP_DIR / "train_features.csv",
                        model_dir=candidate_model_dir,
                        output_dir=candidate_backtest_dir,
                    )
                )
            rows.append(collect_row(label, candidate_model_dir, candidate_backtest_dir))

    summary_df = pd.DataFrame(rows).sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost", "top5_mean_return_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    baseline = summary_df[summary_df["label"] == "baseline_cross_section_rank"].iloc[0]
    summary_df["delta_cumulative_return_after_cost"] = (
        summary_df["cumulative_return_after_cost"] - float(baseline["cumulative_return_after_cost"])
    )
    summary_df["delta_sharpe_after_cost"] = summary_df["sharpe_after_cost"] - float(baseline["sharpe_after_cost"])
    summary_df["delta_top5_mean_return_mean"] = (
        summary_df["top5_mean_return_mean"] - float(baseline["top5_mean_return_mean"])
    )
    summary_df["delta_rank_ic_mean"] = summary_df["rank_ic_mean"] - float(baseline["rank_ic_mean"])

    summary_path = OUTPUT_DIR / "lstm_topk_weight_grid_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary_df, OUTPUT_DIR)

    print(f"[lstm_topk_weight_grid] wrote {summary_path}")
    print(f"[lstm_topk_weight_grid] best_label={summary_df.iloc[0]['label']}")


if __name__ == "__main__":
    main()
