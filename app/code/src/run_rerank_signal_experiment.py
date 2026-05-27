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

OUTPUT_DIR = MODEL_DIR / "rerank_signal_experiment"
RESULTS_DIR = OUTPUT_DIR / "results"
BACKTEST_DIR = OUTPUT_DIR / "backtests"
RERANK_SIGNAL_COLUMN = "rel_strength_accel_5d_v2"


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=ROOT_DIR)


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


def build_test_cmd(output_path: Path, rerank_signal_weight: float) -> list[str]:
    s = BEST_CONFIG["selection"]
    r = BEST_CONFIG["risk_filter_thresholds"]
    e = BEST_CONFIG["execution"]
    cmd = [
        sys.executable, str(SRC_DIR / "test_lstm.py"),
        "--feature_path", str(TEMP_DIR / "predict_features.csv"),
        "--model_dir", str(MODEL_DIR),
        "--output_path", str(output_path),
        "--top_k", str(s["top_k"]),
        "--primary_candidate_size", str(s["primary_candidate_size"]),
        "--max_volatility_20d_pct", str(r["max_volatility_20d_pct"]),
        "--max_volatility_5d_pct", str(r["max_volatility_5d_pct"]),
        "--turnover_rate_lower_pct", str(r["turnover_rate_lower_pct"]),
        "--turnover_rate_upper_pct", str(r["turnover_rate_upper_pct"]),
        "--turnover_ratio_upper_pct", str(r["turnover_ratio_upper_pct"]),
        "--risk_penalty_weight", str(r["risk_penalty_weight"]),
        "--sort_strategy", str(s["sort_strategy"]),
        "--weighting_scheme", str(s["weighting_scheme"]),
        "--max_turnover", str(e["max_turnover"]),
        "--previous_result_path", str(APP_DIR / "output" / "result.csv"),
    ]
    if abs(rerank_signal_weight) > 1e-12:
        cmd.extend(["--rerank_signal_column", RERANK_SIGNAL_COLUMN, "--rerank_signal_weight", str(rerank_signal_weight)])
    return cmd


def build_backtest_cmd(output_dir: Path, rerank_signal_weight: float) -> list[str]:
    s = BEST_CONFIG["selection"]
    r = BEST_CONFIG["risk_filter_thresholds"]
    e = BEST_CONFIG["execution"]
    cmd = [
        sys.executable, str(SRC_DIR / "backtest.py"),
        "--prediction_path", str(MODEL_DIR / "walk_forward_predictions.csv"),
        "--feature_path", str(TEMP_DIR / "train_features.csv"),
        "--model_dir", str(MODEL_DIR),
        "--output_dir", str(output_dir),
        "--top_k", str(s["top_k"]),
        "--primary_candidate_size", str(s["primary_candidate_size"]),
        "--enable_risk_filters", "1",
        "--allow_cash_fallback", "0",
        "--max_volatility_20d_pct", str(r["max_volatility_20d_pct"]),
        "--max_volatility_5d_pct", str(r["max_volatility_5d_pct"]),
        "--turnover_rate_lower_pct", str(r["turnover_rate_lower_pct"]),
        "--turnover_rate_upper_pct", str(r["turnover_rate_upper_pct"]),
        "--turnover_ratio_upper_pct", str(r["turnover_ratio_upper_pct"]),
        "--risk_penalty_weight", str(r["risk_penalty_weight"]),
        "--weighting_scheme", str(s["weighting_scheme"]),
        "--sort_strategy", str(s["sort_strategy"]),
        "--transaction_cost", str(e["transaction_cost"]),
        "--max_turnover", str(e["max_turnover"]),
    ]
    if abs(rerank_signal_weight) > 1e-12:
        cmd.extend(["--rerank_signal_column", RERANK_SIGNAL_COLUMN, "--rerank_signal_weight", str(rerank_signal_weight)])
    return cmd


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "train", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])
    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "predict", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])

    profiles = [
        {"label": "baseline", "rerank_signal_weight": 0.0, "result_path": APP_DIR / "output" / "result.csv", "skip_test": True, "backtest_dir": MODEL_DIR / "alpha_v3_rs_crowding_mini4" / "backtest"},
        {"label": "signal_w005", "rerank_signal_weight": 0.05, "result_path": RESULTS_DIR / "result_signal_w005.csv", "skip_test": False, "backtest_dir": BACKTEST_DIR / "signal_w005"},
        {"label": "signal_w010", "rerank_signal_weight": 0.10, "result_path": RESULTS_DIR / "result_signal_w010.csv", "skip_test": False, "backtest_dir": BACKTEST_DIR / "signal_w010"},
        {"label": "signal_w015", "rerank_signal_weight": 0.15, "result_path": RESULTS_DIR / "result_signal_w015.csv", "skip_test": False, "backtest_dir": BACKTEST_DIR / "signal_w015"},
        {"label": "signal_w020", "rerank_signal_weight": 0.20, "result_path": RESULTS_DIR / "result_signal_w020.csv", "skip_test": False, "backtest_dir": BACKTEST_DIR / "signal_w020"},
    ]

    for profile in profiles:
        if not profile["skip_test"]:
            profile["backtest_dir"].mkdir(parents=True, exist_ok=True)
            run_cmd(build_test_cmd(profile["result_path"], profile["rerank_signal_weight"]))
            run_cmd(build_backtest_cmd(profile["backtest_dir"], profile["rerank_signal_weight"]))

    baseline_bt = pd.read_csv((MODEL_DIR / "alpha_v3_rs_crowding_mini4" / "backtest" / "backtest_summary.csv")).iloc[0]
    rows = []
    baseline_score = score_result_against_case_test(APP_DIR / "output" / "result.csv")
    rows.append({
        "label": "baseline",
        "rerank_signal_column": "",
        "rerank_signal_weight": 0.0,
        "score_self_case_slice": baseline_score,
        "cumulative_return_after_cost": float(baseline_bt["cumulative_return_after_cost"]),
        "sharpe_after_cost": float(baseline_bt["sharpe_after_cost"]),
        "max_drawdown_after_cost": float(baseline_bt["max_drawdown_after_cost"]),
        "avg_turnover": float(baseline_bt["avg_turnover"]),
        "delta_case_slice_score": 0.0,
        "delta_cumulative_return_after_cost": 0.0,
        "delta_sharpe_after_cost": 0.0,
        "delta_max_drawdown_after_cost": 0.0,
    })

    for profile in profiles[1:]:
        bt = pd.read_csv(profile["backtest_dir"] / "backtest_summary.csv").iloc[0]
        score = score_result_against_case_test(profile["result_path"])
        rows.append({
            "label": profile["label"],
            "rerank_signal_column": RERANK_SIGNAL_COLUMN,
            "rerank_signal_weight": profile["rerank_signal_weight"],
            "score_self_case_slice": score,
            "cumulative_return_after_cost": float(bt["cumulative_return_after_cost"]),
            "sharpe_after_cost": float(bt["sharpe_after_cost"]),
            "max_drawdown_after_cost": float(bt["max_drawdown_after_cost"]),
            "avg_turnover": float(bt["avg_turnover"]),
            "delta_case_slice_score": score - baseline_score,
            "delta_cumulative_return_after_cost": float(bt["cumulative_return_after_cost"]) - float(baseline_bt["cumulative_return_after_cost"]),
            "delta_sharpe_after_cost": float(bt["sharpe_after_cost"]) - float(baseline_bt["sharpe_after_cost"]),
            "delta_max_drawdown_after_cost": float(bt["max_drawdown_after_cost"]) - float(baseline_bt["max_drawdown_after_cost"]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "rerank_signal_summary.csv", index=False, encoding="utf-8-sig")

    ranked = df[df["label"] != "baseline"].sort_values(
        ["delta_case_slice_score", "delta_cumulative_return_after_cost", "delta_sharpe_after_cost"],
        ascending=[False, False, False],
    )
    best = ranked.iloc[0]
    lines = [
        "# rs_accel_v2 执行层 rerank 实验",
        "",
        "- 训练保持不变，只在二次排序阶段加入 `rel_strength_accel_5d_v2`。",
        f"- 切片提升最强方案：`{best['label']}`，权重 `{best['rerank_signal_weight']:.2f}`。",
        "",
        "| 方案 | rerank_weight | slice | Δslice | cum_after | Δcum | sharpe | Δsharpe | max_dd | Δdd |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['label']} | {row['rerank_signal_weight']:.2f} | {row['score_self_case_slice']:.6f} | "
            f"{row['delta_case_slice_score']:.6f} | {row['cumulative_return_after_cost']:.6f} | "
            f"{row['delta_cumulative_return_after_cost']:.6f} | {row['sharpe_after_cost']:.6f} | "
            f"{row['delta_sharpe_after_cost']:.6f} | {row['max_drawdown_after_cost']:.6f} | "
            f"{row['delta_max_drawdown_after_cost']:.6f} |"
        )
    (OUTPUT_DIR / "rerank_signal_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[rerank_signal] wrote {OUTPUT_DIR / 'rerank_signal_summary.csv'}")
    print(f"[rerank_signal] wrote {OUTPUT_DIR / 'rerank_signal_report.md'}")


if __name__ == "__main__":
    main()
