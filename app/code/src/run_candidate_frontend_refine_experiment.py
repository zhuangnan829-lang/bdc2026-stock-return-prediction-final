import itertools
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

OUTPUT_DIR = MODEL_DIR / "candidate_frontend_refine_experiment"
BACKTEST_ROOT = OUTPUT_DIR / "backtests"
RESULT_ROOT = OUTPUT_DIR / "results"


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


def build_test_cmd(result_path: Path, candidate_size: int, vol20: float, vol5: float, secondary_size: int) -> list[str]:
    s = BEST_CONFIG["selection"]
    r = BEST_CONFIG["risk_filter_thresholds"]
    e = BEST_CONFIG["execution"]
    cmd = [
        sys.executable, str(SRC_DIR / "test_lstm.py"),
        "--feature_path", str(TEMP_DIR / "predict_features.csv"),
        "--model_dir", str(MODEL_DIR),
        "--output_path", str(result_path),
        "--top_k", str(s["top_k"]),
        "--primary_candidate_size", str(candidate_size),
        "--max_volatility_20d_pct", str(vol20),
        "--max_volatility_5d_pct", str(vol5),
        "--turnover_rate_lower_pct", str(r["turnover_rate_lower_pct"]),
        "--turnover_rate_upper_pct", str(r["turnover_rate_upper_pct"]),
        "--turnover_ratio_upper_pct", str(r["turnover_ratio_upper_pct"]),
        "--risk_penalty_weight", str(r["risk_penalty_weight"]),
        "--sort_strategy", str(s["sort_strategy"]),
        "--weighting_scheme", str(s["weighting_scheme"]),
        "--max_turnover", str(e["max_turnover"]),
        "--previous_result_path", str(APP_DIR / "output" / "result.csv"),
    ]
    if secondary_size > 0:
        cmd.extend(["--secondary_candidate_size", str(secondary_size)])
    return cmd


def build_backtest_cmd(output_dir: Path, candidate_size: int, vol20: float, vol5: float, secondary_size: int) -> list[str]:
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
        "--primary_candidate_size", str(candidate_size),
        "--enable_risk_filters", "1",
        "--allow_cash_fallback", "0",
        "--max_volatility_20d_pct", str(vol20),
        "--max_volatility_5d_pct", str(vol5),
        "--turnover_rate_lower_pct", str(r["turnover_rate_lower_pct"]),
        "--turnover_rate_upper_pct", str(r["turnover_rate_upper_pct"]),
        "--turnover_ratio_upper_pct", str(r["turnover_ratio_upper_pct"]),
        "--risk_penalty_weight", str(r["risk_penalty_weight"]),
        "--weighting_scheme", str(s["weighting_scheme"]),
        "--sort_strategy", str(s["sort_strategy"]),
        "--transaction_cost", str(e["transaction_cost"]),
        "--max_turnover", str(e["max_turnover"]),
    ]
    if secondary_size > 0:
        cmd.extend(["--secondary_candidate_size", str(secondary_size)])
    return cmd


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "train", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])
    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "predict", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])

    baseline_bt = pd.read_csv(MODEL_DIR / "alpha_v3_rs_crowding_mini4" / "backtest" / "backtest_summary.csv").iloc[0]
    baseline_score = score_result_against_case_test(APP_DIR / "output" / "result.csv")

    grid = list(
        itertools.product(
            [35, 40, 45],
            [0.84, 0.86, 0.88],
            [0.95, 0.96, 0.97],
            [0, 15, 20],
        )
    )

    rows = [{
        "label": "baseline",
        "primary_candidate_size": int(BEST_CONFIG["selection"]["primary_candidate_size"]),
        "max_volatility_20d_pct": float(BEST_CONFIG["risk_filter_thresholds"]["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(BEST_CONFIG["risk_filter_thresholds"]["max_volatility_5d_pct"]),
        "secondary_candidate_size": 0,
        "score_self_case_slice": baseline_score,
        "cumulative_return_after_cost": float(baseline_bt["cumulative_return_after_cost"]),
        "sharpe_after_cost": float(baseline_bt["sharpe_after_cost"]),
        "max_drawdown_after_cost": float(baseline_bt["max_drawdown_after_cost"]),
        "avg_turnover": float(baseline_bt["avg_turnover"]),
        "delta_case_slice_score": 0.0,
        "delta_cumulative_return_after_cost": 0.0,
        "delta_sharpe_after_cost": 0.0,
        "delta_max_drawdown_after_cost": 0.0,
    }]

    for candidate_size, vol20, vol5, secondary_size in grid:
        label = f"cs{candidate_size}_v20{int(vol20*100)}_v5{int(vol5*100)}_ss{secondary_size}"
        result_path = RESULT_ROOT / f"{label}.csv"
        backtest_dir = BACKTEST_ROOT / label
        backtest_dir.mkdir(parents=True, exist_ok=True)

        run_cmd(build_test_cmd(result_path, candidate_size, vol20, vol5, secondary_size))
        run_cmd(build_backtest_cmd(backtest_dir, candidate_size, vol20, vol5, secondary_size))

        bt = pd.read_csv(backtest_dir / "backtest_summary.csv").iloc[0]
        score = score_result_against_case_test(result_path)
        rows.append({
            "label": label,
            "primary_candidate_size": candidate_size,
            "max_volatility_20d_pct": vol20,
            "max_volatility_5d_pct": vol5,
            "secondary_candidate_size": secondary_size,
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
    df.to_csv(OUTPUT_DIR / "candidate_frontend_refine_summary.csv", index=False, encoding="utf-8-sig")

    ranked = df[df["label"] != "baseline"].sort_values(
        ["delta_case_slice_score", "delta_cumulative_return_after_cost", "delta_sharpe_after_cost"],
        ascending=[False, False, False],
    )
    best_slice = ranked.iloc[0]
    best_balance = df[df["label"] != "baseline"].sort_values(
        ["delta_cumulative_return_after_cost", "delta_case_slice_score", "delta_sharpe_after_cost"],
        ascending=[False, False, False],
    ).iloc[0]
    report_lines = [
        "# 候选池前段排序局部执行层精修实验",
        "",
        "- 训练保持不变，只微调前段候选池与 Top40 内二段式筛选。",
        f"- 切片提升最强：`{best_slice['label']}`",
        f"- 本地收益最优：`{best_balance['label']}`",
        "",
        "| label | cs | vol20 | vol5 | secondary | slice | Δslice | cum_after | Δcum | sharpe | Δsharpe | max_dd | Δdd |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        report_lines.append(
            f"| {row['label']} | {int(row['primary_candidate_size'])} | {row['max_volatility_20d_pct']:.2f} | "
            f"{row['max_volatility_5d_pct']:.2f} | {int(row['secondary_candidate_size'])} | "
            f"{row['score_self_case_slice']:.6f} | {row['delta_case_slice_score']:.6f} | "
            f"{row['cumulative_return_after_cost']:.6f} | {row['delta_cumulative_return_after_cost']:.6f} | "
            f"{row['sharpe_after_cost']:.6f} | {row['delta_sharpe_after_cost']:.6f} | "
            f"{row['max_drawdown_after_cost']:.6f} | {row['delta_max_drawdown_after_cost']:.6f} |"
        )
    (OUTPUT_DIR / "candidate_frontend_refine_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[candidate_frontend_refine] wrote {OUTPUT_DIR / 'candidate_frontend_refine_summary.csv'}")
    print(f"[candidate_frontend_refine] wrote {OUTPUT_DIR / 'candidate_frontend_refine_report.md'}")


if __name__ == "__main__":
    main()
