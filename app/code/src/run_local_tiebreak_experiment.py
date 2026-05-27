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

OUTPUT_DIR = MODEL_DIR / "local_tiebreak_experiment"
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


def build_test_cmd(result_path: Path, blend_weight: float, start_rank: int, end_rank: int) -> list[str]:
    s = BEST_CONFIG["selection"]
    r = BEST_CONFIG["risk_filter_thresholds"]
    e = BEST_CONFIG["execution"]
    return [
        sys.executable,
        str(SRC_DIR / "test_lstm.py"),
        "--feature_path", str(TEMP_DIR / "predict_features.csv"),
        "--model_dir", str(MODEL_DIR),
        "--output_path", str(result_path),
        "--top_k", str(s["top_k"]),
        "--primary_candidate_size", "40",
        "--max_volatility_20d_pct", "0.86",
        "--max_volatility_5d_pct", "0.96",
        "--turnover_rate_lower_pct", str(r["turnover_rate_lower_pct"]),
        "--turnover_rate_upper_pct", str(r["turnover_rate_upper_pct"]),
        "--turnover_ratio_upper_pct", str(r["turnover_ratio_upper_pct"]),
        "--risk_penalty_weight", str(r["risk_penalty_weight"]),
        "--sort_strategy", str(s["sort_strategy"]),
        "--weighting_scheme", str(s["weighting_scheme"]),
        "--max_turnover", str(e["max_turnover"]),
        "--previous_result_path", str(APP_DIR / "output" / "result.csv"),
        "--secondary_screen_mode", "alpha_local_tiebreak",
        "--secondary_screen_weight", str(blend_weight),
        "--local_tiebreak_start_rank", str(start_rank),
        "--local_tiebreak_end_rank", str(end_rank),
    ]


def build_backtest_cmd(output_dir: Path, blend_weight: float, start_rank: int, end_rank: int) -> list[str]:
    s = BEST_CONFIG["selection"]
    r = BEST_CONFIG["risk_filter_thresholds"]
    e = BEST_CONFIG["execution"]
    return [
        sys.executable,
        str(SRC_DIR / "backtest.py"),
        "--prediction_path", str(MODEL_DIR / "walk_forward_predictions.csv"),
        "--feature_path", str(TEMP_DIR / "train_features.csv"),
        "--model_dir", str(MODEL_DIR),
        "--output_dir", str(output_dir),
        "--top_k", str(s["top_k"]),
        "--primary_candidate_size", "40",
        "--enable_risk_filters", "1",
        "--allow_cash_fallback", "0",
        "--max_volatility_20d_pct", "0.86",
        "--max_volatility_5d_pct", "0.96",
        "--turnover_rate_lower_pct", str(r["turnover_rate_lower_pct"]),
        "--turnover_rate_upper_pct", str(r["turnover_rate_upper_pct"]),
        "--turnover_ratio_upper_pct", str(r["turnover_ratio_upper_pct"]),
        "--risk_penalty_weight", str(r["risk_penalty_weight"]),
        "--weighting_scheme", str(s["weighting_scheme"]),
        "--sort_strategy", str(s["sort_strategy"]),
        "--transaction_cost", str(e["transaction_cost"]),
        "--max_turnover", str(e["max_turnover"]),
        "--secondary_screen_mode", "alpha_local_tiebreak",
        "--secondary_screen_weight", str(blend_weight),
        "--local_tiebreak_start_rank", str(start_rank),
        "--local_tiebreak_end_rank", str(end_rank),
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "train", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])
    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "predict", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])

    baseline_bt = pd.read_csv(MODEL_DIR / "alpha_v3_rs_crowding_mini4" / "backtest" / "backtest_summary.csv").iloc[0]
    baseline_score = score_result_against_case_test(APP_DIR / "output" / "result.csv")
    baseline_cum = float(baseline_bt["cumulative_return_after_cost"])
    baseline_sharpe = float(baseline_bt["sharpe_after_cost"])
    baseline_dd = float(baseline_bt["max_drawdown_after_cost"])

    rows = [{
        "label": "baseline",
        "secondary_screen_mode": "none",
        "secondary_screen_weight": 0.0,
        "local_tiebreak_start_rank": 0,
        "local_tiebreak_end_rank": 0,
        "score_self_case_slice": baseline_score,
        "cumulative_return_after_cost": baseline_cum,
        "sharpe_after_cost": baseline_sharpe,
        "max_drawdown_after_cost": baseline_dd,
        "delta_case_slice_score": 0.0,
        "delta_cumulative_return_after_cost": 0.0,
        "delta_sharpe_after_cost": 0.0,
        "delta_max_drawdown_after_cost": 0.0,
    }]

    rank_windows = [(8, 12), (8, 15), (10, 15)]
    weights = [0.05, 0.10, 0.15, 0.20, 0.25]
    for (start_rank, end_rank), blend_weight in itertools.product(rank_windows, weights):
        label = f"lt_{start_rank}_{end_rank}_w{int(blend_weight*100):02d}"
        result_path = RESULT_ROOT / f"{label}.csv"
        backtest_dir = BACKTEST_ROOT / label
        backtest_dir.mkdir(parents=True, exist_ok=True)

        if not result_path.exists():
            run_cmd(build_test_cmd(result_path, blend_weight, start_rank, end_rank))
        if not (backtest_dir / "backtest_summary.csv").exists():
            run_cmd(build_backtest_cmd(backtest_dir, blend_weight, start_rank, end_rank))

        bt = pd.read_csv(backtest_dir / "backtest_summary.csv").iloc[0]
        score = score_result_against_case_test(result_path)
        rows.append({
            "label": label,
            "secondary_screen_mode": "alpha_local_tiebreak",
            "secondary_screen_weight": blend_weight,
            "local_tiebreak_start_rank": start_rank,
            "local_tiebreak_end_rank": end_rank,
            "score_self_case_slice": score,
            "cumulative_return_after_cost": float(bt["cumulative_return_after_cost"]),
            "sharpe_after_cost": float(bt["sharpe_after_cost"]),
            "max_drawdown_after_cost": float(bt["max_drawdown_after_cost"]),
            "avg_turnover": float(bt["avg_turnover"]),
            "delta_case_slice_score": score - baseline_score,
            "delta_cumulative_return_after_cost": float(bt["cumulative_return_after_cost"]) - baseline_cum,
            "delta_sharpe_after_cost": float(bt["sharpe_after_cost"]) - baseline_sharpe,
            "delta_max_drawdown_after_cost": float(bt["max_drawdown_after_cost"]) - baseline_dd,
        })

    df = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "local_tiebreak_summary.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    exp = df[df["label"] != "baseline"].copy()
    best_slice = exp.sort_values(
        ["delta_case_slice_score", "delta_cumulative_return_after_cost", "delta_sharpe_after_cost"],
        ascending=[False, False, False],
    ).iloc[0]
    best_local = exp.sort_values(
        ["delta_cumulative_return_after_cost", "delta_case_slice_score", "delta_sharpe_after_cost"],
        ascending=[False, False, False],
    ).iloc[0]
    balanced = exp[
        (exp["delta_case_slice_score"] >= 0.0)
        & (exp["delta_cumulative_return_after_cost"] >= -0.02)
    ].sort_values(
        ["delta_case_slice_score", "delta_cumulative_return_after_cost", "delta_sharpe_after_cost"],
        ascending=[False, False, False],
    )

    lines = [
        "# Top8~Top15 局部 Tie-Break 实验",
        "",
        "- 主排序保持不变，只在局部并列区间用 alpha 组合信号做轻微重排。",
        "- alpha 信号来自相对强弱、趋势持续、拥挤反转风险。",
        f"- 切片分数最优：`{best_slice['label']}`，slice={best_slice['score_self_case_slice']:.6f}，成本后累计收益={best_slice['cumulative_return_after_cost']:.6f}",
        f"- 本地收益最优：`{best_local['label']}`，slice={best_local['score_self_case_slice']:.6f}，成本后累计收益={best_local['cumulative_return_after_cost']:.6f}",
    ]
    if balanced.empty:
        lines.append("- 本轮没有出现“切片分数不降且本地累计收益基本保住”的平衡升级组合。")
    else:
        top_balanced = balanced.iloc[0]
        lines.append(
            f"- 平衡候选：`{top_balanced['label']}`，slice={top_balanced['score_self_case_slice']:.6f}，"
            f"Δslice={top_balanced['delta_case_slice_score']:.6f}，Δcum={top_balanced['delta_cumulative_return_after_cost']:.6f}"
        )

    lines.extend([
        "",
        "| label | local_window | weight | slice | Δslice | cum_after | Δcum | sharpe | Δsharpe | max_dd | Δdd |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in df.sort_values(["local_tiebreak_start_rank", "local_tiebreak_end_rank", "secondary_screen_weight"]).iterrows():
        window = "-" if row["label"] == "baseline" else f"{int(row['local_tiebreak_start_rank'])}~{int(row['local_tiebreak_end_rank'])}"
        lines.append(
            f"| {row['label']} | {window} | {row.get('secondary_screen_weight', 0.0):.2f} | "
            f"{row['score_self_case_slice']:.6f} | {row['delta_case_slice_score']:.6f} | "
            f"{row['cumulative_return_after_cost']:.6f} | {row['delta_cumulative_return_after_cost']:.6f} | "
            f"{row['sharpe_after_cost']:.6f} | {row['delta_sharpe_after_cost']:.6f} | "
            f"{row['max_drawdown_after_cost']:.6f} | {row['delta_max_drawdown_after_cost']:.6f} |"
        )

    report_path = OUTPUT_DIR / "local_tiebreak_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[local_tiebreak] wrote {summary_path}")
    print(f"[local_tiebreak] wrote {report_path}")


if __name__ == "__main__":
    main()
