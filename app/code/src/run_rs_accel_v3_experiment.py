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

OUTPUT_DIR = MODEL_DIR / "rs_accel_v3_experiment"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

EXPERIMENTS = [
    {
        "label": "current_mini4_default",
        "feature_set": "base_alpha_v3_rs_crowding_mini4",
        "model_dir": MODEL_DIR,
        "result_path": APP_DIR / "output" / "result.csv",
        "backtest_summary_path": MODEL_DIR / "alpha_v3_rs_crowding_mini4" / "backtest" / "backtest_summary.csv",
        "skip_train": True,
    },
    {
        "label": "rs_accel_v2",
        "feature_set": "base_alpha_v3_rs_crowding_mini4__rel_strength_accel_5d_v2",
        "model_dir": MODELS_DIR / "rs_accel_v2",
        "result_path": RESULTS_DIR / "rs_accel_v2.csv",
        "backtest_summary_path": MODELS_DIR / "rs_accel_v2" / "backtest" / "backtest_summary.csv",
        "skip_train": False,
    },
    {
        "label": "rs_accel_v3a",
        "feature_set": "base_alpha_v3_rs_crowding_mini4__rel_strength_accel_5d_v3a",
        "model_dir": MODELS_DIR / "rs_accel_v3a",
        "result_path": RESULTS_DIR / "rs_accel_v3a.csv",
        "backtest_summary_path": MODELS_DIR / "rs_accel_v3a" / "backtest" / "backtest_summary.csv",
        "skip_train": False,
    },
    {
        "label": "rs_accel_v3b",
        "feature_set": "base_alpha_v3_rs_crowding_mini4__rel_strength_accel_5d_v3b",
        "model_dir": MODELS_DIR / "rs_accel_v3b",
        "result_path": RESULTS_DIR / "rs_accel_v3b.csv",
        "backtest_summary_path": MODELS_DIR / "rs_accel_v3b" / "backtest" / "backtest_summary.csv",
        "skip_train": False,
    },
    {
        "label": "rs_accel_v3c",
        "feature_set": "base_alpha_v3_rs_crowding_mini4__rel_strength_accel_5d_v3c",
        "model_dir": MODELS_DIR / "rs_accel_v3c",
        "result_path": RESULTS_DIR / "rs_accel_v3c.csv",
        "backtest_summary_path": MODELS_DIR / "rs_accel_v3c" / "backtest" / "backtest_summary.csv",
        "skip_train": False,
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


def build_train_cmd(feature_set: str, model_dir: Path) -> list[str]:
    t = BEST_CONFIG["training"]
    return [
        sys.executable, str(SRC_DIR / "train_lstm.py"),
        "--feature_path", str(TEMP_DIR / "train_features.csv"),
        "--model_dir", str(model_dir),
        "--feature_set", feature_set,
        "--sequence_length", str(t["sequence_length"]),
        "--epochs", str(t["epochs"]),
        "--patience", str(t["patience"]),
        "--batch_size", str(t["batch_size"]),
        "--learning_rate", str(t["learning_rate"]),
        "--hidden_size", str(t["hidden_size"]),
        "--num_layers", str(t["num_layers"]),
        "--dropout", str(t["dropout"]),
        "--valid_dates", str(t["valid_dates"]),
        "--num_folds", str(t["num_folds"]),
        "--target_mode", str(t["target_mode"]),
    ]


def build_test_cmd(model_dir: Path, output_path: Path) -> list[str]:
    s = BEST_CONFIG["selection"]
    r = BEST_CONFIG["risk_filter_thresholds"]
    e = BEST_CONFIG["execution"]
    return [
        sys.executable, str(SRC_DIR / "test_lstm.py"),
        "--feature_path", str(TEMP_DIR / "predict_features.csv"),
        "--model_dir", str(model_dir),
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


def build_backtest_cmd(prediction_path: Path, feature_path: Path, model_dir: Path, output_dir: Path) -> list[str]:
    s = BEST_CONFIG["selection"]
    r = BEST_CONFIG["risk_filter_thresholds"]
    e = BEST_CONFIG["execution"]
    return [
        sys.executable, str(SRC_DIR / "backtest.py"),
        "--prediction_path", str(prediction_path),
        "--feature_path", str(feature_path),
        "--model_dir", str(model_dir),
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


def should_retain(delta_slice: float, delta_cum: float, delta_sharpe: float, delta_mdd: float) -> tuple[int, str]:
    checks = [
        ("case_slice_score_up", delta_slice > 0.0),
        ("local_cumulative_not_worse", delta_cum >= -0.02),
        ("local_sharpe_not_worse", delta_sharpe >= -0.20),
        ("drawdown_not_materially_worse", delta_mdd >= -0.01),
    ]
    failed = [name for name, ok in checks if not ok]
    return (0, " / ".join(failed)) if failed else (1, "all_gate_checks_passed")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "train", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])
    run_cmd([sys.executable, str(SRC_DIR / "featurework.py"), "--mode", "predict", "--data_dir", str(DATA_DIR), "--temp_dir", str(TEMP_DIR)])

    for exp in EXPERIMENTS:
        if exp["skip_train"]:
            continue
        exp["model_dir"].mkdir(parents=True, exist_ok=True)
        run_cmd(build_train_cmd(exp["feature_set"], exp["model_dir"]))
        run_cmd(build_test_cmd(exp["model_dir"], exp["result_path"]))
        run_cmd(build_backtest_cmd(exp["model_dir"] / "walk_forward_predictions.csv", TEMP_DIR / "train_features.csv", exp["model_dir"], exp["model_dir"] / "backtest"))

    base = EXPERIMENTS[0]
    base_meta = load_json(base["model_dir"] / "model_meta.json")
    base_bt = pd.read_csv(base["backtest_summary_path"]).iloc[0]
    base_score = score_result_against_case_test(base["result_path"])
    rows = [{
        "label": base["label"],
        "feature_set": base_meta["feature_set"],
        "feature_count": len(base_meta["feature_columns"]),
        "rank_ic_mean": base_meta["walk_forward_summary"]["rank_ic_mean"],
        "top5_mean_return_mean": base_meta["walk_forward_summary"]["top5_mean_return_mean"],
        "cumulative_return_after_cost": float(base_bt["cumulative_return_after_cost"]),
        "sharpe_after_cost": float(base_bt["sharpe_after_cost"]),
        "max_drawdown_after_cost": float(base_bt["max_drawdown_after_cost"]),
        "avg_turnover": float(base_bt["avg_turnover"]),
        "score_self_case_slice": base_score,
        "delta_case_slice_score": 0.0,
        "delta_cumulative_return_after_cost": 0.0,
        "delta_sharpe_after_cost": 0.0,
        "delta_max_drawdown_after_cost": 0.0,
        "retain_candidate": 0,
        "retain_reason": "baseline",
    }]

    for exp in EXPERIMENTS[1:]:
        meta = load_json(exp["model_dir"] / "model_meta.json")
        bt = pd.read_csv(exp["backtest_summary_path"]).iloc[0]
        score = score_result_against_case_test(exp["result_path"])
        delta_slice = score - base_score
        delta_cum = float(bt["cumulative_return_after_cost"]) - float(base_bt["cumulative_return_after_cost"])
        delta_sharpe = float(bt["sharpe_after_cost"]) - float(base_bt["sharpe_after_cost"])
        delta_mdd = float(bt["max_drawdown_after_cost"]) - float(base_bt["max_drawdown_after_cost"])
        keep, reason = should_retain(delta_slice, delta_cum, delta_sharpe, delta_mdd)
        rows.append({
            "label": exp["label"],
            "feature_set": meta["feature_set"],
            "feature_count": len(meta["feature_columns"]),
            "rank_ic_mean": meta["walk_forward_summary"]["rank_ic_mean"],
            "top5_mean_return_mean": meta["walk_forward_summary"]["top5_mean_return_mean"],
            "cumulative_return_after_cost": float(bt["cumulative_return_after_cost"]),
            "sharpe_after_cost": float(bt["sharpe_after_cost"]),
            "max_drawdown_after_cost": float(bt["max_drawdown_after_cost"]),
            "avg_turnover": float(bt["avg_turnover"]),
            "score_self_case_slice": score,
            "delta_case_slice_score": delta_slice,
            "delta_cumulative_return_after_cost": delta_cum,
            "delta_sharpe_after_cost": delta_sharpe,
            "delta_max_drawdown_after_cost": delta_mdd,
            "retain_candidate": keep,
            "retain_reason": reason,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "rs_accel_v3_summary.csv", index=False, encoding="utf-8-sig")
    df[df["retain_candidate"] == 1].to_csv(OUTPUT_DIR / "rs_accel_v3_retained.csv", index=False, encoding="utf-8-sig")

    best_slice = df[df["label"] != "current_mini4_default"].sort_values(
        ["delta_case_slice_score", "delta_cumulative_return_after_cost"], ascending=[False, False]
    ).iloc[0]
    best_def = df[df["label"] != "current_mini4_default"].sort_values(
        ["delta_cumulative_return_after_cost", "delta_case_slice_score"], ascending=[False, False]
    ).iloc[0]
    lines = [
        "# rs_accel v3 保守化实验",
        "",
        f"- 切片提升最强：`{best_slice['label']}`，`{best_slice['delta_case_slice_score']:.6f}`",
        f"- 本地收益损伤最小：`{best_def['label']}`，`{best_def['delta_cumulative_return_after_cost']:.6f}`",
        "",
        "| 方案 | slice | Δslice | cum_after | Δcum | sharpe | Δsharpe | max_dd | Δdd | retain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['label']} | {row['score_self_case_slice']:.6f} | {row['delta_case_slice_score']:.6f} | "
            f"{row['cumulative_return_after_cost']:.6f} | {row['delta_cumulative_return_after_cost']:.6f} | "
            f"{row['sharpe_after_cost']:.6f} | {row['delta_sharpe_after_cost']:.6f} | "
            f"{row['max_drawdown_after_cost']:.6f} | {row['delta_max_drawdown_after_cost']:.6f} | {row['retain_reason']} |"
        )
    (OUTPUT_DIR / "rs_accel_v3_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[rs_accel_v3] wrote {OUTPUT_DIR / 'rs_accel_v3_summary.csv'}")
    print(f"[rs_accel_v3] wrote {OUTPUT_DIR / 'rs_accel_v3_retained.csv'}")
    print(f"[rs_accel_v3] wrote {OUTPUT_DIR / 'rs_accel_v3_report.md'}")


if __name__ == "__main__":
    main()
