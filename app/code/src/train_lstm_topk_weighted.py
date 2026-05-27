from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import load_prediction_frame, run_backtest
from build_formal_model_comparison import compute_hit_rate_at_k, compute_ndcg_at_k
from config import BEST_CONFIG, ROOT_DIR
from evaluate_rank_stability import append_stability_summary, evaluate_rank_stability


APP_DIR = ROOT_DIR / "app"
SRC_DIR = APP_DIR / "code" / "src"
TEMP_DIR = APP_DIR / "temp"
MODEL_DIR = APP_DIR / "model"

DEFAULT_FEATURE_PATH = TEMP_DIR / "train_features.csv"
DEFAULT_BASELINE_MODEL_DIR = MODEL_DIR
DEFAULT_OUTPUT_DIR = MODEL_DIR / "topk_objective_search"
DEFAULT_GAMMAS = "1,2,3"
DEFAULT_TOPKS = "5,10,20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search LSTM top-k weighted rank objectives.")
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--baseline_model_dir", default=str(DEFAULT_BASELINE_MODEL_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gammas", default=DEFAULT_GAMMAS)
    parser.add_argument("--topks", default=DEFAULT_TOPKS)
    parser.add_argument("--retrain", type=int, choices=[0, 1], default=0)
    parser.add_argument("--skip_training", type=int, choices=[0, 1], default=0)
    parser.add_argument("--python_bin", default=sys.executable)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_backtest_config(profile_name: str) -> dict:
    selection = BEST_CONFIG["selection"]
    risk = BEST_CONFIG["risk_filter_thresholds"]
    execution = BEST_CONFIG["execution"]
    return {
        "profile_name": profile_name,
        "top_k": int(selection["top_k"]),
        "primary_candidate_size": int(selection["primary_candidate_size"]),
        "enable_risk_filters": int(bool(selection["enable_risk_filters"])),
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": float(risk["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(risk["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(risk["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(risk["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(risk["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(risk["risk_penalty_weight"]),
        "weighting_scheme": selection["weighting_scheme"],
        "sort_strategy": selection["sort_strategy"],
        "transaction_cost": float(execution["transaction_cost"]),
        "max_turnover": float(execution["max_turnover"]),
    }


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=ROOT_DIR)


def candidate_done(model_dir: Path) -> bool:
    return (
        (model_dir / "model_meta.json").exists()
        and (model_dir / "walk_forward_predictions.csv").exists()
        and (model_dir / "fold_diagnostics.csv").exists()
        and (model_dir / "fold_daily_diagnostics.csv").exists()
    )


def build_train_cmd(python_bin: str, feature_path: Path, model_dir: Path, topk: int, gamma: float) -> list[str]:
    training = BEST_CONFIG["training"]
    return [
        python_bin,
        str(SRC_DIR / "train_lstm.py"),
        "--feature_path",
        str(feature_path),
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
        "--topk_focus_k",
        str(topk),
        "--topk_gamma",
        str(gamma),
    ]


def compute_top5_mean_return(prediction_df: pd.DataFrame) -> float:
    returns: list[float] = []
    for _, day_df in prediction_df.groupby("date", sort=True):
        top5 = day_df.sort_values("pred_return", ascending=False).head(5)
        if not top5.empty:
            returns.append(float(pd.to_numeric(top5["target_return"], errors="coerce").mean()))
    return float(np.mean(returns)) if returns else 0.0


def evaluate_model(
    label: str,
    model_dir: Path,
    feature_path: Path,
    output_dir: Path,
    target_topk: int,
    gamma: float,
) -> dict:
    prediction_path = model_dir / "walk_forward_predictions.csv"
    prediction_df = load_prediction_frame(prediction_path, feature_path)
    prediction_df["date"] = pd.to_datetime(prediction_df["date"])

    backtest_dir = output_dir / "backtests" / label
    backtest_dir.mkdir(parents=True, exist_ok=True)
    backtest_summary_df, backtest_daily_df, _ = run_backtest(
        prediction_df=prediction_df,
        config=build_backtest_config(profile_name=label),
        prediction_source="replay_walk_forward_predictions",
    )
    backtest_summary_path = backtest_dir / "backtest_summary.csv"
    backtest_daily_path = backtest_dir / "backtest_daily.csv"
    backtest_summary_df.to_csv(backtest_summary_path, index=False, encoding="utf-8-sig")
    backtest_daily_df.to_csv(backtest_daily_path, index=False, encoding="utf-8-sig")
    backtest_row = backtest_summary_df.iloc[0]

    stability = evaluate_rank_stability(
        experiment_name=f"topk_objective_search/{label}",
        prediction_path=prediction_path,
        fold_diagnostics_path=model_dir / "fold_diagnostics.csv",
        fold_daily_diagnostics_path=model_dir / "fold_daily_diagnostics.csv",
        backtest_summary_path=backtest_summary_path,
        extra_fields={
            "label": label,
            "target_topk": int(target_topk),
            "gamma": float(gamma),
            "model_dir": str(model_dir),
        },
    )
    append_stability_summary(stability)

    meta = load_json(model_dir / "model_meta.json")
    return {
        "label": label,
        "model_dir": str(model_dir),
        "target_mode": meta.get("target_mode", ""),
        "sample_weight_mode": meta.get("sample_weight_mode", "uniform"),
        "target_topk": int(target_topk),
        "gamma": float(gamma),
        "ndcg_at_5": compute_ndcg_at_k(prediction_df, 5),
        "hit_rate_at_5": compute_hit_rate_at_k(prediction_df, 5),
        "top5_mean_return": compute_top5_mean_return(prediction_df),
        "rank_ic_mean": float(stability["rank_ic_mean"]),
        "rank_ic_std": float(stability["rank_ic_std"]),
        "worst_fold_rank_ic": float(stability["worst_fold_rank_ic"]),
        "negative_day_rank_ic_ratio": float(stability["negative_day_rank_ic_ratio"]),
        "cumulative_return_after_cost": float(backtest_row["cumulative_return_after_cost"]),
        "sharpe_after_cost": float(backtest_row["sharpe_after_cost"]),
        "max_drawdown_after_cost": float(backtest_row["max_drawdown_after_cost"]),
        "avg_turnover": float(backtest_row["avg_turnover"]),
    }


def add_acceptance_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    baseline = out[out["label"].eq("baseline_cross_section_rank")].iloc[0]
    checks = {
        "ndcg_at_5": "delta_ndcg_at_5",
        "hit_rate_at_5": "delta_hit_rate_at_5",
        "top5_mean_return": "delta_top5_mean_return",
        "cumulative_return_after_cost": "delta_cumulative_return_after_cost",
        "worst_fold_rank_ic": "delta_worst_fold_rank_ic",
    }
    for metric, delta_column in checks.items():
        out[delta_column] = out[metric] - float(baseline[metric])
    out["passes_acceptance"] = (
        out["delta_ndcg_at_5"].ge(0)
        & out["delta_hit_rate_at_5"].ge(0)
        & out["delta_top5_mean_return"].ge(0)
        & out["delta_cumulative_return_after_cost"].ge(0)
        & out["delta_worst_fold_rank_ic"].ge(0)
    )
    return out


def write_report(summary_df: pd.DataFrame, output_path: Path) -> None:
    ranked = summary_df.sort_values(
        [
            "passes_acceptance",
            "delta_ndcg_at_5",
            "delta_hit_rate_at_5",
            "delta_top5_mean_return",
            "delta_cumulative_return_after_cost",
            "delta_worst_fold_rank_ic",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    lines = [
        "# TopK Objective Search",
        "",
        "## Search Space",
        "",
        "- target_mode: `topk_weighted_rank`",
        "- gamma: `1, 2, 3`",
        "- topk: `5, 10, 20`",
        "- Acceptance: NDCG@5, HitRate@5, Top5平均收益、回测收益提升，且 worst_fold_rank_ic 不变差。",
        "",
        "## Best Candidate",
        "",
        f"- best_label: `{best['label']}`",
        f"- passes_acceptance: `{bool(best['passes_acceptance'])}`",
        f"- delta_ndcg_at_5: `{best['delta_ndcg_at_5']:.6f}`",
        f"- delta_hit_rate_at_5: `{best['delta_hit_rate_at_5']:.6f}`",
        f"- delta_top5_mean_return: `{best['delta_top5_mean_return']:.6f}`",
        f"- delta_cumulative_return_after_cost: `{best['delta_cumulative_return_after_cost']:.6f}`",
        f"- delta_worst_fold_rank_ic: `{best['delta_worst_fold_rank_ic']:.6f}`",
        "",
        "## Leaderboard",
        "",
        "| label | topk | gamma | NDCG@5 | HitRate@5 | Top5收益 | 回测收益 | worst_fold_rank_ic | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in ranked.iterrows():
        lines.append(
            f"| {row['label']} | {int(row['target_topk'])} | {float(row['gamma']):.1f} | "
            f"{row['ndcg_at_5']:.6f} | {row['hit_rate_at_5']:.6f} | {row['top5_mean_return']:.6f} | "
            f"{row['cumulative_return_after_cost']:.6f} | {row['worst_fold_rank_ic']:.6f} | "
            f"{'yes' if bool(row['passes_acceptance']) else 'no'} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    feature_path = Path(args.feature_path)
    baseline_model_dir = Path(args.baseline_model_dir)
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        evaluate_model(
            label="baseline_cross_section_rank",
            model_dir=baseline_model_dir,
            feature_path=feature_path,
            output_dir=output_dir,
            target_topk=0,
            gamma=0.0,
        )
    ]

    for topk in parse_int_list(args.topks):
        for gamma in parse_float_list(args.gammas):
            label = f"topk{topk}_gamma{str(gamma).replace('.', '_')}"
            model_dir = models_dir / label
            model_dir.mkdir(parents=True, exist_ok=True)
            if (not candidate_done(model_dir) or bool(args.retrain)) and not bool(args.skip_training):
                run_cmd(
                    build_train_cmd(
                        python_bin=args.python_bin,
                        feature_path=feature_path,
                        model_dir=model_dir,
                        topk=topk,
                        gamma=gamma,
                    )
                )
            if candidate_done(model_dir):
                rows.append(
                    evaluate_model(
                        label=label,
                        model_dir=model_dir,
                        feature_path=feature_path,
                        output_dir=output_dir,
                        target_topk=topk,
                        gamma=gamma,
                    )
                )

    summary_df = add_acceptance_columns(pd.DataFrame(rows)).sort_values(
        [
            "passes_acceptance",
            "delta_ndcg_at_5",
            "delta_hit_rate_at_5",
            "delta_top5_mean_return",
            "delta_cumulative_return_after_cost",
            "delta_worst_fold_rank_ic",
        ],
        ascending=[False, False, False, False, False, False],
    )
    summary_path = output_dir / "topk_objective_summary.csv"
    report_path = output_dir / "topk_objective_report.md"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary_df, report_path)

    best = summary_df.iloc[0]
    print(f"[topk_objective_search] wrote {summary_path}")
    print(f"[topk_objective_search] wrote {report_path}")
    print(f"[topk_objective_search] best={best['label']} pass={bool(best['passes_acceptance'])}")


if __name__ == "__main__":
    main()
