from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import load_prediction_frame, run_backtest
from config import ROOT_DIR
from load_submission_config import build_default_inference_args, load_submission_config


SRC_DIR = ROOT_DIR / "app" / "code" / "src"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "topk_objective_search"
DEFAULT_FEATURE_SET = "base_alpha_v3_rs_crowding_mini4"
SUMMARY_COLUMNS = [
    "label",
    "target_topn",
    "gamma",
    "ndcg_at_5",
    "hit_rate_at_5",
    "top5_return_mean",
    "rank_ic_mean",
    "worst_fold_rank_ic",
    "fold1_rank_ic",
    "fold3_rank_ic",
    "negative_day_rank_ic_ratio",
    "cost_after_return",
    "sharpe",
    "max_drawdown",
    "avg_turnover",
    "single_slice_score",
    "passes_acceptance",
    "notes",
    "experiment_dir",
]


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def run_cmd(args: list[str], cwd: Path = ROOT_DIR) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def build_backtest_config() -> dict[str, Any]:
    cfg = load_submission_config()
    args = build_default_inference_args(cfg)
    return {
        "profile_name": "topk_objective_search",
        "top_k": int(args["top_k"]),
        "primary_candidate_size": int(args["primary_candidate_size"]),
        "enable_risk_filters": bool(args["enable_risk_filters"]),
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": float(args["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(args["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(args["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(args["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(args["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(args["risk_penalty_weight"]),
        "weighting_scheme": str(args["weighting_scheme"]),
        "weight_blend_alpha": float(args.get("weight_blend_alpha", 1.0)),
        "max_single_weight": args.get("max_single_weight"),
        "sort_strategy": str(args["sort_strategy"]),
        "transaction_cost": float(args["transaction_cost"]),
        "max_turnover": float(args["max_turnover"]),
    }


def compute_daily_rank_metrics(prediction_path: Path, top_k: int = 5) -> dict[str, float]:
    df = pd.read_csv(prediction_path, encoding="utf-8-sig", dtype={"stock_id": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    rows = []
    for _, day_df in df.groupby("date", sort=True):
        day = day_df.dropna(subset=["pred_return", "target_return"]).copy()
        if day.empty:
            continue
        pred_top = day.sort_values(["pred_return", "stock_id"], ascending=[False, True]).head(top_k)
        true_top = day.sort_values(["target_return", "stock_id"], ascending=[False, True]).head(top_k)
        true_set = set(true_top["stock_id"].astype(str))
        hit_rate = len(set(pred_top["stock_id"].astype(str)) & true_set) / float(top_k)

        # Use positive shifted true returns as relevance for NDCG so large winners matter most.
        min_ret = float(day["target_return"].min())
        relevance = (pred_top["target_return"].astype(float) - min_ret).clip(lower=0.0).to_numpy()
        discounts = 1.0 / np.log2(np.arange(2, len(relevance) + 2))
        dcg = float((relevance * discounts).sum())
        ideal_relevance = (true_top["target_return"].astype(float) - min_ret).clip(lower=0.0).to_numpy()
        ideal_dcg = float((ideal_relevance * discounts[: len(ideal_relevance)]).sum())
        ndcg = dcg / ideal_dcg if ideal_dcg > 1e-12 else 0.0

        valid = day[["pred_return", "target_return"]].apply(pd.to_numeric, errors="coerce").dropna()
        rank_ic = valid["pred_return"].corr(valid["target_return"], method="spearman") if len(valid) > 1 else np.nan
        rows.append(
            {
                "ndcg_at_5": ndcg,
                "hit_rate_at_5": hit_rate,
                "top5_return": float(pred_top["target_return"].mean()),
                "day_rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return {"ndcg_at_5": 0.0, "hit_rate_at_5": 0.0, "top5_return_mean": 0.0, "negative_day_rank_ic_ratio": 0.0}
    rank_ic = pd.to_numeric(daily["day_rank_ic"], errors="coerce").dropna()
    return {
        "ndcg_at_5": float(daily["ndcg_at_5"].mean()),
        "hit_rate_at_5": float(daily["hit_rate_at_5"].mean()),
        "top5_return_mean": float(daily["top5_return"].mean()),
        "negative_day_rank_ic_ratio": float((rank_ic < 0).mean()) if not rank_ic.empty else 0.0,
    }


def write_result_from_backtest(backtest_daily: pd.DataFrame, output_path: Path) -> None:
    if backtest_daily.empty:
        pd.DataFrame(columns=["stock_id", "weight"]).to_csv(output_path, index=False, encoding="utf-8-sig")
        return
    last = backtest_daily.sort_values("date").iloc[-1]
    ids = [item.strip().zfill(6) for item in str(last.get("selected_stock_ids", "")).split(",") if item.strip()]
    weights = [float(item) for item in str(last.get("selected_weights", "")).split(",") if item.strip()]
    pd.DataFrame({"stock_id": ids[: len(weights)], "weight": weights[: len(ids)]}).to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )


def evaluate_experiment(exp_dir: Path, feature_path: Path, target_topn: int, gamma: float) -> dict[str, Any]:
    pred_path = exp_dir / "walk_forward_predictions.csv"
    fold_df = pd.read_csv(exp_dir / "fold_diagnostics.csv", encoding="utf-8-sig")
    metrics = compute_daily_rank_metrics(pred_path)
    prediction_df = load_prediction_frame(pred_path, feature_path)
    bt_summary, bt_daily, _ = run_backtest(prediction_df, build_backtest_config(), prediction_source=str(pred_path))
    bt = bt_summary.iloc[0]

    bt_summary.to_csv(exp_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    bt_daily.to_csv(exp_dir / "backtest_daily.csv", index=False, encoding="utf-8-sig")
    write_result_from_backtest(bt_daily, exp_dir / "result.csv")
    fold_df.to_csv(exp_dir / "fold_results.csv", index=False, encoding="utf-8-sig")
    if (exp_dir / "walk_forward_metrics.csv").exists():
        pd.read_csv(exp_dir / "walk_forward_metrics.csv", encoding="utf-8-sig").to_csv(
            exp_dir / "metrics.csv", index=False, encoding="utf-8-sig"
        )
    else:
        fold_df.to_csv(exp_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    rank_ic_mean = float(fold_df["rank_ic"].mean())
    worst_fold = float(fold_df["rank_ic"].min())
    fold1 = float(fold_df.loc[fold_df["fold_id"].eq(1), "rank_ic"].iloc[0]) if (fold_df["fold_id"].eq(1)).any() else 0.0
    fold3 = float(fold_df.loc[fold_df["fold_id"].eq(3), "rank_ic"].iloc[0]) if (fold_df["fold_id"].eq(3)).any() else 0.0
    single_slice_score = float(pd.to_numeric(bt_daily["net_return"], errors="coerce").fillna(0.0).iloc[-1]) if not bt_daily.empty else 0.0
    row = {
        "label": exp_dir.name,
        "target_topn": int(target_topn),
        "gamma": float(gamma),
        "ndcg_at_5": metrics["ndcg_at_5"],
        "hit_rate_at_5": metrics["hit_rate_at_5"],
        "top5_return_mean": metrics["top5_return_mean"],
        "rank_ic_mean": rank_ic_mean,
        "worst_fold_rank_ic": worst_fold,
        "fold1_rank_ic": fold1,
        "fold3_rank_ic": fold3,
        "negative_day_rank_ic_ratio": metrics["negative_day_rank_ic_ratio"],
        "cost_after_return": float(bt["cumulative_return_after_cost"]),
        "sharpe": float(bt["sharpe_after_cost"]),
        "max_drawdown": float(bt["max_drawdown_after_cost"]),
        "avg_turnover": float(bt["avg_turnover"]),
        "single_slice_score": single_slice_score,
        "passes_acceptance": False,
        "notes": "",
        "experiment_dir": str(exp_dir),
    }
    return row


def write_experiment_report(exp_dir: Path, row: dict[str, Any], baseline: dict[str, Any] | None = None) -> None:
    lines = [
        f"# TopK Objective Experiment: {row['label']}",
        "",
        f"- target_topn: `{row['target_topn']}`",
        f"- gamma: `{row['gamma']}`",
        f"- rank_ic_mean: `{row['rank_ic_mean']:.6f}`",
        f"- worst_fold_rank_ic: `{row['worst_fold_rank_ic']:.6f}`",
        f"- top5_return_mean: `{row['top5_return_mean']:.6f}`",
        f"- cost_after_return: `{row['cost_after_return']:.6f}`",
        f"- sharpe: `{row['sharpe']:.6f}`",
    ]
    if baseline is not None:
        lines.extend(
            [
                "",
                "## Delta vs Baseline",
                "",
                f"- delta_top5_return_mean: `{row['top5_return_mean'] - baseline['top5_return_mean']:.6f}`",
                f"- delta_worst_fold_rank_ic: `{row['worst_fold_rank_ic'] - baseline['worst_fold_rank_ic']:.6f}`",
                f"- delta_sharpe: `{row['sharpe'] - baseline['sharpe']:.6f}`",
            ]
        )
    (exp_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def train_experiment(
    exp_dir: Path,
    feature_path: Path,
    feature_set: str,
    target_topn: int,
    gamma: float,
    epochs: int,
    force: bool,
) -> None:
    pred_path = exp_dir / "walk_forward_predictions.csv"
    if pred_path.exists() and not force:
        return
    exp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SRC_DIR / "train_lstm.py"),
        "--feature_path",
        str(feature_path),
        "--model_dir",
        str(exp_dir),
        "--feature_set",
        feature_set,
        "--sequence_length",
        "20",
        "--valid_dates",
        "20",
        "--num_folds",
        "3",
        "--target_mode",
        "topk_weighted_rank",
        "--topk_focus_k",
        str(target_topn),
        "--topk_gamma",
        str(gamma),
        "--epochs",
        str(epochs),
        "--patience",
        "2",
        "--batch_size",
        "256",
        "--learning_rate",
        "0.001",
        "--hidden_size",
        "64",
        "--num_layers",
        "1",
        "--dropout",
        "0.0",
        "--seed",
        "2026",
    ]
    config = {
        "feature_set": feature_set,
        "model": "lstm",
        "sequence_length": 20,
        "target_mode": "topk_weighted_rank",
        "target_topn": int(target_topn),
        "gamma": float(gamma),
        "epochs": int(epochs),
        "command": cmd,
    }
    (exp_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    run_cmd(cmd)


def baseline_row(feature_path: Path, baseline_dir: Path) -> dict[str, Any]:
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Missing baseline dir: {baseline_dir}")
    return evaluate_experiment(baseline_dir, feature_path, target_topn=0, gamma=0.0)


def apply_acceptance(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    baseline = out[out["target_topn"].eq(0)]
    if baseline.empty:
        return out
    base = baseline.iloc[0]
    for idx, row in out.iterrows():
        if row["target_topn"] == 0:
            out.loc[idx, "passes_acceptance"] = True
            out.loc[idx, "notes"] = "baseline"
            continue
        top5_up = row["top5_return_mean"] > base["top5_return_mean"]
        worst_ok = row["worst_fold_rank_ic"] >= base["worst_fold_rank_ic"] - 0.005
        walk_ok = row["rank_ic_mean"] >= base["rank_ic_mean"] - 0.005 and row["sharpe"] >= base["sharpe"] - 0.25
        slice_only_bad = row["single_slice_score"] > base["single_slice_score"] and not walk_ok
        drawdown_ok = row["max_drawdown"] >= base["max_drawdown"] - 0.01
        passes = bool(top5_up and worst_ok and walk_ok and drawdown_ok and not slice_only_bad)
        out.loc[idx, "passes_acceptance"] = passes
        reasons = []
        if not top5_up:
            reasons.append("top5_not_up")
        if not worst_ok:
            reasons.append("worst_fold_worse")
        if not walk_ok:
            reasons.append("walk_forward_or_sharpe_worse")
        if not drawdown_ok:
            reasons.append("drawdown_worse")
        if slice_only_bad:
            reasons.append("slice_only_gain")
        out.loc[idx, "notes"] = "pass" if passes else ",".join(reasons)
    return out


def write_summary_report(summary: pd.DataFrame, output_dir: Path) -> None:
    baseline = summary[summary["target_topn"].eq(0)].iloc[0]
    candidates = summary[summary["target_topn"].ne(0)].copy()
    if candidates.empty:
        best = baseline
    else:
        best = candidates.sort_values(
            ["passes_acceptance", "top5_return_mean", "worst_fold_rank_ic", "sharpe"],
            ascending=[False, False, False, False],
        ).iloc[0]
    fold13_improved = (
        best["fold1_rank_ic"] >= baseline["fold1_rank_ic"]
        and best["fold3_rank_ic"] >= baseline["fold3_rank_ic"]
        and best["target_topn"] != 0
    )
    ndcg_up_no_return = (
        best["ndcg_at_5"] > baseline["ndcg_at_5"]
        and best["hit_rate_at_5"] >= baseline["hit_rate_at_5"]
        and best["top5_return_mean"] <= baseline["top5_return_mean"]
        and best["target_topn"] != 0
    )
    recommend_replace = bool(best["passes_acceptance"] and best["target_topn"] != 0)
    lines = [
        "# TopK Objective Search Report",
        "",
        "## Required Answers",
        "",
        f"1. Top-K 加权是否提升 Fold 1 和 Fold 3？{'是' if fold13_improved else '否'}。最佳候选 `{best['label']}` 的 Fold1/Fold3 见表。",
        f"2. 是否只提升 HitRate/NDCG 但没有提升收益？{'是' if ndcg_up_no_return else '否'}。",
        f"3. 哪个 gamma/topn 最稳？`{best['label']}`，topn={int(best['target_topn'])}, gamma={best['gamma']:.1f}。",
        f"4. 是否建议替代 cross_section_rank？{'建议替代' if recommend_replace else '不建议替代，只作为融合候选或继续调参'}。",
        "",
        "## Adoption Rules",
        "",
        "- 如果 Top5 收益提升但 worst_fold_rank_ic 明显变差，不采用。",
        "- 如果单切片提升但 Walk-forward 变差，不采用。",
        "",
        "## Leaderboard",
        "",
        "| label | topn | gamma | NDCG@5 | HitRate@5 | Top5 | rank_ic | worst_fold | Fold1 | Fold3 | cost_after | Sharpe | MDD | turnover | slice | pass | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in summary.sort_values(
        ["passes_acceptance", "top5_return_mean", "worst_fold_rank_ic", "sharpe"],
        ascending=[False, False, False, False],
    ).iterrows():
        lines.append(
            f"| {row['label']} | {int(row['target_topn'])} | {row['gamma']:.1f} | {row['ndcg_at_5']:.6f} | "
            f"{row['hit_rate_at_5']:.6f} | {row['top5_return_mean']:.6f} | {row['rank_ic_mean']:.6f} | "
            f"{row['worst_fold_rank_ic']:.6f} | {row['fold1_rank_ic']:.6f} | {row['fold3_rank_ic']:.6f} | "
            f"{row['cost_after_return']:.6f} | {row['sharpe']:.6f} | {row['max_drawdown']:.6f} | "
            f"{row['avg_turnover']:.6f} | {row['single_slice_score']:.6f} | {bool(row['passes_acceptance'])} | {row['notes']} |"
        )
    (output_dir / "topk_objective_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search LSTM top-k weighted rank objective.")
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--feature_set", default=DEFAULT_FEATURE_SET)
    parser.add_argument("--topn", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument("--gamma", nargs="+", type=float, default=[1, 2, 3, 5])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_experiments", type=int, default=0, help="Debug limiter; 0 runs all requested combinations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = resolve_path(args.feature_path)
    output_dir = resolve_path(args.output_dir)
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = output_dir / "baseline_cross_section_rank"
    if not (baseline_dir / "walk_forward_predictions.csv").exists():
        # Reuse the current mainline predictions as a stable baseline artifact.
        baseline_dir.mkdir(parents=True, exist_ok=True)
        for name in ["walk_forward_predictions.csv", "fold_diagnostics.csv", "fold_daily_diagnostics.csv", "walk_forward_metrics.csv", "model_meta.json"]:
            source = ROOT_DIR / "app" / "model" / name
            if source.exists():
                target = baseline_dir / name
                target.write_bytes(source.read_bytes())
        (baseline_dir / "config.json").write_text(
            json.dumps({"target_mode": "cross_section_rank", "source": "app/model"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    rows = [baseline_row(feature_path, baseline_dir)]
    rows[0]["label"] = "baseline_cross_section_rank"
    write_experiment_report(baseline_dir, rows[0])

    combos = [(topn, gamma) for topn in args.topn for gamma in args.gamma]
    if args.max_experiments > 0:
        combos = combos[: args.max_experiments]
    for topn, gamma in combos:
        label = f"topk{topn}_gamma{str(gamma).replace('.', '_')}"
        exp_dir = models_dir / label
        train_experiment(exp_dir, feature_path, args.feature_set, topn, gamma, args.epochs, args.force)
        row = evaluate_experiment(exp_dir, feature_path, topn, gamma)
        rows.append(row)

    summary = apply_acceptance(pd.DataFrame(rows)[SUMMARY_COLUMNS])
    summary_path = output_dir / "topk_objective_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    baseline = summary[summary["target_topn"].eq(0)].iloc[0].to_dict()
    for _, row in summary.iterrows():
        write_experiment_report(Path(row["experiment_dir"]), row.to_dict(), baseline)
    write_summary_report(summary, output_dir)
    print(summary.sort_values(["passes_acceptance", "top5_return_mean"], ascending=[False, False]).to_string(index=False))
    print(f"[topk_objective] wrote {summary_path}")
    print(f"[topk_objective] wrote {output_dir / 'topk_objective_report.md'}")


if __name__ == "__main__":
    main()
