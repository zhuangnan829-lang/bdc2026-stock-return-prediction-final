from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import load_prediction_frame, run_backtest
from config import ROOT_DIR
from load_submission_config import build_default_inference_args, load_submission_config


SRC_DIR = ROOT_DIR / "app" / "code" / "src"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "label_variant_search"
DEFAULT_FEATURE_SET = "base_alpha_v3_rs_crowding_mini4"
LABEL_TYPES = ["original_return", "residual_return", "risk_adjusted_return", "clipped_return"]
OBJECTIVES = ["cross_section_rank", "topk_weighted_rank"]
SUMMARY_COLUMNS = [
    "label_type",
    "objective",
    "rank_ic_mean",
    "worst_fold_rank_ic",
    "top5_return_mean",
    "top5_return_min_by_fold",
    "NDCG@5",
    "HitRate@5",
    "cost_after_return",
    "Sharpe",
    "max_drawdown",
    "avg_turnover",
    "single_slice_score",
    "fold1_rank_ic",
    "fold3_rank_ic",
    "negative_day_rank_ic_ratio",
    "recommend_for_mainline",
    "notes",
    "experiment_dir",
    "status",
]


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def run_cmd(args: list[str], cwd: Path = ROOT_DIR) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSTM label variant experiments under the same Top5 protocol.")
    parser.add_argument("--feature_path", default="", help="Optional prebuilt train_features.csv.")
    parser.add_argument("--data_dir", default=str(ROOT_DIR / "app" / "data"))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--label_types", nargs="+", default=LABEL_TYPES, choices=LABEL_TYPES)
    parser.add_argument("--objectives", nargs="+", default=OBJECTIVES, choices=OBJECTIVES)
    parser.add_argument("--feature_set", default=DEFAULT_FEATURE_SET)
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--valid_dates", type=int, default=20)
    parser.add_argument("--num_folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--topk_focus_k", type=int, default=30)
    parser.add_argument("--topk_gamma", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_base_features(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.feature_path:
        feature_path = resolve_path(args.feature_path)
        if not feature_path.exists():
            raise FileNotFoundError(f"feature_path does not exist: {feature_path}")
    else:
        temp_dir = output_dir / "temp"
        feature_path = temp_dir / "train_features.csv"
        if args.force or not feature_path.exists():
            run_cmd(
                [
                    sys.executable,
                    str(SRC_DIR / "featurework.py"),
                    "--mode",
                    "train",
                    "--data_dir",
                    str(resolve_path(args.data_dir)),
                    "--temp_dir",
                    str(temp_dir),
                ]
            )

    df = pd.read_csv(feature_path, encoding="utf-8-sig", dtype={"stock_id": str})
    df = ensure_label_columns(df)
    enriched_path = output_dir / "temp" / "train_features_with_label_variants.csv"
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    return enriched_path


def ensure_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns or "stock_id" not in out.columns:
        raise ValueError("Feature frame must contain date and stock_id.")
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if "original_return" not in out.columns:
        if "target_return" not in out.columns:
            if {"future_open_1", "future_open_5"}.issubset(out.columns):
                out["target_return"] = (
                    pd.to_numeric(out["future_open_5"], errors="coerce")
                    - pd.to_numeric(out["future_open_1"], errors="coerce")
                ) / pd.to_numeric(out["future_open_1"], errors="coerce")
            else:
                raise ValueError("Cannot build label variants without target_return or future_open_1/future_open_5.")
        out["original_return"] = pd.to_numeric(out["target_return"], errors="coerce")

    out["original_return"] = pd.to_numeric(out["original_return"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    out["target_return"] = out["original_return"]
    out["market_average_future_return"] = out.groupby("date")["original_return"].transform("mean")
    out["residual_return"] = out["original_return"] - out["market_average_future_return"]

    if "volatility_20d" not in out.columns:
        raise ValueError("Feature frame missing volatility_20d required for risk_adjusted_return.")
    volatility = pd.to_numeric(out["volatility_20d"], errors="coerce").abs().clip(lower=1e-4)
    out["risk_adjusted_return"] = out["original_return"] / (volatility + 1e-12)

    lower = out.groupby("date")["original_return"].transform(lambda s: s.quantile(0.05))
    upper = out.groupby("date")["original_return"].transform(lambda s: s.quantile(0.95))
    out["clipped_return"] = out["original_return"].clip(lower=lower, upper=upper)
    return out


def build_variant_feature_file(base_feature_path: Path, label_type: str, output_dir: Path, force: bool) -> Path:
    variant_dir = output_dir / "feature_variants"
    variant_dir.mkdir(parents=True, exist_ok=True)
    variant_path = variant_dir / f"train_features_{label_type}.csv"
    if variant_path.exists() and not force:
        return variant_path
    df = pd.read_csv(base_feature_path, encoding="utf-8-sig", dtype={"stock_id": str})
    if label_type not in df.columns:
        raise ValueError(f"Label column not found in base features: {label_type}")
    df["eval_target_return"] = pd.to_numeric(df["original_return"], errors="coerce")
    df["target_return"] = pd.to_numeric(df[label_type], errors="coerce")
    df.to_csv(variant_path, index=False, encoding="utf-8-sig")
    return variant_path


def build_backtest_config() -> dict[str, Any]:
    cfg = load_submission_config()
    args = build_default_inference_args(cfg)
    return {
        "profile_name": "label_variant_search",
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


def train_one(
    *,
    exp_dir: Path,
    variant_path: Path,
    label_type: str,
    objective: str,
    args: argparse.Namespace,
) -> None:
    pred_path = exp_dir / "walk_forward_predictions.csv"
    if pred_path.exists() and not args.force:
        return
    exp_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "label_type": label_type,
        "objective": objective,
        "feature_set": args.feature_set,
        "sequence_length": int(args.sequence_length),
        "topk_focus_k": int(args.topk_focus_k) if objective == "topk_weighted_rank" else None,
        "topk_gamma": float(args.topk_gamma) if objective == "topk_weighted_rank" else None,
        "epochs": int(args.epochs),
        "evaluation_target": "original_return",
        "notes": "target_return is replaced by label_type only for training; metrics/backtest are recomputed on original_return.",
    }
    (exp_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(SRC_DIR / "train_lstm.py"),
        "--feature_path",
        str(variant_path),
        "--model_dir",
        str(exp_dir),
        "--feature_set",
        args.feature_set,
        "--sequence_length",
        str(args.sequence_length),
        "--valid_dates",
        str(args.valid_dates),
        "--num_folds",
        str(args.num_folds),
        "--target_mode",
        objective,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
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
    if objective == "topk_weighted_rank":
        cmd.extend(["--topk_focus_k", str(args.topk_focus_k), "--topk_gamma", str(args.topk_gamma)])
    run_cmd(cmd)


def build_eval_predictions(exp_dir: Path, variant_path: Path) -> Path:
    pred = pd.read_csv(exp_dir / "walk_forward_predictions.csv", encoding="utf-8-sig", dtype={"stock_id": str})
    features = pd.read_csv(
        variant_path,
        encoding="utf-8-sig",
        dtype={"stock_id": str},
        usecols=["stock_id", "date", "original_return"],
    )
    pred["stock_id"] = pred["stock_id"].astype(str).str.zfill(6)
    features["stock_id"] = features["stock_id"].astype(str).str.zfill(6)
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    pred = pred.rename(columns={"target_return": "train_label_return"})
    pred = pred.merge(features, on=["stock_id", "date"], how="left")
    pred["target_return"] = pd.to_numeric(pred["original_return"], errors="coerce")
    pred = pred.drop(columns=["original_return"])
    out_path = exp_dir / "eval_predictions_original_return.csv"
    pred.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def compute_rank_metrics(prediction_path: Path, top_k: int = 5) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(prediction_path, encoding="utf-8-sig", dtype={"stock_id": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    rows = []
    for date, day_df in df.groupby("date", sort=True):
        day = day_df.dropna(subset=["pred_return", "target_return"]).copy()
        if day.empty:
            continue
        pred_top = day.sort_values(["pred_return", "stock_id"], ascending=[False, True]).head(top_k)
        true_top = day.sort_values(["target_return", "stock_id"], ascending=[False, True]).head(top_k)
        hit_rate = len(set(pred_top["stock_id"]) & set(true_top["stock_id"])) / float(top_k)

        min_ret = float(day["target_return"].min())
        relevance = (pred_top["target_return"].astype(float) - min_ret).clip(lower=0.0).to_numpy()
        ideal_relevance = (true_top["target_return"].astype(float) - min_ret).clip(lower=0.0).to_numpy()
        discounts = 1.0 / np.log2(np.arange(2, top_k + 2))
        dcg = float((relevance * discounts[: len(relevance)]).sum())
        ideal_dcg = float((ideal_relevance * discounts[: len(ideal_relevance)]).sum())
        ndcg = dcg / ideal_dcg if ideal_dcg > 1e-12 else 0.0

        rank_ic = np.nan
        valid = day[["pred_return", "target_return"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(valid) > 1 and valid["pred_return"].nunique() > 1 and valid["target_return"].nunique() > 1:
            rank_ic = valid["pred_return"].corr(valid["target_return"], method="spearman")
        rows.append(
            {
                "date": date,
                "fold_id": int(day["fold_id"].iloc[0]) if "fold_id" in day.columns else 0,
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "top5_return": float(pred_top["target_return"].mean()),
                "NDCG@5": ndcg,
                "HitRate@5": hit_rate,
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return {}, daily, pd.DataFrame()
    fold = (
        daily.groupby("fold_id", as_index=False)
        .agg(
            rank_ic=("rank_ic", "mean"),
            top5_return=("top5_return", "mean"),
            NDCG_at_5=("NDCG@5", "mean"),
            HitRate_at_5=("HitRate@5", "mean"),
        )
        .sort_values("fold_id")
    )
    rank_ic_series = pd.to_numeric(daily["rank_ic"], errors="coerce").dropna()
    metrics = {
        "rank_ic_mean": float(fold["rank_ic"].mean()),
        "worst_fold_rank_ic": float(fold["rank_ic"].min()),
        "top5_return_mean": float(daily["top5_return"].mean()),
        "top5_return_min_by_fold": float(fold["top5_return"].min()),
        "NDCG@5": float(daily["NDCG@5"].mean()),
        "HitRate@5": float(daily["HitRate@5"].mean()),
        "fold1_rank_ic": float(fold.loc[fold["fold_id"].eq(1), "rank_ic"].iloc[0])
        if (fold["fold_id"].eq(1)).any()
        else np.nan,
        "fold3_rank_ic": float(fold.loc[fold["fold_id"].eq(3), "rank_ic"].iloc[0])
        if (fold["fold_id"].eq(3)).any()
        else np.nan,
        "negative_day_rank_ic_ratio": float((rank_ic_series < 0).mean()) if not rank_ic_series.empty else 0.0,
    }
    return metrics, daily, fold


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


def evaluate_one(exp_dir: Path, variant_path: Path, label_type: str, objective: str) -> dict[str, Any]:
    eval_pred_path = build_eval_predictions(exp_dir, variant_path)
    rank_metrics, daily_metrics, fold_metrics = compute_rank_metrics(eval_pred_path)
    prediction_df = load_prediction_frame(eval_pred_path, variant_path)
    bt_summary, bt_daily, holdings = run_backtest(
        prediction_df,
        build_backtest_config(),
        prediction_source=str(eval_pred_path),
    )

    fold_metrics.to_csv(exp_dir / "fold_results.csv", index=False, encoding="utf-8-sig")
    daily_metrics.to_csv(exp_dir / "daily_rank_metrics.csv", index=False, encoding="utf-8-sig")
    bt_summary.to_csv(exp_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    bt_daily.to_csv(exp_dir / "backtest_daily.csv", index=False, encoding="utf-8-sig")
    holdings.to_csv(exp_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    write_result_from_backtest(bt_daily, exp_dir / "result.csv")

    bt = bt_summary.iloc[0]
    row = {
        "label_type": label_type,
        "objective": objective,
        **rank_metrics,
        "cost_after_return": float(bt["cumulative_return_after_cost"]),
        "Sharpe": float(bt["sharpe_after_cost"]),
        "max_drawdown": float(bt["max_drawdown_after_cost"]),
        "avg_turnover": float(bt["avg_turnover"]),
        "single_slice_score": float(pd.to_numeric(bt_daily["net_return"], errors="coerce").fillna(0.0).iloc[-1])
        if not bt_daily.empty
        else 0.0,
        "recommend_for_mainline": False,
        "notes": "",
        "experiment_dir": str(exp_dir),
        "status": "ok",
    }
    pd.DataFrame([row]).to_csv(exp_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    return row


def mark_recommendations(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    ok = out["status"].eq("ok")
    if not ok.any():
        return out
    out["recommend_for_mainline"] = False
    out["notes"] = out["notes"].fillna("")
    for objective in out.loc[ok, "objective"].dropna().unique():
        baseline_rows = out[
            ok & out["label_type"].eq("original_return") & out["objective"].eq(objective)
        ]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        for idx, row in out[ok & out["objective"].eq(objective) & ~out["label_type"].eq("original_return")].iterrows():
            improves_stability = row["worst_fold_rank_ic"] > baseline["worst_fold_rank_ic"] + 1e-4
            improves_top5 = row["top5_return_mean"] > baseline["top5_return_mean"] + 1e-4
            keeps_slice = row["single_slice_score"] >= baseline["single_slice_score"] - 0.005
            keeps_drawdown = row["max_drawdown"] >= baseline["max_drawdown"] - 0.03
            recommend = bool((improves_stability or improves_top5) and keeps_slice and keeps_drawdown)
            out.at[idx, "recommend_for_mainline"] = recommend
            flags = []
            if improves_stability:
                flags.append("stability_improved")
            if improves_top5:
                flags.append("top5_improved")
            if not keeps_slice:
                flags.append("single_slice_drop")
            if not keeps_drawdown:
                flags.append("drawdown_worse")
            out.at[idx, "notes"] = ",".join(flags) if flags else "no_clear_edge"
    return out


def write_experiment_report(exp_dir: Path, row: dict[str, Any]) -> None:
    lines = [
        f"# Label Variant Experiment: {row['label_type']} / {row['objective']}",
        "",
        "- Evaluation target: `original_return`",
        f"- rank_ic_mean: `{row.get('rank_ic_mean', np.nan):.6f}`",
        f"- worst_fold_rank_ic: `{row.get('worst_fold_rank_ic', np.nan):.6f}`",
        f"- top5_return_mean: `{row.get('top5_return_mean', np.nan):.6f}`",
        f"- NDCG@5: `{row.get('NDCG@5', np.nan):.6f}`",
        f"- HitRate@5: `{row.get('HitRate@5', np.nan):.6f}`",
        f"- cost_after_return: `{row.get('cost_after_return', np.nan):.6f}`",
        f"- Sharpe: `{row.get('Sharpe', np.nan):.6f}`",
        f"- max_drawdown: `{row.get('max_drawdown', np.nan):.6f}`",
        "",
        "The training label may differ by variant, but all adoption metrics are recomputed on the original future return.",
        "",
    ]
    (exp_dir / "report.md").write_text("\n".join(lines), encoding="utf-8-sig")


def write_summary_report(summary: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> Path:
    ok = summary[summary["status"].eq("ok")].copy()
    lines = [
        "# Label Variant Search Report",
        "",
        "Protocol: LSTM sl20, base_alpha_v3_rs_crowding_mini4, original-return evaluation, current default backtest selection config.",
        f"TopK objective grid uses topn=`{args.topk_focus_k}`, gamma=`{args.topk_gamma}` when objective is `topk_weighted_rank`.",
        "",
    ]
    if ok.empty:
        lines.extend(["No successful experiments.", ""])
    else:
        ranked = ok.sort_values(["top5_return_mean", "worst_fold_rank_ic", "Sharpe"], ascending=[False, False, False])
        best = ranked.iloc[0]
        lines.extend(
            [
                "## Best By Top5",
                "",
                f"- label_type: `{best['label_type']}`",
                f"- objective: `{best['objective']}`",
                f"- top5_return_mean: `{best['top5_return_mean']:.6f}`",
                f"- worst_fold_rank_ic: `{best['worst_fold_rank_ic']:.6f}`",
                f"- Sharpe: `{best['Sharpe']:.6f}`",
                "",
                "## Adoption Check",
                "",
            ]
        )
        recommended = ok[ok["recommend_for_mainline"].astype(bool)]
        if recommended.empty:
            lines.append(
                "No residual/risk_adjusted/clipped label met the conservative adoption rule versus its same-objective original-return baseline."
            )
        else:
            for _, row in recommended.iterrows():
                lines.append(
                    f"- Candidate `{row['label_type']} / {row['objective']}`: "
                    f"top5=`{row['top5_return_mean']:.6f}`, worst_fold=`{row['worst_fold_rank_ic']:.6f}`, "
                    f"single_slice=`{row['single_slice_score']:.6f}`."
                )
        lines.extend(
            [
                "",
                "## Required Answers",
                "",
                "1. residual_return is mostly rank-equivalent to original_return for cross-sectional rank objectives because it subtracts a same-date market average. Do not treat ties as evidence.",
                "2. risk_adjusted_return is useful only if its stability/Top5 gain survives the single-slice check. In this run, single-slice deterioration blocks adoption when present.",
                "3. clipped_return may be a stability candidate if it improves worst-fold RankIC or Sharpe without materially reducing slice score, but it still needs a full-epoch replay.",
                "4. If a label lifts NDCG/HitRate but hurts worst-fold RankIC or single-slice score, keep it as a fusion candidate only.",
                "5. Any candidate should be replayed with full epochs before touching robust/aggressive configs.",
                "",
            ]
        )
    if summary["status"].ne("ok").any():
        lines.extend(["## Failed Experiments", ""])
        for _, row in summary[summary["status"].ne("ok")].iterrows():
            lines.append(f"- `{row['label_type']} / {row['objective']}`: {row['notes']}")
        lines.append("")
    report_path = output_dir / "label_variant_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return report_path


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_feature_path = ensure_base_features(args, output_dir)

    rows: list[dict[str, Any]] = []
    for label_type in args.label_types:
        variant_path = build_variant_feature_file(base_feature_path, label_type, output_dir, args.force)
        for objective in args.objectives:
            exp_dir = output_dir / "experiments" / f"{label_type}__{objective}"
            print(f"[label_variant] running label_type={label_type} objective={objective}")
            try:
                train_one(exp_dir=exp_dir, variant_path=variant_path, label_type=label_type, objective=objective, args=args)
                row = evaluate_one(exp_dir, variant_path, label_type, objective)
                write_experiment_report(exp_dir, row)
            except Exception as exc:
                row = {
                    "label_type": label_type,
                    "objective": objective,
                    "rank_ic_mean": np.nan,
                    "worst_fold_rank_ic": np.nan,
                    "top5_return_mean": np.nan,
                    "top5_return_min_by_fold": np.nan,
                    "NDCG@5": np.nan,
                    "HitRate@5": np.nan,
                    "cost_after_return": np.nan,
                    "Sharpe": np.nan,
                    "max_drawdown": np.nan,
                    "avg_turnover": np.nan,
                    "single_slice_score": np.nan,
                    "fold1_rank_ic": np.nan,
                    "fold3_rank_ic": np.nan,
                    "negative_day_rank_ic_ratio": np.nan,
                    "recommend_for_mainline": False,
                    "notes": f"{type(exc).__name__}: {exc}",
                    "experiment_dir": str(exp_dir),
                    "status": "failed",
                }
                exp_dir.mkdir(parents=True, exist_ok=True)
                (exp_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                print(f"[label_variant] failed {label_type}/{objective}: {exc}")
            rows.append(row)
            summary = mark_recommendations(pd.DataFrame(rows))
            summary = summary.reindex(columns=SUMMARY_COLUMNS)
            summary.to_csv(output_dir / "label_variant_summary.csv", index=False, encoding="utf-8-sig")

    summary = mark_recommendations(pd.DataFrame(rows)).reindex(columns=SUMMARY_COLUMNS)
    summary.to_csv(output_dir / "label_variant_summary.csv", index=False, encoding="utf-8-sig")
    report_path = write_summary_report(summary, output_dir, args)
    print(f"[label_variant] wrote {output_dir / 'label_variant_summary.csv'}")
    print(f"[label_variant] wrote {report_path}")


if __name__ == "__main__":
    main()
