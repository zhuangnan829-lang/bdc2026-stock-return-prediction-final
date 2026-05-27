from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import ROOT_DIR


DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "stability_eval"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "stability_summary.csv"
DEFAULT_LEADERBOARD_PATH = ROOT_DIR / "app" / "model" / "experiment_leaderboard.csv"
DEFAULT_TARGET_COLUMN = "target_return"
DEFAULT_PREDICTION_COLUMN = "pred_return"
SUMMARY_COLUMNS = [
    "model",
    "feature_set",
    "sequence_length",
    "rank_ic_mean",
    "rank_ic_std",
    "worst_fold_rank_ic",
    "best_fold_rank_ic",
    "negative_fold_count",
    "negative_day_rank_ic_ratio",
    "top5_return_mean",
    "top5_return_min_by_fold",
    "top5_return_std",
    "stability_score",
]


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _spearman_rank_ic(day_df: pd.DataFrame, prediction_column: str, target_column: str) -> float | None:
    pred = _safe_numeric(day_df[prediction_column])
    target = _safe_numeric(day_df[target_column])
    valid = pd.DataFrame({"pred": pred, "target": target}).dropna()
    if len(valid) < 2 or valid["pred"].nunique() <= 1 or valid["target"].nunique() <= 1:
        return None
    corr = valid["pred"].corr(valid["target"], method="spearman")
    return float(corr) if pd.notna(corr) else None


def _topk_return(day_df: pd.DataFrame, prediction_column: str, target_column: str, top_k: int = 5) -> float:
    top = day_df.sort_values([prediction_column, "stock_id"], ascending=[False, True]).head(top_k)
    return float(_safe_numeric(top[target_column]).mean()) if not top.empty else 0.0


def build_daily_rank_ic(
    prediction_df: pd.DataFrame,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    top_k: int = 5,
) -> pd.DataFrame:
    required = {"date", prediction_column, target_column}
    missing = sorted(required - set(prediction_df.columns))
    if missing:
        raise ValueError(f"Prediction data is missing required columns: {missing}")

    working = prediction_df.copy()
    working["date"] = pd.to_datetime(working["date"])
    if "fold_id" not in working.columns:
        working["fold_id"] = 0
    if "stock_id" not in working.columns:
        working["stock_id"] = working.groupby(["fold_id", "date"]).cumcount().astype(str)

    rows: list[dict[str, Any]] = []
    for (fold_id, trade_date), day_df in working.groupby(["fold_id", "date"], sort=True):
        rank_ic = _spearman_rank_ic(day_df, prediction_column, target_column)
        if rank_ic is None:
            continue
        rows.append(
            {
                "fold_id": int(fold_id),
                "date": trade_date.date().isoformat(),
                "day_rank_ic": rank_ic,
                "top5_return": _topk_return(day_df, prediction_column, target_column, top_k=top_k),
                "candidate_count": int(len(day_df)),
            }
        )
    return pd.DataFrame(rows)


def build_fold_rank_ic(daily_rank_ic_df: pd.DataFrame) -> pd.DataFrame:
    if daily_rank_ic_df.empty:
        return pd.DataFrame(columns=["fold_id", "rank_ic", "top5_mean_return"])
    grouped = (
        daily_rank_ic_df.assign(
            day_rank_ic=_safe_numeric(daily_rank_ic_df["day_rank_ic"]),
            top5_return=_safe_numeric(daily_rank_ic_df.get("top5_return", pd.Series(dtype=float))),
        )
        .dropna(subset=["day_rank_ic"])
        .groupby("fold_id", as_index=False)
        .agg(rank_ic=("day_rank_ic", "mean"), top5_mean_return=("top5_return", "mean"))
    )
    return grouped


def read_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_id": str})
    if "target_return" not in df.columns and "y_true" in df.columns:
        df["target_return"] = df["y_true"]
    if "pred_return" not in df.columns and "y_pred" in df.columns:
        df["pred_return"] = df["y_pred"]
    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str).str.zfill(6)
    return df


def load_rank_inputs(
    prediction_path: str | Path | None = None,
    fold_diagnostics_path: str | Path | None = None,
    fold_daily_diagnostics_path: str | Path | None = None,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_path = _resolve_path(fold_diagnostics_path)
    daily_path = _resolve_path(fold_daily_diagnostics_path)
    prediction_resolved = _resolve_path(prediction_path)

    daily_df = pd.DataFrame()
    if daily_path is not None and daily_path.exists():
        daily_df = pd.read_csv(daily_path, encoding="utf-8-sig")
        if "day_rank_ic" not in daily_df.columns and "avg_day_rank_ic" in daily_df.columns:
            daily_df = daily_df.rename(columns={"avg_day_rank_ic": "day_rank_ic"})
        if "day_rank_ic" not in daily_df.columns:
            daily_df = pd.DataFrame()

    if daily_df.empty and prediction_resolved is not None and prediction_resolved.exists():
        daily_df = build_daily_rank_ic(
            read_prediction(prediction_resolved),
            prediction_column=prediction_column,
            target_column=target_column,
        )

    fold_df = pd.DataFrame()
    if fold_path is not None and fold_path.exists():
        fold_df = pd.read_csv(fold_path, encoding="utf-8-sig")
        if "rank_ic" not in fold_df.columns and "avg_day_rank_ic" in fold_df.columns:
            fold_df = fold_df.rename(columns={"avg_day_rank_ic": "rank_ic"})
        if "top5_mean_return" not in fold_df.columns and "top5_return_mean" in fold_df.columns:
            fold_df = fold_df.rename(columns={"top5_return_mean": "top5_mean_return"})
        if "rank_ic" not in fold_df.columns:
            fold_df = pd.DataFrame()

    if fold_df.empty:
        fold_df = build_fold_rank_ic(daily_df)

    return fold_df, daily_df


def metadata_from_path(path: Path | None) -> dict[str, Any]:
    meta: dict[str, Any] = {"model": "", "feature_set": "", "sequence_length": ""}
    candidates = []
    if path is not None:
        if path.is_file():
            candidates.extend([path.parent / "model_meta.json", ROOT_DIR / "app" / "model" / "model_meta.json"])
        else:
            candidates.extend([path / "model_meta.json", ROOT_DIR / "app" / "model" / "model_meta.json"])
    else:
        candidates.append(ROOT_DIR / "app" / "model" / "model_meta.json")
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta["model"] = data.get("model_family") or data.get("backend") or meta["model"]
        meta["feature_set"] = data.get("feature_set") or data.get("default_submission_profile", {}).get("feature_set") or meta["feature_set"]
        meta["sequence_length"] = data.get("sequence_length") or data.get("validation_scheme", {}).get("sequence_length") or meta["sequence_length"]
        break
    return meta


def summarize_stability(
    *,
    model: str,
    feature_set: str,
    sequence_length: Any,
    fold_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> dict[str, Any]:
    fold_rank_ic = _safe_numeric(fold_df["rank_ic"]) if "rank_ic" in fold_df.columns else pd.Series(dtype=float)
    fold_rank_ic = fold_rank_ic.dropna()
    day_rank_ic = _safe_numeric(daily_df["day_rank_ic"]).dropna() if "day_rank_ic" in daily_df.columns else pd.Series(dtype=float)

    if "top5_mean_return" in fold_df.columns:
        fold_top5 = _safe_numeric(fold_df["top5_mean_return"]).dropna()
    elif "top5_return" in daily_df.columns:
        fold_top5 = _safe_numeric(daily_df.groupby("fold_id")["top5_return"].mean()).dropna()
    else:
        fold_top5 = pd.Series(dtype=float)

    rank_ic_mean = float(fold_rank_ic.mean()) if not fold_rank_ic.empty else 0.0
    rank_ic_std = float(fold_rank_ic.std(ddof=0)) if len(fold_rank_ic) > 1 else 0.0
    negative_fold_count = int((fold_rank_ic < 0).sum()) if not fold_rank_ic.empty else 0
    top5_return_mean = float(fold_top5.mean()) if not fold_top5.empty else 0.0
    stability_score = rank_ic_mean - 0.5 * rank_ic_std - 0.02 * negative_fold_count + top5_return_mean
    return {
        "model": model,
        "feature_set": feature_set,
        "sequence_length": sequence_length,
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "worst_fold_rank_ic": float(fold_rank_ic.min()) if not fold_rank_ic.empty else 0.0,
        "best_fold_rank_ic": float(fold_rank_ic.max()) if not fold_rank_ic.empty else 0.0,
        "negative_fold_count": negative_fold_count,
        "negative_day_rank_ic_ratio": float((day_rank_ic < 0).mean()) if not day_rank_ic.empty else 0.0,
        "top5_return_mean": top5_return_mean,
        "top5_return_min_by_fold": float(fold_top5.min()) if not fold_top5.empty else 0.0,
        "top5_return_std": float(fold_top5.std(ddof=0)) if len(fold_top5) > 1 else 0.0,
        "stability_score": stability_score,
    }


def evaluate_one(
    *,
    experiment_name: str,
    prediction_path: str | Path | None = None,
    fold_diagnostics_path: str | Path | None = None,
    fold_daily_diagnostics_path: str | Path | None = None,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    model: str = "",
    feature_set: str = "",
    sequence_length: Any = "",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    fold_df, daily_df = load_rank_inputs(
        prediction_path=prediction_path,
        fold_diagnostics_path=fold_diagnostics_path,
        fold_daily_diagnostics_path=fold_daily_diagnostics_path,
        prediction_column=prediction_column,
        target_column=target_column,
    )
    ref_path = _resolve_path(prediction_path) or _resolve_path(fold_diagnostics_path)
    meta = metadata_from_path(ref_path)
    row = summarize_stability(
        model=model or meta["model"] or experiment_name,
        feature_set=feature_set or meta["feature_set"],
        sequence_length=sequence_length or meta["sequence_length"],
        fold_df=fold_df,
        daily_df=daily_df,
    )
    row["experiment_name"] = experiment_name
    row["evaluated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row["prediction_path"] = str(_resolve_path(prediction_path) or "")
    return row, fold_df, daily_df


def evaluate_rank_stability(
    experiment_name: str,
    prediction_path: str | Path | None = None,
    fold_diagnostics_path: str | Path | None = None,
    fold_daily_diagnostics_path: str | Path | None = None,
    backtest_summary_path: str | Path | None = None,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    extra_fields: dict | None = None,
) -> dict:
    row, _, _ = evaluate_one(
        experiment_name=experiment_name,
        prediction_path=prediction_path,
        fold_diagnostics_path=fold_diagnostics_path,
        fold_daily_diagnostics_path=fold_daily_diagnostics_path,
        prediction_column=prediction_column,
        target_column=target_column,
    )
    backtest_path = _resolve_path(backtest_summary_path)
    if backtest_path is not None and backtest_path.exists():
        backtest_df = pd.read_csv(backtest_path, encoding="utf-8-sig")
        if not backtest_df.empty:
            backtest_row = backtest_df.iloc[0]
            for column in ["cumulative_return_after_cost", "sharpe_after_cost", "max_drawdown_after_cost", "avg_turnover"]:
                if column in backtest_row:
                    row[column] = _safe_float(backtest_row[column])
    if extra_fields:
        row.update(extra_fields)
    return row


def summarize_rank_stability_frames(
    experiment_name: str,
    fold_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    extra_fields: dict | None = None,
) -> dict:
    meta = metadata_from_path(None)
    row = summarize_stability(
        model=meta["model"] or experiment_name,
        feature_set=meta["feature_set"],
        sequence_length=meta["sequence_length"],
        fold_df=fold_df,
        daily_df=daily_df,
    )
    row["experiment_name"] = experiment_name
    row["evaluated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row["prediction_path"] = ""
    if extra_fields:
        row.update(extra_fields)
    return row


def summarize_prediction_rank_stability(
    prediction_df: pd.DataFrame,
    experiment_name: str,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    extra_fields: dict | None = None,
) -> dict:
    daily_df = build_daily_rank_ic(prediction_df, prediction_column=prediction_column, target_column=target_column)
    fold_df = build_fold_rank_ic(daily_df)
    return summarize_rank_stability_frames(experiment_name=experiment_name, fold_df=fold_df, daily_df=daily_df, extra_fields=extra_fields)


def append_stability_summary(row: dict, output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    path = _resolve_path(output_path)
    if path is None:
        raise ValueError("output_path is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if path.exists():
        old_df = pd.read_csv(path, encoding="utf-8-sig")
        summary_df = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        summary_df = new_df
    summary_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def append_experiment_rank_stability(
    experiment_name: str,
    prediction_path: str | Path | None = None,
    fold_diagnostics_path: str | Path | None = None,
    fold_daily_diagnostics_path: str | Path | None = None,
    backtest_summary_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    extra_fields: dict | None = None,
) -> dict:
    row = evaluate_rank_stability(
        experiment_name=experiment_name,
        prediction_path=prediction_path,
        fold_diagnostics_path=fold_diagnostics_path,
        fold_daily_diagnostics_path=fold_daily_diagnostics_path,
        backtest_summary_path=backtest_summary_path,
        extra_fields=extra_fields,
    )
    append_stability_summary(row, output_path=output_path)
    return row


def discover_experiments(fold_result_dir: Path) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    if (fold_result_dir / "walk_forward_predictions.csv").exists():
        paths.append((fold_result_dir.name, fold_result_dir / "walk_forward_predictions.csv"))
    for pred_path in sorted(fold_result_dir.rglob("walk_forward_predictions.csv")):
        if pred_path.parent == fold_result_dir:
            continue
        paths.append((pred_path.parent.name, pred_path))
    for pred_path in sorted(fold_result_dir.rglob("fold_results.csv")):
        paths.append((pred_path.parent.name, pred_path))
    seen = set()
    unique = []
    for name, path in paths:
        key = str(path.resolve())
        if key not in seen:
            unique.append((name, path))
            seen.add(key)
    return unique


def update_leaderboard(summary_df: pd.DataFrame, leaderboard_path: Path = DEFAULT_LEADERBOARD_PATH) -> Path:
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    if leaderboard_path.exists() and leaderboard_path.stat().st_size > 0:
        leaderboard = pd.read_csv(leaderboard_path, encoding="utf-8-sig")
    else:
        leaderboard = pd.DataFrame()
    for _, row in summary_df.iterrows():
        experiment_id = str(row.get("experiment_name") or row.get("model") or "rank_stability")
        update = {
            "experiment_id": experiment_id,
            "model": row.get("model", ""),
            "feature_set": row.get("feature_set", ""),
            "sequence_length": row.get("sequence_length", ""),
            "rank_ic_mean": row.get("rank_ic_mean", 0.0),
            "worst_fold_rank_ic": row.get("worst_fold_rank_ic", 0.0),
            "top5_return_mean": row.get("top5_return_mean", 0.0),
            "notes": "rank_stability_eval",
        }
        if "experiment_id" in leaderboard.columns:
            leaderboard = leaderboard[leaderboard["experiment_id"].astype(str) != experiment_id]
        leaderboard = pd.concat([leaderboard, pd.DataFrame([update])], ignore_index=True, sort=False)
    leaderboard.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")
    return leaderboard_path


def render_report(summary_df: pd.DataFrame, fold_df: pd.DataFrame | None = None) -> str:
    row = summary_df.sort_values("stability_score", ascending=False).iloc[0]
    fold_notes = ""
    if fold_df is not None and not fold_df.empty and "fold_id" in fold_df.columns:
        fold_map = {int(r["fold_id"]): float(r["rank_ic"]) for _, r in fold_df.iterrows() if pd.notna(r.get("rank_ic"))}
        fold1_bad = fold_map.get(1, 0.0) < 0
        fold3_bad = fold_map.get(3, 0.0) <= 0.01
        fold_notes = f"Fold 1 rank_ic={fold_map.get(1, 0.0):.6f}, Fold 3 rank_ic={fold_map.get(3, 0.0):.6f}."
    else:
        fold1_bad = False
        fold3_bad = False
    positive_return_unstable_rankic = row["top5_return_mean"] > 0 and (row["negative_fold_count"] > 0 or row["worst_fold_rank_ic"] < 0)
    lines = [
        "# RankIC Stability Report",
        "",
        "## Required Answers",
        "",
        f"1. Fold 1 和 Fold 3 是否是主要不稳定来源？{'是' if (fold1_bad or fold3_bad) else '否'}。{fold_notes}",
        f"2. 当前模型是否属于“收益正但 RankIC 不稳”？{'是' if positive_return_unstable_rankic else '否'}。"
        f" top5_return_mean={row['top5_return_mean']:.6f}, worst_fold_rank_ic={row['worst_fold_rank_ic']:.6f}。",
        "3. 后续实验采用与否的硬性门槛：top5_return_min_by_fold > 0；worst_fold_rank_ic 不低于当前主线；negative_day_rank_ic_ratio 下降。",
        "4. 推荐门槛已写入本报告，并应同步用于 experiment_leaderboard 的 adopted 判断。",
        "",
        "## Summary",
        "",
        "| model | feature_set | sl | rank_ic_mean | rank_ic_std | worst_fold | neg_folds | neg_day_ratio | top5_mean | top5_min_fold | stability_score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, item in summary_df.iterrows():
        lines.append(
            f"| {item['model']} | {item['feature_set']} | {item['sequence_length']} | "
            f"{item['rank_ic_mean']:.6f} | {item['rank_ic_std']:.6f} | {item['worst_fold_rank_ic']:.6f} | "
            f"{int(item['negative_fold_count'])} | {item['negative_day_rank_ic_ratio']:.6f} | "
            f"{item['top5_return_mean']:.6f} | {item['top5_return_min_by_fold']:.6f} | {item['stability_score']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RankIC stability by fold and by day.")
    parser.add_argument("--fold_result_dir", nargs="*", default=[])
    parser.add_argument("--pred_path", nargs="*", default=[])
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--experiment_name", default="current_mainline")
    parser.add_argument("--prediction_column", default=DEFAULT_PREDICTION_COLUMN)
    parser.add_argument("--target_column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--update_leaderboard", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve_path(args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path]] = []
    for item in args.pred_path:
        path = _resolve_path(item)
        if path is not None:
            jobs.append((args.experiment_name if len(args.pred_path) == 1 else path.parent.name, path))
    for item in args.fold_result_dir:
        path = _resolve_path(item)
        if path is not None:
            jobs.extend(discover_experiments(path))
    if not jobs:
        default_pred = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"
        jobs.append((args.experiment_name, default_pred))

    rows = []
    last_fold_df = pd.DataFrame()
    for name, path in jobs:
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction/fold result file: {path}")
        if path.name in {"fold_results.csv", "fold_diagnostics.csv", "walk_forward_metrics.csv"}:
            fold_df = pd.read_csv(path, encoding="utf-8-sig")
            if "rank_ic" not in fold_df.columns and "avg_day_rank_ic" in fold_df.columns:
                fold_df = fold_df.rename(columns={"avg_day_rank_ic": "rank_ic"})
            daily_df = pd.DataFrame()
            row = summarize_stability(
                model=name,
                feature_set="",
                sequence_length="",
                fold_df=fold_df,
                daily_df=daily_df,
            )
            row["experiment_name"] = name
        else:
            row, fold_df, daily_df = evaluate_one(
                experiment_name=name,
                prediction_path=path,
                prediction_column=args.prediction_column,
                target_column=args.target_column,
            )
        rows.append(row)
        last_fold_df = fold_df

    summary_df = pd.DataFrame(rows)
    summary_df[SUMMARY_COLUMNS].to_csv(output_dir / "stability_summary.csv", index=False, encoding="utf-8-sig")
    report_path = output_dir / "stability_report.md"
    report_path.write_text(render_report(summary_df[SUMMARY_COLUMNS], last_fold_df), encoding="utf-8-sig")
    if args.update_leaderboard:
        leaderboard_path = update_leaderboard(summary_df)
    else:
        leaderboard_path = None

    print(summary_df[SUMMARY_COLUMNS].to_string(index=False))
    print(f"[rank_stability] wrote {output_dir / 'stability_summary.csv'}")
    print(f"[rank_stability] wrote {report_path}")
    if leaderboard_path is not None:
        print(f"[rank_stability] updated {leaderboard_path}")


if __name__ == "__main__":
    main()
