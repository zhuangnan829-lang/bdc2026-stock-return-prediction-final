import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from result_validator import validate_result_file


DEFAULT_CANDIDATE = (
    "app/model/case_slice_submission_search/generated_results/"
    "recent_strength_pred__allow_600115__top6_take5_2__pred_full_cap0.20.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review single-slice score candidate stability before default sync."
    )
    parser.add_argument("--candidate_result", default=DEFAULT_CANDIDATE)
    parser.add_argument("--app_root", default="app")
    parser.add_argument("--make_aggressive_package", action="store_true", default=True)
    return parser.parse_args()


def zfill_stock(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def annualized_sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    std = returns.std(ddof=1)
    if std == 0 or math.isnan(std):
        return float("nan")
    return float(returns.mean() / std * math.sqrt(252))


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_case_details(app_root: Path, label: str, stock_ids: list[str]) -> pd.DataFrame:
    path = app_root / "model/case_slice_submission_search/case_slice_generated_candidate_details.csv"
    details = read_csv_if_exists(path)
    if details.empty:
        return pd.DataFrame()
    details["stock_id"] = zfill_stock(details["stock_id"])
    rows = details[(details["candidate_label"] == label) & (details["stock_id"].isin(stock_ids))]
    if rows.empty:
        rows = details[details["stock_id"].isin(stock_ids)].copy()
    return rows


def infer_candidate_label(candidate_path: Path) -> str:
    return candidate_path.stem


def latest_feature_snapshot(app_root: Path, stock_ids: list[str]) -> tuple[pd.DataFrame, str]:
    features = pd.read_csv(app_root / "temp/train_features.csv")
    features["stock_id"] = zfill_stock(features["stock_id"])
    features["date"] = pd.to_datetime(features["date"])
    latest_date = str(features["date"].max().date())
    latest = features[features["date"] == features["date"].max()].copy()
    selected = latest[latest["stock_id"].isin(stock_ids)].copy()

    percentile_cols = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "turnover_rate",
        "turnover_mean_20d",
        "volatility_5d",
        "volatility_20d",
        "close_position_20d",
        "crowding_risk_5d",
        "reversal_risk_score",
        "overheat_score",
    ]
    available = [c for c in percentile_cols if c in latest.columns]
    for col in available:
        pct = latest[col].rank(pct=True)
        selected[col + "_pctile"] = selected.index.map(pct).astype(float)
    keep = ["stock_id", "date"] + available + [c + "_pctile" for c in available]
    return selected[keep].sort_values("stock_id"), latest_date


def walk_forward_review(app_root: Path, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    wf = pd.read_csv(app_root / "model/walk_forward_predictions.csv")
    wf["stock_id"] = zfill_stock(wf["stock_id"])
    wf["date"] = pd.to_datetime(wf["date"])
    candidate = candidate.copy()
    candidate["stock_id"] = zfill_stock(candidate["stock_id"])
    weights = candidate.set_index("stock_id")["weight"].astype(float).to_dict()
    stock_ids = list(weights)

    sel = wf[wf["stock_id"].isin(stock_ids)].copy()
    stock_summary = (
        sel.groupby("stock_id")
        .agg(
            wf_obs=("target_return", "size"),
            wf_mean_return=("target_return", "mean"),
            wf_median_return=("target_return", "median"),
            wf_min_return=("target_return", "min"),
            wf_max_return=("target_return", "max"),
            wf_positive_rate=("target_return", lambda x: float((x > 0).mean())),
            wf_pred_mean=("pred_return", "mean"),
            wf_train_target_mean=("train_target", "mean"),
        )
        .reset_index()
    )
    stock_summary["weight"] = stock_summary["stock_id"].map(weights)

    fold_summary = (
        sel.groupby(["stock_id", "fold_id"])
        .agg(
            fold_obs=("target_return", "size"),
            fold_mean_return=("target_return", "mean"),
            fold_positive_rate=("target_return", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )

    pivot = sel.pivot_table(index="date", columns="stock_id", values="target_return", aggfunc="mean")
    basket_returns = pd.Series(0.0, index=pivot.index)
    for stock_id, weight in weights.items():
        if stock_id in pivot:
            basket_returns = basket_returns.add(pivot[stock_id].fillna(0.0) * weight, fill_value=0.0)

    basket_by_fold = (
        sel[["date", "fold_id"]]
        .drop_duplicates()
        .set_index("date")
        .join(basket_returns.rename("basket_return"), how="inner")
        .reset_index()
    )
    fold_basket = (
        basket_by_fold.groupby("fold_id")
        .agg(
            basket_days=("basket_return", "size"),
            basket_mean_return=("basket_return", "mean"),
            basket_positive_rate=("basket_return", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    fold_basket["basket_cumulative_return"] = basket_by_fold.groupby("fold_id")["basket_return"].apply(
        lambda x: float((1.0 + x).prod() - 1.0)
    ).values
    fold_basket["basket_max_drawdown"] = basket_by_fold.groupby("fold_id")["basket_return"].apply(max_drawdown).values

    basket_stats = {
        "wf_days": int(len(basket_returns)),
        "wf_basket_mean_return": float(basket_returns.mean()),
        "wf_basket_cumulative_return": float((1.0 + basket_returns).prod() - 1.0),
        "wf_basket_positive_rate": float((basket_returns > 0).mean()),
        "wf_basket_sharpe": annualized_sharpe(basket_returns),
        "wf_basket_max_drawdown": max_drawdown(basket_returns),
        "wf_basket_worst_day": float(basket_returns.min()),
        "wf_worst_fold_cumulative_return": float(fold_basket["basket_cumulative_return"].min()),
        "wf_worst_fold_mean_return": float(fold_basket["basket_mean_return"].min()),
    }
    return stock_summary, fold_summary, basket_stats, fold_basket


def misrank_review(app_root: Path, stock_ids: list[str]) -> pd.DataFrame:
    rows = []
    files = {
        "false_positives": app_root / "model/misrank_diagnostics/false_positives.csv",
        "missed_winners": app_root / "model/misrank_diagnostics/missed_winners.csv",
        "misrank_samples": app_root / "model/misrank_diagnostics/misrank_samples.csv",
    }
    for name, path in files.items():
        df = read_csv_if_exists(path)
        if df.empty or "stock_id" not in df.columns:
            continue
        df["stock_id"] = zfill_stock(df["stock_id"])
        for stock_id in stock_ids:
            sub = df[df["stock_id"] == stock_id]
            row = {
                "stock_id": stock_id,
                "diagnostic_file": name,
                "count": int(len(sub)),
            }
            if "is_poor_return" in sub.columns:
                row["poor_return_count"] = int(sub["is_poor_return"].fillna(False).astype(bool).sum())
            if "is_bad_pred_top5" in sub.columns:
                row["bad_pred_top5_count"] = int(sub["is_bad_pred_top5"].fillna(False).astype(bool).sum())
            if "target_return" in sub.columns and len(sub):
                row["mean_target_return"] = float(sub["target_return"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def regime_review(app_root: Path, candidate: pd.DataFrame) -> pd.DataFrame:
    regimes = read_csv_if_exists(app_root / "model/market_regime_analysis/daily_market_regimes.csv")
    wf = read_csv_if_exists(app_root / "model/walk_forward_predictions.csv")
    if regimes.empty or wf.empty:
        return pd.DataFrame()
    regimes["date"] = pd.to_datetime(regimes["date"])
    wf["date"] = pd.to_datetime(wf["date"])
    wf["stock_id"] = zfill_stock(wf["stock_id"])
    candidate["stock_id"] = zfill_stock(candidate["stock_id"])
    weights = candidate.set_index("stock_id")["weight"].to_dict()
    sel = wf[wf["stock_id"].isin(weights)].copy()
    sel["weighted_return"] = sel["target_return"] * sel["stock_id"].map(weights)
    daily = sel.groupby("date")["weighted_return"].sum().reset_index(name="basket_return")
    joined = daily.merge(regimes, on="date", how="left")
    if "primary_regime" not in joined.columns:
        return pd.DataFrame()
    return (
        joined.groupby("primary_regime")
        .agg(
            days=("basket_return", "size"),
            mean_return=("basket_return", "mean"),
            cumulative_return=("basket_return", lambda x: float((1.0 + x).prod() - 1.0)),
            positive_rate=("basket_return", lambda x: float((x > 0).mean())),
            max_drawdown=("basket_return", max_drawdown),
        )
        .reset_index()
        .sort_values("mean_return", ascending=False)
    )


def add_risk_flags(summary: pd.DataFrame, features: pd.DataFrame, misrank: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    if not features.empty:
        feature_cols = [c for c in features.columns if c != "date"]
        summary = summary.merge(features[feature_cols], on="stock_id", how="left")

    fp = misrank[misrank["diagnostic_file"] == "false_positives"] if not misrank.empty else pd.DataFrame()
    if not fp.empty:
        summary = summary.merge(
            fp[["stock_id", "count", "poor_return_count"]].rename(
                columns={"count": "false_positive_count", "poor_return_count": "poor_false_positive_count"}
            ),
            on="stock_id",
            how="left",
        )
    else:
        summary["false_positive_count"] = 0
        summary["poor_false_positive_count"] = 0
    summary[["false_positive_count", "poor_false_positive_count"]] = summary[
        ["false_positive_count", "poor_false_positive_count"]
    ].fillna(0)

    flags = []
    for _, row in summary.iterrows():
        stock_flags = []
        if row.get("wf_mean_return", 0) <= 0:
            stock_flags.append("wf_mean_nonpositive")
        if row.get("wf_positive_rate", 1) < 0.48:
            stock_flags.append("wf_positive_rate_low")
        if row.get("wf_min_return", 0) < -0.08:
            stock_flags.append("large_single_period_loss")
        if row.get("volatility_20d_pctile", 0) >= 0.85:
            stock_flags.append("latest_volatility_high")
        if row.get("turnover_rate_pctile", 0) >= 0.9:
            stock_flags.append("latest_turnover_high")
        if row.get("poor_false_positive_count", 0) >= 2:
            stock_flags.append("misrank_false_positive_repeat")
        flags.append(";".join(stock_flags) if stock_flags else "ok")
    summary["risk_flags"] = flags
    return summary


def decide(basket_stats: dict, stock_review: pd.DataFrame, case_score: float) -> tuple[str, str]:
    severe_stock_flags = int((stock_review["risk_flags"] != "ok").sum())
    nonpositive_stocks = int((stock_review["wf_mean_return"] <= 0).sum())
    weak_positive_rate = int((stock_review["wf_positive_rate"] < 0.48).sum())
    default_sync_bad = (
        nonpositive_stocks >= 2
        or weak_positive_rate >= 2
        or basket_stats["wf_basket_cumulative_return"] <= 0
        or basket_stats["wf_basket_max_drawdown"] < -0.2
    )
    aggressive_ok = (
        case_score > 0.03
        and basket_stats["wf_basket_cumulative_return"] > 0
        and basket_stats["wf_worst_fold_cumulative_return"] > -0.08
        and severe_stock_flags <= 4
    )
    if default_sync_bad:
        return (
            "do_not_sync_default; optional_aggressive_score_package_only" if aggressive_ok else "reject_for_submission",
            "单切片分数强，但历史稳定性不足以替换当前 HV rerank/sl20 默认主线。",
        )
    if aggressive_ok:
        return (
            "aggressive_package_ok; default_sync_requires_manual_confirmation",
            "历史稳定性可接受，但该候选仍是单切片冲分导向，默认配置需要人工确认后再同步。",
        )
    return (
        "watchlist_only",
        "证据不够支持直接提交，只适合作为观察候选。",
    )


def write_package(app_root: Path, candidate_path: Path, candidate: pd.DataFrame, decision: str, case_score: float) -> Path:
    package_dir = app_root / "model/aggressive_score_submission_candidate"
    package_dir.mkdir(parents=True, exist_ok=True)
    result_path = package_dir / "result_aggressive_score.csv"
    shutil.copy2(candidate_path, result_path)
    config = {
        "profile_name": "aggressive_score_recent_strength_top6_take5_cap020",
        "status": "candidate_only_do_not_sync_default",
        "scenario": "single_slice_score_chase",
        "source_candidate_result": str(candidate_path),
        "result_path": str(result_path),
        "case_slice_score": case_score,
        "candidate_stocks": candidate["stock_id"].tolist(),
        "weights": [float(x) for x in candidate["weight"].tolist()],
        "selection_logic": {
            "top_k": 5,
            "candidate_family": "recent_strength_pred",
            "take_rule": "top6_take5_2",
            "weighting_scheme": "equal_full_cap0.20",
            "max_single_weight": 0.2,
        },
        "decision": decision,
        "notes": [
            "Generated for manual aggressive score review only.",
            "This file does not overwrite app/output/result.csv or default_submission_config.json.",
            "Use only if the operator explicitly chooses competition score chasing over stability.",
        ],
    }
    (package_dir / "submission_aggressive_score_candidate.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_result_file(result_path)
    return result_path


def fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.6f}"


def write_report(
    out_dir: Path,
    candidate_path: Path,
    candidate_label: str,
    candidate: pd.DataFrame,
    case_score: float,
    stock_review: pd.DataFrame,
    fold_basket: pd.DataFrame,
    basket_stats: dict,
    misrank: pd.DataFrame,
    regime: pd.DataFrame,
    decision: str,
    reason: str,
    package_result: Path | None,
) -> None:
    lines = [
        "# Aggressive Score Candidate Stability Review",
        "",
        "本报告复核单切片冲分候选是否只是偶然命中，并给出是否同步默认配置的建议。脚本不会覆盖默认配置。",
        "",
        "## Candidate",
        "",
        f"- candidate label: `{candidate_label}`",
        f"- candidate result: `{candidate_path}`",
        f"- stocks: `{','.join(candidate['stock_id'].tolist())}`",
        f"- weight sum: `{candidate['weight'].sum():.6f}`",
        f"- same-protocol single-slice score: `{case_score:.6f}`",
        "",
        "## Decision",
        "",
        f"- decision: `{decision}`",
        f"- reason: {reason}",
        "- default sync: `不建议自动同步 default_submission_config.json`",
        "- current default: 保留已人工确认的 HV rerank/sl20 主线，除非用户明确选择比赛冲分。",
    ]
    if package_result is not None:
        lines.append(f"- aggressive score package result: `{package_result}`")
    lines += [
        "",
        "## Walk-Forward Basket Check",
        "",
        f"- days: `{basket_stats['wf_days']}`",
        f"- cumulative return: `{basket_stats['wf_basket_cumulative_return']:.6f}`",
        f"- mean daily return: `{basket_stats['wf_basket_mean_return']:.6f}`",
        f"- positive rate: `{basket_stats['wf_basket_positive_rate']:.6f}`",
        f"- sharpe: `{basket_stats['wf_basket_sharpe']:.6f}`",
        f"- max drawdown: `{basket_stats['wf_basket_max_drawdown']:.6f}`",
        f"- worst fold cumulative return: `{basket_stats['wf_worst_fold_cumulative_return']:.6f}`",
        "",
        "## Per-Stock Evidence",
        "",
        "| stock_id | weight | case_return | case_contribution | wf_mean | wf_positive_rate | wf_min | latest_vol20_pctile | latest_turnover_pctile | false_positive | risk_flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in stock_review.sort_values("stock_id").iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['stock_id']}`",
                    fmt(row.get("weight")),
                    fmt(row.get("case_slice_return")),
                    fmt(row.get("case_slice_contribution")),
                    fmt(row.get("wf_mean_return")),
                    fmt(row.get("wf_positive_rate")),
                    fmt(row.get("wf_min_return")),
                    fmt(row.get("volatility_20d_pctile")),
                    fmt(row.get("turnover_rate_pctile")),
                    str(int(row.get("false_positive_count", 0))),
                    f"`{row.get('risk_flags', '')}`",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Fold Basket Evidence",
        "",
        "| fold_id | days | mean_return | cumulative_return | positive_rate | max_drawdown |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in fold_basket.iterrows():
        lines.append(
            f"| {int(row['fold_id'])} | {int(row['basket_days'])} | {fmt(row['basket_mean_return'])} | "
            f"{fmt(row['basket_cumulative_return'])} | {fmt(row['basket_positive_rate'])} | "
            f"{fmt(row['basket_max_drawdown'])} |"
        )
    if not regime.empty:
        lines += [
            "",
            "## Regime Evidence",
            "",
            "| regime | days | mean_return | cumulative_return | positive_rate | max_drawdown |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _, row in regime.iterrows():
            lines.append(
                f"| `{row['primary_regime']}` | {int(row['days'])} | {fmt(row['mean_return'])} | "
                f"{fmt(row['cumulative_return'])} | {fmt(row['positive_rate'])} | {fmt(row['max_drawdown'])} |"
            )
    if not misrank.empty:
        lines += [
            "",
            "## Misrank Diagnostic Counts",
            "",
            "| stock_id | file | count | poor_return_count | bad_pred_top5_count | mean_target_return |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for _, row in misrank.sort_values(["stock_id", "diagnostic_file"]).iterrows():
            lines.append(
                f"| `{row['stock_id']}` | `{row['diagnostic_file']}` | {int(row.get('count', 0))} | "
                f"{int(row.get('poor_return_count', 0)) if not pd.isna(row.get('poor_return_count', np.nan)) else 0} | "
                f"{int(row.get('bad_pred_top5_count', 0)) if not pd.isna(row.get('bad_pred_top5_count', np.nan)) else 0} | "
                f"{fmt(row.get('mean_target_return', np.nan))} |"
            )
    lines += [
        "",
        "## Submit/Sync Rule",
        "",
        "- 比赛冲分：可人工选择 aggressive score 包，但要接受它不是默认稳健主线。",
        "- 稳定策略：继续使用当前 HV rerank/sl20 默认结果。",
        "- 默认同步：本轮不自动同步；只有用户明确确认后才可替换 `app/output/result.csv` 和默认配置。",
    ]
    (out_dir / "score_candidate_stability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    app_root = Path(args.app_root)
    candidate_path = Path(args.candidate_result)
    out_dir = app_root / "model/case_slice_submission_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = pd.read_csv(candidate_path, dtype={"stock_id": str})
    candidate["stock_id"] = zfill_stock(candidate["stock_id"])
    candidate["weight"] = candidate["weight"].astype(float)
    validate_result_file(candidate_path)
    stock_ids = candidate["stock_id"].tolist()
    candidate_label = infer_candidate_label(candidate_path)

    leaderboard = pd.read_csv(out_dir / "case_slice_generated_leaderboard.csv")
    label_col = "label" if "label" in leaderboard.columns else "candidate_label"
    score_col = "score" if "score" in leaderboard.columns else "case_slice_score"
    label_rows = leaderboard[leaderboard[label_col] == candidate_label]
    case_score = float(label_rows.iloc[0][score_col]) if len(label_rows) else float("nan")

    case_details = load_case_details(app_root, candidate_label, stock_ids)
    stock_summary, fold_summary, basket_stats, fold_basket = walk_forward_review(app_root, candidate)
    latest_features, _ = latest_feature_snapshot(app_root, stock_ids)
    misrank = misrank_review(app_root, stock_ids)
    regime = regime_review(app_root, candidate.copy())

    stock_review = stock_summary.merge(
        case_details[["stock_id", "case_slice_return", "case_slice_contribution"]].drop_duplicates("stock_id"),
        on="stock_id",
        how="left",
    )
    stock_review = add_risk_flags(stock_review, latest_features, misrank)
    decision, reason = decide(basket_stats, stock_review, case_score)

    package_result = None
    if args.make_aggressive_package and "aggressive" in decision:
        package_result = write_package(app_root, candidate_path, candidate, decision, case_score)

    stock_review.to_csv(out_dir / "score_candidate_stability_by_stock.csv", index=False, encoding="utf-8")
    fold_summary.to_csv(out_dir / "score_candidate_stability_by_stock_fold.csv", index=False, encoding="utf-8")
    fold_basket.to_csv(out_dir / "score_candidate_stability_by_fold.csv", index=False, encoding="utf-8")
    misrank.to_csv(out_dir / "score_candidate_misrank_counts.csv", index=False, encoding="utf-8")
    regime.to_csv(out_dir / "score_candidate_regime_review.csv", index=False, encoding="utf-8")
    pd.DataFrame([{"decision": decision, "reason": reason, **basket_stats, "case_slice_score": case_score}]).to_csv(
        out_dir / "score_candidate_stability_decision.csv", index=False, encoding="utf-8"
    )
    write_report(
        out_dir,
        candidate_path,
        candidate_label,
        candidate,
        case_score,
        stock_review,
        fold_basket,
        basket_stats,
        misrank,
        regime,
        decision,
        reason,
        package_result,
    )

    print(f"[score_candidate_stability] decision={decision}")
    print(f"[score_candidate_stability] case_slice_score={case_score:.6f}")
    print(f"[score_candidate_stability] wf_cumulative_return={basket_stats['wf_basket_cumulative_return']:.6f}")
    print(f"[score_candidate_stability] wf_max_drawdown={basket_stats['wf_basket_max_drawdown']:.6f}")
    print(f"[score_candidate_stability] report={out_dir / 'score_candidate_stability_report.md'}")
    if package_result:
        print(f"[score_candidate_stability] aggressive_package_result={package_result}")


if __name__ == "__main__":
    main()
