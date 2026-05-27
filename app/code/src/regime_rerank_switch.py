from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import MERGE_FEATURE_COLUMNS, run_backtest
from config import ROOT_DIR
from load_submission_config import build_default_inference_args, load_submission_config
from market_regime_split import DEFAULT_OUTPUT_PATH as DEFAULT_REGIME_PATH
from market_regime_split import load_and_split_market_regimes


DEFAULT_PRED_PATH = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "regime_rerank_switch"
RISK_SIGNALS = ["close_position_20d", "reversal_risk_score"]
REGIME_FLAGS = ["is_high_volatility", "is_high_volatility_range"]


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply defensive rerank signals only in high-risk market regimes.")
    parser.add_argument("--pred_path", default=str(DEFAULT_PRED_PATH))
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--regime_path", default=str(DEFAULT_REGIME_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--weight", type=float, default=-0.05)
    return parser.parse_args()


def build_backtest_config(profile_name: str, rerank_column: str | None = None, rerank_weight: float = 0.0) -> dict[str, Any]:
    cfg = load_submission_config()
    args = build_default_inference_args(cfg)
    return {
        "profile_name": profile_name,
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
        "rerank_signal_column": rerank_column,
        "rerank_signal_weight": float(rerank_weight),
    }


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out


def load_prediction_with_features(pred_path: Path, feature_path: Path) -> pd.DataFrame:
    pred = normalize_keys(pd.read_csv(pred_path, encoding="utf-8-sig", dtype={"stock_id": str}))
    features = normalize_keys(pd.read_csv(feature_path, encoding="utf-8-sig", dtype={"stock_id": str}))
    required = ["stock_id", "date", "target_return", *MERGE_FEATURE_COLUMNS, *RISK_SIGNALS]
    missing = [column for column in required if column not in features.columns and column not in pred.columns]
    if missing:
        raise ValueError(f"Missing required columns for regime rerank switch: {missing}")

    merge_columns = [column for column in required if column in features.columns]
    merge_columns = list(dict.fromkeys(merge_columns))
    need_merge = [column for column in merge_columns if column not in pred.columns or pred[column].isna().any()]
    if need_merge:
        pred = pred.merge(
            features[["stock_id", "date", *[c for c in merge_columns if c not in {"stock_id", "date"}]]]
            .drop_duplicates(["stock_id", "date"]),
            on=["stock_id", "date"],
            how="left",
            suffixes=("", "_feature"),
        )
        for column in merge_columns:
            feature_column = f"{column}_feature"
            if feature_column in pred.columns:
                if column in pred.columns:
                    pred[column] = pred[column].where(pred[column].notna(), pred[feature_column])
                    pred = pred.drop(columns=[feature_column])
                else:
                    pred = pred.rename(columns={feature_column: column})

    still_missing = [column for column in ["pred_return", "target_return", *MERGE_FEATURE_COLUMNS, *RISK_SIGNALS] if column not in pred.columns]
    if still_missing:
        raise ValueError(f"Prediction frame still missing columns after merge: {still_missing}")
    return pred


def load_regimes(feature_path: Path, regime_path: Path) -> pd.DataFrame:
    regime_df = load_and_split_market_regimes(feature_path=feature_path, output_path=regime_path)
    regime_df = regime_df.copy()
    regime_df["date"] = pd.to_datetime(regime_df["date"], errors="coerce").dt.normalize()
    for flag in REGIME_FLAGS:
        if flag not in regime_df.columns:
            raise ValueError(f"Regime table missing `{flag}`")
    return regime_df[["date", *REGIME_FLAGS]].drop_duplicates("date")


def add_switch_signal(
    prediction_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    signal: str,
    regime_flag: str | None,
    output_column: str,
) -> pd.DataFrame:
    out = prediction_df.copy()
    if regime_flag is None:
        active = pd.Series(1, index=out.index, dtype=int)
    else:
        out = out.merge(regime_df[["date", regime_flag]], on="date", how="left")
        active = out[regime_flag].fillna(0).astype(int)
        out = out.drop(columns=[regime_flag])

    if signal == "combo":
        close_rank = out.groupby("date")["close_position_20d"].rank(pct=True)
        reversal_rank = out.groupby("date")["reversal_risk_score"].rank(pct=True)
        raw_signal = 0.5 * close_rank + 0.5 * reversal_rank
    else:
        raw_signal = pd.to_numeric(out[signal], errors="coerce").fillna(0.0)

    out[output_column] = np.where(active.eq(1), raw_signal, 0.0)
    return out


def parse_selected_returns(value: object) -> list[float]:
    returns = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            returns.append(float(item))
        except ValueError:
            continue
    return returns


def summarize_selection(
    prediction_df: pd.DataFrame,
    backtest_daily: pd.DataFrame,
    regime_df: pd.DataFrame,
    profile_name: str,
) -> dict[str, Any]:
    pred = prediction_df[["date", "stock_id", "target_return", "fold_id"]].copy()
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce").dt.normalize()
    true_top = (
        pred.sort_values(["date", "target_return", "stock_id"], ascending=[True, False, True])
        .groupby("date")
        .head(5)
        .groupby("date")["stock_id"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )
    fold_by_date = pred.drop_duplicates("date").set_index("date")["fold_id"].to_dict()
    daily = backtest_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily = daily.merge(regime_df, on="date", how="left")

    rows = []
    for _, row in daily.iterrows():
        ids = [item.strip().zfill(6) for item in str(row.get("selected_stock_ids", "")).split(",") if item.strip()]
        rets = parse_selected_returns(row.get("selected_target_returns", ""))
        selected_set = set(ids)
        true_set = true_top.get(row["date"], set())
        false_count = len(selected_set - true_set)
        median_return = float(pred.loc[pred["date"].eq(row["date"]), "target_return"].median())
        poor_false = 0
        if ids and rets:
            for stock_id, ret in zip(ids, rets):
                if stock_id not in true_set and ret <= median_return:
                    poor_false += 1
        rows.append(
            {
                "profile_name": profile_name,
                "date": row["date"],
                "fold_id": int(fold_by_date.get(row["date"], 0)),
                "selected_top5_return": float(np.mean(rets)) if rets else 0.0,
                "hit_rate_at5": float(len(selected_set & true_set) / 5.0) if true_set else 0.0,
                "false_positives": int(false_count),
                "poor_false_positives": int(poor_false),
                "is_high_volatility": int(row.get("is_high_volatility", 0) or 0),
                "is_high_volatility_range": int(row.get("is_high_volatility_range", 0) or 0),
            }
        )
    detail = pd.DataFrame(rows)
    fold = (
        detail.groupby("fold_id", as_index=False)
        .agg(
            selected_top5_return=("selected_top5_return", "mean"),
            hit_rate_at5=("hit_rate_at5", "mean"),
            false_positives=("false_positives", "sum"),
            poor_false_positives=("poor_false_positives", "sum"),
        )
        .rename(columns={"selected_top5_return": "fold_selected_top5_return"})
    )
    hv = detail[detail["is_high_volatility"].eq(1)]
    hvr = detail[detail["is_high_volatility_range"].eq(1)]
    return {
        "selected_top5_return_mean": float(detail["selected_top5_return"].mean()) if not detail.empty else 0.0,
        "hit_rate_at5": float(detail["hit_rate_at5"].mean()) if not detail.empty else 0.0,
        "false_positives": int(detail["false_positives"].sum()) if not detail.empty else 0,
        "poor_false_positives": int(detail["poor_false_positives"].sum()) if not detail.empty else 0,
        "fold1_selected_top5_return": float(fold.loc[fold["fold_id"].eq(1), "fold_selected_top5_return"].iloc[0]) if (fold["fold_id"].eq(1)).any() else np.nan,
        "fold3_selected_top5_return": float(fold.loc[fold["fold_id"].eq(3), "fold_selected_top5_return"].iloc[0]) if (fold["fold_id"].eq(3)).any() else np.nan,
        "high_volatility_selected_top5_return": float(hv["selected_top5_return"].mean()) if not hv.empty else np.nan,
        "high_volatility_range_selected_top5_return": float(hvr["selected_top5_return"].mean()) if not hvr.empty else np.nan,
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


def run_profile(
    base_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    output_dir: Path,
    profile_name: str,
    signal: str | None,
    regime_flag: str | None,
    weight: float,
) -> dict[str, Any]:
    profile_dir = output_dir / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    if signal is None:
        prediction_df = base_df.copy()
        rerank_column = None
        rerank_weight = 0.0
    else:
        rerank_column = f"switch_{profile_name}"
        prediction_df = add_switch_signal(base_df, regime_df, signal=signal, regime_flag=regime_flag, output_column=rerank_column)
        rerank_weight = float(weight)

    config = build_backtest_config(profile_name, rerank_column=rerank_column, rerank_weight=rerank_weight)
    summary_df, daily_df, holdings_df = run_backtest(prediction_df, config, prediction_source=profile_name)
    prediction_df.to_csv(profile_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(profile_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(profile_dir / "backtest_daily.csv", index=False, encoding="utf-8-sig")
    holdings_df.to_csv(profile_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    write_result_from_backtest(daily_df, profile_dir / "result.csv")

    bt = summary_df.iloc[0].to_dict()
    row = {
        "profile_name": profile_name,
        "signal": signal or "",
        "regime_flag": regime_flag or "global_or_none",
        "weight": float(rerank_weight),
        "cost_after_return": float(bt["cumulative_return_after_cost"]),
        "sharpe": float(bt["sharpe_after_cost"]),
        "max_drawdown": float(bt["max_drawdown_after_cost"]),
        "avg_turnover": float(bt["avg_turnover"]),
        "max_single_contribution_share": float(bt.get("max_single_contribution_share", 0.0)),
    }
    row.update(summarize_selection(prediction_df, daily_df, regime_df, profile_name))
    return row


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    baseline = summary[summary["profile_name"].eq("baseline")].iloc[0]
    candidates = summary[~summary["profile_name"].eq("baseline")].copy()
    ranked = candidates.sort_values(
        ["high_volatility_selected_top5_return", "fold3_selected_top5_return", "cost_after_return"],
        ascending=[False, False, False],
    )
    best = ranked.iloc[0]
    lines = [
        "# Regime Rerank Switch Report",
        "",
        "This experiment keeps the current aggressive baseline outside high-risk regimes and applies small defensive rerank penalties only on selected regime dates.",
        "",
        "## Best High-Vol Candidate",
        "",
        f"- profile_name: `{best['profile_name']}`",
        f"- signal: `{best['signal']}`",
        f"- regime_flag: `{best['regime_flag']}`",
        f"- high_volatility_selected_top5_return: `{best['high_volatility_selected_top5_return']:.6f}`",
        f"- fold3_selected_top5_return: `{best['fold3_selected_top5_return']:.6f}`",
        f"- cost_after_return: `{best['cost_after_return']:.6f}`",
        "",
        "## Decision Notes",
        "",
        f"- Baseline high_volatility_selected_top5_return: `{baseline['high_volatility_selected_top5_return']:.6f}`",
        f"- Baseline fold3_selected_top5_return: `{baseline['fold3_selected_top5_return']:.6f}`",
        f"- Baseline cost_after_return: `{baseline['cost_after_return']:.6f}`",
        "- Adopt only if high-volatility/Fold3 improves without a large cost_after_return or single-slice hit.",
        "",
        "## Summary Table",
        "",
        "| profile | signal | regime | cost_after | sharpe | max_dd | selected_top5 | fold1 | fold3 | high_vol | high_vol_range | poor_fp |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['profile_name']} | {row['signal']} | {row['regime_flag']} | "
            f"{row['cost_after_return']:.6f} | {row['sharpe']:.6f} | {row['max_drawdown']:.6f} | "
            f"{row['selected_top5_return_mean']:.6f} | {row['fold1_selected_top5_return']:.6f} | "
            f"{row['fold3_selected_top5_return']:.6f} | {row['high_volatility_selected_top5_return']:.6f} | "
            f"{row['high_volatility_range_selected_top5_return']:.6f} | {int(row['poor_false_positives'])} |"
        )
    (output_dir / "regime_rerank_switch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    pred_path = resolve_path(args.pred_path)
    feature_path = resolve_path(args.feature_path)
    regime_path = resolve_path(args.regime_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_prediction_with_features(pred_path, feature_path)
    regime_df = load_regimes(feature_path, regime_path)

    profiles: list[tuple[str, str | None, str | None]] = [
        ("baseline", None, None),
        ("global_close_position_20d_m005", "close_position_20d", None),
        ("global_reversal_risk_score_m005", "reversal_risk_score", None),
        ("hv_close_position_20d_m005", "close_position_20d", "is_high_volatility"),
        ("hv_reversal_risk_score_m005", "reversal_risk_score", "is_high_volatility"),
        ("hvrange_close_position_20d_m005", "close_position_20d", "is_high_volatility_range"),
        ("hvrange_reversal_risk_score_m005", "reversal_risk_score", "is_high_volatility_range"),
        ("hv_combo_m005", "combo", "is_high_volatility"),
        ("hvrange_combo_m005", "combo", "is_high_volatility_range"),
    ]

    rows = []
    for profile_name, signal, regime_flag in profiles:
        print(f"[regime_rerank] running {profile_name}")
        rows.append(
            run_profile(
                base_df=base_df,
                regime_df=regime_df,
                output_dir=output_dir,
                profile_name=profile_name,
                signal=signal,
                regime_flag=regime_flag,
                weight=float(args.weight),
            )
        )
    summary = pd.DataFrame(rows)
    baseline = summary[summary["profile_name"].eq("baseline")].iloc[0]
    for column in [
        "cost_after_return",
        "sharpe",
        "max_drawdown",
        "selected_top5_return_mean",
        "fold1_selected_top5_return",
        "fold3_selected_top5_return",
        "high_volatility_selected_top5_return",
        "high_volatility_range_selected_top5_return",
        "poor_false_positives",
    ]:
        summary[f"delta_{column}"] = summary[column] - baseline[column]
    summary.to_csv(output_dir / "regime_rerank_switch_summary.csv", index=False, encoding="utf-8-sig")
    write_report(summary, output_dir)
    print(f"[regime_rerank] wrote {output_dir / 'regime_rerank_switch_summary.csv'}")
    print(f"[regime_rerank] wrote {output_dir / 'regime_rerank_switch_report.md'}")


if __name__ == "__main__":
    main()
