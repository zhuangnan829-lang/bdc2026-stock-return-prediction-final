from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import (
    calculate_max_drawdown,
    calculate_sharpe,
    load_prediction_frame,
)
from config import ROOT_DIR
from evaluate_rank_stability import build_daily_rank_ic, build_fold_rank_ic
from load_submission_config import build_default_inference_args, load_submission_config
from result_validator import validate_result_file
from utils import (
    apply_turnover_cap,
    build_portfolio_weights,
    calculate_turnover,
    select_top_candidates,
)


DEFAULT_PRED_PATH = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_PREDICT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "predict_features.csv"
DEFAULT_AGGRESSIVE_CONFIG = ROOT_DIR / "app" / "model" / "configs" / "submission_aggressive_candidate.json"
DEFAULT_ROBUST_CONFIG = ROOT_DIR / "app" / "model" / "configs" / "submission_robust_candidate.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "regime_switch"

SUMMARY_COLUMNS = [
    "profile_name",
    "active_rule",
    "single_slice_score",
    "cost_after_return",
    "Sharpe",
    "max_drawdown",
    "avg_turnover",
    "rank_ic_mean",
    "worst_fold_rank_ic",
    "high_vol_range_cost_after_return",
    "high_vol_range_max_drawdown",
    "high_vol_range_avg_return",
    "robust_day_count",
    "total_days",
    "latest_regime",
    "latest_selected_config",
    "adopted",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simple rule-based aggressive/robust regime switching.")
    parser.add_argument("--pred_path", default=str(DEFAULT_PRED_PATH))
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--predict_feature_path", default=str(DEFAULT_PREDICT_FEATURE_PATH))
    parser.add_argument("--aggressive_config", default=str(DEFAULT_AGGRESSIVE_CONFIG))
    parser.add_argument("--robust_config", default=str(DEFAULT_ROBUST_CONFIG))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--volatility_quantile", type=float, default=0.70)
    parser.add_argument("--range_quantile", type=float, default=0.30)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else 0.0


def build_simple_regimes(
    feature_path: Path,
    *,
    volatility_quantile: float = 0.70,
    range_quantile: float = 0.30,
    output_path: Path | None = None,
) -> pd.DataFrame:
    features = pd.read_csv(feature_path, encoding="utf-8-sig", dtype={"stock_id": str})
    required = {"date", "volatility_20d", "ret_1d"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Feature data missing regime columns: {missing}")
    features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.normalize()
    daily = (
        features.dropna(subset=["date"])
        .groupby("date", as_index=False)
        .agg(
            market_volatility_20d=("volatility_20d", _safe_mean),
            market_return_1d=("ret_1d", _safe_mean),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["market_return_20d"] = daily["market_return_1d"].rolling(20, min_periods=5).sum()
    vol_threshold = float(daily["market_volatility_20d"].quantile(volatility_quantile))
    range_threshold = float(daily["market_return_20d"].abs().dropna().quantile(range_quantile))
    daily["volatility_threshold_70q"] = vol_threshold
    daily["range_abs_return_20d_threshold_30q"] = range_threshold
    daily["is_high_volatility"] = daily["market_volatility_20d"].ge(vol_threshold).astype(int)
    daily["is_range"] = daily["market_return_20d"].abs().le(range_threshold).fillna(False).astype(int)
    daily["is_high_volatility_range"] = (daily["is_high_volatility"].eq(1) & daily["is_range"].eq(1)).astype(int)
    daily["regime"] = np.where(
        daily["is_high_volatility_range"].eq(1),
        "high_volatility_range",
        np.where(daily["is_high_volatility"].eq(1), "high_volatility_trend_or_normal", "normal_or_low_volatility"),
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(output_path, index=False, encoding="utf-8-sig")
    return daily


def build_current_regime(
    train_feature_path: Path,
    predict_feature_path: Path,
    *,
    volatility_quantile: float,
    range_quantile: float,
) -> dict[str, Any]:
    train_regime = build_simple_regimes(
        train_feature_path,
        volatility_quantile=volatility_quantile,
        range_quantile=range_quantile,
    )
    if not predict_feature_path.exists():
        latest = train_regime.iloc[-1]
        return {
            "latest_date": str(pd.to_datetime(latest["date"]).date()),
            "latest_regime": str(latest["regime"]),
            "selected_config": "robust" if int(latest["is_high_volatility_range"]) else "aggressive",
            "source": "train_features_last_date",
        }
    predict = pd.read_csv(predict_feature_path, encoding="utf-8-sig", dtype={"stock_id": str})
    if not {"date", "volatility_20d", "ret_1d"}.issubset(predict.columns):
        latest = train_regime.iloc[-1]
        return {
            "latest_date": str(pd.to_datetime(latest["date"]).date()),
            "latest_regime": str(latest["regime"]),
            "selected_config": "robust" if int(latest["is_high_volatility_range"]) else "aggressive",
            "source": "train_features_last_date_missing_predict_columns",
        }
    combined = pd.concat(
        [
            pd.read_csv(train_feature_path, encoding="utf-8-sig", dtype={"stock_id": str}),
            predict,
        ],
        ignore_index=True,
    )
    combined_path = None
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
    daily = (
        combined.dropna(subset=["date"])
        .groupby("date", as_index=False)
        .agg(market_volatility_20d=("volatility_20d", _safe_mean), market_return_1d=("ret_1d", _safe_mean))
        .sort_values("date")
        .reset_index(drop=True)
    )
    vol_threshold = float(train_regime["volatility_threshold_70q"].iloc[0])
    range_threshold = float(train_regime["range_abs_return_20d_threshold_30q"].iloc[0])
    daily["market_return_20d"] = daily["market_return_1d"].rolling(20, min_periods=5).sum()
    daily["is_high_volatility"] = daily["market_volatility_20d"].ge(vol_threshold).astype(int)
    daily["is_range"] = daily["market_return_20d"].abs().le(range_threshold).fillna(False).astype(int)
    daily["is_high_volatility_range"] = (daily["is_high_volatility"].eq(1) & daily["is_range"].eq(1)).astype(int)
    daily["regime"] = np.where(
        daily["is_high_volatility_range"].eq(1),
        "high_volatility_range",
        np.where(daily["is_high_volatility"].eq(1), "high_volatility_trend_or_normal", "normal_or_low_volatility"),
    )
    latest = daily.iloc[-1]
    return {
        "latest_date": str(pd.to_datetime(latest["date"]).date()),
        "latest_regime": str(latest["regime"]),
        "selected_config": "robust" if int(latest["is_high_volatility_range"]) else "aggressive",
        "source": "predict_features_latest_date",
        "market_volatility_20d": float(latest["market_volatility_20d"]),
        "market_return_20d": float(latest["market_return_20d"]),
        "volatility_threshold_70q": vol_threshold,
        "range_abs_return_20d_threshold_30q": range_threshold,
        "combined_path": str(combined_path or ""),
    }


def config_to_backtest_config(config_path: Path, profile_name: str) -> dict[str, Any]:
    defaults = build_default_inference_args(load_submission_config(config_path))
    return {
        "profile_name": profile_name,
        "top_k": int(defaults["top_k"]),
        "primary_candidate_size": int(defaults["primary_candidate_size"]),
        "enable_risk_filters": bool(defaults["enable_risk_filters"]),
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": float(defaults["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(defaults["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(defaults["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(defaults["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(defaults["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(defaults["risk_penalty_weight"]),
        "weighting_scheme": str(defaults["weighting_scheme"]),
        "weight_blend_alpha": float(defaults["weight_blend_alpha"]),
        "max_single_weight": defaults["max_single_weight"],
        "sort_strategy": str(defaults["sort_strategy"]),
        "transaction_cost": float(defaults["transaction_cost"]),
        "max_turnover": float(defaults["max_turnover"]),
        "rerank_signal_column": defaults.get("rerank_signal_column") or None,
        "rerank_signal_weight": float(defaults.get("rerank_signal_weight", 0.0)),
        "secondary_candidate_size": 0,
        "secondary_screen_mode": "none",
        "secondary_screen_weight": 0.0,
        "local_tiebreak_start_rank": 8,
        "local_tiebreak_end_rank": 15,
    }


def choose_config(
    profile_name: str,
    is_high_volatility_range: bool,
    aggressive_config: dict[str, Any],
    robust_config: dict[str, Any],
) -> dict[str, Any]:
    if profile_name == "aggressive_static":
        return aggressive_config
    if profile_name == "robust_static":
        return robust_config
    if profile_name == "regime_switch":
        return robust_config if is_high_volatility_range else aggressive_config
    raise ValueError(f"Unknown profile: {profile_name}")


def calculate_high_vol_range_stats(daily_df: pd.DataFrame) -> dict[str, float]:
    hv = daily_df[daily_df["is_high_volatility_range"].eq(1)].copy()
    if hv.empty:
        return {
            "high_vol_range_cost_after_return": 0.0,
            "high_vol_range_max_drawdown": 0.0,
            "high_vol_range_avg_return": 0.0,
        }
    net = (1.0 + pd.to_numeric(hv["net_return"], errors="coerce").fillna(0.0)).cumprod()
    return {
        "high_vol_range_cost_after_return": float(net.iloc[-1] - 1.0),
        "high_vol_range_max_drawdown": calculate_max_drawdown(net),
        "high_vol_range_avg_return": float(pd.to_numeric(hv["net_return"], errors="coerce").fillna(0.0).mean()),
    }


def run_dynamic_backtest(
    prediction_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    *,
    profile_name: str,
    aggressive_config: dict[str, Any],
    robust_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = prediction_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    regimes = regime_df[["date", "regime", "is_high_volatility_range"]].copy()
    regimes["date"] = pd.to_datetime(regimes["date"], errors="coerce").dt.normalize()
    working = working.merge(regimes, on="date", how="left")
    working["is_high_volatility_range"] = working["is_high_volatility_range"].fillna(0).astype(int)
    working["regime"] = working["regime"].fillna("unknown")

    daily_rows: list[dict[str, Any]] = []
    holdings_rows: list[dict[str, Any]] = []
    previous_weights: dict[str, float] = {}
    prev_net_before = 1.0
    prev_net_after = 1.0

    for trade_date, day_df in working.groupby("date", sort=True):
        is_hvr = bool(int(day_df["is_high_volatility_range"].iloc[0]))
        active = choose_config(profile_name, is_hvr, aggressive_config, robust_config)
        selected, diagnostics = select_top_candidates(
            latest_df=day_df,
            top_k=active["top_k"],
            primary_candidate_size=active["primary_candidate_size"],
            max_volatility_20d_pct=active["max_volatility_20d_pct"],
            max_volatility_5d_pct=active["max_volatility_5d_pct"],
            turnover_rate_lower_pct=active["turnover_rate_lower_pct"],
            turnover_rate_upper_pct=active["turnover_rate_upper_pct"],
            turnover_ratio_upper_pct=active["turnover_ratio_upper_pct"],
            risk_penalty_weight=active["risk_penalty_weight"],
            sort_strategy=active["sort_strategy"],
            rerank_signal_column=active.get("rerank_signal_column"),
            rerank_signal_weight=float(active.get("rerank_signal_weight", 0.0)),
            secondary_candidate_size=None,
            secondary_screen_mode="none",
            secondary_screen_weight=0.0,
            local_tiebreak_start_rank=8,
            local_tiebreak_end_rank=15,
            enable_risk_filters=bool(active["enable_risk_filters"]),
            allow_cash_fallback=False,
        )
        selected = build_portfolio_weights(
            selected,
            top_k=active["top_k"],
            weighting_scheme=active["weighting_scheme"],
            max_single_weight=active.get("max_single_weight"),
            weight_blend_alpha=float(active.get("weight_blend_alpha", 1.0)),
        )
        target_weights = dict(zip(selected["stock_id"], selected["weight"]))
        current_weights, desired_turnover, execution_strength = apply_turnover_cap(
            previous_weights=previous_weights,
            target_weights=target_weights,
            max_turnover=active["max_turnover"],
        )
        selected["executed_weight"] = selected["stock_id"].map(current_weights).fillna(0.0)
        executed = selected[selected["executed_weight"] > 1e-12].copy()
        turnover = calculate_turnover(previous_weights, current_weights)
        cost = turnover * active["transaction_cost"]
        gross_return = float((executed["executed_weight"] * executed["target_return"]).sum()) if not executed.empty else 0.0
        net_return = gross_return - cost
        net_before = prev_net_before * (1.0 + gross_return)
        net_after = prev_net_after * (1.0 + net_return)
        executed_sorted = executed.sort_values(
            ["selection_score_final", "selection_score", "pred_return", "stock_id"],
            ascending=[False, False, False, True],
        )
        for _, row in executed_sorted.iterrows():
            holdings_rows.append(
                {
                    "profile_name": profile_name,
                    "active_config": "robust" if is_hvr else "aggressive",
                    "date": trade_date.date().isoformat(),
                    "stock_id": row["stock_id"],
                    "executed_weight": float(row["executed_weight"]),
                    "target_return": float(row["target_return"]),
                    "pred_return": float(row["pred_return"]),
                }
            )
        daily_rows.append(
            {
                "profile_name": profile_name,
                "date": trade_date.date().isoformat(),
                "regime": str(day_df["regime"].iloc[0]),
                "is_high_volatility_range": int(is_hvr),
                "active_config": "robust" if is_hvr else "aggressive",
                "selected_count": int(len(executed_sorted)),
                "gross_return": gross_return,
                "transaction_cost": cost,
                "net_return": net_return,
                "turnover": turnover,
                "desired_turnover": desired_turnover,
                "execution_strength": execution_strength,
                "net_value_before_cost": net_before,
                "net_value_after_cost": net_after,
                "selected_stock_ids": ",".join(executed_sorted["stock_id"].astype(str).tolist()),
                "selected_weights": ",".join(f"{value:.6f}" for value in executed_sorted["executed_weight"].tolist()),
                "selected_target_returns": ",".join(f"{value:.6f}" for value in executed_sorted["target_return"].tolist()),
                "filter_after_risk_filters": diagnostics.get("after_risk_filters", 0),
            }
        )
        previous_weights = current_weights
        prev_net_before = net_before
        prev_net_after = net_after

    daily_df = pd.DataFrame(daily_rows)
    holdings_df = pd.DataFrame(holdings_rows)
    if not daily_df.empty:
        rolling_peak_after = daily_df["net_value_after_cost"].cummax()
        daily_df["drawdown_after_cost"] = daily_df["net_value_after_cost"] / rolling_peak_after - 1.0
    rank_daily = build_daily_rank_ic(prediction_df, prediction_column="pred_return", target_column="target_return")
    rank_fold = build_fold_rank_ic(rank_daily)
    fold_rank_ic = pd.to_numeric(rank_fold.get("rank_ic", pd.Series(dtype=float)), errors="coerce").dropna()
    hv_stats = calculate_high_vol_range_stats(daily_df)
    summary = {
        "profile_name": profile_name,
        "active_rule": "robust_if_high_volatility_range_else_aggressive" if profile_name == "regime_switch" else profile_name,
        "single_slice_score": float(daily_df["net_return"].iloc[-1]) if not daily_df.empty else 0.0,
        "cost_after_return": float(daily_df["net_value_after_cost"].iloc[-1] - 1.0) if not daily_df.empty else 0.0,
        "Sharpe": calculate_sharpe(daily_df["net_return"]) if not daily_df.empty else 0.0,
        "max_drawdown": calculate_max_drawdown(daily_df["net_value_after_cost"]) if not daily_df.empty else 0.0,
        "avg_turnover": float(daily_df["turnover"].mean()) if not daily_df.empty else 0.0,
        "rank_ic_mean": float(fold_rank_ic.mean()) if not fold_rank_ic.empty else 0.0,
        "worst_fold_rank_ic": float(fold_rank_ic.min()) if not fold_rank_ic.empty else 0.0,
        **hv_stats,
        "robust_day_count": int(daily_df["is_high_volatility_range"].sum()) if not daily_df.empty else 0,
        "total_days": int(len(daily_df)),
    }
    return pd.DataFrame([summary]), daily_df, holdings_df


def write_result_snapshot(daily_df: pd.DataFrame, output_path: Path) -> None:
    if daily_df.empty:
        pd.DataFrame(columns=["stock_id", "weight"]).to_csv(output_path, index=False, encoding="utf-8")
        return
    last = daily_df.sort_values("date").iloc[-1]
    ids = [item.strip().zfill(6) for item in str(last["selected_stock_ids"]).split(",") if item.strip()]
    weights = [float(item) for item in str(last["selected_weights"]).split(",") if item.strip()]
    result = pd.DataFrame({"stock_id": ids[: len(weights)], "weight": weights[: len(ids)]})
    result.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")
    validate_result_file(output_path)


def write_report(summary: pd.DataFrame, current_regime: dict[str, Any], output_dir: Path) -> None:
    aggressive = summary[summary["profile_name"].eq("aggressive_static")].iloc[0]
    switch = summary[summary["profile_name"].eq("regime_switch")].iloc[0]
    stable_up = switch["worst_fold_rank_ic"] > aggressive["worst_fold_rank_ic"] + 1e-12
    hvr_dd_up = switch["high_vol_range_max_drawdown"] >= aggressive["high_vol_range_max_drawdown"]
    return_loss = float(aggressive["cost_after_return"] - switch["cost_after_return"])
    not_too_much_return_loss = return_loss <= 0.05
    adopted = bool(stable_up and hvr_dd_up and not_too_much_return_loss and switch["cost_after_return"] >= aggressive["cost_after_return"] - 0.02)
    lines = [
        "# Regime Switch Submission Report",
        "",
        "Rule: use robust config only when the recent market state is high-volatility range; otherwise use aggressive config.",
        "",
        "## Current Regime",
        "",
        f"- latest_date: `{current_regime.get('latest_date', '')}`",
        f"- latest_regime: `{current_regime.get('latest_regime', '')}`",
        f"- selected_config: `{current_regime.get('selected_config', '')}`",
        f"- source: `{current_regime.get('source', '')}`",
        f"- volatility_threshold_70q: `{current_regime.get('volatility_threshold_70q', 0.0):.6f}`",
        f"- range_abs_return_20d_threshold_30q: `{current_regime.get('range_abs_return_20d_threshold_30q', 0.0):.6f}`",
        "",
        "## Required Answers",
        "",
        f"1. Regime switching 是否提升稳定性: {'yes' if stable_up else 'no'}, worst_fold_rank_ic `{switch['worst_fold_rank_ic']:.6f}` vs aggressive `{aggressive['worst_fold_rank_ic']:.6f}`.",
        f"2. 是否降低高波动震荡阶段回撤: {'yes' if hvr_dd_up else 'no'}, high_vol_range_max_drawdown `{switch['high_vol_range_max_drawdown']:.6f}` vs aggressive `{aggressive['high_vol_range_max_drawdown']:.6f}`.",
        f"3. 是否牺牲过多收益: {'no' if not_too_much_return_loss else 'yes'}, cost_after_return loss `{return_loss:.6f}`.",
        "4. 阈值是否简单可解释: yes, volatility_20d uses the training 70% quantile and range uses the 30% quantile of abs(market_return_20d).",
        f"5. 是否建议用于最终提交: {'yes' if adopted else 'no'}, recommendation is `{'final_submission_candidate' if adopted else 'research_only'}`.",
        "",
        "## Summary",
        "",
        "| profile | rule | slice | cost_after | sharpe | mdd | turnover | rank_ic | worst_fold | hvr_cost | hvr_mdd | hvr_avg | robust_days | adopted |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['profile_name']} | {row['active_rule']} | {row['single_slice_score']:.6f} | "
            f"{row['cost_after_return']:.6f} | {row['Sharpe']:.6f} | {row['max_drawdown']:.6f} | "
            f"{row['avg_turnover']:.6f} | {row['rank_ic_mean']:.6f} | {row['worst_fold_rank_ic']:.6f} | "
            f"{row['high_vol_range_cost_after_return']:.6f} | {row['high_vol_range_max_drawdown']:.6f} | "
            f"{row['high_vol_range_avg_return']:.6f} | {int(row['robust_day_count'])} | {str(bool(row['adopted'])).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This switch uses the turnover-stress robust candidate, not the earlier rerank robust candidate. If the switch does not beat the aggressive walk-forward profile, keep it as research evidence only.",
        ]
    )
    (output_dir / "regime_switch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = resolve_path(args.pred_path)
    feature_path = resolve_path(args.feature_path)
    aggressive_config_path = resolve_path(args.aggressive_config)
    robust_config_path = resolve_path(args.robust_config)

    regime_df = build_simple_regimes(
        feature_path,
        volatility_quantile=float(args.volatility_quantile),
        range_quantile=float(args.range_quantile),
        output_path=output_dir / "daily_simple_regimes.csv",
    )
    current_regime = build_current_regime(
        feature_path,
        resolve_path(args.predict_feature_path),
        volatility_quantile=float(args.volatility_quantile),
        range_quantile=float(args.range_quantile),
    )
    prediction_df = load_prediction_frame(pred_path, feature_path)
    aggressive_config = config_to_backtest_config(aggressive_config_path, "aggressive_static")
    robust_config = config_to_backtest_config(robust_config_path, "robust_static")

    rows = []
    for profile_name in ["aggressive_static", "robust_static", "regime_switch"]:
        print(f"[regime_switch] running {profile_name}")
        summary_df, daily_df, holdings_df = run_dynamic_backtest(
            prediction_df,
            regime_df,
            profile_name=profile_name,
            aggressive_config=aggressive_config,
            robust_config=robust_config,
        )
        profile_dir = output_dir / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(profile_dir / "backtest_daily_results.csv", index=False, encoding="utf-8-sig")
        holdings_df.to_csv(profile_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
        write_result_snapshot(daily_df, profile_dir / "result.csv")
        rows.append(summary_df.iloc[0].to_dict())

    summary = pd.DataFrame(rows)
    aggressive = summary[summary["profile_name"].eq("aggressive_static")].iloc[0]
    for idx, row in summary.iterrows():
        adopted = (
            row["profile_name"] == "regime_switch"
            and row["worst_fold_rank_ic"] >= aggressive["worst_fold_rank_ic"]
            and row["high_vol_range_max_drawdown"] >= aggressive["high_vol_range_max_drawdown"]
            and row["cost_after_return"] >= aggressive["cost_after_return"] - 0.02
        )
        summary.loc[idx, "latest_regime"] = current_regime.get("latest_regime", "")
        summary.loc[idx, "latest_selected_config"] = current_regime.get("selected_config", "")
        summary.loc[idx, "adopted"] = bool(adopted)
        summary.loc[idx, "notes"] = (
            "simple_threshold_switch" if row["profile_name"] == "regime_switch" else "static_reference"
        )
    summary[SUMMARY_COLUMNS].to_csv(output_dir / "regime_switch_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "current_regime_decision.json").write_text(
        json.dumps(current_regime, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(summary[SUMMARY_COLUMNS], current_regime, output_dir)
    print(summary[SUMMARY_COLUMNS].to_string(index=False))
    print(f"[regime_switch] wrote {output_dir / 'regime_switch_summary.csv'}")
    print(f"[regime_switch] wrote {output_dir / 'regime_switch_report.md'}")


if __name__ == "__main__":
    main()
