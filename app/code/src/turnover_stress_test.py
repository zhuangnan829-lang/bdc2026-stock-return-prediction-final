from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import DEFAULT_MODEL_DIR, load_or_generate_predictions, make_config, run_backtest
from load_submission_config import build_default_inference_args, load_submission_config


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PREDICTION_PATH = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_BASE_CONFIG = ROOT_DIR / "app" / "model" / "default_submission_config.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "turnover_stress_test"
DEFAULT_CONFIG_DIR = ROOT_DIR / "app" / "model" / "configs"

SUMMARY_COLUMNS = [
    "max_turnover",
    "transaction_cost",
    "weight_strategy",
    "max_single_weight",
    "cost_after_return",
    "sharpe",
    "max_drawdown",
    "avg_turnover",
    "total_cost",
    "win_rate",
    "top5_return_mean",
    "rank_ic_mean",
    "robust_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test turnover, transaction cost, weight strategy, and max single weight.")
    parser.add_argument("--pred_path", "--prediction_path", dest="pred_path", default=str(DEFAULT_PREDICTION_PATH))
    parser.add_argument("--data_path", "--feature_path", dest="data_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--base_config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config_dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--max_turnovers", nargs="+", type=float, default=[0.50, 0.65, 0.75, 0.85, 1.00])
    parser.add_argument("--transaction_costs", nargs="+", type=float, default=[0.001, 0.002, 0.003, 0.005])
    parser.add_argument("--weight_strategies", nargs="+", default=["pred", "equal", "blend_0.5"])
    parser.add_argument("--max_single_weights", nargs="+", default=["none", "0.20", "0.18"])
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate


def parse_cap(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "null", "nan", ""}:
        return None
    return float(text)


def cap_label(cap: float | None) -> str:
    return "none" if cap is None else f"{cap:.2f}"


def strategy_to_backtest(strategy: str) -> tuple[str, float]:
    normalized = strategy.strip().lower()
    if normalized == "pred":
        return "pred", 1.0
    if normalized == "equal":
        return "equal", 0.0
    if normalized in {"blend_0.5", "blend_0_5", "pred_equal_blend"}:
        return "pred_equal_blend", 0.5
    raise ValueError(f"Unsupported weight_strategy: {strategy}")


def build_base_args(base_config: dict[str, Any]) -> argparse.Namespace:
    defaults = build_default_inference_args(base_config)
    return argparse.Namespace(
        top_k=int(defaults["top_k"]),
        primary_candidate_size=int(defaults["primary_candidate_size"]),
        enable_risk_filters=int(defaults["enable_risk_filters"]),
        allow_cash_fallback=0,
        max_volatility_20d_pct=float(defaults["max_volatility_20d_pct"]),
        max_volatility_5d_pct=float(defaults["max_volatility_5d_pct"]),
        turnover_rate_lower_pct=float(defaults["turnover_rate_lower_pct"]),
        turnover_rate_upper_pct=float(defaults["turnover_rate_upper_pct"]),
        turnover_ratio_upper_pct=float(defaults["turnover_ratio_upper_pct"]),
        risk_penalty_weight=float(defaults["risk_penalty_weight"]),
        weighting_scheme="pred",
        weight_blend_alpha=1.0,
        max_single_weight=1.0,
        sort_strategy=str(defaults["sort_strategy"]),
        transaction_cost=float(defaults["transaction_cost"]),
        max_turnover=float(defaults["max_turnover"]),
        rerank_signal_column=None,
        rerank_signal_weight=0.0,
        secondary_candidate_size=None,
        secondary_screen_mode="none",
        secondary_screen_weight=0.0,
        local_tiebreak_start_rank=8,
        local_tiebreak_end_rank=15,
    )


def compute_rank_ic_mean(prediction_df: pd.DataFrame) -> float:
    work = prediction_df.copy()
    if "pred_return" not in work or "target_return" not in work:
        return 0.0
    values = []
    for _, day_df in work.groupby("date"):
        if day_df["pred_return"].nunique() <= 1 or day_df["target_return"].nunique() <= 1:
            continue
        corr = day_df["pred_return"].corr(day_df["target_return"], method="spearman")
        if pd.notna(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else 0.0


def normalize_prediction_frame(prediction_df: pd.DataFrame) -> pd.DataFrame:
    out = prediction_df.copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "pred_return" in out:
        out["pred_return"] = pd.to_numeric(out["pred_return"], errors="coerce")
    if "target_return" in out:
        out["target_return"] = pd.to_numeric(out["target_return"], errors="coerce")
    return out.dropna(subset=["stock_id", "date", "pred_return", "target_return"]).copy()


def minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index)
    lower = float(valid.min())
    upper = float(valid.max())
    if abs(upper - lower) <= 1e-12:
        scored = pd.Series(np.where(values.notna(), 1.0, 0.0), index=series.index)
    else:
        scored = (values - lower) / (upper - lower)
    if not higher_is_better:
        scored = 1.0 - scored
    return scored.fillna(0.0).clip(0.0, 1.0)


def add_robust_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    successful = out["error"].fillna("").astype(str).eq("")
    out["robust_score"] = np.nan
    if not successful.any():
        return out
    sub = out.loc[successful].copy()
    drawdown_penalty = minmax(sub["max_drawdown"].abs(), higher_is_better=True)
    turnover_penalty = minmax(sub["avg_turnover"], higher_is_better=True)
    score = (
        minmax(sub["cost_after_return"], higher_is_better=True)
        + minmax(sub["sharpe"], higher_is_better=True)
        - drawdown_penalty
        - turnover_penalty
    )
    out.loc[successful, "robust_score"] = score
    out["robust_score"] = out["robust_score"].fillna(-999.0)
    return out


def profile_name(max_turnover: float, transaction_cost: float, strategy: str, cap: float | None) -> str:
    return (
        f"mt{int(round(max_turnover * 100)):03d}_"
        f"tc{int(round(transaction_cost * 10000)):04d}_"
        f"{strategy}_cap{cap_label(cap)}"
    )


def run_one(
    *,
    prediction_df: pd.DataFrame,
    prediction_source: str,
    base_args: argparse.Namespace,
    rank_ic_mean: float,
    max_turnover: float,
    transaction_cost: float,
    weight_strategy: str,
    cap: float | None,
    output_dir: Path,
) -> dict[str, Any]:
    weighting_scheme, alpha = strategy_to_backtest(weight_strategy)
    label = profile_name(max_turnover, transaction_cost, weight_strategy, cap)
    config = make_config(
        base_args,
        overrides={
            "profile_name": label,
            "max_turnover": float(max_turnover),
            "transaction_cost": float(transaction_cost),
            "weighting_scheme": weighting_scheme,
            "weight_blend_alpha": float(alpha),
            "max_single_weight": 1.0 if cap is None else float(cap),
        },
    )
    combo_dir = output_dir / "profiles" / label
    combo_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary_df, daily_df, holdings_df = run_backtest(
            prediction_df=prediction_df,
            config=config,
            prediction_source=prediction_source,
        )
        daily_df.to_csv(combo_dir / "backtest_daily_results.csv", index=False, encoding="utf-8-sig")
        holdings_df.to_csv(combo_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(combo_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
        row = summary_df.iloc[0]
        return {
            "max_turnover": float(max_turnover),
            "transaction_cost": float(transaction_cost),
            "weight_strategy": weight_strategy,
            "max_single_weight": cap_label(cap),
            "cost_after_return": float(row["cumulative_return_after_cost"]),
            "sharpe": float(row["sharpe_after_cost"]),
            "max_drawdown": float(row["max_drawdown_after_cost"]),
            "avg_turnover": float(row["avg_turnover"]),
            "total_cost": float(row["total_transaction_cost"]),
            "win_rate": float(row["win_rate_after_cost"]),
            "top5_return_mean": float(row["mean_period_return_before_cost"]),
            "rank_ic_mean": float(rank_ic_mean),
            "robust_score": np.nan,
            "error": "",
            "profile_name": label,
        }
    except Exception as exc:
        return {
            "max_turnover": float(max_turnover),
            "transaction_cost": float(transaction_cost),
            "weight_strategy": weight_strategy,
            "max_single_weight": cap_label(cap),
            "cost_after_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_turnover": 0.0,
            "total_cost": 0.0,
            "win_rate": 0.0,
            "top5_return_mean": 0.0,
            "rank_ic_mean": float(rank_ic_mean),
            "robust_score": -999.0,
            "error": str(exc),
            "profile_name": label,
        }


def update_config_for_candidate(base_config: dict[str, Any], row: pd.Series, status: str) -> dict[str, Any]:
    out = json.loads(json.dumps(base_config, ensure_ascii=False))
    out["status"] = status
    out["profile_name"] = f"{status}_{row['profile_name']}"
    strategy = str(row["weight_strategy"])
    weighting_scheme, alpha = strategy_to_backtest(strategy)
    cap = parse_cap(str(row["max_single_weight"]))
    out["selection_logic"]["weighting_scheme"] = weighting_scheme
    out["selection_logic"]["weight_blend_alpha"] = alpha
    out["selection_logic"]["max_single_weight"] = cap
    out["execution_logic"]["max_turnover"] = float(row["max_turnover"])
    out["execution_logic"]["transaction_cost"] = float(row["transaction_cost"])
    out["model_name"] = out.get("model_family", "lstm")
    out["sequence_length"] = int(out["validation_scheme"]["sequence_length"])
    out["sort_strategy"] = out["selection_logic"]["sort_strategy"]
    out["weight_strategy"] = strategy
    out["top_k"] = int(out["selection_logic"]["top_k"])
    out["candidate_size"] = int(out["selection_logic"]["primary_candidate_size"])
    out["risk_penalty_weight"] = float(out["risk_filter_thresholds"]["risk_penalty_weight"])
    out["max_turnover"] = float(row["max_turnover"])
    out["transaction_cost"] = float(row["transaction_cost"])
    out["max_single_weight"] = cap
    notes = list(out.get("notes", []))
    notes.append(
        "Generated by turnover_stress_test.py as a candidate config; default_submission_config.json was not overwritten."
    )
    out["notes"] = notes
    return out


def write_candidate_configs(summary: pd.DataFrame, base_config: dict[str, Any], config_dir: Path) -> tuple[Path, Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    valid = summary[summary["error"].fillna("").astype(str).eq("")].copy()
    low_cost = valid[valid["transaction_cost"] == valid["transaction_cost"].min()].copy()
    aggressive_row = low_cost.sort_values(
        ["cost_after_return", "sharpe", "avg_turnover"],
        ascending=[False, False, True],
    ).iloc[0]
    robust_candidates = valid[
        (valid["transaction_cost"].isin([0.002, 0.003, valid["transaction_cost"].max()]))
        & (valid["max_turnover"] <= 0.85)
    ].copy()
    if robust_candidates.empty:
        robust_candidates = valid.copy()
    robust_row = robust_candidates.sort_values(
        ["robust_score", "cost_after_return", "sharpe"],
        ascending=[False, False, False],
    ).iloc[0]

    aggressive_path = config_dir / "submission_aggressive_candidate.json"
    robust_path = config_dir / "submission_robust_candidate.json"
    aggressive_path.write_text(
        json.dumps(update_config_for_candidate(base_config, aggressive_row, "aggressive_candidate"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    robust_path.write_text(
        json.dumps(update_config_for_candidate(base_config, robust_row, "robust_candidate"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return aggressive_path, robust_path


def render_report(summary: pd.DataFrame, aggressive_path: Path, robust_path: Path) -> str:
    valid = summary[summary["error"].fillna("").astype(str).eq("")].copy()
    current = valid[
        (valid["max_turnover"].round(8) == 1.0)
        & (valid["transaction_cost"].round(8) == 0.001)
        & (valid["weight_strategy"] == "pred")
        & (valid["max_single_weight"] == "0.18")
    ]
    tc001_best = valid[valid["transaction_cost"] == 0.001].sort_values("cost_after_return", ascending=False).iloc[0]
    tc002_best = valid[valid["transaction_cost"] == 0.002].sort_values("cost_after_return", ascending=False).iloc[0]
    tc003_best = valid[valid["transaction_cost"] == 0.003].sort_values("cost_after_return", ascending=False).iloc[0]
    aggressive = valid.sort_values(["cost_after_return", "sharpe"], ascending=[False, False]).iloc[0]
    robust_pool = valid[
        (valid["transaction_cost"].isin([0.002, 0.003, valid["transaction_cost"].max()]))
        & (valid["max_turnover"] <= 0.85)
    ].copy()
    if robust_pool.empty:
        robust_pool = valid.copy()
    robust = robust_pool.sort_values(["robust_score", "cost_after_return"], ascending=[False, False]).iloc[0]

    over_aggressive = False
    if not current.empty:
        cur = current.iloc[0]
        lower_turnover = valid[
            (valid["transaction_cost"] == 0.001)
            & (valid["weight_strategy"] == "pred")
            & (valid["max_single_weight"] == "0.18")
            & (valid["max_turnover"] < 1.0)
        ]
        if not lower_turnover.empty:
            best_lower = lower_turnover.sort_values("cost_after_return", ascending=False).iloc[0]
            over_aggressive = float(cur["cost_after_return"] - best_lower["cost_after_return"]) < 0.02

    collapse_002 = float((tc001_best["cost_after_return"] - tc002_best["cost_after_return"]) / max(abs(tc001_best["cost_after_return"]), 1e-12))
    collapse_003 = float((tc001_best["cost_after_return"] - tc003_best["cost_after_return"]) / max(abs(tc001_best["cost_after_return"]), 1e-12))

    lines = [
        "# Turnover Stress Test Report",
        "",
        f"- total_grid_rows: `{len(summary)}`",
        f"- failed_rows: `{int((summary['error'].fillna('').astype(str) != '').sum())}`",
        "",
        "## Required Answers",
        "",
        f"1. 当前 max_turnover=1.00 是否过于激进？{'是' if over_aggressive else '不明显'}。"
        "判断依据是低换手候选在收益接近时能降低交易强度。",
        f"2. 交易成本升到 0.002 或 0.003 后是否明显崩塌？"
        f"tc=0.002 最优收益降幅 `{collapse_002:.2%}`，tc=0.003 最优收益降幅 `{collapse_003:.2%}`。",
        f"3. robust 稳健版建议：`{robust['profile_name']}`，robust_score `{robust['robust_score']:.6f}`。"
        "该选择优先从 tc>=0.002 且 max_turnover<=0.85 的压力场景中产生。",
        f"4. aggressive 冲分版建议：`{aggressive['profile_name']}`，cost_after_return `{aggressive['cost_after_return']:.6f}`。",
        "5. 建议保留 aggressive/robust 双配置：是。收益最优和稳健最优并不完全等价。",
        "",
        "## Candidate Configs",
        "",
        f"- aggressive candidate: `{aggressive_path}`",
        f"- robust candidate: `{robust_path}`",
        "",
        "## Top Robust Score Rows",
        "",
        "| robust_rank | max_turnover | tc | strategy | cap | cost_after | sharpe | max_dd | avg_turnover | robust_score |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    top_robust = valid.sort_values(["robust_score", "cost_after_return"], ascending=[False, False]).head(12)
    for rank, (_, row) in enumerate(top_robust.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['max_turnover']:.2f} | {row['transaction_cost']:.3f} | {row['weight_strategy']} | "
            f"{row['max_single_weight']} | {row['cost_after_return']:.6f} | {row['sharpe']:.6f} | "
            f"{row['max_drawdown']:.6f} | {row['avg_turnover']:.6f} | {row['robust_score']:.6f} |"
        )
    if (summary["error"].fillna("").astype(str) != "").any():
        lines.extend(["", "## Failed Experiments", "", "| profile | error |", "|---|---|"])
        for _, row in summary[summary["error"].fillna("").astype(str) != ""].iterrows():
            lines.append(f"| {row['profile_name']} | {row['error']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = resolve_path(args.config_dir)
    base_config = load_submission_config(resolve_path(args.base_config))
    base_args = build_base_args(base_config)
    prediction_df, prediction_source = load_or_generate_predictions(
        prediction_path=resolve_path(args.pred_path),
        feature_path=resolve_path(args.data_path),
        model_dir=resolve_path(args.model_dir),
    )
    prediction_df = normalize_prediction_frame(prediction_df)
    rank_ic_mean = compute_rank_ic_mean(prediction_df)

    caps = [parse_cap(value) for value in args.max_single_weights]
    rows = []
    for max_turnover, transaction_cost, strategy, cap in itertools.product(
        args.max_turnovers,
        args.transaction_costs,
        args.weight_strategies,
        caps,
    ):
        rows.append(
            run_one(
                prediction_df=prediction_df,
                prediction_source=prediction_source,
                base_args=base_args,
                rank_ic_mean=rank_ic_mean,
                max_turnover=max_turnover,
                transaction_cost=transaction_cost,
                weight_strategy=strategy,
                cap=cap,
                output_dir=output_dir,
            )
        )

    full_summary = add_robust_score(pd.DataFrame(rows))
    public_columns = SUMMARY_COLUMNS + ["error", "profile_name"]
    summary_path = output_dir / "turnover_stress_summary.csv"
    full_summary[public_columns].to_csv(summary_path, index=False, encoding="utf-8-sig")
    aggressive_path, robust_path = write_candidate_configs(full_summary, base_config, config_dir)
    report_path = output_dir / "turnover_stress_report.md"
    report_path.write_text(render_report(full_summary, aggressive_path, robust_path), encoding="utf-8-sig")
    metadata = {
        "pred_path": str(resolve_path(args.pred_path)),
        "data_path": str(resolve_path(args.data_path)),
        "base_config": str(resolve_path(args.base_config)),
        "prediction_source": prediction_source,
        "max_turnovers": args.max_turnovers,
        "transaction_costs": args.transaction_costs,
        "weight_strategies": args.weight_strategies,
        "max_single_weights": [cap_label(cap) for cap in caps],
    }
    (output_dir / "turnover_stress_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[turnover_stress_test] rows={len(full_summary)} failed={(full_summary['error'].fillna('').astype(str) != '').sum()}")
    print(f"[turnover_stress_test] wrote {summary_path}")
    print(f"[turnover_stress_test] wrote {report_path}")
    print(f"[turnover_stress_test] wrote {aggressive_path}")
    print(f"[turnover_stress_test] wrote {robust_path}")
    top = full_summary[full_summary["error"].fillna("").astype(str).eq("")].sort_values(
        ["robust_score", "cost_after_return"], ascending=[False, False]
    ).head(10)
    print(top[SUMMARY_COLUMNS].to_string(index=False))


if __name__ == "__main__":
    main()
