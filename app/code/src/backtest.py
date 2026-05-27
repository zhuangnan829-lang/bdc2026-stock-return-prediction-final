import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    BEST_PROFILE_NAME,
    EXECUTION_DEFAULTS,
    RISK_FILTER_DEFAULTS,
    SELECTION_DEFAULTS,
    build_default_inference_args,
)
from utils import (
    apply_turnover_cap,
    build_portfolio_weights,
    calculate_turnover,
    ensure_dir,
    load_feature_frame,
    select_top_candidates,
)


DEFAULT_PREDICTION_PATH = "app/model/walk_forward_predictions.csv"
DEFAULT_FEATURE_PATH = "app/temp/train_features.csv"
DEFAULT_MODEL_DIR = "app/model"
DEFAULT_OUTPUT_DIR = "app/model"
DEFAULTS = build_default_inference_args()
DEFAULT_TOP_K = DEFAULTS["top_k"]
DEFAULT_PRIMARY_CANDIDATE_SIZE = DEFAULTS["primary_candidate_size"]
DEFAULT_MAX_VOLATILITY_20D_PCT = DEFAULTS["max_volatility_20d_pct"]
DEFAULT_MAX_VOLATILITY_5D_PCT = DEFAULTS["max_volatility_5d_pct"]
DEFAULT_TURNOVER_RATE_LOWER_PCT = DEFAULTS["turnover_rate_lower_pct"]
DEFAULT_TURNOVER_RATE_UPPER_PCT = DEFAULTS["turnover_rate_upper_pct"]
DEFAULT_TURNOVER_RATIO_UPPER_PCT = DEFAULTS["turnover_ratio_upper_pct"]
DEFAULT_RISK_PENALTY_WEIGHT = DEFAULTS["risk_penalty_weight"]
DEFAULT_WEIGHTING_SCHEME = DEFAULTS["weighting_scheme"]
DEFAULT_WEIGHT_BLEND_ALPHA = DEFAULTS["weight_blend_alpha"]
DEFAULT_MAX_SINGLE_WEIGHT = DEFAULTS["max_single_weight"]
DEFAULT_TRANSACTION_COST = DEFAULTS["transaction_cost"]
DEFAULT_MAX_TURNOVER = DEFAULTS["max_turnover"]
DEFAULT_SORT_STRATEGY = DEFAULTS["sort_strategy"]

DEFAULT_COMPARE_PROFILES = [
    {
        "profile_name": "default_risk_adjusted",
        "enable_risk_filters": 1,
        "allow_cash_fallback": 0,
        "primary_candidate_size": int(SELECTION_DEFAULTS["primary_candidate_size"]),
        "max_volatility_20d_pct": float(RISK_FILTER_DEFAULTS["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(RISK_FILTER_DEFAULTS["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(RISK_FILTER_DEFAULTS["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(RISK_FILTER_DEFAULTS["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(RISK_FILTER_DEFAULTS["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(RISK_FILTER_DEFAULTS["risk_penalty_weight"]),
        "weighting_scheme": SELECTION_DEFAULTS["weighting_scheme"],
        "max_single_weight": float(SELECTION_DEFAULTS.get("max_single_weight", 1.0)),
        "sort_strategy": SELECTION_DEFAULTS["sort_strategy"],
        "max_turnover": 1.0,
    },
    {
        "profile_name": "looser_risk",
        "enable_risk_filters": 1,
        "allow_cash_fallback": 0,
        "primary_candidate_size": 40,
        "max_volatility_20d_pct": 0.90,
        "max_volatility_5d_pct": 0.90,
        "turnover_rate_lower_pct": 0.03,
        "turnover_rate_upper_pct": 0.97,
        "turnover_ratio_upper_pct": 0.95,
        "risk_penalty_weight": 0.25,
        "weighting_scheme": "risk_adjusted",
        "sort_strategy": "risk_adjusted",
        "max_turnover": 1.0,
    },
    {
        "profile_name": "stricter_risk",
        "enable_risk_filters": 1,
        "allow_cash_fallback": 0,
        "primary_candidate_size": 25,
        "max_volatility_20d_pct": 0.70,
        "max_volatility_5d_pct": 0.70,
        "turnover_rate_lower_pct": 0.10,
        "turnover_rate_upper_pct": 0.90,
        "turnover_ratio_upper_pct": 0.80,
        "risk_penalty_weight": 0.45,
        "weighting_scheme": "risk_adjusted",
        "sort_strategy": "risk_adjusted",
        "max_turnover": 1.0,
    },
    {
        "profile_name": "equal",
        "enable_risk_filters": 1,
        "allow_cash_fallback": 0,
        "primary_candidate_size": 40,
        "max_volatility_20d_pct": 0.90,
        "max_volatility_5d_pct": 0.90,
        "turnover_rate_lower_pct": 0.03,
        "turnover_rate_upper_pct": 0.97,
        "turnover_ratio_upper_pct": 0.95,
        "risk_penalty_weight": 0.25,
        "weighting_scheme": "equal",
        "sort_strategy": "risk_adjusted",
        "max_turnover": 1.0,
    },
    {
        "profile_name": "pred",
        "enable_risk_filters": 1,
        "allow_cash_fallback": 0,
        "primary_candidate_size": 40,
        "max_volatility_20d_pct": 0.90,
        "max_volatility_5d_pct": 0.90,
        "turnover_rate_lower_pct": 0.03,
        "turnover_rate_upper_pct": 0.97,
        "turnover_ratio_upper_pct": 0.95,
        "risk_penalty_weight": 0.25,
        "weighting_scheme": "pred",
        "sort_strategy": "risk_adjusted",
        "max_turnover": 1.0,
    },
    {
        "profile_name": "risk_adjusted",
        "enable_risk_filters": 1,
        "allow_cash_fallback": 0,
        "primary_candidate_size": 40,
        "max_volatility_20d_pct": 0.90,
        "max_volatility_5d_pct": 0.90,
        "turnover_rate_lower_pct": 0.03,
        "turnover_rate_upper_pct": 0.97,
        "turnover_ratio_upper_pct": 0.95,
        "risk_penalty_weight": 0.25,
        "weighting_scheme": "risk_adjusted",
        "sort_strategy": "risk_adjusted",
        "max_turnover": 1.0,
    },
    {
        "profile_name": "looser_risk_low_turnover",
        "enable_risk_filters": 1,
        "allow_cash_fallback": 0,
        "primary_candidate_size": 40,
        "max_volatility_20d_pct": 0.90,
        "max_volatility_5d_pct": 0.90,
        "turnover_rate_lower_pct": 0.03,
        "turnover_rate_upper_pct": 0.97,
        "turnover_ratio_upper_pct": 0.95,
        "risk_penalty_weight": 0.25,
        "weighting_scheme": "risk_adjusted",
        "sort_strategy": "risk_adjusted",
        "max_turnover": float(EXECUTION_DEFAULTS["max_turnover"]),
    },
]

MERGE_FEATURE_COLUMNS = [
    "volatility_5d",
    "volatility_20d",
    "turnover_rate",
    "turnover_ratio_10d",
    "amplitude_ratio_5d",
    "turnover_spike_5d",
    "crowding_reversal_risk_5d",
    "rel_strength_accel_5d_v2",
    "trend_persistence_score_10d_v2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local rolling backtest on validation predictions.")
    parser.add_argument("--prediction_path", default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--feature_path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--primary_candidate_size", type=int, default=DEFAULT_PRIMARY_CANDIDATE_SIZE)
    parser.add_argument("--enable_risk_filters", type=int, choices=[0, 1], default=1)
    parser.add_argument("--allow_cash_fallback", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max_volatility_20d_pct", type=float, default=DEFAULT_MAX_VOLATILITY_20D_PCT)
    parser.add_argument("--max_volatility_5d_pct", type=float, default=DEFAULT_MAX_VOLATILITY_5D_PCT)
    parser.add_argument("--turnover_rate_lower_pct", type=float, default=DEFAULT_TURNOVER_RATE_LOWER_PCT)
    parser.add_argument("--turnover_rate_upper_pct", type=float, default=DEFAULT_TURNOVER_RATE_UPPER_PCT)
    parser.add_argument("--turnover_ratio_upper_pct", type=float, default=DEFAULT_TURNOVER_RATIO_UPPER_PCT)
    parser.add_argument("--risk_penalty_weight", type=float, default=DEFAULT_RISK_PENALTY_WEIGHT)
    parser.add_argument(
        "--weighting_scheme",
        choices=["equal", "pred", "risk_adjusted", "pred_equal_blend"],
        default=DEFAULT_WEIGHTING_SCHEME,
    )
    parser.add_argument("--weight_blend_alpha", type=float, default=DEFAULT_WEIGHT_BLEND_ALPHA)
    parser.add_argument("--max_single_weight", type=float, default=DEFAULT_MAX_SINGLE_WEIGHT)
    parser.add_argument(
        "--sort_strategy",
        choices=["pure_prediction", "risk_adjusted"],
        default=DEFAULT_SORT_STRATEGY,
    )
    parser.add_argument("--transaction_cost", type=float, default=DEFAULT_TRANSACTION_COST)
    parser.add_argument("--max_turnover", type=float, default=DEFAULT_MAX_TURNOVER)
    parser.add_argument("--rerank_signal_column")
    parser.add_argument("--rerank_signal_weight", type=float, default=0.0)
    parser.add_argument("--secondary_candidate_size", type=int)
    parser.add_argument(
        "--secondary_screen_mode",
        choices=["none", "alpha_combo", "alpha_blend", "alpha_local_tiebreak", "quality_layer"],
        default="none",
    )
    parser.add_argument("--secondary_screen_weight", type=float, default=0.0)
    parser.add_argument("--local_tiebreak_start_rank", type=int, default=8)
    parser.add_argument("--local_tiebreak_end_rank", type=int, default=15)
    parser.add_argument("--compare_profiles", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "model_meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_prediction_frame(prediction_path: Path, feature_path: Path) -> pd.DataFrame:
    prediction_df = load_feature_frame(prediction_path)
    if "pred_return" in prediction_df.columns:
        base = prediction_df.copy()
    else:
        raise ValueError("Prediction frame must contain pred_return. Use walk_forward_predictions.csv or generated scores.")

    feature_df = load_feature_frame(feature_path)
    required_feature_columns = ["stock_id", "date", "target_return", *MERGE_FEATURE_COLUMNS]
    missing_feature_columns = [col for col in required_feature_columns if col not in feature_df.columns]
    if missing_feature_columns:
        raise ValueError(f"Feature file missing required columns for backtest: {missing_feature_columns}")

    merge_columns = ["stock_id", "date", *MERGE_FEATURE_COLUMNS]
    merged = base.merge(
        feature_df[merge_columns].drop_duplicates(["stock_id", "date"]),
        on=["stock_id", "date"],
        how="left",
    )
    missing_after_merge = [col for col in MERGE_FEATURE_COLUMNS if merged[col].isna().any()]
    if missing_after_merge:
        raise ValueError(f"Missing merged risk columns in prediction frame: {missing_after_merge}")
    return merged


def generate_prediction_frame(feature_path: Path, model_dir: Path) -> pd.DataFrame:
    metadata = load_metadata(model_dir)
    model_path = Path(metadata["model_path"])
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    feature_columns = metadata["feature_columns"]
    df = load_feature_frame(feature_path)
    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns for prediction generation: {missing_columns}")
    model = joblib.load(model_path)
    df = df[df["target_return"].notna()].copy()
    df["pred_return"] = model.predict(df[feature_columns])
    return df


def load_or_generate_predictions(
    prediction_path: Path,
    feature_path: Path,
    model_dir: Path,
) -> tuple[pd.DataFrame, str]:
    if prediction_path.exists():
        return load_prediction_frame(prediction_path, feature_path), "replay_walk_forward_predictions"
    return generate_prediction_frame(feature_path, model_dir), "regenerated_from_model"


def calculate_max_drawdown(net_values: pd.Series) -> float:
    if net_values.empty:
        return 0.0
    rolling_peak = net_values.cummax()
    drawdown = net_values / rolling_peak - 1.0
    return float(drawdown.min())


def calculate_sharpe(returns: pd.Series) -> float:
    if len(returns) <= 1:
        return 0.0
    std = float(returns.std(ddof=0))
    if std < 1e-12:
        return 0.0
    return float(returns.mean() / std * np.sqrt(52.0))


def calculate_max_contribution_share(holding_df: pd.DataFrame) -> float:
    if holding_df.empty or "executed_weight" not in holding_df or "target_return" not in holding_df:
        return 0.0
    contribution = (holding_df["executed_weight"].astype(float) * holding_df["target_return"].astype(float)).abs()
    total = float(contribution.sum())
    if total <= 1e-12:
        return 0.0
    return float(contribution.max() / total)


def maybe_write_backtest_plots(daily_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    if daily_df.empty:
        return []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    written_paths: list[Path] = []
    for profile_name, profile_df in daily_df.groupby("profile_name"):
        plot_df = profile_df.copy()
        plot_df["date"] = pd.to_datetime(plot_df["date"])

        equity_path = figure_dir / f"backtest_equity_{profile_name}.png"
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(plot_df["date"], plot_df["net_value_before_cost"], label="Before Cost")
        ax.plot(plot_df["date"], plot_df["net_value_after_cost"], label="After Cost")
        ax.set_title(f"Backtest Equity Curve - {profile_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Net Value")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(equity_path, dpi=160)
        plt.close(fig)
        written_paths.append(equity_path)

        drawdown_path = figure_dir / f"backtest_drawdown_{profile_name}.png"
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(plot_df["date"], plot_df["drawdown_before_cost"], label="Before Cost")
        ax.plot(plot_df["date"], plot_df["drawdown_after_cost"], label="After Cost")
        ax.set_title(f"Backtest Drawdown - {profile_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(drawdown_path, dpi=160)
        plt.close(fig)
        written_paths.append(drawdown_path)

    return written_paths


def write_backtest_report(summary_df: pd.DataFrame, output_dir: Path, comparison_path: Path) -> Path:
    ranked = summary_df.sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost", "avg_turnover"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranked["is_best_profile"] = False
    if not ranked.empty:
        ranked.loc[0, "is_best_profile"] = True
    report_path = output_dir / "backtest_report.md"

    lines = [
        "# Backtest Report",
        "",
        "## Best Profile",
        "",
    ]
    if ranked.empty:
        lines.extend(["No backtest rows available.", ""])
    else:
        best = ranked.iloc[0]
        lines.extend(
            [
                f"- profile_name: `{best['profile_name']}`",
                f"- cumulative_return_after_cost: `{best['cumulative_return_after_cost']:.6f}`",
                f"- sharpe_after_cost: `{best['sharpe_after_cost']:.6f}`",
                f"- max_drawdown_after_cost: `{best['max_drawdown_after_cost']:.6f}`",
                f"- max_single_contribution_share: `{best.get('max_single_contribution_share', 0.0):.6f}`",
                f"- avg_turnover: `{best['avg_turnover']:.6f}`",
                "",
                "## Ranked Summary",
                "",
                "| profile_name | cum_after_cost | sharpe_after_cost | max_dd_after_cost | max_contrib_share | avg_turnover | best |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for _, row in ranked.iterrows():
            lines.append(
                f"| {row['profile_name']} | {row['cumulative_return_after_cost']:.6f} | "
                f"{row['sharpe_after_cost']:.6f} | {row['max_drawdown_after_cost']:.6f} | "
                f"{row.get('max_single_contribution_share', 0.0):.6f} | "
                f"{row['avg_turnover']:.6f} | {'yes' if row['is_best_profile'] else 'no'} |"
            )
        lines.extend(["", f"Reference CSV: `{comparison_path.name}`", ""])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_result_snapshot(summary_df: pd.DataFrame, holdings_df: pd.DataFrame, output_dir: Path) -> Path:
    result_path = output_dir / "result.csv"
    if summary_df.empty or holdings_df.empty:
        pd.DataFrame(columns=["stock_id", "weight"]).to_csv(result_path, index=False, encoding="utf-8-sig")
        return result_path

    best_profile = (
        summary_df.sort_values(
            ["cumulative_return_after_cost", "sharpe_after_cost", "avg_turnover"],
            ascending=[False, False, True],
        )
        .iloc[0]["profile_name"]
    )
    profile_holdings = holdings_df[holdings_df["profile_name"] == best_profile].copy()
    if profile_holdings.empty:
        result_df = pd.DataFrame(columns=["stock_id", "weight"])
    else:
        latest_date = profile_holdings["date"].max()
        result_df = (
            profile_holdings[profile_holdings["date"] == latest_date]
            .sort_values(["executed_weight", "stock_id"], ascending=[False, True])
            .loc[:, ["stock_id", "executed_weight"]]
            .rename(columns={"executed_weight": "weight"})
        )
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    return result_path


def make_config(args: argparse.Namespace, overrides: dict | None = None) -> dict:
    config = {
        "profile_name": BEST_PROFILE_NAME if not overrides else overrides.get("profile_name", BEST_PROFILE_NAME),
        "top_k": int(args.top_k),
        "primary_candidate_size": int(args.primary_candidate_size),
        "enable_risk_filters": int(args.enable_risk_filters),
        "allow_cash_fallback": int(args.allow_cash_fallback),
        "max_volatility_20d_pct": float(args.max_volatility_20d_pct),
        "max_volatility_5d_pct": float(args.max_volatility_5d_pct),
        "turnover_rate_lower_pct": float(args.turnover_rate_lower_pct),
        "turnover_rate_upper_pct": float(args.turnover_rate_upper_pct),
        "turnover_ratio_upper_pct": float(args.turnover_ratio_upper_pct),
        "risk_penalty_weight": float(args.risk_penalty_weight),
        "weighting_scheme": args.weighting_scheme,
        "weight_blend_alpha": float(args.weight_blend_alpha),
        "max_single_weight": float(args.max_single_weight),
        "sort_strategy": args.sort_strategy,
        "transaction_cost": float(args.transaction_cost),
        "max_turnover": float(args.max_turnover),
        "rerank_signal_column": args.rerank_signal_column,
        "rerank_signal_weight": float(args.rerank_signal_weight),
        "secondary_candidate_size": int(args.secondary_candidate_size) if args.secondary_candidate_size else 0,
        "secondary_screen_mode": args.secondary_screen_mode,
        "secondary_screen_weight": float(args.secondary_screen_weight),
        "local_tiebreak_start_rank": int(args.local_tiebreak_start_rank),
        "local_tiebreak_end_rank": int(args.local_tiebreak_end_rank),
    }
    if overrides:
        config.update(overrides)
    return config


def run_backtest(
    prediction_df: pd.DataFrame,
    config: dict,
    prediction_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = prediction_df.copy()
    working["date"] = pd.to_datetime(working["date"])
    working = working.sort_values(["date", "stock_id"]).reset_index(drop=True)

    daily_rows: list[dict] = []
    holdings_rows: list[dict] = []
    previous_weights: dict[str, float] = {}
    prev_net_before = 1.0
    prev_net_after = 1.0

    for trade_date, day_df in working.groupby("date"):
        selected, diagnostics = select_top_candidates(
            latest_df=day_df,
            top_k=config["top_k"],
            primary_candidate_size=config["primary_candidate_size"],
            max_volatility_20d_pct=config["max_volatility_20d_pct"],
            max_volatility_5d_pct=config["max_volatility_5d_pct"],
            turnover_rate_lower_pct=config["turnover_rate_lower_pct"],
            turnover_rate_upper_pct=config["turnover_rate_upper_pct"],
            turnover_ratio_upper_pct=config["turnover_ratio_upper_pct"],
            risk_penalty_weight=config["risk_penalty_weight"],
            sort_strategy=config["sort_strategy"],
            rerank_signal_column=config.get("rerank_signal_column"),
            rerank_signal_weight=float(config.get("rerank_signal_weight", 0.0)),
            secondary_candidate_size=int(config.get("secondary_candidate_size", 0)) or None,
            secondary_screen_mode=config.get("secondary_screen_mode", "none"),
            secondary_screen_weight=float(config.get("secondary_screen_weight", 0.0)),
            local_tiebreak_start_rank=int(config.get("local_tiebreak_start_rank", 8)),
            local_tiebreak_end_rank=int(config.get("local_tiebreak_end_rank", 15)),
            enable_risk_filters=bool(config["enable_risk_filters"]),
            allow_cash_fallback=bool(config["allow_cash_fallback"]),
        )
        selected = build_portfolio_weights(
            selected,
            top_k=config["top_k"],
            weighting_scheme=config["weighting_scheme"],
            max_single_weight=config.get("max_single_weight"),
            weight_blend_alpha=float(config.get("weight_blend_alpha", 1.0)),
        )

        target_weights = dict(zip(selected["stock_id"], selected["weight"]))
        current_weights, desired_turnover, execution_strength = apply_turnover_cap(
            previous_weights=previous_weights,
            target_weights=target_weights,
            max_turnover=config["max_turnover"],
        )
        selected["executed_weight"] = selected["stock_id"].map(current_weights).fillna(0.0)
        executed = selected[selected["executed_weight"] > 1e-12].copy()

        turnover = calculate_turnover(previous_weights, current_weights)
        cost = turnover * config["transaction_cost"]
        gross_return = float((executed["executed_weight"] * executed["target_return"]).sum()) if not executed.empty else 0.0
        net_return = gross_return - cost
        cash_weight = max(0.0, 1.0 - float(sum(current_weights.values()))) if current_weights else 1.0
        net_before = prev_net_before * (1.0 + gross_return)
        net_after = prev_net_after * (1.0 + net_return)

        executed_sorted = executed.sort_values(
            ["selection_score_final", "selection_score", "pred_return", "stock_id"], ascending=[False, False, False, True]
        ).reset_index(drop=True)

        for _, row in executed_sorted.iterrows():
            holdings_rows.append(
                {
                    "profile_name": config["profile_name"],
                    "date": trade_date.date().isoformat(),
                    "stock_id": row["stock_id"],
                    "target_weight": float(row["weight"]),
                    "executed_weight": float(row["executed_weight"]),
                    "pred_return": float(row["pred_return"]),
                    "target_return": float(row["target_return"]),
                    "selection_score": float(row["selection_score"]),
                    "selection_score_final": float(row["selection_score_final"]),
                    "rerank_signal_score": float(row["rerank_signal_score"]),
                    "risk_penalty": float(row["risk_penalty"]),
                    "secondary_candidate_size": int(config.get("secondary_candidate_size", 0)),
                    "secondary_screen_mode": config.get("secondary_screen_mode", "none"),
                    "secondary_signal_score": float(row.get("secondary_signal_score", 0.0)),
                    "secondary_screen_weight": float(config.get("secondary_screen_weight", 0.0)),
                    "local_tiebreak_start_rank": int(config.get("local_tiebreak_start_rank", 8)),
                    "local_tiebreak_end_rank": int(config.get("local_tiebreak_end_rank", 15)),
                    "volatility_20d": float(row["volatility_20d"]),
                    "volatility_5d": float(row["volatility_5d"]),
                    "turnover_rate": float(row["turnover_rate"]),
                    "turnover_ratio_10d": float(row["turnover_ratio_10d"]),
                    "amplitude_ratio_5d": float(row["amplitude_ratio_5d"]),
                    "turnover_spike_5d": float(row.get("turnover_spike_5d", 0.0)),
                    "crowding_reversal_risk_5d": float(row.get("crowding_reversal_risk_5d", 0.0)),
                }
            )

        daily_rows.append(
            {
                "profile_name": config["profile_name"],
                "date": trade_date.date().isoformat(),
                "selected_count": int(len(executed_sorted)),
                "signal_count": int(len(selected)),
                "cash_weight": cash_weight,
                "gross_return": gross_return,
                "transaction_cost": cost,
                "net_return": net_return,
                "turnover": turnover,
                "desired_turnover": desired_turnover,
                "execution_strength": execution_strength,
                "net_value_before_cost": net_before,
                "net_value_after_cost": net_after,
                "selected_stock_ids": ",".join(executed_sorted["stock_id"].tolist()),
                "selected_weights": ",".join(f"{weight:.6f}" for weight in executed_sorted["executed_weight"].tolist()),
                "target_weights": ",".join(f"{weight:.6f}" for weight in selected["weight"].tolist()),
                "selected_pred_returns": ",".join(f"{value:.6f}" for value in executed_sorted["pred_return"].tolist()),
                "selected_target_returns": ",".join(f"{value:.6f}" for value in executed_sorted["target_return"].tolist()),
                "max_single_weight": float(executed_sorted["executed_weight"].max()) if not executed_sorted.empty else 0.0,
                "top2_weight_sum": float(executed_sorted["executed_weight"].nlargest(2).sum()) if not executed_sorted.empty else 0.0,
                "max_single_contribution_share": calculate_max_contribution_share(executed_sorted),
                "filter_initial_candidates": diagnostics.get("initial_candidates", 0),
                "filter_after_primary_screen": diagnostics.get("after_primary_screen", 0),
                "filter_after_risk_filters": diagnostics.get("after_risk_filters", 0),
                "filter_after_secondary_screen": diagnostics.get("after_secondary_screen", 0),
                "filter_secondary_quality_filter_count": diagnostics.get("secondary_quality_filter_count", 0),
                "filter_fallback_used": diagnostics.get("fallback_used", False),
            }
        )

        previous_weights = current_weights
        prev_net_before = net_before
        prev_net_after = net_after

    daily_df = pd.DataFrame(daily_rows)
    holdings_df = pd.DataFrame(holdings_rows)
    if not daily_df.empty:
        rolling_peak_before = daily_df["net_value_before_cost"].cummax()
        rolling_peak_after = daily_df["net_value_after_cost"].cummax()
        daily_df["drawdown_before_cost"] = daily_df["net_value_before_cost"] / rolling_peak_before - 1.0
        daily_df["drawdown_after_cost"] = daily_df["net_value_after_cost"] / rolling_peak_after - 1.0

    summary = {
        "profile_name": config["profile_name"],
        "prediction_source": prediction_source,
        "weighting_scheme": config["weighting_scheme"],
        "weight_blend_alpha": float(config.get("weight_blend_alpha", 1.0)),
        "max_single_weight_param": float(config.get("max_single_weight", 1.0)),
        "sort_strategy": config["sort_strategy"],
        "top_k": int(config["top_k"]),
        "enable_risk_filters": bool(config["enable_risk_filters"]),
        "allow_cash_fallback": bool(config["allow_cash_fallback"]),
        "max_volatility_20d_pct": float(config["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(config["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(config["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(config["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(config["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(config["risk_penalty_weight"]),
        "transaction_cost": float(config["transaction_cost"]),
        "max_turnover": float(config["max_turnover"]),
        "rerank_signal_column": config.get("rerank_signal_column") or "",
        "rerank_signal_weight": float(config.get("rerank_signal_weight", 0.0)),
        "secondary_candidate_size": int(config.get("secondary_candidate_size", 0)),
        "secondary_screen_mode": config.get("secondary_screen_mode", "none"),
        "secondary_screen_weight": float(config.get("secondary_screen_weight", 0.0)),
        "local_tiebreak_start_rank": int(config.get("local_tiebreak_start_rank", 8)),
        "local_tiebreak_end_rank": int(config.get("local_tiebreak_end_rank", 15)),
        "periods": int(len(daily_df)),
        "cumulative_return_before_cost": float(daily_df["net_value_before_cost"].iloc[-1] - 1.0) if not daily_df.empty else 0.0,
        "cumulative_return_after_cost": float(daily_df["net_value_after_cost"].iloc[-1] - 1.0) if not daily_df.empty else 0.0,
        "mean_period_return_before_cost": float(daily_df["gross_return"].mean()) if not daily_df.empty else 0.0,
        "mean_period_return_after_cost": float(daily_df["net_return"].mean()) if not daily_df.empty else 0.0,
        "return_volatility_before_cost": float(daily_df["gross_return"].std(ddof=0)) if len(daily_df) > 1 else 0.0,
        "return_volatility_after_cost": float(daily_df["net_return"].std(ddof=0)) if len(daily_df) > 1 else 0.0,
        "max_drawdown_before_cost": calculate_max_drawdown(daily_df["net_value_before_cost"]) if not daily_df.empty else 0.0,
        "max_drawdown_after_cost": calculate_max_drawdown(daily_df["net_value_after_cost"]) if not daily_df.empty else 0.0,
        "sharpe_before_cost": calculate_sharpe(daily_df["gross_return"]) if not daily_df.empty else 0.0,
        "sharpe_after_cost": calculate_sharpe(daily_df["net_return"]) if not daily_df.empty else 0.0,
        "win_rate_before_cost": float((daily_df["gross_return"] > 0).mean()) if not daily_df.empty else 0.0,
        "win_rate_after_cost": float((daily_df["net_return"] > 0).mean()) if not daily_df.empty else 0.0,
        "avg_selected_count": float(daily_df["selected_count"].mean()) if not daily_df.empty else 0.0,
        "avg_signal_count": float(daily_df["signal_count"].mean()) if not daily_df.empty else 0.0,
        "avg_cash_weight": float(daily_df["cash_weight"].mean()) if not daily_df.empty else 0.0,
        "avg_max_single_weight": float(daily_df["max_single_weight"].mean()) if not daily_df.empty else 0.0,
        "max_single_weight_observed": float(daily_df["max_single_weight"].max()) if not daily_df.empty else 0.0,
        "avg_top2_weight_sum": float(daily_df["top2_weight_sum"].mean()) if not daily_df.empty else 0.0,
        "avg_max_single_contribution_share": float(daily_df["max_single_contribution_share"].mean()) if not daily_df.empty else 0.0,
        "max_single_contribution_share": float(daily_df["max_single_contribution_share"].max()) if not daily_df.empty else 0.0,
        "avg_turnover": float(daily_df["turnover"].mean()) if not daily_df.empty else 0.0,
        "avg_desired_turnover": float(daily_df["desired_turnover"].mean()) if not daily_df.empty else 0.0,
        "avg_execution_strength": float(daily_df["execution_strength"].mean()) if not daily_df.empty else 0.0,
        "total_transaction_cost": float(daily_df["transaction_cost"].sum()) if not daily_df.empty else 0.0,
    }
    summary_df = pd.DataFrame([summary])
    return summary_df, daily_df, holdings_df


def run_backtest_profiles(
    prediction_df: pd.DataFrame,
    prediction_source: str,
    profile_configs: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    daily_frames = []
    holdings_frames = []
    for profile in profile_configs:
        summary_df, daily_df, holdings_df = run_backtest(
            prediction_df=prediction_df,
            config=profile,
            prediction_source=prediction_source,
        )
        summary_frames.append(summary_df)
        daily_frames.append(daily_df)
        holdings_frames.append(holdings_df)

    all_summary = pd.concat(summary_frames, ignore_index=True)
    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_holdings = pd.concat(holdings_frames, ignore_index=True)
    return all_summary, all_daily, all_holdings


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.prediction_path)
    feature_path = Path(args.feature_path)
    model_dir = Path(args.model_dir)
    output_dir = ensure_dir(args.output_dir)

    prediction_df, prediction_source = load_or_generate_predictions(
        prediction_path=prediction_path,
        feature_path=feature_path,
        model_dir=model_dir,
    )
    if args.compare_profiles:
        profile_configs = [make_config(args, overrides=profile) for profile in DEFAULT_COMPARE_PROFILES]
    else:
        profile_configs = [make_config(args)]
    summary_df, daily_df, holdings_df = run_backtest_profiles(
        prediction_df=prediction_df,
        prediction_source=prediction_source,
        profile_configs=profile_configs,
    )

    summary_path = output_dir / "backtest_summary.csv"
    daily_path = output_dir / "backtest_daily_results.csv"
    holdings_path = output_dir / "backtest_holdings.csv"
    comparison_path = output_dir / "backtest_config_comparison.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    holdings_df.to_csv(holdings_path, index=False, encoding="utf-8-sig")
    summary_df.sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost", "avg_turnover"],
        ascending=[False, False, True],
    ).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    report_path = write_backtest_report(summary_df, output_dir, comparison_path)
    plot_paths = maybe_write_backtest_plots(daily_df, output_dir)
    result_path = write_result_snapshot(summary_df, holdings_df, output_dir)

    best = summary_df.sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost", "avg_turnover"],
        ascending=[False, False, True],
    ).iloc[0]
    print(f"[backtest] prediction_source={prediction_source}")
    print(f"[backtest] profiles_run={len(summary_df)}")
    print(
        "[backtest] best_profile="
        f"{best['profile_name']} "
        f"cum_after={best['cumulative_return_after_cost']:.6f} "
        f"sharpe_after={best['sharpe_after_cost']:.6f} "
        f"avg_turnover={best['avg_turnover']:.6f}"
    )
    print(f"[backtest] wrote {summary_path}")
    print(f"[backtest] wrote {daily_path}")
    print(f"[backtest] wrote {holdings_path}")
    print(f"[backtest] wrote {comparison_path}")
    print(f"[backtest] wrote {report_path}")
    print(f"[backtest] wrote {result_path}")
    for plot_path in plot_paths:
        print(f"[backtest] wrote {plot_path}")


if __name__ == "__main__":
    main()
