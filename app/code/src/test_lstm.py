import argparse
import json
from pathlib import Path

import pandas as pd

from config import BEST_PROFILE_NAME, build_default_inference_args, resolve_metadata_artifact_path
from lstm_utils import build_sequence_dataset, load_lstm_checkpoint, predict_sequences, transform_sequences
from market_regime_split import build_daily_market_regimes
from utils import (
    apply_turnover_cap,
    build_candidate_debug_frame,
    build_portfolio_weights,
    load_feature_frame,
    load_result_portfolio,
    select_top_candidates,
)


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
DEFAULT_MAX_TURNOVER = DEFAULTS["max_turnover"]
DEFAULT_SORT_STRATEGY = DEFAULTS["sort_strategy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM inference entrypoint.")
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--history_feature_path")
    parser.add_argument("--score_output_path")
    parser.add_argument("--debug_candidates_path")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--primary_candidate_size", type=int, default=DEFAULT_PRIMARY_CANDIDATE_SIZE)
    parser.add_argument("--max_volatility_20d_pct", type=float, default=DEFAULT_MAX_VOLATILITY_20D_PCT)
    parser.add_argument("--max_volatility_5d_pct", type=float, default=DEFAULT_MAX_VOLATILITY_5D_PCT)
    parser.add_argument("--turnover_rate_lower_pct", type=float, default=DEFAULT_TURNOVER_RATE_LOWER_PCT)
    parser.add_argument("--turnover_rate_upper_pct", type=float, default=DEFAULT_TURNOVER_RATE_UPPER_PCT)
    parser.add_argument("--turnover_ratio_upper_pct", type=float, default=DEFAULT_TURNOVER_RATIO_UPPER_PCT)
    parser.add_argument("--risk_penalty_weight", type=float, default=DEFAULT_RISK_PENALTY_WEIGHT)
    parser.add_argument(
        "--sort_strategy",
        choices=["pure_prediction", "risk_adjusted"],
        default=DEFAULT_SORT_STRATEGY,
    )
    parser.add_argument(
        "--weighting_scheme",
        choices=["equal", "pred", "risk_adjusted", "pred_equal_blend"],
        default=DEFAULT_WEIGHTING_SCHEME,
    )
    parser.add_argument("--weight_blend_alpha", type=float, default=DEFAULT_WEIGHT_BLEND_ALPHA)
    parser.add_argument("--max_single_weight", type=float, default=DEFAULT_MAX_SINGLE_WEIGHT)
    parser.add_argument("--previous_result_path")
    parser.add_argument("--max_turnover", type=float, default=DEFAULT_MAX_TURNOVER)
    parser.add_argument("--rerank_signal_column")
    parser.add_argument("--rerank_signal_weight", type=float, default=0.0)
    parser.add_argument("--regime_rerank_enabled", action="store_true")
    parser.add_argument("--regime_rerank_flag", default="")
    parser.add_argument("--regime_rerank_signal", default="")
    parser.add_argument("--regime_rerank_weight", type=float, default=0.0)
    parser.add_argument("--secondary_candidate_size", type=int)
    parser.add_argument(
        "--secondary_screen_mode",
        choices=["none", "alpha_combo", "alpha_blend", "alpha_local_tiebreak", "quality_layer"],
        default="none",
    )
    parser.add_argument("--secondary_screen_weight", type=float, default=0.0)
    parser.add_argument("--local_tiebreak_start_rank", type=int, default=8)
    parser.add_argument("--local_tiebreak_end_rank", type=int, default=15)
    return parser.parse_args()


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "model_meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def validate_result(df: pd.DataFrame) -> None:
    if list(df.columns) != ["stock_id", "weight"]:
        raise ValueError("result.csv must have exactly two columns: stock_id, weight")
    if len(df) > 5:
        raise ValueError("result.csv must contain at most 5 rows")
    if not df["stock_id"].astype(str).str.fullmatch(r"\d{6}").all():
        raise ValueError("All stock_id values must be 6-digit strings")
    if not df["stock_id"].is_unique:
        raise ValueError("stock_id values must be unique")
    if not (df["weight"] >= 0).all():
        raise ValueError("All weights must be non-negative")
    if float(df["weight"].sum()) > 1.0 + 1e-9:
        raise ValueError("Weight sum must be <= 1")


def sanitize_result_weights(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    total_weight = float(out["weight"].sum())
    if total_weight > 1.0 and total_weight <= 1.0 + 1e-6:
        out["weight"] = out["weight"] / total_weight
    return out


def resolve_history_frame(feature_path: Path, history_feature_path: str | None) -> pd.DataFrame:
    if history_feature_path:
        history_path = Path(history_feature_path)
    else:
        candidate = feature_path.parent / "train_features.csv"
        history_path = candidate if candidate.exists() and candidate.resolve() != feature_path.resolve() else None

    if history_path is None or not history_path.exists():
        return pd.DataFrame()

    history_df = load_feature_frame(history_path)
    return history_df


def resolve_effective_rerank(
    *,
    args: argparse.Namespace,
    context_df: pd.DataFrame,
    latest_date: pd.Timestamp,
) -> tuple[str | None, float, dict]:
    diagnostics = {
        "regime_rerank_enabled": bool(args.regime_rerank_enabled),
        "regime_rerank_active": False,
        "regime_rerank_flag": args.regime_rerank_flag or "",
        "regime_rerank_signal": args.regime_rerank_signal or "",
        "regime_rerank_weight": float(args.regime_rerank_weight),
        "latest_regime": "",
    }
    if not args.regime_rerank_enabled:
        return args.rerank_signal_column, float(args.rerank_signal_weight), diagnostics

    if not args.regime_rerank_flag or not args.regime_rerank_signal:
        raise ValueError("--regime_rerank_flag and --regime_rerank_signal are required when regime rerank is enabled")
    if args.regime_rerank_signal not in context_df.columns:
        raise ValueError(f"Missing regime rerank signal column: {args.regime_rerank_signal}")

    regime_df = build_daily_market_regimes(context_df)
    regime_df["date"] = pd.to_datetime(regime_df["date"], errors="coerce").dt.normalize()
    flag = args.regime_rerank_flag
    if flag not in regime_df.columns:
        raise ValueError(f"Unknown regime flag `{flag}`. Available columns: {list(regime_df.columns)}")

    latest_key = pd.Timestamp(latest_date).normalize()
    latest_rows = regime_df[regime_df["date"].eq(latest_key)]
    if latest_rows.empty:
        raise ValueError(f"No market regime row found for latest date {latest_key.date()}")
    latest_regime = latest_rows.iloc[-1]
    active = bool(int(latest_regime[flag]))
    diagnostics.update(
        {
            "regime_rerank_active": active,
            "latest_regime": str(latest_regime.get("primary_regime", "")),
            "latest_market_volatility_20d": float(latest_regime.get("market_volatility_20d", 0.0)),
            "latest_volatility_threshold": float(latest_regime.get("volatility_threshold", 0.0)),
        }
    )
    if active:
        return args.regime_rerank_signal, float(args.regime_rerank_weight), diagnostics
    return args.rerank_signal_column, float(args.rerank_signal_weight), diagnostics


def main() -> None:
    args = parse_args()
    feature_path = Path(args.feature_path)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output_path)

    metadata = load_metadata(model_dir)
    model_path = resolve_metadata_artifact_path(model_dir, metadata["model_path"])
    feature_columns = metadata["feature_columns"]
    sequence_length = int(metadata["sequence_length"])
    batch_size = int(metadata.get("batch_size", 256))

    target_df = load_feature_frame(feature_path)
    history_df = resolve_history_frame(feature_path, args.history_feature_path)
    if not history_df.empty:
        min_target_date = target_df["date"].min()
        history_df = history_df[history_df["date"] < min_target_date].copy()
        context_df = pd.concat([history_df, target_df], ignore_index=True)
    else:
        context_df = target_df.copy()
    context_df = (
        context_df.sort_values(["stock_id", "date"])
        .drop_duplicates(["stock_id", "date"], keep="last")
        .reset_index(drop=True)
    )

    missing_columns = [column for column in feature_columns if column not in context_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns for LSTM inference: {missing_columns}")

    target_dates = set(pd.to_datetime(target_df["date"]).tolist())
    bundle = build_sequence_dataset(
        context_df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        target_dates=target_dates,
        label_column=None,
        raw_label_column=None,
    )
    if len(bundle.x) == 0:
        raise ValueError(
            "No inference sequences were built. Check whether predict_features has enough history "
            "or provide --history_feature_path."
        )

    model, checkpoint = load_lstm_checkpoint(model_path)
    score_x = transform_sequences(bundle.x, checkpoint["scaler_mean"], checkpoint["scaler_std"])
    pred = predict_sequences(model, score_x, batch_size=batch_size, device=next(model.parameters()).device)

    scored = bundle.meta.copy()
    scored["pred_return"] = pred
    latest_date = scored["date"].max()
    latest_df = target_df[target_df["date"] == latest_date].copy()
    latest_scores = scored[scored["date"] == latest_date][["stock_id", "date", "pred_return"]].copy()
    latest_df = latest_df.merge(latest_scores, on=["stock_id", "date"], how="inner")
    if latest_df.empty:
        raise ValueError("No prediction rows found for the latest date after sequence construction")
    effective_rerank_column, effective_rerank_weight, regime_diagnostics = resolve_effective_rerank(
        args=args,
        context_df=context_df,
        latest_date=latest_date,
    )

    selected, diagnostics = select_top_candidates(
        latest_df=latest_df,
        top_k=args.top_k,
        primary_candidate_size=args.primary_candidate_size,
        max_volatility_20d_pct=args.max_volatility_20d_pct,
        max_volatility_5d_pct=args.max_volatility_5d_pct,
        turnover_rate_lower_pct=args.turnover_rate_lower_pct,
        turnover_rate_upper_pct=args.turnover_rate_upper_pct,
        turnover_ratio_upper_pct=args.turnover_ratio_upper_pct,
        risk_penalty_weight=args.risk_penalty_weight,
        sort_strategy=args.sort_strategy,
        rerank_signal_column=effective_rerank_column,
        rerank_signal_weight=effective_rerank_weight,
        secondary_candidate_size=args.secondary_candidate_size,
        secondary_screen_mode=args.secondary_screen_mode,
        secondary_screen_weight=args.secondary_screen_weight,
        local_tiebreak_start_rank=args.local_tiebreak_start_rank,
        local_tiebreak_end_rank=args.local_tiebreak_end_rank,
        enable_risk_filters=True,
        allow_cash_fallback=False,
    )
    diagnostics.update(regime_diagnostics)

    weighted = build_portfolio_weights(
        selected,
        top_k=args.top_k,
        weighting_scheme=args.weighting_scheme,
        max_single_weight=args.max_single_weight,
        weight_blend_alpha=args.weight_blend_alpha,
    )
    target_weights = dict(zip(weighted["stock_id"], weighted["weight"]))

    previous_weights: dict[str, float] = {}
    if args.previous_result_path:
        previous_result_path = Path(args.previous_result_path)
        if previous_result_path.exists():
            previous_weights = load_result_portfolio(previous_result_path)

    if previous_weights:
        executed_weights, desired_turnover, execution_strength = apply_turnover_cap(
            previous_weights=previous_weights,
            target_weights=target_weights,
            max_turnover=args.max_turnover,
        )
    else:
        executed_weights = target_weights
        desired_turnover = float(sum(abs(weight) for weight in target_weights.values()))
        execution_strength = 1.0

    rows = [
        {"stock_id": stock_id, "weight": float(weight)}
        for stock_id, weight in executed_weights.items()
        if weight > 1e-12
    ]
    if rows:
        result = (
            pd.DataFrame(rows)
            .sort_values(["weight", "stock_id"], ascending=[False, True])
            .head(args.top_k)
            .reset_index(drop=True)
        )
    else:
        result = pd.DataFrame(columns=["stock_id", "weight"])
    result = sanitize_result_weights(result)
    validate_result(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")

    if args.score_output_path:
        score_output_path = Path(args.score_output_path)
        score_output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(score_output_path, index=False, encoding="utf-8-sig")

    if args.debug_candidates_path:
        debug_path = Path(args.debug_candidates_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_df = build_candidate_debug_frame(
            latest_df=latest_df,
            selected_df=selected,
            weighted_df=weighted,
            executed_weights=executed_weights,
            diagnostics=diagnostics,
            top_k=args.top_k,
        )
        debug_df.to_csv(debug_path, index=False, encoding="utf-8-sig")

    print(f"[test_lstm] feature_path={feature_path}")
    print(f"[test_lstm] model_path={model_path}")
    print(f"[test_lstm] latest_date={latest_date.date()} candidate_count={len(latest_df)}")
    print(f"[test_lstm] default_profile={BEST_PROFILE_NAME}")
    print(f"[test_lstm] sequence_length={sequence_length}")
    print(f"[test_lstm] diagnostics={diagnostics}")
    print(
        f"[test_lstm] weighting_scheme={args.weighting_scheme} "
        f"weight_blend_alpha={args.weight_blend_alpha:.6f} "
        f"max_single_weight={args.max_single_weight:.6f} "
        f"sort_strategy={args.sort_strategy} "
        f"rerank_signal_column={effective_rerank_column or 'none'} "
        f"rerank_signal_weight={effective_rerank_weight:.6f} "
        f"regime_rerank_enabled={int(args.regime_rerank_enabled)} "
        f"regime_rerank_active={int(regime_diagnostics['regime_rerank_active'])} "
        f"secondary_candidate_size={args.secondary_candidate_size or 0} "
        f"secondary_screen_mode={args.secondary_screen_mode} "
        f"secondary_screen_weight={args.secondary_screen_weight:.6f} "
        f"local_tiebreak={args.local_tiebreak_start_rank}~{args.local_tiebreak_end_rank} "
        f"desired_turnover={desired_turnover:.6f} "
        f"executed_turnover_cap={args.max_turnover:.6f} "
        f"execution_strength={execution_strength:.6f}"
    )
    print(f"[test_lstm] wrote result to {output_path}")
    if args.debug_candidates_path:
        print(f"[test_lstm] wrote debug candidates to {args.debug_candidates_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
