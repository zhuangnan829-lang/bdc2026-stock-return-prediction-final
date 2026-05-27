import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from config import BEST_PROFILE_NAME, build_default_inference_args, resolve_metadata_artifact_path
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
    parser = argparse.ArgumentParser(description="Inference entrypoint.")
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_path", required=True)
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


def main() -> None:
    args = parse_args()
    feature_path = Path(args.feature_path)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output_path)

    metadata = load_metadata(model_dir)
    model_path = resolve_metadata_artifact_path(model_dir, metadata["model_path"])
    feature_columns = metadata["feature_columns"]

    predict_df = load_feature_frame(feature_path)
    missing_columns = [column for column in feature_columns if column not in predict_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns for inference: {missing_columns}")

    latest_date = predict_df["date"].max()
    latest_df = predict_df[predict_df["date"] == latest_date].copy()
    if latest_df.empty:
        raise ValueError("No prediction rows found for the latest date")

    model = joblib.load(model_path)
    latest_df["pred_return"] = model.predict(latest_df[feature_columns])
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
        rerank_signal_column=args.rerank_signal_column,
        rerank_signal_weight=args.rerank_signal_weight,
        enable_risk_filters=True,
        allow_cash_fallback=False,
    )

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
        result = pd.DataFrame(rows).sort_values(["weight", "stock_id"], ascending=[False, True]).head(args.top_k).reset_index(drop=True)
    else:
        result = pd.DataFrame(columns=["stock_id", "weight"])
    validate_result(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")

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

    print(f"[test] feature_path={feature_path}")
    print(f"[test] model_path={model_path}")
    print(f"[test] latest_date={latest_date.date()} candidate_count={len(latest_df)}")
    print(f"[test] default_profile={BEST_PROFILE_NAME}")
    print(f"[test] diagnostics={diagnostics}")
    print(
        f"[test] weighting_scheme={args.weighting_scheme} "
        f"weight_blend_alpha={args.weight_blend_alpha:.6f} "
        f"max_single_weight={args.max_single_weight:.6f} "
        f"sort_strategy={args.sort_strategy} "
        f"rerank_signal_column={args.rerank_signal_column or 'none'} "
        f"rerank_signal_weight={args.rerank_signal_weight:.6f} "
        f"desired_turnover={desired_turnover:.6f} "
        f"executed_turnover_cap={args.max_turnover:.6f} "
        f"execution_strength={execution_strength:.6f}"
    )
    print(f"[test] wrote result to {output_path}")
    if args.debug_candidates_path:
        print(f"[test] wrote debug candidates to {args.debug_candidates_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
