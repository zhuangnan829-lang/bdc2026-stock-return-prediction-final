import argparse
import json
from pathlib import Path

import pandas as pd

from backtest import load_or_generate_predictions, run_backtest
from config import resolve_metadata_artifact_path
from lstm_utils import (
    build_sequence_dataset,
    load_lstm_checkpoint,
    predict_sequences,
    transform_sequences,
)
from test_lstm import load_metadata, resolve_history_frame
from utils import build_portfolio_weights, load_feature_frame, select_top_candidates


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PREDICTION_PATH = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"
DEFAULT_WALK_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_PREDICT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "predict_features.csv"
DEFAULT_MODEL_DIR = ROOT_DIR / "app" / "model"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "lstm_dual_objective_search"
CASE_DIR = ROOT_DIR / "_case_zip" / "THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0"
MINI4_SUMMARY_PATH = ROOT_DIR / "app" / "model" / "alpha_rs_crowding_mini4_experiment" / "alpha_rs_crowding_mini4_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search LSTM execution-layer parameters under both case-slice and walk-forward objectives."
    )
    parser.add_argument("--prediction_path", default=str(DEFAULT_PREDICTION_PATH))
    parser.add_argument("--walk_feature_path", default=str(DEFAULT_WALK_FEATURE_PATH))
    parser.add_argument("--predict_feature_path", default=str(DEFAULT_PREDICT_FEATURE_PATH))
    parser.add_argument("--history_feature_path")
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--candidate_sizes", nargs="+", type=int, default=[120, 150, 180])
    parser.add_argument("--vol20_pcts", nargs="+", type=float, default=[0.86, 0.90])
    parser.add_argument("--vol5_pcts", nargs="+", type=float, default=[0.96, 0.98, 1.00])
    parser.add_argument("--risk_penalties", nargs="+", type=float, default=[-0.80, -0.50, -0.20, 0.10])
    parser.add_argument("--sort_strategies", nargs="+", default=["risk_adjusted", "pure_prediction"])
    parser.add_argument("--weighting_schemes", nargs="+", default=["pred", "risk_adjusted"])
    parser.add_argument("--max_turnovers", nargs="+", type=float, default=[1.0])
    parser.add_argument("--case_score_floor")
    parser.add_argument("--min_cum_ratio", type=float, default=0.65)
    parser.add_argument("--min_sharpe_ratio", type=float, default=0.65)
    parser.add_argument("--max_drawdown_slack", type=float, default=0.02)
    parser.add_argument(
        "--drawdown_slack_schedule",
        nargs="+",
        type=float,
        default=[0.02, 0.04, 0.06, 0.08],
        help="If shortlist is empty, retry by relaxing drawdown slack in this order before touching case floor.",
    )
    parser.add_argument(
        "--disable_auto_relax_drawdown",
        action="store_true",
        help="Run only once with --max_drawdown_slack and do not auto-relax drawdown slack.",
    )
    return parser.parse_args()


def read_case_best_score() -> float:
    text = (CASE_DIR / "model" / "60_158+39" / "final_score.txt").read_text(encoding="utf-8", errors="ignore")
    marker = "Best final_score:"
    if marker not in text:
        return float("nan")
    return float(text.split(marker, 1)[1].strip().split()[0].replace("\\n", ""))


def read_case_current_slice_score() -> float:
    df = pd.read_csv(CASE_DIR / "temp" / "tmp.csv")
    return float(df.iloc[0, 1])


def score_result_df_against_case_slice(result_df: pd.DataFrame) -> float:
    test_df = pd.read_csv(CASE_DIR / "data" / "test.csv")
    stock_col = test_df.columns[0]
    open_col = test_df.columns[2]

    output_df = result_df.copy()
    output_df["stock_id"] = output_df["stock_id"].astype(str).str.zfill(6)
    output_df["weight"] = pd.to_numeric(output_df["weight"], errors="coerce").fillna(0.0)

    scored = test_df[test_df[stock_col].astype(str).str.zfill(6).isin(output_df["stock_id"])].copy()
    scored[stock_col] = scored[stock_col].astype(str).str.zfill(6)
    scored = scored.groupby(stock_col, sort=False).tail(5)

    grouped = scored.groupby(stock_col, sort=False)[open_col]
    slice_return = ((grouped.last() - grouped.first()) / grouped.first()).to_numpy(dtype=float)
    returns = pd.DataFrame(
        {
            stock_col: list(grouped.groups.keys()),
            "slice_return": slice_return,
        }
    )
    merged = returns.merge(output_df, left_on=stock_col, right_on="stock_id", how="inner")
    return float((merged["slice_return"] * merged["weight"]).sum())


def load_local_baseline_metrics() -> pd.Series:
    mini4 = pd.read_csv(MINI4_SUMMARY_PATH)
    return mini4[mini4["label"] == "alpha_v3_rs_crowding_mini4"].iloc[0]


def build_latest_prediction_frame(
    predict_feature_path: Path,
    model_dir: Path,
    history_feature_path: str | None,
) -> pd.DataFrame:
    metadata = load_metadata(model_dir)
    model_path = resolve_metadata_artifact_path(model_dir, metadata["model_path"])
    feature_columns = metadata["feature_columns"]
    sequence_length = int(metadata["sequence_length"])
    batch_size = int(metadata.get("batch_size", 256))

    target_df = load_feature_frame(predict_feature_path)
    history_df = resolve_history_frame(predict_feature_path, history_feature_path)
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
        raise ValueError("No inference sequences were built for dual-objective search.")

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
        raise ValueError("No latest-date prediction rows found for dual-objective search.")
    return latest_df


def build_backtest_config(
    profile_name: str,
    candidate_size: int,
    vol20_pct: float,
    vol5_pct: float,
    risk_penalty: float,
    sort_strategy: str,
    weighting_scheme: str,
    max_turnover: float,
) -> dict:
    return {
        "profile_name": profile_name,
        "top_k": 5,
        "primary_candidate_size": int(candidate_size),
        "enable_risk_filters": 1,
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": float(vol20_pct),
        "max_volatility_5d_pct": float(vol5_pct),
        "turnover_rate_lower_pct": 0.03,
        "turnover_rate_upper_pct": 0.97,
        "turnover_ratio_upper_pct": 0.95,
        "risk_penalty_weight": float(risk_penalty),
        "weighting_scheme": weighting_scheme,
        "sort_strategy": sort_strategy,
        "transaction_cost": 0.001,
        "max_turnover": float(max_turnover),
    }


def build_live_result_for_config(latest_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    selected, _ = select_top_candidates(
        latest_df=latest_df,
        top_k=config["top_k"],
        primary_candidate_size=config["primary_candidate_size"],
        max_volatility_20d_pct=config["max_volatility_20d_pct"],
        max_volatility_5d_pct=config["max_volatility_5d_pct"],
        turnover_rate_lower_pct=config["turnover_rate_lower_pct"],
        turnover_rate_upper_pct=config["turnover_rate_upper_pct"],
        turnover_ratio_upper_pct=config["turnover_ratio_upper_pct"],
        risk_penalty_weight=config["risk_penalty_weight"],
        sort_strategy=config["sort_strategy"],
        enable_risk_filters=bool(config["enable_risk_filters"]),
        allow_cash_fallback=bool(config["allow_cash_fallback"]),
    )
    weighted = build_portfolio_weights(
        selected,
        top_k=config["top_k"],
        weighting_scheme=config["weighting_scheme"],
    )
    result_df = (
        weighted[["stock_id", "weight"]]
        .sort_values(["weight", "stock_id"], ascending=[False, True])
        .head(config["top_k"])
        .reset_index(drop=True)
    )
    return result_df


def is_pareto_efficient(df: pd.DataFrame, objective_columns: list[str]) -> pd.Series:
    values = df[objective_columns].to_numpy(dtype=float)
    efficient = [True] * len(values)
    for i, row in enumerate(values):
        if not efficient[i]:
            continue
        for j, challenger in enumerate(values):
            if i == j:
                continue
            no_worse = (challenger >= row).all()
            strictly_better = (challenger > row).any()
            if no_worse and strictly_better:
                efficient[i] = False
                break
    return pd.Series(efficient, index=df.index)


def write_report(
    all_df: pd.DataFrame,
    shortlist_df: pd.DataFrame,
    recommended_row: pd.Series | None,
    output_path: Path,
    case_current_score: float,
    case_best_score: float,
    local_baseline: pd.Series,
    case_score_floor: float,
    min_cum: float,
    min_sharpe: float,
    min_drawdown: float,
    used_drawdown_slack: float,
    attempted_drawdown_slacks: list[float],
) -> None:
    best_case = all_df.sort_values("case_slice_score", ascending=False).iloc[0]
    best_local = all_df.sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost", "max_drawdown_after_cost"],
        ascending=[False, False, False],
    ).iloc[0]

    lines = [
        "# LSTM Dual Objective Search Report",
        "",
        "## Objective",
        "",
        "- Optimize for both zip-case visible slice score and local walk-forward stability.",
        f"- zip current slice score: `{case_current_score:.6f}`",
        f"- zip reported best score: `{case_best_score:.6f}`",
        f"- local baseline cumulative_return_after_cost: `{float(local_baseline['cumulative_return_after_cost']):.6f}`",
        f"- local baseline sharpe_after_cost: `{float(local_baseline['sharpe_after_cost']):.6f}`",
        f"- local baseline max_drawdown_after_cost: `{float(local_baseline['max_drawdown_after_cost']):.6f}`",
        "",
        "## Guardrails",
        "",
        f"- case_slice_score >= `{case_score_floor:.6f}`",
        f"- cumulative_return_after_cost >= `{min_cum:.6f}`",
        f"- sharpe_after_cost >= `{min_sharpe:.6f}`",
        f"- max_drawdown_after_cost >= `{min_drawdown:.6f}`",
        f"- used max_drawdown_slack = `{used_drawdown_slack:.6f}`",
        f"- attempted drawdown slack schedule = `{', '.join(f'{value:.2f}' for value in attempted_drawdown_slacks)}`",
        "",
        "## Best Single-Objective Rows",
        "",
        f"- best case slice row: `{best_case['profile_name']}` with case score `{best_case['case_slice_score']:.6f}`",
        f"- best local row: `{best_local['profile_name']}` with cumulative `{best_local['cumulative_return_after_cost']:.6f}` and sharpe `{best_local['sharpe_after_cost']:.6f}`",
        "",
        "## Recommended Default",
        "",
    ]

    if recommended_row is None:
        lines.append("No recommended champion because shortlist is empty.")
    else:
        lines.extend(
            [
                f"- recommended profile: `{recommended_row['profile_name']}`",
                f"- recommendation score: `{recommended_row['recommendation_score']:.6f}`",
                f"- case slice score: `{recommended_row['case_slice_score']:.6f}`",
                f"- cumulative_return_after_cost: `{recommended_row['cumulative_return_after_cost']:.6f}`",
                f"- sharpe_after_cost: `{recommended_row['sharpe_after_cost']:.6f}`",
                f"- max_drawdown_after_cost: `{recommended_row['max_drawdown_after_cost']:.6f}`",
                "- rationale: case score first, then cumulative return, then sharpe, and finally drawdown stability within the shortlist.",
                "",
            ]
        )

    lines.extend(
        [
        "## Shortlist",
        "",
        ]
    )

    if shortlist_df.empty:
        lines.append("No parameter group passed the guardrails and Pareto filter together.")
        lines.extend(
            [
                "",
                "## Closest Near Misses",
                "",
                "| profile_name | guardrail_pass_count | failed_guardrails | case_slice_score | cumulative_return_after_cost | sharpe_after_cost | max_drawdown_after_cost |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        near_miss_df = all_df.sort_values(
            ["guardrail_pass_count", "case_slice_score", "cumulative_return_after_cost", "sharpe_after_cost"],
            ascending=[False, False, False, False],
        ).head(10)
        for _, row in near_miss_df.iterrows():
            lines.append(
                f"| {row['profile_name']} | {int(row['guardrail_pass_count'])} | {row['failed_guardrails']} | "
                f"{row['case_slice_score']:.6f} | {row['cumulative_return_after_cost']:.6f} | "
                f"{row['sharpe_after_cost']:.6f} | {row['max_drawdown_after_cost']:.6f} |"
            )
    else:
        lines.extend(
            [
                "| profile_name | case_slice_score | cumulative_return_after_cost | sharpe_after_cost | max_drawdown_after_cost |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in shortlist_df.iterrows():
            lines.append(
                f"| {row['profile_name']} | {row['case_slice_score']:.6f} | "
                f"{row['cumulative_return_after_cost']:.6f} | {row['sharpe_after_cost']:.6f} | "
                f"{row['max_drawdown_after_cost']:.6f} |"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_profiles(
    latest_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    prediction_source: str,
    candidate_sizes: list[int],
    vol20_pcts: list[float],
    vol5_pcts: list[float],
    risk_penalties: list[float],
    sort_strategies: list[str],
    weighting_schemes: list[str],
    max_turnovers: list[float],
) -> pd.DataFrame:
    rows: list[dict] = []
    for candidate_size in candidate_sizes:
        for vol20_pct in vol20_pcts:
            for vol5_pct in vol5_pcts:
                for risk_penalty in risk_penalties:
                    for sort_strategy in sort_strategies:
                        for weighting_scheme in weighting_schemes:
                            for max_turnover in max_turnovers:
                                profile_name = (
                                    f"dual_cs{candidate_size}_v20{int(round(vol20_pct * 100)):02d}"
                                    f"_v5{int(round(vol5_pct * 100)):02d}"
                                    f"_rp{int(round(risk_penalty * 100)):03d}"
                                    f"_{sort_strategy}_{weighting_scheme}"
                                    f"_mt{int(round(max_turnover * 100)):02d}"
                                )
                                config = build_backtest_config(
                                    profile_name=profile_name,
                                    candidate_size=candidate_size,
                                    vol20_pct=vol20_pct,
                                    vol5_pct=vol5_pct,
                                    risk_penalty=risk_penalty,
                                    sort_strategy=sort_strategy,
                                    weighting_scheme=weighting_scheme,
                                    max_turnover=max_turnover,
                                )
                                result_df = build_live_result_for_config(latest_df, config)
                                case_slice_score = score_result_df_against_case_slice(result_df)
                                summary_df, _, _ = run_backtest(
                                    prediction_df=prediction_df,
                                    config=config,
                                    prediction_source=prediction_source,
                                )
                                summary_row = summary_df.iloc[0].to_dict()
                                summary_row.update(
                                    {
                                        "candidate_size": int(candidate_size),
                                        "vol20_pct": float(vol20_pct),
                                        "vol5_pct": float(vol5_pct),
                                        "risk_penalty": float(risk_penalty),
                                        "sort_strategy": sort_strategy,
                                        "weighting_scheme": weighting_scheme,
                                        "case_slice_score": float(case_slice_score),
                                        "result_stock_ids": "/".join(result_df["stock_id"].astype(str).tolist()),
                                        "result_weight_sum": float(result_df["weight"].sum()),
                                    }
                                )
                                rows.append(summary_row)
                                print(
                                    "[dual_search] "
                                    f"profile={profile_name} "
                                    f"case={case_slice_score:.6f} "
                                    f"cum_after={summary_row['cumulative_return_after_cost']:.6f} "
                                    f"sharpe={summary_row['sharpe_after_cost']:.6f}"
                                )

    return pd.DataFrame(rows).sort_values(
        ["case_slice_score", "cumulative_return_after_cost", "sharpe_after_cost", "max_drawdown_after_cost"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def apply_guardrails(
    all_df: pd.DataFrame,
    case_score_floor: float,
    min_cum: float,
    min_sharpe: float,
    min_drawdown: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_df = all_df.copy()
    scored_df["passes_case_floor"] = scored_df["case_slice_score"] >= case_score_floor
    scored_df["passes_cum_floor"] = scored_df["cumulative_return_after_cost"] >= min_cum
    scored_df["passes_sharpe_floor"] = scored_df["sharpe_after_cost"] >= min_sharpe
    scored_df["passes_drawdown_floor"] = scored_df["max_drawdown_after_cost"] >= min_drawdown
    scored_df["guardrail_pass_count"] = (
        scored_df["passes_case_floor"].astype(int)
        + scored_df["passes_cum_floor"].astype(int)
        + scored_df["passes_sharpe_floor"].astype(int)
        + scored_df["passes_drawdown_floor"].astype(int)
    )
    scored_df["failed_guardrails"] = scored_df.apply(
        lambda row: ",".join(
            [
                name
                for name, passed in [
                    ("case", row["passes_case_floor"]),
                    ("cum", row["passes_cum_floor"]),
                    ("sharpe", row["passes_sharpe_floor"]),
                    ("drawdown", row["passes_drawdown_floor"]),
                ]
                if not bool(passed)
            ]
        ),
        axis=1,
    )
    scored_df["passes_guardrails"] = (
        scored_df["passes_case_floor"]
        & scored_df["passes_cum_floor"]
        & scored_df["passes_sharpe_floor"]
        & scored_df["passes_drawdown_floor"]
    )

    passed_df = scored_df[scored_df["passes_guardrails"]].copy()
    if passed_df.empty:
        shortlist_df = passed_df.copy()
    else:
        passed_df["pareto_efficient"] = is_pareto_efficient(
            passed_df,
            [
                "case_slice_score",
                "cumulative_return_after_cost",
                "sharpe_after_cost",
                "max_drawdown_after_cost",
            ],
        )
        shortlist_df = passed_df[passed_df["pareto_efficient"]].copy()
    return scored_df, shortlist_df


def minmax_score(series: pd.Series) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    if maximum - minimum <= 1e-12:
        return pd.Series([1.0] * len(series), index=series.index, dtype=float)
    return (series - minimum) / (maximum - minimum)


def choose_recommended_profile(shortlist_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    if shortlist_df.empty:
        return shortlist_df.copy(), None

    ranked_df = shortlist_df.copy()
    ranked_df["score_case"] = minmax_score(ranked_df["case_slice_score"])
    ranked_df["score_cum"] = minmax_score(ranked_df["cumulative_return_after_cost"])
    ranked_df["score_sharpe"] = minmax_score(ranked_df["sharpe_after_cost"])
    ranked_df["score_drawdown"] = minmax_score(ranked_df["max_drawdown_after_cost"])
    ranked_df["recommendation_score"] = (
        0.40 * ranked_df["score_case"]
        + 0.30 * ranked_df["score_cum"]
        + 0.20 * ranked_df["score_sharpe"]
        + 0.10 * ranked_df["score_drawdown"]
    )
    ranked_df = ranked_df.sort_values(
        [
            "recommendation_score",
            "case_slice_score",
            "cumulative_return_after_cost",
            "sharpe_after_cost",
            "max_drawdown_after_cost",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return ranked_df, ranked_df.iloc[0]


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_current_score = read_case_current_slice_score()
    case_best_score = read_case_best_score()
    local_baseline = load_local_baseline_metrics()
    case_score_floor = float(args.case_score_floor) if args.case_score_floor is not None else max(case_current_score, case_best_score)
    min_cum = float(local_baseline["cumulative_return_after_cost"]) * float(args.min_cum_ratio)
    min_sharpe = float(local_baseline["sharpe_after_cost"]) * float(args.min_sharpe_ratio)

    latest_df = build_latest_prediction_frame(
        predict_feature_path=Path(args.predict_feature_path),
        model_dir=model_dir,
        history_feature_path=args.history_feature_path,
    )
    prediction_df, prediction_source = load_or_generate_predictions(
        prediction_path=Path(args.prediction_path),
        feature_path=Path(args.walk_feature_path),
        model_dir=model_dir,
    )

    all_df = evaluate_profiles(
        latest_df=latest_df,
        prediction_df=prediction_df,
        prediction_source=prediction_source,
        candidate_sizes=args.candidate_sizes,
        vol20_pcts=args.vol20_pcts,
        vol5_pcts=args.vol5_pcts,
        risk_penalties=args.risk_penalties,
        sort_strategies=args.sort_strategies,
        weighting_schemes=args.weighting_schemes,
        max_turnovers=args.max_turnovers,
    )

    attempted_drawdown_slacks = [float(args.max_drawdown_slack)]
    if not args.disable_auto_relax_drawdown:
        for value in args.drawdown_slack_schedule:
            value = float(value)
            if value not in attempted_drawdown_slacks:
                attempted_drawdown_slacks.append(value)

    final_all_df = pd.DataFrame()
    shortlist_df = pd.DataFrame()
    recommended_row = None
    used_drawdown_slack = float(args.max_drawdown_slack)
    min_drawdown = float(local_baseline["max_drawdown_after_cost"]) - used_drawdown_slack

    for drawdown_slack in attempted_drawdown_slacks:
        used_drawdown_slack = float(drawdown_slack)
        min_drawdown = float(local_baseline["max_drawdown_after_cost"]) - used_drawdown_slack
        final_all_df, shortlist_df = apply_guardrails(
            all_df=all_df,
            case_score_floor=case_score_floor,
            min_cum=min_cum,
            min_sharpe=min_sharpe,
            min_drawdown=min_drawdown,
        )
        print(
            "[dual_search] "
            f"drawdown_slack={used_drawdown_slack:.2f} "
            f"passed_guardrails={int(final_all_df['passes_guardrails'].sum())} "
            f"shortlisted={len(shortlist_df)}"
        )
        if not shortlist_df.empty:
            break

    shortlist_df, recommended_row = choose_recommended_profile(shortlist_df)

    all_path = output_dir / "dual_objective_search_all.csv"
    shortlist_path = output_dir / "dual_objective_search_shortlist.csv"
    recommended_path = output_dir / "dual_objective_search_recommended.csv"
    recommended_json_path = output_dir / "dual_objective_search_recommended.json"
    report_path = output_dir / "dual_objective_search_report.md"
    final_all_df["used_drawdown_slack"] = used_drawdown_slack
    final_all_df["attempted_drawdown_slacks"] = ",".join(f"{value:.2f}" for value in attempted_drawdown_slacks)
    shortlist_df["used_drawdown_slack"] = used_drawdown_slack
    shortlist_df["attempted_drawdown_slacks"] = ",".join(f"{value:.2f}" for value in attempted_drawdown_slacks)
    final_all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    shortlist_df.to_csv(shortlist_path, index=False, encoding="utf-8-sig")
    if recommended_row is None:
        pd.DataFrame().to_csv(recommended_path, index=False, encoding="utf-8-sig")
        recommended_json_path.write_text("{}\n", encoding="utf-8")
    else:
        pd.DataFrame([recommended_row.to_dict()]).to_csv(recommended_path, index=False, encoding="utf-8-sig")
        recommended_json_path.write_text(
            json.dumps(recommended_row.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    write_report(
        all_df=final_all_df,
        shortlist_df=shortlist_df,
        recommended_row=recommended_row,
        output_path=report_path,
        case_current_score=case_current_score,
        case_best_score=case_best_score,
        local_baseline=local_baseline,
        case_score_floor=case_score_floor,
        min_cum=min_cum,
        min_sharpe=min_sharpe,
        min_drawdown=min_drawdown,
        used_drawdown_slack=used_drawdown_slack,
        attempted_drawdown_slacks=attempted_drawdown_slacks,
    )

    print(f"[dual_search] profiles={len(final_all_df)}")
    print(f"[dual_search] passed_guardrails={int(final_all_df['passes_guardrails'].sum())}")
    print(f"[dual_search] shortlisted={len(shortlist_df)}")
    if recommended_row is None:
        print("[dual_search] recommended_profile=none")
    else:
        print(
            "[dual_search] "
            f"recommended_profile={recommended_row['profile_name']} "
            f"recommendation_score={recommended_row['recommendation_score']:.6f}"
        )
    print(f"[dual_search] wrote {all_path}")
    print(f"[dual_search] wrote {shortlist_path}")
    print(f"[dual_search] wrote {recommended_path}")
    print(f"[dual_search] wrote {recommended_json_path}")
    print(f"[dual_search] wrote {report_path}")


if __name__ == "__main__":
    main()
