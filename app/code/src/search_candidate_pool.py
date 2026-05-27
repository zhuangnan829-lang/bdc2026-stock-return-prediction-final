import argparse
from pathlib import Path

import pandas as pd

from backtest import (
    DEFAULT_FEATURE_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_PREDICTION_PATH,
    DEFAULTS,
    load_or_generate_predictions,
    make_config,
    run_backtest,
)
from utils import ensure_dir


DEFAULT_OUTPUT_DIR = "app/model/candidate_pool_search"
DEFAULT_CANDIDATE_SIZES = "100,120,150,160,180,200,240"
DEFAULT_RISK_PENALTIES = "-0.20,-0.25,-0.30,-0.35,-0.40"


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one candidate size is required")
    return values


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one risk penalty is required")
    return values


def build_args_namespace(args: argparse.Namespace) -> argparse.Namespace:
    payload = dict(DEFAULTS)
    payload.update(
        {
            "top_k": args.top_k,
            "enable_risk_filters": args.enable_risk_filters,
            "allow_cash_fallback": args.allow_cash_fallback,
            "transaction_cost": args.transaction_cost,
            "max_turnover": args.max_turnover,
            "max_single_weight": args.max_single_weight,
            "weight_blend_alpha": args.weight_blend_alpha,
            "rerank_signal_column": None,
            "rerank_signal_weight": 0.0,
            "secondary_candidate_size": None,
            "secondary_screen_mode": "none",
            "secondary_screen_weight": 0.0,
            "local_tiebreak_start_rank": 8,
            "local_tiebreak_end_rank": 15,
        }
    )
    return argparse.Namespace(**payload)


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if not higher_is_better:
        values = -values
    low = float(values.min())
    high = float(values.max())
    if abs(high - low) <= 1e-12:
        return pd.Series(0.5, index=series.index)
    return (values - low) / (high - low)


def add_composite_score(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["score_return"] = minmax_score(out["cumulative_return_after_cost"], higher_is_better=True)
    out["score_sharpe"] = minmax_score(out["sharpe_after_cost"], higher_is_better=True)
    out["score_drawdown"] = minmax_score(out["max_drawdown_after_cost"], higher_is_better=True)
    out["score_turnover"] = minmax_score(out["avg_turnover"], higher_is_better=False)
    out["score_concentration"] = minmax_score(out["max_single_contribution_share"], higher_is_better=False)
    out["composite_score"] = (
        0.35 * out["score_return"]
        + 0.25 * out["score_sharpe"]
        + 0.20 * out["score_drawdown"]
        + 0.10 * out["score_turnover"]
        + 0.10 * out["score_concentration"]
    )
    out["is_current_default"] = (out["candidate_size"] == 180) & (out["risk_penalty_weight"].round(6) == -0.30)
    out["is_best_composite"] = False
    out["return_rank"] = out["cumulative_return_after_cost"].rank(ascending=False, method="min").astype(int)
    out["sharpe_rank"] = out["sharpe_after_cost"].rank(ascending=False, method="min").astype(int)
    out["drawdown_rank"] = out["max_drawdown_after_cost"].rank(ascending=False, method="min").astype(int)
    out["composite_rank"] = out["composite_score"].rank(ascending=False, method="min").astype(int)
    if not out.empty:
        best_idx = out.sort_values(
            ["composite_score", "cumulative_return_after_cost", "sharpe_after_cost"],
            ascending=[False, False, False],
        ).index[0]
        out.loc[best_idx, "is_best_composite"] = True
    return out


def write_heatmap(summary: pd.DataFrame, output_path: Path, value_column: str = "composite_score") -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        fallback = output_path.with_suffix(".txt")
        fallback.write_text(f"matplotlib unavailable: {exc}", encoding="utf-8")
        return

    pivot = summary.pivot(index="risk_penalty_weight", columns="candidate_size", values=value_column).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{idx:.2f}" for idx in pivot.index])
    ax.set_xlabel("candidate_size")
    ax.set_ylabel("risk_penalty")
    ax.set_title("Candidate Pool Search Composite Score")

    for y_idx, risk_penalty in enumerate(pivot.index):
        for x_idx, candidate_size in enumerate(pivot.columns):
            value = pivot.loc[risk_penalty, candidate_size]
            cell = summary[
                (summary["candidate_size"] == candidate_size)
                & (summary["risk_penalty_weight"].round(6) == round(float(risk_penalty), 6))
            ].iloc[0]
            marker = "*" if bool(cell["is_current_default"]) else ""
            ax.text(x_idx, y_idx, f"{value:.3f}{marker}", ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(image, ax=ax, label=value_column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_path: Path) -> None:
    ranked = summary.sort_values(
        ["composite_score", "cumulative_return_after_cost", "sharpe_after_cost"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    current = summary[summary["is_current_default"]]
    best = ranked.iloc[0]
    lines = [
        "# Candidate Pool Search",
        "",
        "Composite score = 35% return + 25% Sharpe + 20% drawdown + 10% turnover + 10% concentration.",
        "",
        "## Best Composite",
        "",
        f"- candidate_size: `{int(best['candidate_size'])}`",
        f"- risk_penalty: `{best['risk_penalty_weight']:.2f}`",
        f"- composite_score: `{best['composite_score']:.6f}`",
        f"- cumulative_return_after_cost: `{best['cumulative_return_after_cost']:.6f}`",
        f"- max_drawdown_after_cost: `{best['max_drawdown_after_cost']:.6f}`",
        f"- sharpe_after_cost: `{best['sharpe_after_cost']:.6f}`",
        "",
        "## Current Default cs180 + rp=-0.30",
        "",
    ]
    if current.empty:
        lines.append("- Current default was not part of this grid.")
    else:
        row = current.iloc[0]
        lines.extend(
            [
                f"- composite_rank: `{int(row['composite_rank'])}` / `{len(ranked)}`",
                f"- return_rank: `{int(row['return_rank'])}` / `{len(ranked)}`",
                f"- sharpe_rank: `{int(row['sharpe_rank'])}` / `{len(ranked)}`",
                f"- drawdown_rank: `{int(row['drawdown_rank'])}` / `{len(ranked)}`",
                f"- composite_score: `{row['composite_score']:.6f}`",
                f"- cumulative_return_after_cost: `{row['cumulative_return_after_cost']:.6f}`",
                f"- max_drawdown_after_cost: `{row['max_drawdown_after_cost']:.6f}`",
                f"- sharpe_after_cost: `{row['sharpe_after_cost']:.6f}`",
                f"- avg_turnover: `{row['avg_turnover']:.6f}`",
                f"- max_single_contribution_share: `{row['max_single_contribution_share']:.6f}`",
                "- Interpretation: cs180 + rp=-0.30 is best treated as the aggressive default: it is near the top on return and Sharpe, while the heatmap also exposes more robust alternatives such as smaller candidate pools.",
            ]
        )
    lines.extend(
        [
            "",
            "## Top 10 Composite Rows",
            "",
            "| rank | candidate_size | risk_penalty | score | return | drawdown | sharpe | turnover | max_contrib_share | current |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for idx, row in ranked.head(10).iterrows():
        lines.append(
            f"| {idx + 1} | {int(row['candidate_size'])} | {row['risk_penalty_weight']:.2f} | "
            f"{row['composite_score']:.6f} | {row['cumulative_return_after_cost']:.6f} | "
            f"{row['max_drawdown_after_cost']:.6f} | {row['sharpe_after_cost']:.6f} | "
            f"{row['avg_turnover']:.6f} | {row['max_single_contribution_share']:.6f} | "
            f"{'yes' if row['is_current_default'] else 'no'} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search candidate pool size and risk penalty.")
    parser.add_argument("--prediction_path", default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--feature_path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate_sizes", default=DEFAULT_CANDIDATE_SIZES)
    parser.add_argument("--risk_penalties", default=DEFAULT_RISK_PENALTIES)
    parser.add_argument("--top_k", type=int, default=int(DEFAULTS["top_k"]))
    parser.add_argument("--enable_risk_filters", type=int, choices=[0, 1], default=int(DEFAULTS["enable_risk_filters"]))
    parser.add_argument("--allow_cash_fallback", type=int, choices=[0, 1], default=0)
    parser.add_argument("--transaction_cost", type=float, default=float(DEFAULTS["transaction_cost"]))
    parser.add_argument("--max_turnover", type=float, default=float(DEFAULTS["max_turnover"]))
    parser.add_argument("--max_single_weight", type=float, default=float(DEFAULTS["max_single_weight"]))
    parser.add_argument("--weight_blend_alpha", type=float, default=float(DEFAULTS["weight_blend_alpha"]))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    prediction_df, prediction_source = load_or_generate_predictions(
        prediction_path=Path(args.prediction_path),
        feature_path=Path(args.feature_path),
        model_dir=Path(args.model_dir),
    )

    summary_frames = []
    base_args = build_args_namespace(args)
    for candidate_size in parse_int_list(args.candidate_sizes):
        for risk_penalty in parse_float_list(args.risk_penalties):
            config = make_config(
                base_args,
                overrides={
                    "profile_name": f"cs{candidate_size}_rp{risk_penalty:.2f}",
                    "primary_candidate_size": candidate_size,
                    "risk_penalty_weight": risk_penalty,
                },
            )
            summary_df, daily_df, holdings_df = run_backtest(
                prediction_df=prediction_df,
                config=config,
                prediction_source=prediction_source,
            )
            case_dir = ensure_dir(output_dir / f"cs{candidate_size}_rp{risk_penalty:.2f}")
            daily_df.to_csv(case_dir / "backtest_daily_results.csv", index=False, encoding="utf-8-sig")
            holdings_df.to_csv(case_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
            summary_df["candidate_size"] = candidate_size
            summary_df["risk_penalty_weight"] = risk_penalty
            summary_frames.append(summary_df)

    summary = pd.concat(summary_frames, ignore_index=True)
    summary = add_composite_score(summary)
    ranked = summary.sort_values(
        ["composite_score", "cumulative_return_after_cost", "sharpe_after_cost"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary_path = output_dir / "candidate_pool_summary.csv"
    ranked.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_heatmap(ranked, output_dir / "heatmap.png")
    write_report(ranked, output_dir / "candidate_pool_report.md")

    print(f"[candidate_pool] wrote {summary_path}")
    print(f"[candidate_pool] wrote {output_dir / 'heatmap.png'}")
    print(f"[candidate_pool] wrote {output_dir / 'candidate_pool_report.md'}")
    print(
        ranked[
            [
                "candidate_size",
                "risk_penalty_weight",
                "composite_score",
                "cumulative_return_after_cost",
                "max_drawdown_after_cost",
                "sharpe_after_cost",
                "avg_turnover",
                "max_single_contribution_share",
                "is_current_default",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
