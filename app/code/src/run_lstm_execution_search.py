import argparse
from pathlib import Path

import pandas as pd

from backtest import load_or_generate_predictions, run_backtest
from config import BEST_CONFIG


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PREDICTION_PATH = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_MODEL_DIR = ROOT_DIR / "app" / "model"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "lstm_execution_search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local execution-layer search for current LSTM default model."
    )
    parser.add_argument("--prediction_path", default=str(DEFAULT_PREDICTION_PATH))
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--candidate_sizes", nargs="+", type=int, default=[40, 80, 120, 150, 200])
    parser.add_argument("--vol20_pcts", nargs="+", type=float, default=[0.86, 0.90, 0.95, 1.00])
    parser.add_argument("--vol5_pcts", nargs="+", type=float, default=[0.96, 0.98, 1.00])
    parser.add_argument("--risk_penalties", nargs="+", type=float, default=[-0.80, -0.50, -0.20, 0.00, 0.10, 0.30])
    parser.add_argument("--sort_strategies", nargs="+", default=["risk_adjusted", "pure_prediction"])
    parser.add_argument("--weighting_schemes", nargs="+", default=["equal", "pred", "risk_adjusted"])
    parser.add_argument("--max_turnovers", nargs="+", type=float, default=[1.0])
    return parser.parse_args()


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
        "top_k": int(BEST_CONFIG["selection"]["top_k"]),
        "primary_candidate_size": int(candidate_size),
        "enable_risk_filters": int(bool(BEST_CONFIG["selection"]["enable_risk_filters"])),
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": float(vol20_pct),
        "max_volatility_5d_pct": float(vol5_pct),
        "turnover_rate_lower_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(risk_penalty),
        "weighting_scheme": weighting_scheme,
        "sort_strategy": sort_strategy,
        "transaction_cost": float(BEST_CONFIG["execution"]["transaction_cost"]),
        "max_turnover": float(max_turnover),
    }


def write_report(summary_df: pd.DataFrame, report_path: Path) -> None:
    ranked = summary_df.sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost", "max_drawdown_after_cost"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    lines = [
        "# LSTM 执行层局部搜索报告",
        "",
        "## 搜索范围",
        "",
        "- 基础模型：`LSTM sl10 + base`",
        "- 搜索维度：`候选池大小 × 波动率阈值 × 风险惩罚 × max_turnover`",
        "",
        "## 排名结果",
        "",
        "| 排名 | candidate_size | vol20_pct | vol5_pct | risk_penalty | max_turnover | cumulative_return_after_cost | sharpe_after_cost | max_drawdown_after_cost | avg_turnover | top5_mean_return_mean | rank_ic_mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, (_, row) in enumerate(ranked.head(20).iterrows(), start=1):
        lines.append(
            f"| {idx} | {int(row['candidate_size'])} | {row['vol20_pct']:.2f} | {row['vol5_pct']:.2f} | "
            f"{row['risk_penalty']:.2f} | {row['max_turnover']:.2f} | "
            f"{row['cumulative_return_after_cost']:.6f} | {row['sharpe_after_cost']:.6f} | "
            f"{row['max_drawdown_after_cost']:.6f} | {row['avg_turnover']:.6f} | "
            f"{row['top5_mean_return_mean']:.6f} | {row['rank_ic_mean']:.6f} |"
        )

    if not ranked.empty:
        best = ranked.iloc[0]
        lines.extend(
            [
                "",
                "## 当前最优参数",
                "",
                f"- 候选池大小：`{int(best['candidate_size'])}`",
                f"- `max_volatility_20d_pct`：`{best['vol20_pct']:.2f}`",
                f"- `max_volatility_5d_pct`：`{best['vol5_pct']:.2f}`",
                f"- `risk_penalty_weight`：`{best['risk_penalty']:.2f}`",
                f"- `max_turnover`：`{best['max_turnover']:.2f}`",
                f"- 成本后累计收益：`{best['cumulative_return_after_cost']:.6f}`",
                f"- 成本后夏普：`{best['sharpe_after_cost']:.6f}`",
                f"- 成本后最大回撤：`{best['max_drawdown_after_cost']:.6f}`",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.prediction_path)
    feature_path = Path(args.feature_path)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_df, prediction_source = load_or_generate_predictions(
        prediction_path=prediction_path,
        feature_path=feature_path,
        model_dir=model_dir,
    )

    rows: list[dict] = []
    # current model's walk-forward diagnostics stay fixed across execution-layer search
    model_meta_path = model_dir / "model_meta.json"
    model_meta = {}
    if model_meta_path.exists():
        import json

        model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
    wf_summary = model_meta.get("walk_forward_summary", {})
    rank_ic_mean = float(wf_summary.get("rank_ic_mean", 0.0))
    top5_mean_return_mean = float(wf_summary.get("top5_mean_return_mean", 0.0))

    for candidate_size in args.candidate_sizes:
        for vol20_pct in args.vol20_pcts:
            for vol5_pct in args.vol5_pcts:
                for risk_penalty in args.risk_penalties:
                    for max_turnover in args.max_turnovers:
                        for sort_strategy in args.sort_strategies:
                            for weighting_scheme in args.weighting_schemes:
                                profile_name = (
                                    f"lstm_exec_cs{candidate_size}_v20{int(round(vol20_pct * 100)):02d}"
                                    f"_v5{int(round(vol5_pct * 100)):02d}_rp{int(round(risk_penalty * 100)):03d}"
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
                                summary_df, _, _ = run_backtest(
                                    prediction_df=prediction_df,
                                    config=config,
                                    prediction_source=prediction_source,
                                )
                                row = summary_df.iloc[0].to_dict()
                                row.update(
                                    {
                                        "candidate_size": int(candidate_size),
                                        "vol20_pct": float(vol20_pct),
                                        "vol5_pct": float(vol5_pct),
                                        "risk_penalty": float(risk_penalty),
                                        "sort_strategy": sort_strategy,
                                        "weighting_scheme": weighting_scheme,
                                        "rank_ic_mean": rank_ic_mean,
                                        "top5_mean_return_mean": top5_mean_return_mean,
                                    }
                                )
                                rows.append(row)
                                print(
                                    "[lstm_exec_search] "
                                    f"profile={profile_name} "
                                    f"cum_after={row['cumulative_return_after_cost']:.6f} "
                                    f"sharpe_after={row['sharpe_after_cost']:.6f}"
                                )

    summary = pd.DataFrame(rows).sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost", "max_drawdown_after_cost"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary_path = output_dir / "lstm_execution_search_summary.csv"
    report_path = output_dir / "lstm_execution_search_report.md"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary, report_path)

    best = summary.iloc[0]
    print(f"[lstm_exec_search] profiles={len(summary)}")
    print(
        "[lstm_exec_search] best_profile="
        f"{best['profile_name']} "
        f"cum_after={best['cumulative_return_after_cost']:.6f} "
        f"sharpe_after={best['sharpe_after_cost']:.6f}"
    )
    print(f"[lstm_exec_search] wrote {summary_path}")
    print(f"[lstm_exec_search] wrote {report_path}")


if __name__ == "__main__":
    main()
