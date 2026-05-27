import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import pandas as pd

from backtest import run_backtest
from config import BEST_CONFIG
from stability_diagnostics import (
    build_analysis_config,
    build_fold_diagnostics,
    load_prediction_artifact,
)
from evaluate_rank_stability import append_experiment_rank_stability


ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = ROOT_DIR / "app"
SRC_DIR = APP_DIR / "code" / "src"
TEMP_DIR = APP_DIR / "temp"
MODEL_DIR = APP_DIR / "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stability-oriented parameter comparison for walk-forward LSTM experiments."
    )
    parser.add_argument("--feature_path", default=str(TEMP_DIR / "train_features.csv"))
    parser.add_argument("--base_model_dir", default=str(MODEL_DIR))
    parser.add_argument("--output_dir", default=str(MODEL_DIR / "stability_parameter_compare"))
    parser.add_argument("--sequence_lengths", default="")
    parser.add_argument("--risk_penalty_weights", default="")
    parser.add_argument("--sort_strategies", default="")
    parser.add_argument("--primary_candidate_sizes", default="")
    parser.add_argument("--retrain", type=int, choices=[0, 1], default=0)
    parser.add_argument("--reuse_base_model_for_default_sequence", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def parse_int_list(raw: str, fallback: list[int]) -> list[int]:
    if not raw.strip():
        return fallback
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_float_list(raw: str, fallback: list[float]) -> list[float]:
    if not raw.strip():
        return fallback
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_str_list(raw: str, fallback: list[str]) -> list[str]:
    if not raw.strip():
        return fallback
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=ROOT_DIR)


def build_train_cmd(feature_path: Path, model_dir: Path, sequence_length: int) -> list[str]:
    training = BEST_CONFIG["training"]
    return [
        sys.executable,
        str(SRC_DIR / "train_lstm.py"),
        "--feature_path",
        str(feature_path),
        "--model_dir",
        str(model_dir),
        "--feature_set",
        str(training["feature_set"]),
        "--sequence_length",
        str(sequence_length),
        "--epochs",
        str(training["epochs"]),
        "--patience",
        str(training["patience"]),
        "--batch_size",
        str(training["batch_size"]),
        "--learning_rate",
        str(training["learning_rate"]),
        "--hidden_size",
        str(training["hidden_size"]),
        "--num_layers",
        str(training["num_layers"]),
        "--dropout",
        str(training["dropout"]),
        "--valid_dates",
        str(training["valid_dates"]),
        "--num_folds",
        str(training["num_folds"]),
        "--target_mode",
        str(training["target_mode"]),
    ]


def ensure_model_artifacts(feature_path: Path, model_dir: Path, sequence_length: int, retrain: bool) -> None:
    required = [
        model_dir / "model_meta.json",
        model_dir / "walk_forward_predictions.csv",
    ]
    if retrain or any(not path.exists() for path in required):
        model_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(build_train_cmd(feature_path=feature_path, model_dir=model_dir, sequence_length=sequence_length))


def build_profile_name(sequence_length: int, risk_penalty_weight: float, sort_strategy: str, candidate_size: int) -> str:
    rp_tag = str(int(round(risk_penalty_weight * 100)))
    return f"sl{sequence_length}__{sort_strategy}__cs{candidate_size}__rp{rp_tag}"


def write_report(summary_df: pd.DataFrame, report_path: Path) -> None:
    ranked = summary_df.sort_values(
        [
            "worst_fold_rank_ic",
            "rank_ic_mean",
            "cumulative_return_after_cost",
            "sharpe_after_cost",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    lines = [
        "# Stability Parameter Comparison",
        "",
        "## Overview",
        "",
        "- This report compares sequence length, risk penalty, sort strategy, and primary candidate size.",
        "- Ranking priority emphasizes worst-fold stability first, then overall rank_ic_mean and backtest quality.",
        "",
        "## Ranked Table",
        "",
        "| profile_name | sequence_length | sort_strategy | primary_candidate_size | risk_penalty_weight | worst_fold_rank_ic | rank_ic_mean | avg_negative_day_ratio | cum_after_cost | sharpe_after_cost | max_dd_after_cost |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in ranked.iterrows():
        lines.append(
            f"| {row['profile_name']} | {int(row['sequence_length'])} | {row['sort_strategy']} | "
            f"{int(row['primary_candidate_size'])} | {float(row['risk_penalty_weight']):.2f} | "
            f"{float(row['worst_fold_rank_ic']):.6f} | {float(row['rank_ic_mean']):.6f} | "
            f"{float(row['avg_negative_day_rank_ic_ratio']):.6f} | "
            f"{float(row['cumulative_return_after_cost']):.6f} | {float(row['sharpe_after_cost']):.6f} | "
            f"{float(row['max_drawdown_after_cost']):.6f} |"
        )

    if not ranked.empty:
        best = ranked.iloc[0]
        lines.extend(
            [
                "",
                "## Suggested Focus",
                "",
                f"- Current top candidate: `{best['profile_name']}`",
                f"- Sequence length: `{int(best['sequence_length'])}`",
                f"- Sort strategy: `{best['sort_strategy']}`",
                f"- Primary candidate size: `{int(best['primary_candidate_size'])}`",
                f"- Risk penalty weight: `{float(best['risk_penalty_weight']):.2f}`",
                f"- Worst fold rank_ic: `{float(best['worst_fold_rank_ic']):.6f}`",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    feature_path = Path(args.feature_path)
    base_model_dir = Path(args.base_model_dir)
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    current_training = BEST_CONFIG["training"]
    current_selection = BEST_CONFIG["selection"]
    current_risk = BEST_CONFIG["risk_filter_thresholds"]

    sequence_lengths = parse_int_list(args.sequence_lengths, [int(current_training["sequence_length"])])
    risk_penalty_weights = parse_float_list(
        args.risk_penalty_weights,
        [float(current_risk["risk_penalty_weight"])],
    )
    sort_strategies = parse_str_list(args.sort_strategies, [str(current_selection["sort_strategy"])])
    primary_candidate_sizes = parse_int_list(
        args.primary_candidate_sizes,
        [int(current_selection["primary_candidate_size"])],
    )

    trained_models: dict[int, dict] = {}
    default_sequence = int(current_training["sequence_length"])
    for sequence_length in sequence_lengths:
        if (
            sequence_length == default_sequence
            and bool(args.reuse_base_model_for_default_sequence)
            and (base_model_dir / "walk_forward_predictions.csv").exists()
        ):
            model_dir = base_model_dir
        else:
            model_dir = models_dir / f"lstm_sl{sequence_length}"
            ensure_model_artifacts(
                feature_path=feature_path,
                model_dir=model_dir,
                sequence_length=sequence_length,
                retrain=bool(args.retrain),
            )

        meta = load_json(model_dir / "model_meta.json")
        merged_prediction_df = load_prediction_artifact(
            prediction_path=model_dir / "walk_forward_predictions.csv",
            feature_path=feature_path,
        )
        trained_models[int(sequence_length)] = {
            "model_dir": model_dir,
            "meta": meta,
            "prediction_df": merged_prediction_df,
        }

    rows: list[dict] = []
    fold_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []

    for sequence_length, risk_penalty_weight, sort_strategy, candidate_size in product(
        sequence_lengths,
        risk_penalty_weights,
        sort_strategies,
        primary_candidate_sizes,
    ):
        profile_name = build_profile_name(
            sequence_length=sequence_length,
            risk_penalty_weight=risk_penalty_weight,
            sort_strategy=sort_strategy,
            candidate_size=candidate_size,
        )
        config = build_analysis_config(
            profile_name=profile_name,
            overrides={
                "primary_candidate_size": int(candidate_size),
                "risk_penalty_weight": float(risk_penalty_weight),
                "sort_strategy": sort_strategy,
            },
        )
        model_info = trained_models[int(sequence_length)]
        prediction_df = model_info["prediction_df"]

        fold_diag_df, daily_diag_df = build_fold_diagnostics(prediction_df=prediction_df, config=config)
        backtest_summary_df, daily_backtest_df, holdings_backtest_df = run_backtest(
            prediction_df=prediction_df,
            config=config,
            prediction_source="replay_walk_forward_predictions",
        )

        fold_diag_df["sequence_length"] = int(sequence_length)
        fold_diag_df["risk_penalty_weight"] = float(risk_penalty_weight)
        fold_diag_df["sort_strategy"] = sort_strategy
        fold_diag_df["primary_candidate_size"] = int(candidate_size)
        daily_diag_df["sequence_length"] = int(sequence_length)
        daily_diag_df["risk_penalty_weight"] = float(risk_penalty_weight)
        daily_diag_df["sort_strategy"] = sort_strategy
        daily_diag_df["primary_candidate_size"] = int(candidate_size)
        fold_rows.append(fold_diag_df)
        daily_rows.append(daily_diag_df)

        backtest_row = backtest_summary_df.iloc[0]
        meta = model_info["meta"]
        rank_ic_values = pd.to_numeric(fold_diag_df["rank_ic"], errors="coerce").dropna()
        negative_day_rank_ic_ratio = (
            float(pd.to_numeric(daily_diag_df["day_rank_ic"], errors="coerce").dropna().lt(0).mean())
            if not daily_diag_df.empty
            else 0.0
        )
        rows.append(
            {
                "profile_name": profile_name,
                "model_dir": str(model_info["model_dir"]),
                "sequence_length": int(sequence_length),
                "risk_penalty_weight": float(risk_penalty_weight),
                "sort_strategy": sort_strategy,
                "primary_candidate_size": int(candidate_size),
                "rank_ic_mean": float(rank_ic_values.mean()) if not rank_ic_values.empty else 0.0,
                "rank_ic_std": float(rank_ic_values.std(ddof=0)) if len(rank_ic_values) > 1 else 0.0,
                "top5_mean_return_mean": float(meta["walk_forward_summary"]["top5_mean_return_mean"]),
                "worst_fold_rank_ic": float(rank_ic_values.min()) if not rank_ic_values.empty else 0.0,
                "best_fold_rank_ic": float(rank_ic_values.max()) if not rank_ic_values.empty else 0.0,
                "negative_day_rank_ic_ratio": negative_day_rank_ic_ratio,
                "avg_negative_day_rank_ic_ratio": negative_day_rank_ic_ratio,
                "avg_after_risk_filters": float(fold_diag_df["avg_after_risk_filters"].mean())
                if not fold_diag_df.empty
                else 0.0,
                "avg_selected_target_return": float(fold_diag_df["avg_selected_target_return"].mean())
                if not fold_diag_df.empty
                else 0.0,
                "cumulative_return_after_cost": float(backtest_row["cumulative_return_after_cost"]),
                "sharpe_after_cost": float(backtest_row["sharpe_after_cost"]),
                "max_drawdown_after_cost": float(backtest_row["max_drawdown_after_cost"]),
                "avg_turnover": float(backtest_row["avg_turnover"]),
                "avg_selected_count": float(backtest_row["avg_selected_count"]),
                "periods": int(backtest_row["periods"]),
            }
        )

        per_profile_dir = output_dir / profile_name
        per_profile_dir.mkdir(parents=True, exist_ok=True)
        fold_diag_df.to_csv(per_profile_dir / "fold_diagnostics.csv", index=False, encoding="utf-8-sig")
        daily_diag_df.to_csv(per_profile_dir / "fold_daily_diagnostics.csv", index=False, encoding="utf-8-sig")
        daily_backtest_df.to_csv(per_profile_dir / "backtest_daily.csv", index=False, encoding="utf-8-sig")
        holdings_backtest_df.to_csv(
            per_profile_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig"
        )
        backtest_summary_df.to_csv(
            per_profile_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig"
        )
        append_experiment_rank_stability(
            experiment_name=f"stability_parameter_compare/{profile_name}",
            prediction_path=model_info["model_dir"] / "walk_forward_predictions.csv",
            fold_diagnostics_path=per_profile_dir / "fold_diagnostics.csv",
            fold_daily_diagnostics_path=per_profile_dir / "fold_daily_diagnostics.csv",
            backtest_summary_path=per_profile_dir / "backtest_summary.csv",
            extra_fields={
                "profile_name": profile_name,
                "model_dir": str(model_info["model_dir"]),
                "sequence_length": int(sequence_length),
                "risk_penalty_weight": float(risk_penalty_weight),
                "sort_strategy": sort_strategy,
                "primary_candidate_size": int(candidate_size),
            },
        )

    summary_df = pd.DataFrame(rows)
    fold_summary_df = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    daily_summary_df = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()

    summary_path = output_dir / "parameter_comparison_summary.csv"
    fold_path = output_dir / "parameter_comparison_fold_diagnostics.csv"
    daily_path = output_dir / "parameter_comparison_daily_diagnostics.csv"
    report_path = output_dir / "parameter_comparison_report.md"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    fold_summary_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    daily_summary_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    write_report(summary_df, report_path)

    print(f"[stability_parameter_compare] wrote {summary_path}")
    print(f"[stability_parameter_compare] wrote {fold_path}")
    print(f"[stability_parameter_compare] wrote {daily_path}")
    print(f"[stability_parameter_compare] wrote {report_path}")


if __name__ == "__main__":
    main()
