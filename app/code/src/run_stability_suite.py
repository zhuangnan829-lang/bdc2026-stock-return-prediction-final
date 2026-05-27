import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import load_prediction_frame, run_backtest
from build_formal_model_comparison import compute_rank_ic, compute_top5_mean_return
from config import BEST_CONFIG, ROOT_DIR
from evaluate_rank_stability import append_stability_summary, summarize_prediction_rank_stability


DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "stability_suite"
DEFAULT_MODEL_ROOT = DEFAULT_OUTPUT_DIR / "ensemble_models"
DEFAULT_SEEDS = "2026,2027,2028"
DEFAULT_TURNOVER_VALUES = "0.5,0.6,0.7"
DEFAULT_SMOOTH_WINDOWS = "1,2,3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stability enhancement experiments for the default LSTM pipeline.")
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--turnover_values", default=DEFAULT_TURNOVER_VALUES)
    parser.add_argument("--smooth_windows", default=DEFAULT_SMOOTH_WINDOWS)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--skip_training", action="store_true")
    return parser.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def train_seed_models(
    feature_path: Path,
    model_root: Path,
    seeds: list[int],
    python_bin: str,
) -> list[Path]:
    training = BEST_CONFIG["training"]
    model_dirs: list[Path] = []
    for seed in seeds:
        model_dir = model_root / f"seed_{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            python_bin,
            str(ROOT_DIR / "app" / "code" / "src" / "train_lstm.py"),
            "--feature_path",
            str(feature_path),
            "--model_dir",
            str(model_dir),
            "--feature_set",
            str(training["feature_set"]),
            "--target_mode",
            str(training["target_mode"]),
            "--valid_dates",
            str(training["valid_dates"]),
            "--num_folds",
            str(training["num_folds"]),
            "--sequence_length",
            str(training["sequence_length"]),
            "--hidden_size",
            str(training["hidden_size"]),
            "--num_layers",
            str(training["num_layers"]),
            "--dropout",
            str(training["dropout"]),
            "--learning_rate",
            str(training["learning_rate"]),
            "--batch_size",
            str(training["batch_size"]),
            "--epochs",
            str(training["epochs"]),
            "--patience",
            str(training["patience"]),
            "--seed",
            str(seed),
        ]
        print(f"[stability_suite] training seed={seed} -> {model_dir}")
        subprocess.run(cmd, check=True)
        model_dirs.append(model_dir)
    return model_dirs


def load_seed_predictions(model_dirs: list[Path]) -> tuple[pd.DataFrame, list[dict]]:
    merged = None
    seed_rows: list[dict] = []
    for idx, model_dir in enumerate(model_dirs):
        meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
        pred_df = pd.read_csv(model_dir / "walk_forward_predictions.csv", encoding="utf-8-sig", dtype={"stock_id": str})
        pred_df["stock_id"] = pred_df["stock_id"].astype(str).str.zfill(6)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        pred_df = pred_df.rename(columns={"pred_return": f"pred_return_seed_{idx}"})
        keep_cols = ["stock_id", "date", "target_return", "train_target", "fold_id", f"pred_return_seed_{idx}"]
        pred_df = pred_df[keep_cols]
        if merged is None:
            merged = pred_df
        else:
            merged = merged.merge(
                pred_df,
                on=["stock_id", "date", "target_return", "train_target", "fold_id"],
                how="inner",
                validate="one_to_one",
            )
        seed_rows.append(
            {
                "seed": int(meta["seed"]),
                "model_dir": str(model_dir),
                "rank_ic_mean": float(meta["walk_forward_summary"]["rank_ic_mean"]),
                "top5_mean_return_mean": float(meta["walk_forward_summary"]["top5_mean_return_mean"]),
                "rmse_mean": float(meta["walk_forward_summary"]["rmse_mean"]),
                "mae_mean": float(meta["walk_forward_summary"]["mae_mean"]),
            }
        )
    if merged is None:
        raise ValueError("No seed predictions were loaded")
    return merged, seed_rows


def build_ensemble_predictions(merged_seed_df: pd.DataFrame) -> pd.DataFrame:
    pred_columns = [column for column in merged_seed_df.columns if column.startswith("pred_return_seed_")]
    out = merged_seed_df.copy()
    out["pred_return"] = out[pred_columns].mean(axis=1)
    return out[["stock_id", "date", "target_return", "train_target", "fold_id", "pred_return"]].copy()


def apply_temporal_smoothing(prediction_df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = prediction_df.copy()
    out = out.sort_values(["stock_id", "date"]).reset_index(drop=True)
    if window <= 1:
        out["pred_return_smoothed"] = out["pred_return"]
    else:
        out["pred_return_smoothed"] = (
            out.groupby("stock_id")["pred_return"]
            .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        )
    out["pred_return"] = out["pred_return_smoothed"]
    return out.drop(columns=["pred_return_smoothed"])


def build_backtest_config(max_turnover: float, profile_name: str) -> dict:
    return {
        "profile_name": profile_name,
        "top_k": int(BEST_CONFIG["selection"]["top_k"]),
        "primary_candidate_size": int(BEST_CONFIG["selection"]["primary_candidate_size"]),
        "enable_risk_filters": int(bool(BEST_CONFIG["selection"]["enable_risk_filters"])),
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": float(BEST_CONFIG["risk_filter_thresholds"]["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(BEST_CONFIG["risk_filter_thresholds"]["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(BEST_CONFIG["risk_filter_thresholds"]["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(BEST_CONFIG["risk_filter_thresholds"]["risk_penalty_weight"]),
        "weighting_scheme": BEST_CONFIG["selection"]["weighting_scheme"],
        "sort_strategy": BEST_CONFIG["selection"]["sort_strategy"],
        "transaction_cost": float(BEST_CONFIG["execution"]["transaction_cost"]),
        "max_turnover": float(max_turnover),
    }


def evaluate_predictions(prediction_df: pd.DataFrame, feature_path: Path, smooth_window: int, max_turnover: float) -> dict:
    temp_prediction_path = ROOT_DIR / "app" / "temp" / "_stability_suite_predictions.csv"
    prediction_df.to_csv(temp_prediction_path, index=False, encoding="utf-8-sig")
    backtest_ready = load_prediction_frame(temp_prediction_path, feature_path)
    summary_df, _, _ = run_backtest(
        prediction_df=backtest_ready,
        config=build_backtest_config(
            max_turnover=max_turnover,
            profile_name=f"ensemble_sw{smooth_window}_mt{int(round(max_turnover * 100))}",
        ),
        prediction_source="stability_suite_ensemble",
    )
    summary = summary_df.iloc[0].to_dict()
    stability_summary = summarize_prediction_rank_stability(
        prediction_df=prediction_df,
        experiment_name=f"stability_suite/sw{smooth_window}_mt{int(round(max_turnover * 100))}",
        extra_fields={
            "smooth_window": int(smooth_window),
            "max_turnover": float(max_turnover),
        },
    )
    append_stability_summary({**stability_summary, **summary})
    return {
        "smooth_window": int(smooth_window),
        "max_turnover": float(max_turnover),
        "rank_ic_mean": float(stability_summary["rank_ic_mean"]),
        "rank_ic_std": float(stability_summary["rank_ic_std"]),
        "worst_fold_rank_ic": float(stability_summary["worst_fold_rank_ic"]),
        "negative_day_rank_ic_ratio": float(stability_summary["negative_day_rank_ic_ratio"]),
        "top5_mean_return_mean": float(compute_top5_mean_return(prediction_df)),
        "cumulative_return_after_cost": float(summary["cumulative_return_after_cost"]),
        "sharpe_after_cost": float(summary["sharpe_after_cost"]),
        "max_drawdown_after_cost": float(summary["max_drawdown_after_cost"]),
        "avg_turnover": float(summary["avg_turnover"]),
        "win_rate_after_cost": float(summary["win_rate_after_cost"]),
        "avg_execution_strength": float(summary["avg_execution_strength"]),
        "total_transaction_cost": float(summary["total_transaction_cost"]),
    }


def write_markdown(summary_df: pd.DataFrame, output_path: Path, seeds: list[int]) -> None:
    ranked = summary_df.sort_values(
        ["worst_fold_rank_ic", "negative_day_rank_ic_ratio", "sharpe_after_cost", "cumulative_return_after_cost"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    lines = [
        "# Stability Suite Summary",
        "",
        f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- Base feature set: `{BEST_CONFIG['training']['feature_set']}`",
        f"- Base sequence length: `{BEST_CONFIG['training']['sequence_length']}`",
        "",
        "| Rank | smooth_window | max_turnover | worst_fold_rank_ic | negative_day_rank_ic_ratio | rank_ic_mean | cum_after_cost | sharpe_after_cost | max_dd_after_cost | avg_turnover |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(ranked.itertuples(index=False), start=1):
        lines.append(
            f"| {idx} | {row.smooth_window} | {row.max_turnover:.2f} | "
            f"{row.worst_fold_rank_ic:.6f} | {row.negative_day_rank_ic_ratio:.6f} | "
            f"{row.rank_ic_mean:.6f} | {row.cumulative_return_after_cost:.6f} | "
            f"{row.sharpe_after_cost:.6f} | {row.max_drawdown_after_cost:.6f} | {row.avg_turnover:.6f} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    feature_path = Path(args.feature_path)
    output_dir = Path(args.output_dir)
    model_root = Path(args.model_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    seeds = parse_csv_ints(args.seeds)
    turnover_values = parse_csv_floats(args.turnover_values)
    smooth_windows = parse_csv_ints(args.smooth_windows)

    if args.skip_training:
        model_dirs = [model_root / f"seed_{seed}" for seed in seeds]
    else:
        model_dirs = train_seed_models(
            feature_path=feature_path,
            model_root=model_root,
            seeds=seeds,
            python_bin=args.python_bin,
        )

    merged_seed_df, seed_rows = load_seed_predictions(model_dirs)
    seed_summary_df = pd.DataFrame(seed_rows)
    ensemble_raw_df = build_ensemble_predictions(merged_seed_df)

    ensemble_raw_path = output_dir / "ensemble_walk_forward_predictions_raw.csv"
    ensemble_raw_df.to_csv(ensemble_raw_path, index=False, encoding="utf-8-sig")
    seed_summary_df.to_csv(output_dir / "seed_training_summary.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict] = []
    for smooth_window in smooth_windows:
        smoothed_df = apply_temporal_smoothing(ensemble_raw_df, smooth_window)
        smoothed_path = output_dir / f"ensemble_walk_forward_predictions_sw{smooth_window}.csv"
        smoothed_df.to_csv(smoothed_path, index=False, encoding="utf-8-sig")
        for max_turnover in turnover_values:
            row = evaluate_predictions(
                prediction_df=smoothed_df,
                feature_path=feature_path,
                smooth_window=smooth_window,
                max_turnover=max_turnover,
            )
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["worst_fold_rank_ic", "negative_day_rank_ic_ratio", "sharpe_after_cost", "cumulative_return_after_cost"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    summary_csv_path = output_dir / "stability_suite_summary.csv"
    summary_md_path = output_dir / "stability_suite_summary.md"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    write_markdown(summary_df, summary_md_path, seeds)

    best = summary_df.iloc[0]
    print(f"[stability_suite] wrote {summary_csv_path}")
    print(f"[stability_suite] wrote {summary_md_path}")
    print(
        "[stability_suite] best="
        f"sw{int(best['smooth_window'])}_mt{best['max_turnover']:.2f} "
        f"sharpe={best['sharpe_after_cost']:.6f} "
        f"cum_after={best['cumulative_return_after_cost']:.6f}"
    )


if __name__ == "__main__":
    main()
