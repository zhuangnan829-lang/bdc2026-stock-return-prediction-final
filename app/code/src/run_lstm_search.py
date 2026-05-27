import argparse
import json
from pathlib import Path

import pandas as pd

from backtest import load_or_generate_predictions, run_backtest
from config import BEST_CONFIG
from lstm_utils import save_lstm_checkpoint
from train import (
    MODEL_LABEL_COLUMN,
    RAW_LABEL_COLUMN,
    SEED,
    add_training_target,
    load_training_frame,
    resolve_feature_columns,
    set_seed,
    summarise_metrics,
)
from train_lstm import fit_final_lstm, run_walk_forward_lstm


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "lstm_search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight LSTM search over sequence length, feature set, and max_turnover.")
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--valid_dates", type=int, default=int(BEST_CONFIG["training"]["valid_dates"]))
    parser.add_argument("--num_folds", type=int, default=int(BEST_CONFIG["training"]["num_folds"]))
    parser.add_argument("--target_mode", default=BEST_CONFIG["training"]["target_mode"])
    parser.add_argument("--sequence_lengths", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--feature_sets", nargs="+", default=["base", "base_technical", "base_technical_risk"])
    parser.add_argument("--turnovers", nargs="+", type=float, default=[0.60, 0.70, 0.75])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    return parser.parse_args()


def build_backtest_config(profile_name: str, max_turnover: float) -> dict:
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


def train_single_lstm_combo(
    df: pd.DataFrame,
    feature_path: Path,
    model_dir: Path,
    feature_set: str,
    target_mode: str,
    valid_dates: int,
    num_folds: int,
    sequence_length: int,
    hidden_size: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
) -> dict:
    feature_columns = resolve_feature_columns(feature_set)
    fold_metrics, walk_forward_predictions, fold_training_summaries = run_walk_forward_lstm(
        df=df,
        feature_columns=feature_columns,
        valid_dates=valid_dates,
        num_folds=num_folds,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.0,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
    )
    metric_summary = summarise_metrics(fold_metrics)
    final_model, scaler_mean, scaler_std, final_bundle, final_training_info = fit_final_lstm(
        df=df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.0,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
    )

    model_path = model_dir / "lstm_model.pt"
    save_lstm_checkpoint(
        path=model_path,
        model=final_model,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.0,
    )
    walk_forward_predictions.to_csv(model_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_metrics).to_csv(model_dir / "walk_forward_metrics.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "status": "trained",
        "backend": "torch_lstm",
        "walk_forward_backend": "torch_lstm",
        "feature_path": str(feature_path),
        "model_path": str(model_path),
        "feature_columns": feature_columns,
        "feature_set": feature_set,
        "raw_label_column": RAW_LABEL_COLUMN,
        "model_label_column": MODEL_LABEL_COLUMN,
        "target_mode": target_mode,
        "model_family": "lstm",
        "seed": SEED,
        "valid_dates": valid_dates,
        "num_folds": num_folds,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "train_rows_full": int(len(df)),
        "train_sequence_rows_full": int(len(final_bundle.x)),
        "train_date_range_full": [
            str(df["date"].min().date()),
            str(df["date"].max().date()),
        ],
        "walk_forward_summary": metric_summary,
        "walk_forward_folds": fold_metrics,
        "walk_forward_training": fold_training_summaries,
        "final_training": final_training_info,
    }
    (model_dir / "model_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def write_report(summary_df: pd.DataFrame, report_path: Path) -> None:
    ranked = summary_df.sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost", "top5_mean_return_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    lines = [
        "# LSTM 轻量搜索报告",
        "",
        "## 搜索范围",
        "",
        "- 序列窗口长度：`10 / 20`",
        "- 特征集：`base / base_technical / base_technical_risk`",
        "- 换手上限：`0.60 / 0.70 / 0.75`",
        "",
        "## 排名结果",
        "",
        "| 排名 | sequence_length | feature_set | max_turnover | rank_ic_mean | top5_mean_return_mean | cumulative_return_after_cost | sharpe_after_cost | max_drawdown_after_cost | avg_turnover |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, (_, row) in enumerate(ranked.iterrows(), start=1):
        lines.append(
            f"| {idx} | {int(row['sequence_length'])} | {row['feature_set']} | {row['max_turnover']:.2f} | "
            f"{row['rank_ic_mean']:.6f} | {row['top5_mean_return_mean']:.6f} | "
            f"{row['cumulative_return_after_cost']:.6f} | {row['sharpe_after_cost']:.6f} | "
            f"{row['max_drawdown_after_cost']:.6f} | {row['avg_turnover']:.6f} |"
        )
    if not ranked.empty:
        best = ranked.iloc[0]
        lines.extend(
            [
                "",
                "## 当前最优组合",
                "",
                f"- 序列窗口长度：`{int(best['sequence_length'])}`",
                f"- 特征集：`{best['feature_set']}`",
                f"- 最大换手：`{best['max_turnover']:.2f}`",
                f"- 成本后累计收益：`{best['cumulative_return_after_cost']:.6f}`",
                f"- 成本后夏普：`{best['sharpe_after_cost']:.6f}`",
                f"- 成本后最大回撤：`{best['max_drawdown_after_cost']:.6f}`",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    feature_path = Path(args.feature_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    df = load_training_frame(feature_path)
    df = add_training_target(df, args.target_mode)

    train_rows: list[dict] = []
    combo_cache: dict[tuple[int, str], dict] = {}

    for sequence_length in args.sequence_lengths:
        for feature_set in args.feature_sets:
            combo_name = f"lstm_sl{sequence_length}_{feature_set}"
            model_dir = output_dir / combo_name
            model_dir.mkdir(parents=True, exist_ok=True)
            print(f"[lstm_search] training combo={combo_name}")
            metadata = train_single_lstm_combo(
                df=df,
                feature_path=feature_path,
                model_dir=model_dir,
                feature_set=feature_set,
                target_mode=args.target_mode,
                valid_dates=args.valid_dates,
                num_folds=args.num_folds,
                sequence_length=sequence_length,
                hidden_size=args.hidden_size,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
            )
            combo_cache[(sequence_length, feature_set)] = metadata

    for sequence_length in args.sequence_lengths:
        for feature_set in args.feature_sets:
            model_dir = output_dir / f"lstm_sl{sequence_length}_{feature_set}"
            prediction_df, prediction_source = load_or_generate_predictions(
                prediction_path=model_dir / "walk_forward_predictions.csv",
                feature_path=feature_path,
                model_dir=model_dir,
            )
            metadata = combo_cache[(sequence_length, feature_set)]
            walk_forward_summary = metadata["walk_forward_summary"]

            for max_turnover in args.turnovers:
                profile_name = f"lstm_sl{sequence_length}_{feature_set}_mt{int(round(max_turnover * 100)):02d}"
                config = build_backtest_config(profile_name=profile_name, max_turnover=max_turnover)
                backtest_summary_df, _, _ = run_backtest(
                    prediction_df=prediction_df,
                    config=config,
                    prediction_source=prediction_source,
                )
                backtest_summary = backtest_summary_df.iloc[0].to_dict()
                train_rows.append(
                    {
                        "model_family": "lstm",
                        "sequence_length": sequence_length,
                        "feature_set": feature_set,
                        "feature_count": len(metadata["feature_columns"]),
                        "max_turnover": float(max_turnover),
                        "rank_ic_mean": float(walk_forward_summary["rank_ic_mean"]),
                        "top5_mean_return_mean": float(walk_forward_summary["top5_mean_return_mean"]),
                        "rmse_mean": float(walk_forward_summary["rmse_mean"]),
                        "mae_mean": float(walk_forward_summary["mae_mean"]),
                        "cumulative_return_after_cost": float(backtest_summary["cumulative_return_after_cost"]),
                        "sharpe_after_cost": float(backtest_summary["sharpe_after_cost"]),
                        "max_drawdown_after_cost": float(backtest_summary["max_drawdown_after_cost"]),
                        "avg_turnover": float(backtest_summary["avg_turnover"]),
                        "win_rate_after_cost": float(backtest_summary["win_rate_after_cost"]),
                        "avg_cash_weight": float(backtest_summary["avg_cash_weight"]),
                        "model_dir": str(model_dir),
                        "profile_name": profile_name,
                    }
                )
                print(
                    f"[lstm_search] evaluated profile={profile_name} "
                    f"cum_after={backtest_summary['cumulative_return_after_cost']:.6f} "
                    f"sharpe_after={backtest_summary['sharpe_after_cost']:.6f}"
                )

    summary_df = pd.DataFrame(train_rows).sort_values(
        ["cumulative_return_after_cost", "sharpe_after_cost", "top5_mean_return_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary_path = output_dir / "lstm_search_summary.csv"
    report_path = output_dir / "lstm_search_report.md"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary_df, report_path)

    best = summary_df.iloc[0]
    print(f"[lstm_search] profiles={len(summary_df)}")
    print(
        "[lstm_search] best_profile="
        f"{best['profile_name']} "
        f"cum_after={best['cumulative_return_after_cost']:.6f} "
        f"sharpe_after={best['sharpe_after_cost']:.6f}"
    )
    print(f"[lstm_search] wrote {summary_path}")
    print(f"[lstm_search] wrote {report_path}")


if __name__ == "__main__":
    main()
