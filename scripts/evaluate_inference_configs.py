import argparse
import importlib.util
from pathlib import Path

import pandas as pd


DEFAULT_CONFIGS = [
    {
        "name": "default",
        "top_k": 5,
        "primary_candidate_size": 30,
        "max_volatility_20d_pct": 0.80,
        "max_volatility_5d_pct": 0.80,
        "turnover_rate_lower_pct": 0.05,
        "turnover_rate_upper_pct": 0.95,
        "turnover_ratio_upper_pct": 0.90,
        "risk_penalty_weight": 0.35,
    },
    {
        "name": "stricter_risk",
        "top_k": 5,
        "primary_candidate_size": 25,
        "max_volatility_20d_pct": 0.70,
        "max_volatility_5d_pct": 0.70,
        "turnover_rate_lower_pct": 0.10,
        "turnover_rate_upper_pct": 0.90,
        "turnover_ratio_upper_pct": 0.80,
        "risk_penalty_weight": 0.45,
    },
    {
        "name": "looser_risk",
        "top_k": 5,
        "primary_candidate_size": 40,
        "max_volatility_20d_pct": 0.90,
        "max_volatility_5d_pct": 0.90,
        "turnover_rate_lower_pct": 0.03,
        "turnover_rate_upper_pct": 0.97,
        "turnover_ratio_upper_pct": 0.95,
        "risk_penalty_weight": 0.25,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple inference parameter settings on walk-forward predictions.")
    parser.add_argument("--prediction_path", default="app/model/walk_forward_predictions.csv")
    parser.add_argument("--feature_path", default="app/temp/train_features.csv")
    parser.add_argument("--output_path", default="app/model/inference_config_comparison.csv")
    return parser.parse_args()


def load_inference_module():
    module_path = Path("app/code/src/test.py").resolve()
    spec = importlib.util.spec_from_file_location("app_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArgNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main() -> None:
    args = parse_args()
    app_test = load_inference_module()

    pred = pd.read_csv(args.prediction_path, encoding="utf-8-sig", dtype={"stock_id": str})
    feat = pd.read_csv(args.feature_path, encoding="utf-8-sig", dtype={"stock_id": str})
    for df in (pred, feat):
        df["stock_id"] = df["stock_id"].astype(str).str.zfill(6)
        df["date"] = pd.to_datetime(df["date"])

    risk_cols = [
        "volatility_20d",
        "volatility_5d",
        "turnover_rate",
        "turnover_ratio_10d",
        "amplitude_ratio_5d",
    ]
    merged = pred.merge(
        feat[["stock_id", "date"] + risk_cols],
        on=["stock_id", "date"],
        how="left",
        validate="many_to_one",
    )
    for column in risk_cols:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    rows = []
    for config in DEFAULT_CONFIGS:
        cfg_args = ArgNamespace(**config)
        fold_rows = []
        for fold_id, fold_df in merged.groupby("fold_id"):
            daily_returns = []
            for _, day_df in fold_df.groupby("date"):
                candidates, diagnostics = app_test.apply_candidate_filters(day_df.copy(), cfg_args)
                reranked = app_test.rerank_with_risk_controls(candidates.copy(), cfg_args)
                topk = reranked.sort_values(
                    ["selection_score", "pred_return", "stock_id"],
                    ascending=[False, False, True],
                ).head(cfg_args.top_k)
                ret = float(topk["target_return"].mean()) if not topk.empty else 0.0
                daily_returns.append(ret)
            fold_rows.append(
                {
                    "config_name": config["name"],
                    "fold_id": int(fold_id),
                    "top5_mean_return": float(pd.Series(daily_returns).mean()),
                    "top5_return_std": float(pd.Series(daily_returns).std(ddof=0)),
                }
            )

        fold_df = pd.DataFrame(fold_rows)
        rows.append(
            {
                "config_name": config["name"],
                "top5_mean_return_mean": float(fold_df["top5_mean_return"].mean()),
                "top5_mean_return_std": float(fold_df["top5_mean_return"].std(ddof=0)),
                "top5_return_std_mean": float(fold_df["top5_return_std"].mean()),
            }
        )

    result_df = pd.DataFrame(rows).sort_values(
        ["top5_mean_return_mean", "top5_return_std_mean"],
        ascending=[False, True],
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(result_df.to_string(index=False))
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
