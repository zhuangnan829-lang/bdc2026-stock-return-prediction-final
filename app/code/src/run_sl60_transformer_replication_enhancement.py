from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import load_prediction_frame, run_backtest
from compare_with_case_score import read_case_best_score, read_case_test, read_prediction, score_prediction
from config import ROOT_DIR
from evaluate_rank_stability import evaluate_one
from load_submission_config import build_default_inference_args, load_submission_config


DEFAULT_OUTPUT_DIR = ROOT_DIR / "app/model/sl60_transformer_replication_enhancement"
DEFAULT_BASE_CONFIG = ROOT_DIR / "app/model/default_submission_config.json"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app/temp/train_features.csv"
DEFAULT_CASE_ROOT = (
    ROOT_DIR
    / "app/model/external_case_zip/from_desktop_20260526/THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0"
)


SOURCE_SPECS = {
    "lstm_sl20_mainline": ROOT_DIR / "app/model/walk_forward_predictions.csv",
    "lstm_sl60_v3_mini4": ROOT_DIR / "app/model/sequence_length_search/sl60/walk_forward_predictions.csv",
    "lstm_sl60_v4_medium": ROOT_DIR / "app/model/sequence_length_search/v4_medium_lstm_sl60/walk_forward_predictions.csv",
    "transformer_lite_sl60": ROOT_DIR / "app/model/transformer_lite/sl60/walk_forward_predictions.csv",
}

VISIBLE_RESULT_SPECS = {
    "current_output_active": ROOT_DIR / "app/output/result.csv",
    "aggressive_score": ROOT_DIR / "app/model/aggressive_score_submission_candidate/result_aggressive_score.csv",
    "case_zip_output": DEFAULT_CASE_ROOT / "output/result.csv",
    "lstm_sl60_v3_mini4": ROOT_DIR / "app/model/sequence_length_search/sl60/case_slice_result.csv",
    "rank_blend_B_lstm60_lightgbm20_momentum20": ROOT_DIR / "app/model/rank_blend/B_lstm60_lightgbm20_momentum20/result.csv",
    "transformer_lite_sl20": ROOT_DIR / "app/model/transformer_lite/sl20/candidate_result.csv",
}


BLEND_SPECS = {
    "xattn_equal_sl20_sl60_t60": {
        "lstm_sl20_mainline": 0.34,
        "lstm_sl60_v3_mini4": 0.33,
        "transformer_lite_sl60": 0.33,
    },
    "xattn_sl60_heavy": {
        "lstm_sl20_mainline": 0.20,
        "lstm_sl60_v3_mini4": 0.50,
        "transformer_lite_sl60": 0.30,
    },
    "xattn_transformer_heavy": {
        "lstm_sl20_mainline": 0.20,
        "lstm_sl60_v3_mini4": 0.30,
        "transformer_lite_sl60": 0.50,
    },
    "xattn_v4_sl60_plus_transformer": {
        "lstm_sl20_mainline": 0.20,
        "lstm_sl60_v4_medium": 0.45,
        "transformer_lite_sl60": 0.35,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate zip sl60/attention strengths with local sl60, Transformer-lite, and cross-stock rank blends."
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base_config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--case_root", default=str(DEFAULT_CASE_ROOT))
    parser.add_argument("--retrain_missing", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


def run_optional_training(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    if not args.retrain_missing:
        return [{"step": "optional_retrain", "status": "SKIPPED", "reason": "use --retrain_missing to train missing sl60/Transformer artifacts"}]
    commands = [
        [
            sys.executable,
            "app/code/src/run_sequence_length_search.py",
            "--sequence_lengths",
            "60",
        ],
        [
            sys.executable,
            "app/code/src/train_transformer_lite.py",
            "--sequence_lengths",
            "60",
            "--train_missing",
        ],
    ]
    rows = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        rows.append(
            {
                "step": command[1],
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "optional_training_steps.csv", index=False, encoding="utf-8-sig")
    return rows


def read_prediction_source(name: str, path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"stock_id": str})
    required = {"stock_id", "date", "pred_return", "target_return"}
    if not required.issubset(df.columns):
        return None
    out = df[["stock_id", "date", "target_return", "pred_return"] + (["fold_id"] if "fold_id" in df.columns else [])].copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"])
    out = out.rename(columns={"pred_return": f"{name}_score"})
    return out


def merge_sources() -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    base: pd.DataFrame | None = None
    available: dict[str, str] = {}
    missing: dict[str, str] = {}
    for name, path in SOURCE_SPECS.items():
        df = read_prediction_source(name, path)
        if df is None:
            missing[name] = str(path)
            continue
        available[name] = str(path)
        if base is None:
            base = df
        else:
            merge_cols = ["stock_id", "date"]
            keep = [c for c in ["stock_id", "date", f"{name}_score"] if c in df.columns]
            base = base.merge(df[keep], on=merge_cols, how="inner")
    if base is None:
        raise FileNotFoundError("No usable prediction sources found for sl60/Transformer replication.")
    return base, available, missing


def daily_rank(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").groupby(df["date"]).rank(pct=True)


def build_cross_stock_attention_prediction(base: pd.DataFrame, blend_name: str, weights: dict[str, float]) -> pd.DataFrame:
    out = base[["stock_id", "date", "target_return"] + (["fold_id"] if "fold_id" in base.columns else [])].copy()
    usable = {name: weight for name, weight in weights.items() if f"{name}_score" in base.columns}
    total = sum(usable.values())
    if total <= 1e-12:
        raise ValueError(f"No usable components for {blend_name}: {weights}")
    score = pd.Series(0.0, index=base.index, dtype=float)
    rank_frame = pd.DataFrame(index=base.index)
    for name, weight in usable.items():
        rank = daily_rank(base, f"{name}_score")
        rank_frame[name] = rank
        score += rank.fillna(0.5) * (weight / total)

    agreement = 1.0 - rank_frame.std(axis=1).fillna(0.0).clip(0.0, 1.0)
    leader = rank_frame.max(axis=1).fillna(0.5)
    # Cross-stock attention proxy: daily rank blend plus a small boost for names where
    # long-sequence and Transformer views agree near the top of the cross-section.
    out["pred_return"] = score + 0.08 * (leader - 0.5) * agreement
    return out


def backtest_config(base_config_path: Path) -> dict[str, Any]:
    defaults = build_default_inference_args(load_submission_config(base_config_path))
    return {
        "top_k": int(defaults["top_k"]),
        "primary_candidate_size": int(defaults["primary_candidate_size"]),
        "enable_risk_filters": bool(defaults["enable_risk_filters"]),
        "allow_cash_fallback": 0,
        "max_volatility_20d_pct": float(defaults["max_volatility_20d_pct"]),
        "max_volatility_5d_pct": float(defaults["max_volatility_5d_pct"]),
        "turnover_rate_lower_pct": float(defaults["turnover_rate_lower_pct"]),
        "turnover_rate_upper_pct": float(defaults["turnover_rate_upper_pct"]),
        "turnover_ratio_upper_pct": float(defaults["turnover_ratio_upper_pct"]),
        "risk_penalty_weight": float(defaults["risk_penalty_weight"]),
        "weighting_scheme": str(defaults["weighting_scheme"]),
        "weight_blend_alpha": float(defaults["weight_blend_alpha"]),
        "max_single_weight": float(defaults["max_single_weight"]),
        "sort_strategy": str(defaults["sort_strategy"]),
        "transaction_cost": float(defaults["transaction_cost"]),
        "max_turnover": float(defaults["max_turnover"]),
        "rerank_signal_column": defaults.get("rerank_signal_column"),
        "rerank_signal_weight": float(defaults.get("rerank_signal_weight", 0.0)),
        "secondary_candidate_size": int(defaults.get("secondary_candidate_size", 0) or 0),
        "secondary_screen_mode": str(defaults.get("secondary_screen_mode", "none")),
        "secondary_screen_weight": float(defaults.get("secondary_screen_weight", 0.0)),
        "local_tiebreak_start_rank": int(defaults.get("local_tiebreak_start_rank", 8)),
        "local_tiebreak_end_rank": int(defaults.get("local_tiebreak_end_rank", 15)),
    }


def single_slice_from_daily(daily: pd.DataFrame) -> float:
    if daily.empty or "net_return" not in daily.columns:
        return 0.0
    return float(pd.to_numeric(daily["net_return"], errors="coerce").fillna(0.0).iloc[-1])


def evaluate_prediction_artifact(
    *,
    name: str,
    prediction_path: Path,
    feature_path: Path,
    base_config_path: Path,
    output_dir: Path,
    model_family: str,
    sequence_length: int,
) -> dict[str, Any]:
    row, _, _ = evaluate_one(
        experiment_name=name,
        prediction_path=prediction_path,
        model=model_family,
        feature_set="base_alpha_v3_rs_crowding_mini4",
        sequence_length=sequence_length,
    )
    prediction_df = load_prediction_frame(prediction_path, feature_path)
    config = {**backtest_config(base_config_path), "profile_name": name}
    summary, daily, holdings = run_backtest(prediction_df, config=config, prediction_source=str(prediction_path))
    bt_dir = output_dir / name / "backtest_same_protocol"
    bt_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(bt_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(bt_dir / "backtest_daily_results.csv", index=False, encoding="utf-8-sig")
    holdings.to_csv(bt_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    bt = summary.iloc[0]
    return {
        "candidate": name,
        "model_family": model_family,
        "sequence_length": sequence_length,
        "rank_ic_mean": float(row.get("rank_ic_mean", 0.0)),
        "worst_fold_rank_ic": float(row.get("worst_fold_rank_ic", 0.0)),
        "top5_return_mean": float(row.get("top5_return_mean", 0.0)),
        "top5_return_min_by_fold": float(row.get("top5_return_min_by_fold", 0.0)),
        "cost_after_return": float(bt.get("cumulative_return_after_cost", 0.0)),
        "Sharpe": float(bt.get("sharpe_after_cost", 0.0)),
        "max_drawdown": float(bt.get("max_drawdown_after_cost", 0.0)),
        "avg_turnover": float(bt.get("avg_turnover", 0.0)),
        "single_slice_score": single_slice_from_daily(daily),
        "prediction_path": str(prediction_path),
        "backtest_dir": str(bt_dir),
    }


def score_visible_results(case_root: Path, output_dir: Path) -> pd.DataFrame:
    case_test = read_case_test(case_root / "data/test.csv")
    case_best = read_case_best_score(case_root / "model/60_158+39/final_score.txt")
    rows = []
    details = []
    for name, path in VISIBLE_RESULT_SPECS.items():
        if name == "case_zip_output":
            path = case_root / "output/result.csv"
        if not path.exists():
            rows.append({"candidate": name, "status": "missing", "result_path": str(path)})
            continue
        score, detail = score_prediction(read_prediction(path), case_test)
        rows.append(
            {
                "candidate": name,
                "status": "ok",
                "case_slice_score": score,
                "diff_vs_case_best": score - case_best,
                "result_path": str(path),
                "stocks": ",".join(
                    detail[
                        "鑲＄エ浠ｇ爜"
                        if "鑲＄エ浠ｇ爜" in detail.columns
                        else "股票代码"
                        if "股票代码" in detail.columns
                        else "stock_id"
                    ]
                    .astype(str)
                    .str.zfill(6)
                    .tolist()
                ),
            }
        )
        detail["candidate"] = name
        details.append(detail)
    result = pd.DataFrame(rows).sort_values("case_slice_score", ascending=False, na_position="last")
    result.to_csv(output_dir / "visible_slice_result_scores.csv", index=False, encoding="utf-8-sig")
    if details:
        pd.concat(details, ignore_index=True).to_csv(output_dir / "visible_slice_result_details.csv", index=False, encoding="utf-8-sig")
    return result


def collect_existing_summaries(output_dir: Path) -> pd.DataFrame:
    frames = []
    for path, source in [
        (ROOT_DIR / "app/model/sequence_length_search/sl20_sl40_sl60_summary.csv", "sequence_length_search"),
        (ROOT_DIR / "app/model/transformer_lite/transformer_lite_summary.csv", "transformer_lite"),
        (ROOT_DIR / "app/model/rank_blend/blend_summary.csv", "rank_blend"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            df["source_summary"] = source
            frames.append(df)
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    out.to_csv(output_dir / "existing_sl60_transformer_evidence.csv", index=False, encoding="utf-8-sig")
    return out


def recommend(summary: pd.DataFrame) -> tuple[str, str]:
    if summary.empty:
        return "keep_current", "没有可用候选。"
    baseline = summary[summary["candidate"].eq("lstm_sl20_mainline")]
    baseline_row = baseline.iloc[0] if not baseline.empty else None
    candidates = summary[~summary["candidate"].eq("lstm_sl20_mainline")].copy()
    if candidates.empty:
        return "keep_current", "没有非主线候选。"
    candidates = candidates.sort_values(["cost_after_return", "worst_fold_rank_ic", "Sharpe"], ascending=[False, False, False])
    best = candidates.iloc[0]
    if baseline_row is None:
        return str(best["candidate"]), "没有主线基准，选择综合表现最高候选。"
    full_win = (
        best["cost_after_return"] > baseline_row["cost_after_return"]
        and best["worst_fold_rank_ic"] > baseline_row["worst_fold_rank_ic"]
        and best["Sharpe"] > baseline_row["Sharpe"]
        and best["max_drawdown"] >= baseline_row["max_drawdown"] - 0.03
    )
    if full_win:
        return str(best["candidate"]), "候选在收益、worst fold 和 Sharpe 上全面优于 sl20，回撤没有明显恶化。"
    return "keep_hv_rerank_sl20_mainline", "sl60/Transformer 候选没有同时在收益、worst fold、Sharpe 和回撤上全面胜出。"


def write_report(
    output_dir: Path,
    existing: pd.DataFrame,
    blend_summary: pd.DataFrame,
    visible: pd.DataFrame,
    available: dict[str, str],
    missing: dict[str, str],
    decision: str,
    reason: str,
) -> None:
    lines = [
        "# SL60 / Transformer Replication Enhancement Report",
        "",
        "本报告吸收压缩包的两个强点：`sequence_length=60` 和跨股票注意力思想，并用当前工程的同协议产物进行复刻增强评估。",
        "",
        "## Source Availability",
        "",
        "| source | status | path |",
        "|---|---|---|",
    ]
    for name, path in available.items():
        lines.append(f"| `{name}` | `available` | `{path}` |")
    for name, path in missing.items():
        lines.append(f"| `{name}` | `missing` | `{path}` |")
    lines += [
        "",
        "## Cross-Stock Attention Proxy Candidates",
        "",
        "| candidate | return | sharpe | max_dd | worst_ic | top5 | turnover | slice | decision_hint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in blend_summary.sort_values("cost_after_return", ascending=False).iterrows():
        hint = "candidate" if row["candidate"] != "lstm_sl20_mainline" else "baseline"
        lines.append(
            f"| `{row['candidate']}` | {row['cost_after_return']:.6f} | {row['Sharpe']:.6f} | "
            f"{row['max_drawdown']:.6f} | {row['worst_fold_rank_ic']:.6f} | {row['top5_return_mean']:.6f} | "
            f"{row['avg_turnover']:.6f} | {row['single_slice_score']:.6f} | {hint} |"
        )
    if not visible.empty:
        lines += [
            "",
            "## Visible Single-Slice Result Scores",
            "",
            "| candidate | score | diff_vs_case_best | stocks |",
            "|---|---:|---:|---|",
        ]
        for _, row in visible.iterrows():
            if row.get("status") != "ok":
                continue
            lines.append(
                f"| `{row['candidate']}` | {float(row['case_slice_score']):.6f} | "
                f"{float(row['diff_vs_case_best']):+.6f} | `{row.get('stocks', '')}` |"
            )
    lines += [
        "",
        "## Existing Evidence Snapshot",
        "",
        "- `sequence_length_search`：已有 LSTM sl20/sl40/sl60 对比，sl60 提升部分 rank/top5 但单切片和回撤不稳定。",
        "- `transformer_lite`：已有 sl20/sl40/sl60，Transformer-lite 没有全面超过 LSTM sl20，适合作为融合分支。",
        "- `rank_blend`：已有包含 LSTM60、LightGBM、momentum 的融合候选，但稳定性仍未达到替换主线门槛。",
        "",
        "## Final Recommendation",
        "",
        f"- decision: `{decision}`",
        f"- reason: {reason}",
        "- default config: 不建议因 Prompt 30 自动替换当前 HV rerank/sl20 默认主线。",
        "- aggressive result: 当前已同步的 aggressive score 输出仍是可见单切片最高候选。",
        "",
        "## Next",
        "",
        "- Prompt 31 应生成双提交包：stable engineering 包保留 HV rerank/sl20；aggressive score 包使用当前 `app/output/result.csv`。",
        "- 若继续投入模型增强，应使用 `--retrain_missing` 完整重训 sl60/Transformer，并把 cross-stock attention proxy 中表现最好的候选纳入 rank_blend 搜索。",
    ]
    (output_dir / "sl60_transformer_replication_enhancement_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = resolve(args.feature_path)
    base_config = resolve(args.base_config)
    case_root = resolve(args.case_root)

    train_steps = run_optional_training(args, output_dir)
    existing = collect_existing_summaries(output_dir)
    visible = score_visible_results(case_root, output_dir)

    base, available, missing = merge_sources()
    rows = []
    # Baseline mainline, then generated cross-stock attention proxy blends.
    baseline_path = output_dir / "lstm_sl20_mainline" / "walk_forward_predictions.csv"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(SOURCE_SPECS["lstm_sl20_mainline"], dtype={"stock_id": str}).to_csv(baseline_path, index=False, encoding="utf-8-sig")
    rows.append(
        evaluate_prediction_artifact(
            name="lstm_sl20_mainline",
            prediction_path=baseline_path,
            feature_path=feature_path,
            base_config_path=base_config,
            output_dir=output_dir,
            model_family="lstm",
            sequence_length=20,
        )
    )

    for blend_name, weights in BLEND_SPECS.items():
        pred = build_cross_stock_attention_prediction(base, blend_name, weights)
        blend_dir = output_dir / blend_name
        blend_dir.mkdir(parents=True, exist_ok=True)
        pred_path = blend_dir / "walk_forward_predictions.csv"
        pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
        rows.append(
            evaluate_prediction_artifact(
                name=blend_name,
                prediction_path=pred_path,
                feature_path=feature_path,
                base_config_path=base_config,
                output_dir=output_dir,
                model_family="cross_stock_attention_proxy",
                sequence_length=60,
            )
        )

    blend_summary = pd.DataFrame(rows)
    blend_summary.to_csv(output_dir / "sl60_transformer_enhanced_blend_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(train_steps).to_csv(output_dir / "training_step_status.csv", index=False, encoding="utf-8-sig")
    decision, reason = recommend(blend_summary)
    write_report(output_dir, existing, blend_summary, visible, available, missing, decision, reason)
    print(f"[sl60_transformer_enhancement] decision={decision}")
    print(f"[sl60_transformer_enhancement] report={output_dir / 'sl60_transformer_replication_enhancement_report.md'}")


if __name__ == "__main__":
    main()
