from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest import load_prediction_frame
from config import ROOT_DIR
from regime_switch_submission import (
    SUMMARY_COLUMNS,
    build_current_regime,
    build_simple_regimes,
    config_to_backtest_config,
    run_dynamic_backtest,
)


DEFAULT_PRED_PATH = ROOT_DIR / "app" / "model" / "walk_forward_predictions.csv"
DEFAULT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "train_features.csv"
DEFAULT_PREDICT_FEATURE_PATH = ROOT_DIR / "app" / "temp" / "predict_features.csv"
DEFAULT_MAINLINE_CONFIG = ROOT_DIR / "app" / "model" / "default_submission_config.json"
DEFAULT_AGGRESSIVE_CONFIG = ROOT_DIR / "app" / "model" / "configs" / "submission_aggressive.json"
DEFAULT_ROBUST_CONFIG = ROOT_DIR / "app" / "model" / "configs" / "submission_robust.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "high_volatility_range_optimization"


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def run_named_profile(
    prediction_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    output_dir: Path,
    *,
    output_name: str,
    engine_profile_name: str,
    aggressive_config: dict[str, Any],
    robust_config: dict[str, Any],
) -> dict[str, Any]:
    summary_df, daily_df, holdings_df = run_dynamic_backtest(
        prediction_df,
        regime_df,
        profile_name=engine_profile_name,
        aggressive_config=aggressive_config,
        robust_config=robust_config,
    )
    profile_dir = output_dir / output_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(profile_dir / "backtest_daily_results.csv", index=False, encoding="utf-8-sig")
    holdings_df.to_csv(profile_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    row = summary_df.iloc[0].to_dict()
    row["profile_name"] = output_name
    if output_name == "regime_switch":
        row["active_rule"] = "robust_if_high_volatility_range_else_aggressive"
    elif output_name == "mainline_static":
        row["active_rule"] = "current_default_submission_config"
    elif output_name == "aggressive_static":
        row["active_rule"] = "submission_aggressive_static"
    elif output_name == "robust_static":
        row["active_rule"] = "submission_robust_static"
    return row


def add_decision_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    mainline = out[out["profile_name"].eq("mainline_static")].iloc[0]
    aggressive = out[out["profile_name"].eq("aggressive_static")].iloc[0]
    robust = out[out["profile_name"].eq("robust_static")].iloc[0]
    switch = out[out["profile_name"].eq("regime_switch")].iloc[0]

    mainline_return = float(mainline["cost_after_return"])
    for idx, row in out.iterrows():
        return_loss_vs_mainline = mainline_return - float(row["cost_after_return"])
        hvr_drawdown_improved = float(row["high_vol_range_max_drawdown"]) >= float(mainline["high_vol_range_max_drawdown"])
        turnover_improved = float(row["avg_turnover"]) < float(mainline["avg_turnover"])
        hvr_return_improved = float(row["high_vol_range_cost_after_return"]) > float(mainline["high_vol_range_cost_after_return"])
        acceptable_return_loss = return_loss_vs_mainline <= 0.05
        recommended = bool(
            row["profile_name"] in {"robust_static", "regime_switch"}
            and (hvr_drawdown_improved or turnover_improved or hvr_return_improved)
            and acceptable_return_loss
        )
        out.loc[idx, "return_loss_vs_mainline"] = return_loss_vs_mainline
        out.loc[idx, "hvr_drawdown_improved_vs_mainline"] = hvr_drawdown_improved
        out.loc[idx, "turnover_improved_vs_mainline"] = turnover_improved
        out.loc[idx, "hvr_return_improved_vs_mainline"] = hvr_return_improved
        out.loc[idx, "recommended_for_next_step"] = recommended

    switch_return_loss = float(mainline["cost_after_return"]) - float(switch["cost_after_return"])
    switch_risk_better = (
        float(switch["avg_turnover"]) < float(mainline["avg_turnover"])
        or float(switch["high_vol_range_max_drawdown"]) >= float(mainline["high_vol_range_max_drawdown"])
    )
    out.attrs["final_decision"] = {
        "recommend_regime_switch": bool(switch_risk_better and switch_return_loss <= 0.05),
        "recommend_robust_observation": bool(float(robust["avg_turnover"]) < float(mainline["avg_turnover"])),
        "switch_return_loss_vs_mainline": switch_return_loss,
        "aggressive_return_delta_vs_mainline": float(aggressive["cost_after_return"]) - float(mainline["cost_after_return"]),
    }
    return out


def write_report(summary: pd.DataFrame, current_regime: dict[str, Any], output_dir: Path) -> None:
    decision = summary.attrs.get("final_decision", {})
    lines = [
        "# High Volatility Range Optimization Report",
        "",
        "目标：只针对高波动震荡阶段做专项优化，不推翻 LSTM sl20 主线。",
        "",
        "## Current Regime Decision",
        "",
        f"- latest_date: `{current_regime.get('latest_date', '')}`",
        f"- latest_regime: `{current_regime.get('latest_regime', '')}`",
        f"- selected_config_by_rule: `{current_regime.get('selected_config', '')}`",
        f"- source: `{current_regime.get('source', '')}`",
        "",
        "## Profile Comparison",
        "",
        "| profile | rule | total_return | sharpe | max_dd | avg_turnover | hvr_return | hvr_max_dd | hvr_avg_return | return_loss_vs_mainline | next_step |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['profile_name']} | {row['active_rule']} | {fmt(row['cost_after_return'])} | "
            f"{fmt(row['Sharpe'])} | {fmt(row['max_drawdown'])} | {fmt(row['avg_turnover'])} | "
            f"{fmt(row['high_vol_range_cost_after_return'])} | {fmt(row['high_vol_range_max_drawdown'])} | "
            f"{fmt(row['high_vol_range_avg_return'])} | {fmt(row['return_loss_vs_mainline'])} | "
            f"{'candidate' if bool(row['recommended_for_next_step']) else 'hold'} |"
        )

    if decision.get("recommend_regime_switch"):
        final = "建议进入下一步：把 regime_switch 作为候选继续做更细的阈值网格和不利阶段复核。"
    elif decision.get("recommend_robust_observation"):
        final = "暂不采用 regime_switch；robust_static 可作为风险观察配置，但收益损失需要继续压缩。"
    else:
        final = "暂不推进切换策略；保留 sl20 主线，继续寻找更低收益损失的防御规则。"

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- recommend_regime_switch: `{str(bool(decision.get('recommend_regime_switch'))).lower()}`",
            f"- recommend_robust_observation: `{str(bool(decision.get('recommend_robust_observation'))).lower()}`",
            f"- switch_return_loss_vs_mainline: `{fmt(decision.get('switch_return_loss_vs_mainline'))}`",
            f"- aggressive_return_delta_vs_mainline: `{fmt(decision.get('aggressive_return_delta_vs_mainline'))}`",
            f"- final: {final}",
            "",
            "## Next Step",
            "",
            "如果继续推进，下一步应该做阈值网格：volatility quantile 0.60/0.65/0.70/0.75，range quantile 0.25/0.30/0.35/0.40，并要求收益损失不超过 0.05、换手下降、HVR 回撤不恶化。",
            "",
        ]
    )
    (output_dir / "high_volatility_range_optimization_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run high-volatility range targeted optimization.")
    parser.add_argument("--pred_path", default=str(DEFAULT_PRED_PATH))
    parser.add_argument("--feature_path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--predict_feature_path", default=str(DEFAULT_PREDICT_FEATURE_PATH))
    parser.add_argument("--mainline_config", default=str(DEFAULT_MAINLINE_CONFIG))
    parser.add_argument("--aggressive_config", default=str(DEFAULT_AGGRESSIVE_CONFIG))
    parser.add_argument("--robust_config", default=str(DEFAULT_ROBUST_CONFIG))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--volatility_quantile", type=float, default=0.70)
    parser.add_argument("--range_quantile", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = resolve_path(args.feature_path)
    pred_path = resolve_path(args.pred_path)
    regime_df = build_simple_regimes(
        feature_path,
        volatility_quantile=float(args.volatility_quantile),
        range_quantile=float(args.range_quantile),
        output_path=output_dir / "daily_high_volatility_range_regimes.csv",
    )
    current_regime = build_current_regime(
        feature_path,
        resolve_path(args.predict_feature_path),
        volatility_quantile=float(args.volatility_quantile),
        range_quantile=float(args.range_quantile),
    )
    prediction_df = load_prediction_frame(pred_path, feature_path)

    mainline_config = config_to_backtest_config(resolve_path(args.mainline_config), "mainline_static")
    aggressive_config = config_to_backtest_config(resolve_path(args.aggressive_config), "aggressive_static")
    robust_config = config_to_backtest_config(resolve_path(args.robust_config), "robust_static")

    rows = [
        run_named_profile(
            prediction_df,
            regime_df,
            output_dir,
            output_name="mainline_static",
            engine_profile_name="aggressive_static",
            aggressive_config=mainline_config,
            robust_config=robust_config,
        ),
        run_named_profile(
            prediction_df,
            regime_df,
            output_dir,
            output_name="aggressive_static",
            engine_profile_name="aggressive_static",
            aggressive_config=aggressive_config,
            robust_config=robust_config,
        ),
        run_named_profile(
            prediction_df,
            regime_df,
            output_dir,
            output_name="robust_static",
            engine_profile_name="robust_static",
            aggressive_config=aggressive_config,
            robust_config=robust_config,
        ),
        run_named_profile(
            prediction_df,
            regime_df,
            output_dir,
            output_name="regime_switch",
            engine_profile_name="regime_switch",
            aggressive_config=aggressive_config,
            robust_config=robust_config,
        ),
    ]

    summary = add_decision_columns(pd.DataFrame(rows))
    output_columns = [
        *SUMMARY_COLUMNS,
        "return_loss_vs_mainline",
        "hvr_drawdown_improved_vs_mainline",
        "turnover_improved_vs_mainline",
        "hvr_return_improved_vs_mainline",
        "recommended_for_next_step",
    ]
    for idx, row in summary.iterrows():
        summary.loc[idx, "latest_regime"] = current_regime.get("latest_regime", "")
        summary.loc[idx, "latest_selected_config"] = current_regime.get("selected_config", "")
        summary.loc[idx, "adopted"] = bool(row["recommended_for_next_step"])
        summary.loc[idx, "notes"] = "high_volatility_range_targeted_optimization"
    summary[output_columns].to_csv(output_dir / "high_volatility_range_optimization_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "current_regime_decision.json").write_text(
        json.dumps(current_regime, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(summary[output_columns], current_regime, output_dir)
    print(summary[["profile_name", "cost_after_return", "max_drawdown", "avg_turnover", "high_vol_range_cost_after_return", "high_vol_range_max_drawdown", "recommended_for_next_step"]].to_string(index=False))
    print(f"[hvr_optimization] wrote {output_dir / 'high_volatility_range_optimization_summary.csv'}")
    print(f"[hvr_optimization] wrote {output_dir / 'high_volatility_range_optimization_report.md'}")


if __name__ == "__main__":
    main()
