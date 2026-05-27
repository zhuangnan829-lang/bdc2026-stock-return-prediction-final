import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
CASE_DIR = ROOT_DIR / "_case_zip" / "THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0"
DEFAULT_CONFIG = ROOT_DIR / "app/model/default_submission_config.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app/model/case_zip_reverse_diagnosis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reverse-diagnose why case-zip high-score stocks were missed.")
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--case_result_path", default=str(CASE_DIR / "output/result.csv"))
    parser.add_argument("--case_test_path", default=str(CASE_DIR / "data/test.csv"))
    parser.add_argument("--predict_scores_path", default=str(ROOT_DIR / "app/output/predict_scores.csv"))
    parser.add_argument("--feature_path", default=str(ROOT_DIR / "app/temp/train_features.csv"))
    return parser.parse_args()


def zfill_stock(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_prediction_frame(feature_path: Path, predict_scores_path: Path) -> pd.DataFrame:
    features = pd.read_csv(feature_path, dtype={"stock_id": str})
    features["stock_id"] = zfill_stock(features["stock_id"])
    features["date"] = pd.to_datetime(features["date"])
    latest_date = features["date"].max()
    latest_features = features[features["date"] == latest_date].copy()

    scores = pd.read_csv(predict_scores_path, dtype={"stock_id": str})
    scores["stock_id"] = zfill_stock(scores["stock_id"])
    scores["date"] = pd.to_datetime(scores["date"])
    latest_score_date = scores["date"].max()
    latest_scores = scores[scores["date"] == latest_score_date][["stock_id", "pred_return"]].copy()

    merged = latest_features.merge(latest_scores, on="stock_id", how="inner")
    merged["prediction_date"] = latest_score_date
    return merged


def case_result_detail(case_result_path: Path, case_test_path: Path) -> tuple[pd.DataFrame, float]:
    result = pd.read_csv(case_result_path, dtype={"stock_id": str})
    result["stock_id"] = zfill_stock(result["stock_id"])
    result["weight"] = pd.to_numeric(result["weight"], errors="coerce")

    test = pd.read_csv(case_test_path, dtype={"股票代码": str})
    test["stock_id"] = zfill_stock(test["股票代码"])
    test["开盘"] = pd.to_numeric(test["开盘"], errors="coerce")
    selected = test[test["stock_id"].isin(result["stock_id"])].copy()
    selected = selected.groupby("stock_id", group_keys=False).tail(5)
    returns = (
        selected.groupby("stock_id", sort=False)
        .apply(lambda g: float((g.iloc[-1]["开盘"] - g.iloc[0]["开盘"]) / g.iloc[0]["开盘"]), include_groups=False)
        .reset_index(name="case_slice_return")
    )
    detail = result.merge(returns, on="stock_id", how="left")
    detail["case_slice_contribution"] = detail["weight"] * detail["case_slice_return"]
    return detail, float(detail["case_slice_contribution"].sum())


def risk_filter_reason(row: pd.Series, thresholds: dict) -> str:
    reasons = []
    if row["volatility_20d"] > thresholds["vol20_threshold"]:
        reasons.append("volatility_20d_above_threshold")
    if row["volatility_5d"] > thresholds["vol5_threshold"]:
        reasons.append("volatility_5d_above_threshold")
    if row["turnover_rate"] < thresholds["turnover_low"]:
        reasons.append("turnover_rate_below_threshold")
    if row["turnover_rate"] > thresholds["turnover_high"]:
        reasons.append("turnover_rate_above_threshold")
    if row["turnover_ratio_10d"] > thresholds["turnover_ratio_high"]:
        reasons.append("turnover_ratio_10d_above_threshold")
    return ";".join(reasons) if reasons else "pass"


def score_current_model(latest: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    selection = config["selection_logic"]
    risk = config["risk_filter_thresholds"]
    top_k = int(selection["top_k"])
    primary_size = int(selection["primary_candidate_size"])

    full = latest.copy()
    full["pred_rank_all"] = full["pred_return"].rank(ascending=False, method="min").astype(int)
    full["pred_rank_pct_all"] = full["pred_return"].rank(pct=True)

    primary = full.sort_values(["pred_return", "stock_id"], ascending=[False, True]).head(primary_size).copy()
    thresholds = {
        "vol20_threshold": float(full["volatility_20d"].quantile(float(risk["max_volatility_20d_pct"]))),
        "vol5_threshold": float(full["volatility_5d"].quantile(float(risk["max_volatility_5d_pct"]))),
        "turnover_low": float(full["turnover_rate"].quantile(float(risk["turnover_rate_lower_pct"]))),
        "turnover_high": float(full["turnover_rate"].quantile(float(risk["turnover_rate_upper_pct"]))),
        "turnover_ratio_high": float(full["turnover_ratio_10d"].quantile(float(risk["turnover_ratio_upper_pct"]))),
    }
    primary["risk_filter_reason"] = primary.apply(lambda row: risk_filter_reason(row, thresholds), axis=1)
    filtered = primary[primary["risk_filter_reason"] == "pass"].copy()

    filtered["pred_rank_pct"] = filtered["pred_return"].rank(pct=True)
    filtered["risk_vol20_pct"] = filtered["volatility_20d"].rank(pct=True)
    filtered["risk_vol5_pct"] = filtered["volatility_5d"].rank(pct=True)
    filtered["risk_turnover_pct"] = filtered["turnover_ratio_10d"].rank(pct=True)
    filtered["risk_amplitude_pct"] = filtered["amplitude_ratio_5d"].rank(pct=True)
    filtered["risk_penalty"] = (
        0.4 * filtered["risk_vol20_pct"]
        + 0.25 * filtered["risk_vol5_pct"]
        + 0.2 * filtered["risk_turnover_pct"]
        + 0.15 * filtered["risk_amplitude_pct"]
    )
    risk_penalty_weight = float(risk["risk_penalty_weight"])
    filtered["selection_score"] = filtered["pred_rank_pct"] - risk_penalty_weight * filtered["risk_penalty"]

    regime_rerank = config.get("regime_rerank", {})
    signal = regime_rerank.get("signal") if regime_rerank.get("enabled") else None
    signal_weight = float(regime_rerank.get("weight", 0.0) or 0.0)
    if signal and signal in filtered.columns and abs(signal_weight) > 1e-12:
        filtered["rerank_signal_score"] = filtered[signal].rank(pct=True) - 0.5
        filtered["selection_score_final"] = filtered["selection_score"] + signal_weight * filtered["rerank_signal_score"]
    else:
        filtered["rerank_signal_score"] = 0.0
        filtered["selection_score_final"] = filtered["selection_score"]

    filtered = filtered.sort_values(
        ["selection_score_final", "selection_score", "pred_return", "stock_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    filtered["selection_rank"] = np.arange(1, len(filtered) + 1)
    filtered["is_selected_by_current_model"] = filtered["selection_rank"] <= top_k

    scored = full.merge(
        primary[["stock_id", "risk_filter_reason"]],
        on="stock_id",
        how="left",
    )
    scored["in_primary_topn"] = scored["risk_filter_reason"].notna()
    scored["risk_filter_reason"] = scored["risk_filter_reason"].fillna("not_in_primary_topn")
    score_cols = [
        "stock_id",
        "pred_rank_pct",
        "risk_penalty",
        "selection_score",
        "rerank_signal_score",
        "selection_score_final",
        "selection_rank",
        "is_selected_by_current_model",
        "risk_vol20_pct",
        "risk_vol5_pct",
        "risk_turnover_pct",
        "risk_amplitude_pct",
    ]
    scored = scored.merge(filtered[score_cols], on="stock_id", how="left")
    scored["passed_risk_filter"] = scored["risk_filter_reason"].eq("pass")
    scored["is_selected_by_current_model"] = scored["is_selected_by_current_model"].fillna(False).astype(bool)
    diagnostics = {
        **thresholds,
        "top_k": top_k,
        "primary_candidate_size": primary_size,
        "after_primary_screen": int(len(primary)),
        "after_risk_filters": int(len(filtered)),
        "risk_penalty_weight": risk_penalty_weight,
        "rerank_signal": signal or "",
        "rerank_signal_weight": signal_weight,
    }
    return scored, diagnostics


def find_blockers(scored: pd.DataFrame, case_ids: list[str], top_k: int) -> pd.DataFrame:
    selected_or_ranked = scored[scored["passed_risk_filter"]].copy()
    selected_or_ranked = selected_or_ranked.sort_values("selection_rank")
    rows = []
    top_rows = selected_or_ranked.head(max(top_k, 10))
    for stock_id in case_ids:
        target = scored[scored["stock_id"] == stock_id]
        if target.empty:
            continue
        target_rank = target.iloc[0].get("selection_rank", np.nan)
        if pd.isna(target_rank):
            blockers = top_rows.head(top_k)
            blocker_type = "not_ranked_after_filter"
        else:
            blockers = selected_or_ranked[selected_or_ranked["selection_rank"] < target_rank].head(10)
            blocker_type = "ranked_below_blockers"
        for _, row in blockers.iterrows():
            rows.append(
                {
                    "case_stock_id": stock_id,
                    "blocker_type": blocker_type,
                    "blocker_stock_id": row["stock_id"],
                    "blocker_selection_rank": row.get("selection_rank"),
                    "blocker_pred_return": row.get("pred_return"),
                    "blocker_selection_score_final": row.get("selection_score_final"),
                    "blocker_risk_penalty": row.get("risk_penalty"),
                    "blocker_is_selected": bool(row.get("is_selected_by_current_model", False)),
                }
            )
    return pd.DataFrame(rows)


def decision_reason(row: pd.Series) -> str:
    if not row.get("in_primary_topn", False):
        return "预测分排名未进入当前 TopN 初筛池。"
    if not row.get("passed_risk_filter", False):
        return f"进入初筛池，但被风控过滤：{row.get('risk_filter_reason', '')}。"
    rank = row.get("selection_rank")
    if pd.isna(rank):
        return "进入初筛池但没有获得最终排序分，需检查特征缺失。"
    if int(rank) <= 5:
        return "当前模型本应选中；若结果未出现，可能是后续人工/候选包覆盖。"
    return f"通过风控，但最终 risk_adjusted 排名第 {int(rank)}，被排名更高的候选挤掉。"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.6f}" if not pd.isna(val) else "")
            else:
                vals.append(f"`{val}`" if "stock" in col or col in {"risk_filter_reason", "miss_reason"} else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_report(
    path: Path,
    case_score: float,
    diagnosis: pd.DataFrame,
    blockers: pd.DataFrame,
    top_current: pd.DataFrame,
    diagnostics: dict,
) -> None:
    missed = diagnosis[~diagnosis["is_selected_by_current_model"]]
    lines = [
        "# Case Zip Stock Reverse Diagnosis",
        "",
        "本报告分析压缩包 5 只股票为什么在可见单切片得分高，以及它们为什么没有进入当前模型最终结果。",
        "",
        "## Summary",
        "",
        f"- case zip visible score: `{case_score:.6f}`",
        f"- case zip stocks: `{','.join(diagnosis['stock_id'].tolist())}`",
        f"- current primary candidate size: `{diagnostics['primary_candidate_size']}`",
        f"- after risk filters: `{diagnostics['after_risk_filters']}`",
        f"- rerank signal: `{diagnostics['rerank_signal']}` weight `{diagnostics['rerank_signal_weight']}`",
        f"- missed by current model: `{len(missed)}/{len(diagnosis)}`",
        "",
        "## Per-Stock Diagnosis",
        "",
    ]
    table_cols = [
        "stock_id",
        "weight",
        "case_slice_return",
        "case_slice_contribution",
        "pred_return",
        "pred_rank_all",
        "in_primary_topn",
        "risk_filter_reason",
        "selection_score_final",
        "selection_rank",
        "miss_reason",
    ]
    lines.extend(markdown_table(diagnosis[table_cols], table_cols))
    lines.extend(
        [
            "",
            "## Current Model Top 10 By Final Score",
            "",
        ]
    )
    top_cols = ["stock_id", "pred_return", "pred_rank_all", "selection_score_final", "selection_rank", "risk_penalty"]
    lines.extend(markdown_table(top_current[top_cols], top_cols))
    if not blockers.empty:
        lines.extend(
            [
                "",
                "## Who Pushed Them Out",
                "",
            ]
        )
        blocker_cols = [
            "case_stock_id",
            "blocker_stock_id",
            "blocker_selection_rank",
            "blocker_pred_return",
            "blocker_selection_score_final",
            "blocker_is_selected",
        ]
        lines.extend(markdown_table(blockers.head(50), blocker_cols))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 如果压缩包股票未进入 TopN 初筛，说明当前模型原始预测分不足，Prompt 29 需要做 `当前 TopN + 压缩包 Top5` 补入。",
            "- 如果通过风控但排名靠后，说明主要问题是排序目标/风险重排，不是硬过滤。",
            "- 如果被风控过滤，Prompt 29 应测试“压缩包候选补入但保留权重 cap”的 hybrid，而不是完全放开风控。",
            "- 本报告只做诊断，不修改 `app/output/result.csv` 或默认配置。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config_path))
    latest = latest_prediction_frame(Path(args.feature_path), Path(args.predict_scores_path))
    scored, diagnostics = score_current_model(latest, config)
    case_detail, case_score = case_result_detail(Path(args.case_result_path), Path(args.case_test_path))
    case_ids = case_detail["stock_id"].tolist()

    diagnosis = case_detail.merge(scored, on="stock_id", how="left")
    diagnosis["miss_reason"] = diagnosis.apply(decision_reason, axis=1)
    blockers = find_blockers(scored, case_ids, int(diagnostics["top_k"]))
    top_current = scored[scored["passed_risk_filter"]].sort_values("selection_rank").head(10).copy()

    compact_cols = [
        "stock_id",
        "weight",
        "case_slice_return",
        "case_slice_contribution",
        "pred_return",
        "pred_rank_all",
        "pred_rank_pct_all",
        "in_primary_topn",
        "passed_risk_filter",
        "risk_filter_reason",
        "risk_penalty",
        "selection_score",
        "rerank_signal_score",
        "selection_score_final",
        "selection_rank",
        "is_selected_by_current_model",
        "volatility_20d",
        "volatility_5d",
        "turnover_rate",
        "turnover_ratio_10d",
        "close_position_20d",
        "miss_reason",
    ]
    diagnosis[compact_cols].to_csv(output_dir / "case_zip_stock_reverse_diagnosis.csv", index=False, encoding="utf-8-sig")
    blockers.to_csv(output_dir / "case_zip_stock_blockers.csv", index=False, encoding="utf-8-sig")
    top_current.to_csv(output_dir / "current_model_top10_final_score.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([diagnostics | {"case_zip_score": case_score}]).to_csv(
        output_dir / "case_zip_reverse_diagnosis_summary.csv", index=False, encoding="utf-8-sig"
    )
    write_report(
        output_dir / "case_zip_reverse_diagnosis_report.md",
        case_score,
        diagnosis,
        blockers,
        top_current,
        diagnostics,
    )
    print(f"[case_zip_diagnosis] case_zip_score={case_score:.6f}")
    print(f"[case_zip_diagnosis] wrote {output_dir / 'case_zip_reverse_diagnosis_report.md'}")


if __name__ == "__main__":
    main()
