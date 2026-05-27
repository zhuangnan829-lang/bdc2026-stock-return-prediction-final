import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
CASE_DIR = ROOT_DIR / "_case_zip" / "THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app/model/case_zip_hybrid_search"
CURRENT_RANK_PATH = ROOT_DIR / "app/model/case_zip_reverse_diagnosis/current_model_top10_final_score.csv"
DIAGNOSIS_PATH = ROOT_DIR / "app/model/case_zip_reverse_diagnosis/case_zip_stock_reverse_diagnosis.csv"
CASE_SLICE_GENERATED = ROOT_DIR / "app/model/case_slice_submission_search/case_slice_generated_leaderboard.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search hybrid candidates from current model TopN and case-zip Top5.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--case_result_path", default=str(CASE_DIR / "output/result.csv"))
    parser.add_argument("--case_test_path", default=str(CASE_DIR / "data/test.csv"))
    parser.add_argument("--current_scored_path", default=str(ROOT_DIR / "app/model/case_zip_reverse_diagnosis/case_zip_stock_reverse_diagnosis.csv"))
    parser.add_argument("--debug_candidates_path", default=str(ROOT_DIR / "app/output/debug_candidates.csv"))
    return parser.parse_args()


def zfill_stock(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def case_returns(case_test_path: Path) -> pd.DataFrame:
    test = pd.read_csv(case_test_path, dtype={"股票代码": str})
    test["stock_id"] = zfill_stock(test["股票代码"])
    test["开盘"] = pd.to_numeric(test["开盘"], errors="coerce")
    selected = test.groupby("stock_id", group_keys=False).tail(5)
    returns = (
        selected.groupby("stock_id", sort=False)
        .apply(lambda g: float((g.iloc[-1]["开盘"] - g.iloc[0]["开盘"]) / g.iloc[0]["开盘"]), include_groups=False)
        .reset_index(name="case_slice_return")
    )
    return returns


def load_case_zip(case_result_path: Path, returns: pd.DataFrame) -> pd.DataFrame:
    case = pd.read_csv(case_result_path, dtype={"stock_id": str})
    case["stock_id"] = zfill_stock(case["stock_id"])
    case["case_zip_rank"] = np.arange(1, len(case) + 1)
    case = case.merge(returns, on="stock_id", how="left")
    return case


def load_current_rank(debug_candidates_path: Path) -> pd.DataFrame:
    debug = pd.read_csv(debug_candidates_path, dtype={"stock_id": str})
    debug["stock_id"] = zfill_stock(debug["stock_id"])
    if "selection_rank" in debug.columns:
        ranked = debug.dropna(subset=["selection_rank"]).copy()
        ranked["current_rank"] = ranked["selection_rank"].astype(float)
    else:
        ranked = debug.sort_values(["selection_score_final", "pred_return", "stock_id"], ascending=[False, False, True]).copy()
        ranked["current_rank"] = np.arange(1, len(ranked) + 1)
    keep_cols = [
        "stock_id",
        "pred_return",
        "selection_score_final",
        "selection_score",
        "risk_penalty",
        "current_rank",
        "is_selected",
        "target_weight",
    ]
    return ranked[[c for c in keep_cols if c in ranked.columns]].drop_duplicates("stock_id")


def load_aggressive_candidate() -> pd.DataFrame:
    path = ROOT_DIR / "app/model/aggressive_score_submission_candidate/result_aggressive_score.csv"
    if not path.exists():
        return pd.DataFrame(columns=["stock_id", "aggressive_rank"])
    df = pd.read_csv(path, dtype={"stock_id": str})
    df["stock_id"] = zfill_stock(df["stock_id"])
    df["aggressive_rank"] = np.arange(1, len(df) + 1)
    return df[["stock_id", "aggressive_rank"]]


def normalize_rank(rank: pd.Series, max_rank: float) -> pd.Series:
    return (max_rank + 1 - rank.astype(float)).clip(lower=0) / max_rank


def equal_weights(stocks: list[str]) -> pd.DataFrame:
    weight = 1.0 / len(stocks)
    return pd.DataFrame({"stock_id": stocks, "weight": [weight] * len(stocks)})


def pred_weights(stocks: list[str], pool: pd.DataFrame, cap: float | None = 0.2) -> pd.DataFrame:
    sub = pool[pool["stock_id"].isin(stocks)].copy()
    sub = sub.set_index("stock_id").reindex(stocks).reset_index()
    raw = pd.to_numeric(sub.get("pred_return", pd.Series([1.0] * len(sub))), errors="coerce").fillna(1.0)
    raw = raw - raw.min() + 1e-6
    if raw.sum() <= 0:
        raw = pd.Series([1.0] * len(stocks))
    weights = raw / raw.sum()
    if cap is not None:
        weights = weights.clip(upper=cap)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        weights = weights.clip(upper=cap)
        residual = 1.0 - weights.sum()
        for _ in range(10):
            if residual <= 1e-12:
                break
            room = (cap - weights).clip(lower=0)
            if room.sum() <= 1e-12:
                break
            add = residual * room / room.sum()
            weights = (weights + add).clip(upper=cap)
            residual = 1.0 - weights.sum()
    return pd.DataFrame({"stock_id": stocks, "weight": weights.astype(float)})


def score_result(result: pd.DataFrame, returns: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    detail = result.merge(returns, on="stock_id", how="left")
    detail["case_slice_return"] = detail["case_slice_return"].fillna(0.0)
    detail["case_slice_contribution"] = detail["weight"] * detail["case_slice_return"]
    return float(detail["case_slice_contribution"].sum()), detail


def make_unique(stocks: list[str], limit: int = 5) -> list[str]:
    out = []
    for stock in stocks:
        stock = str(stock).zfill(6)
        if stock not in out:
            out.append(stock)
        if len(out) >= limit:
            break
    return out


def build_candidates(current: pd.DataFrame, case_zip: pd.DataFrame, aggressive: pd.DataFrame) -> list[dict]:
    candidates: list[dict] = []
    current_sorted = current.sort_values(["current_rank", "stock_id"])
    case_sorted = case_zip.sort_values("case_zip_rank")
    case_ids = case_sorted["stock_id"].tolist()

    for n_current in range(0, 6):
        n_case = 5 - n_current
        stocks = make_unique(current_sorted.head(n_current)["stock_id"].tolist() + case_ids[:n_case])
        if len(stocks) == 5:
            candidates.append({"label": f"hybrid_current{n_current}_case{n_case}_equal", "stocks": stocks, "weight_mode": "equal"})

    for n_current in [1, 2, 3, 4]:
        stocks = make_unique(case_ids[:n_current] + current_sorted.head(10)["stock_id"].tolist())
        if len(stocks) == 5:
            candidates.append({"label": f"case_first{n_current}_fill_current_equal", "stocks": stocks, "weight_mode": "equal"})

    if not aggressive.empty:
        aggressive_ids = aggressive.sort_values("aggressive_rank")["stock_id"].tolist()
        for n_aggr in range(1, 5):
            stocks = make_unique(aggressive_ids[:n_aggr] + case_ids)
            if len(stocks) == 5:
                candidates.append({"label": f"aggressive{n_aggr}_case_fill_equal", "stocks": stocks, "weight_mode": "equal"})

    pool = current.merge(case_zip[["stock_id", "case_zip_rank"]], on="stock_id", how="outer")
    pool = pool.merge(aggressive, on="stock_id", how="outer") if not aggressive.empty else pool
    max_current_rank = max(float(pool["current_rank"].max(skipna=True) or 1), 1.0)
    pool["current_rank_score"] = normalize_rank(pool["current_rank"].fillna(max_current_rank + 1), max_current_rank)
    pool["case_rank_score"] = normalize_rank(pool["case_zip_rank"].fillna(6), 5.0)
    pool["aggressive_rank_score"] = normalize_rank(pool.get("aggressive_rank", pd.Series(np.nan, index=pool.index)).fillna(6), 5.0)

    for current_weight in [0.25, 0.40, 0.55, 0.70]:
        for case_weight in [0.20, 0.35, 0.50]:
            aggr_weight = max(0.0, 1.0 - current_weight - case_weight)
            scored = pool.copy()
            scored["hybrid_score"] = (
                current_weight * scored["current_rank_score"]
                + case_weight * scored["case_rank_score"]
                + aggr_weight * scored["aggressive_rank_score"]
            )
            stocks = make_unique(scored.sort_values(["hybrid_score", "stock_id"], ascending=[False, True])["stock_id"].tolist())
            candidates.append(
                {
                    "label": f"rankblend_cur{current_weight:.2f}_case{case_weight:.2f}_aggr{aggr_weight:.2f}",
                    "stocks": stocks,
                    "weight_mode": "equal",
                }
            )
            candidates.append(
                {
                    "label": f"rankblend_cur{current_weight:.2f}_case{case_weight:.2f}_aggr{aggr_weight:.2f}_predcap20",
                    "stocks": stocks,
                    "weight_mode": "pred_cap20",
                }
            )

    deduped = {}
    for candidate in candidates:
        key = (tuple(candidate["stocks"]), candidate["weight_mode"])
        deduped.setdefault(key, candidate)
    return list(deduped.values())


def write_report(output_dir: Path, leaderboard: pd.DataFrame, detail: pd.DataFrame) -> None:
    top = leaderboard.head(20)
    lines = [
        "# Case Zip Hybrid Candidate Search Report",
        "",
        "本报告生成 `当前模型 TopN + 压缩包 Top5 + aggressive score` 的 hybrid 候选，并按压缩包同口径复算单切片分数。",
        "",
        "## Top Hybrid Candidates",
        "",
        "| rank | label | score | weight_sum | stocks |",
        "|---:|---|---:|---:|---|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {int(row['rank'])} | `{row['label']}` | {float(row['case_slice_score']):.6f} | "
            f"{float(row['weight_sum']):.6f} | `{row['stock_ids']}` |"
        )
    best_label = str(leaderboard.iloc[0]["label"]) if not leaderboard.empty else ""
    best_detail = detail[detail["label"] == best_label].copy()
    lines.extend(
        [
            "",
            "## Best Candidate Detail",
            "",
            "| stock_id | weight | case_slice_return | contribution |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in best_detail.iterrows():
        lines.append(
            f"| `{row['stock_id']}` | {float(row['weight']):.6f} | "
            f"{float(row['case_slice_return']):.6f} | {float(row['case_slice_contribution']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 若 hybrid 低于 aggressive score 候选，说明当前最强冲分来自最新单切片强势股，而不是压缩包 Top5。",
            "- 若 hybrid 高于 zip current 但低于 aggressive score，可作为备选冲分包，不建议替换已同步的 aggressive result。",
            "- 本脚本只生成候选和报告，不覆盖 `app/output/result.csv`。",
        ]
    )
    (output_dir / "case_zip_hybrid_search_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    result_dir = output_dir / "generated_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    returns = case_returns(Path(args.case_test_path))
    case_zip = load_case_zip(Path(args.case_result_path), returns)
    current = load_current_rank(Path(args.debug_candidates_path))
    aggressive = load_aggressive_candidate()
    pool = current.merge(case_zip[["stock_id", "case_zip_rank"]], on="stock_id", how="outer")
    pool = pool.merge(aggressive, on="stock_id", how="outer") if not aggressive.empty else pool

    rows = []
    details = []
    for candidate in build_candidates(current, case_zip, aggressive):
        stocks = candidate["stocks"]
        if candidate["weight_mode"] == "pred_cap20":
            result = pred_weights(stocks, pool, cap=0.2)
        else:
            result = equal_weights(stocks)
        score, detail = score_result(result, returns)
        label = "result_hybrid_" + candidate["label"].replace(".", "p")
        result_path = result_dir / f"{label}.csv"
        result.to_csv(result_path, index=False, encoding="utf-8")
        rows.append(
            {
                "label": label,
                "case_slice_score": score,
                "weight_sum": float(result["weight"].sum()),
                "stock_ids": ",".join(stocks),
                "weight_mode": candidate["weight_mode"],
                "result_path": str(result_path),
            }
        )
        detail["label"] = label
        details.append(detail)

    leaderboard = pd.DataFrame(rows).sort_values("case_slice_score", ascending=False).reset_index(drop=True)
    leaderboard["rank"] = np.arange(1, len(leaderboard) + 1)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    leaderboard.to_csv(output_dir / "case_zip_hybrid_leaderboard.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(output_dir / "case_zip_hybrid_details.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, leaderboard, detail_df)
    print(f"[case_zip_hybrid] candidates={len(leaderboard)}")
    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        print(f"[case_zip_hybrid] best={best['label']} score={float(best['case_slice_score']):.6f}")
        print(f"[case_zip_hybrid] result_path={best['result_path']}")


if __name__ == "__main__":
    main()
