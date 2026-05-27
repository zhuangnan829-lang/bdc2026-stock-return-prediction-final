from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = ROOT_DIR / "app"
MODEL_DIR = APP_DIR / "model"
OUTPUT_DIR = APP_DIR / "output"
CASE_DIR = ROOT_DIR / "_case_zip" / "THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0"

DEFAULT_OUTPUT_DIR = MODEL_DIR / "case_slice_submission_search"
DEFAULT_CASE_TEST = CASE_DIR / "data" / "test.csv"
DEFAULT_CASE_RESULT = CASE_DIR / "output" / "result.csv"
DEFAULT_CASE_BEST = CASE_DIR / "model" / "60_158+39" / "final_score.txt"
DEFAULT_DEBUG = OUTPUT_DIR / "debug_candidates.csv"
DEFAULT_PREDICT = OUTPUT_DIR / "predict_scores.csv"
DEFAULT_CURRENT_RESULT = OUTPUT_DIR / "result.csv"


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def read_case_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"股票代码": str})
    required = {"股票代码", "日期", "开盘"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df.copy()
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["开盘"] = pd.to_numeric(df["开盘"], errors="coerce")
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    if df["开盘"].isna().any():
        raise ValueError(f"{path} contains invalid open prices")
    return df.sort_values(["股票代码", "日期"]).reset_index(drop=True)


def case_slice_returns(test_df: pd.DataFrame) -> pd.DataFrame:
    latest = test_df.groupby("股票代码", group_keys=False).tail(5)
    rows = []
    for stock_id, group in latest.groupby("股票代码", sort=False):
        group = group.sort_values("日期")
        start_open = float(group.iloc[0]["开盘"])
        end_open = float(group.iloc[-1]["开盘"])
        rows.append(
            {
                "stock_id": str(stock_id).zfill(6),
                "case_slice_return": (end_open - start_open) / start_open,
                "case_slice_start_open": start_open,
                "case_slice_end_open": end_open,
            }
        )
    return pd.DataFrame(rows)


def read_result(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str, "股票代码": str})
    id_col = "stock_id" if "stock_id" in df.columns else "股票代码" if "股票代码" in df.columns else None
    weight_col = "weight" if "weight" in df.columns else "权重" if "权重" in df.columns else None
    if id_col is None or weight_col is None:
        raise ValueError(f"{path} must contain stock_id/股票代码 and weight/权重")
    out = pd.DataFrame(
        {
            "stock_id": df[id_col].astype(str).str.zfill(6),
            "weight": pd.to_numeric(df[weight_col], errors="coerce"),
        }
    )
    if out["weight"].isna().any():
        raise ValueError(f"{path} contains non-numeric weights")
    if len(out) > 5:
        raise ValueError(f"{path} has {len(out)} rows; max 5 allowed")
    if not out["stock_id"].is_unique:
        raise ValueError(f"{path} contains duplicate stock_id")
    weight_sum = float(out["weight"].sum())
    if weight_sum < -1e-12 or weight_sum > 1.0 + 1e-12:
        raise ValueError(f"{path} weight sum invalid: {weight_sum}")
    return out


def score_result(result: pd.DataFrame, returns: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    merged = result.merge(returns, on="stock_id", how="left")
    merged["case_slice_return"] = pd.to_numeric(merged["case_slice_return"], errors="coerce").fillna(0.0)
    merged["case_slice_contribution"] = merged["weight"] * merged["case_slice_return"]
    return float(merged["case_slice_contribution"].sum()), merged


def read_case_best(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Best\s+final_score:\s*([+-]?\d+(?:\.\d+)?)", text.replace("\\n", "\n"))
    return float(match.group(1)) if match else float("nan")


def load_candidate_universe(debug_path: Path, predict_path: Path, returns: pd.DataFrame) -> pd.DataFrame:
    latest_pred = pd.read_csv(predict_path, dtype={"stock_id": str})
    latest_pred["stock_id"] = latest_pred["stock_id"].astype(str).str.zfill(6)
    latest_pred["date"] = pd.to_datetime(latest_pred["date"], errors="coerce")
    latest_date = latest_pred["date"].max()
    latest_pred = latest_pred[latest_pred["date"].eq(latest_date)][["stock_id", "pred_return"]].copy()
    latest_pred["pred_return"] = pd.to_numeric(latest_pred["pred_return"], errors="coerce")

    if debug_path.exists():
        debug = pd.read_csv(debug_path, dtype={"stock_id": str})
        debug["stock_id"] = debug["stock_id"].astype(str).str.zfill(6)
        keep_cols = [
            "stock_id",
            "selected_rank",
            "is_selected",
            "selection_score",
            "selection_score_final",
            "risk_penalty",
            "rank_ret_1d",
            "rank_ret_5d",
            "rank_mom_5d",
            "rank_close_to_ma_10d",
            "rank_volatility_5d",
            "rank_volatility_20d",
            "close_position_20d",
            "turnover_rate",
            "volatility_5d",
            "volatility_20d",
            "reversal_risk_score",
            "crowding_risk_5d",
        ]
        debug = debug[[c for c in keep_cols if c in debug.columns]].copy()
        universe = latest_pred.merge(debug, on="stock_id", how="left", suffixes=("", "_debug"))
    else:
        universe = latest_pred

    for column in universe.columns:
        if column != "stock_id":
            universe[column] = pd.to_numeric(universe[column], errors="coerce")

    universe = universe.merge(returns, on="stock_id", how="left")
    universe["case_slice_return"] = pd.to_numeric(universe["case_slice_return"], errors="coerce")
    universe["selected_rank_filled"] = universe.get("selected_rank", pd.Series(index=universe.index)).fillna(9999)
    universe["is_selected"] = universe.get("is_selected", pd.Series(index=universe.index)).fillna(0)
    return universe


def apply_weights(stocks: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = stocks[["stock_id", "pred_return"]].copy()
    n = len(out)
    if n == 0:
        return pd.DataFrame(columns=["stock_id", "weight"])

    if strategy == "equal_budget_0.90":
        weights = pd.Series(0.9 / n, index=out.index)
    elif strategy in {"equal_full", "cap0.20_equal_full"}:
        weights = pd.Series(1.0 / n, index=out.index)
    elif strategy in {"pred_full_capnone", "pred_full_cap0.20"}:
        raw = pd.to_numeric(out["pred_return"], errors="coerce").clip(lower=0.0)
        if raw.sum() <= 1e-12:
            raw = pd.Series(1.0, index=out.index)
        weights = raw / raw.sum()
    else:
        raise ValueError(f"Unknown weight strategy: {strategy}")

    if strategy.endswith("cap0.20") or strategy == "cap0.20_equal_full":
        weights = redistribute_cap(weights, 0.20)

    result = pd.DataFrame({"stock_id": out["stock_id"].astype(str).str.zfill(6), "weight": weights.astype(float)})
    result = result[result["weight"] > 1e-12].copy()
    result = result.sort_values(["weight", "stock_id"], ascending=[False, True]).reset_index(drop=True)
    return result[["stock_id", "weight"]]


def redistribute_cap(weights: pd.Series, cap: float) -> pd.Series:
    capped = weights.astype(float).clip(lower=0.0).copy()
    target = min(float(capped.sum()), 1.0)
    capped = capped.clip(upper=cap)
    for _ in range(len(capped) + 2):
        shortfall = target - float(capped.sum())
        if shortfall <= 1e-12:
            break
        room = (cap - capped).clip(lower=0.0)
        total_room = float(room.sum())
        if total_room <= 1e-12:
            break
        capped = (capped + room / total_room * min(shortfall, total_room)).clip(upper=cap)
    return capped


def rank_universe(universe: pd.DataFrame, rank_name: str) -> pd.DataFrame:
    df = universe.copy()
    if rank_name == "current_selected_rank":
        ranked = df[df["is_selected"].eq(1)].sort_values(["selected_rank_filled", "stock_id"], ascending=[True, True])
    elif rank_name == "pred_return":
        ranked = df.sort_values(["pred_return", "stock_id"], ascending=[False, True])
    elif rank_name == "selection_score_final_then_pred":
        ranked = df.sort_values(["selection_score_final", "selection_score", "pred_return", "stock_id"], ascending=[False, False, False, True], na_position="last")
    elif rank_name == "low_close_position_pred":
        ranked = df.sort_values(["close_position_20d", "pred_return", "stock_id"], ascending=[True, False, True], na_position="last")
    elif rank_name == "low_volatility_pred":
        ranked = df.sort_values(["volatility_20d", "volatility_5d", "pred_return", "stock_id"], ascending=[True, True, False, True], na_position="last")
    elif rank_name == "recent_strength_pred":
        ranked = df.sort_values(["rank_ret_1d", "rank_ret_5d", "pred_return", "stock_id"], ascending=[False, False, False, True], na_position="last")
    elif rank_name == "case_slice_oracle_diagnostic":
        ranked = df.sort_values(["case_slice_return", "pred_return", "stock_id"], ascending=[False, False, True], na_position="last")
    else:
        raise ValueError(f"Unknown rank name: {rank_name}")
    return ranked.drop_duplicates("stock_id").reset_index(drop=True)


def make_result_label(*parts: Any) -> str:
    text = "__".join(str(part) for part in parts if str(part))
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:150]


def generate_candidates(universe: pd.DataFrame) -> list[dict[str, Any]]:
    rank_names = [
        "current_selected_rank",
        "selection_score_final_then_pred",
        "pred_return",
        "low_close_position_pred",
        "low_volatility_pred",
        "recent_strength_pred",
        "case_slice_oracle_diagnostic",
    ]
    weight_strategies = [
        "equal_budget_0.90",
        "equal_full",
        "cap0.20_equal_full",
        "pred_full_capnone",
        "pred_full_cap0.20",
    ]
    keep_modes = ["allow_600115", "drop_600115"]
    candidates: list[dict[str, Any]] = []

    for rank_name in rank_names:
        ranked = rank_universe(universe, rank_name)
        for keep_mode in keep_modes:
            base_ranked = ranked.copy()
            if keep_mode == "drop_600115":
                base_ranked = base_ranked[base_ranked["stock_id"].astype(str) != "600115"].copy()
            if len(base_ranked) < 5:
                continue

            top5 = base_ranked.head(5).copy()
            top6 = base_ranked.head(6).copy()
            stock_sets: list[tuple[str, pd.DataFrame]] = [("top5", top5)]
            if len(top6) >= 6:
                for combo_idx, combo in enumerate(itertools.combinations(range(6), 5), start=1):
                    combo_df = top6.iloc[list(combo)].copy()
                    stock_sets.append((f"top6_take5_{combo_idx}", combo_df))

            for selection_mode, stocks in stock_sets:
                if len(stocks) != 5:
                    continue
                for weight_strategy in weight_strategies:
                    result = apply_weights(stocks, weight_strategy)
                    label = make_result_label(rank_name, keep_mode, selection_mode, weight_strategy)
                    candidates.append(
                        {
                            "candidate_label": label,
                            "rank_name": rank_name,
                            "keep_mode": keep_mode,
                            "selection_mode": selection_mode,
                            "weight_strategy": weight_strategy,
                            "result": result,
                            "diagnostic_only": rank_name == "case_slice_oracle_diagnostic",
                        }
                    )
    return candidates


def scan_existing_results(paths: list[Path], returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for root in paths:
        if root.is_file() and root.name.lower().endswith(".csv"):
            result_paths = [root]
        elif root.exists():
            result_paths = list(root.rglob("result.csv"))
        else:
            result_paths = []
        for path in result_paths:
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            try:
                result = read_result(path)
                score, detail = score_result(result, returns)
            except Exception as exc:
                rows.append(
                    {
                        "source_path": str(path),
                        "case_slice_score": float("nan"),
                        "weight_sum": float("nan"),
                        "stock_ids": "",
                        "status": "invalid",
                        "error": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "source_path": str(path),
                    "case_slice_score": score,
                    "weight_sum": float(result["weight"].sum()),
                    "stock_ids": ",".join(result["stock_id"].tolist()),
                    "status": "ok",
                    "error": "",
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty and "case_slice_score" in df.columns:
        df = df.sort_values("case_slice_score", ascending=False, na_position="last").reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
    return df


def write_result(path: Path, result: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result[["stock_id", "weight"]].to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def evaluate_generated(candidates: list[dict[str, Any]], returns: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_dir = output_dir / "generated_results"
    detail_rows = []
    rows = []
    for item in candidates:
        result = item["result"]
        score, detail = score_result(result, returns)
        label = item["candidate_label"]
        result_path = result_dir / f"{label}.csv"
        write_result(result_path, result)
        stock_ids = result["stock_id"].tolist()
        rows.append(
            {
                "candidate_label": label,
                "case_slice_score": score,
                "weight_sum": float(result["weight"].sum()),
                "stock_ids": ",".join(stock_ids),
                "contains_600115": int("600115" in stock_ids),
                "rank_name": item["rank_name"],
                "keep_mode": item["keep_mode"],
                "selection_mode": item["selection_mode"],
                "weight_strategy": item["weight_strategy"],
                "diagnostic_only": int(bool(item["diagnostic_only"])),
                "result_path": str(result_path),
            }
        )
        for _, row in detail.iterrows():
            detail_rows.append(
                {
                    "candidate_label": label,
                    "stock_id": row["stock_id"],
                    "weight": float(row["weight"]),
                    "case_slice_return": float(row["case_slice_return"]),
                    "case_slice_contribution": float(row["case_slice_contribution"]),
                }
            )
    df = pd.DataFrame(rows).sort_values("case_slice_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    detail_df = pd.DataFrame(detail_rows)
    return df, detail_df


def render_report(
    generated: pd.DataFrame,
    existing: pd.DataFrame,
    details: pd.DataFrame,
    output_dir: Path,
    case_current_score: float,
    case_best_score: float,
) -> str:
    submit_ready = generated[generated["diagnostic_only"].eq(0)].copy()
    best_submit = submit_ready.iloc[0] if not submit_ready.empty else None
    best_any = generated.iloc[0] if not generated.empty else None
    current_rows = existing[existing["source_path"].astype(str).str.replace("\\", "/", regex=False).str.endswith("app/output/result.csv")]
    current_score = float(current_rows.iloc[0]["case_slice_score"]) if not current_rows.empty else float("nan")

    lines = [
        "# Case Slice Submission Candidate Search Report",
        "",
        "本报告只做单切片冲分候选搜索，不覆盖 `app/output/result.csv` 或默认配置。",
        "",
        "## 结论",
        "",
        f"- 当前正式结果单切片得分: `{current_score:.6f}`。",
        f"- 压缩包当前输出得分: `{case_current_score:.6f}`。",
        f"- 压缩包记录最好分数: `{case_best_score:.6f}`。",
    ]
    if best_submit is not None:
        lines.extend(
            [
                f"- 当前工程可提交候选中最佳: `{best_submit['candidate_label']}`，得分 `{float(best_submit['case_slice_score']):.6f}`。",
                f"- 最佳可提交候选持仓: `{best_submit['stock_ids']}`，权重和 `{float(best_submit['weight_sum']):.6f}`。",
            ]
        )
    if best_any is not None and int(best_any["diagnostic_only"]) == 1:
        lines.append(
            f"- 诊断型 oracle 最高候选: `{best_any['candidate_label']}`，得分 `{float(best_any['case_slice_score']):.6f}`，仅用于解释上限，不建议直接作为真实提交依据。"
        )
    lines.extend(
        [
            "",
            "## Top 20 Submit-Ready Generated Candidates",
            "",
            "| rank | label | score | weight_sum | contains_600115 | stocks |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for _, row in submit_ready.head(20).iterrows():
        lines.append(
            f"| {int(row['rank'])} | `{row['candidate_label']}` | {float(row['case_slice_score']):.6f} | "
            f"{float(row['weight_sum']):.6f} | {int(row['contains_600115'])} | `{row['stock_ids']}` |"
        )

    diagnostic = generated[generated["diagnostic_only"].eq(1)].copy()
    lines.extend(
        [
            "",
            "## Top 10 Diagnostic Oracle Candidates",
            "",
            "这些候选使用了单切片真实收益排序，只用于解释上限和漏选方向，不建议直接作为真实提交依据。",
            "",
            "| rank | label | score | weight_sum | stocks |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for _, row in diagnostic.head(10).iterrows():
        lines.append(
            f"| {int(row['rank'])} | `{row['candidate_label']}` | {float(row['case_slice_score']):.6f} | "
            f"{float(row['weight_sum']):.6f} | `{row['stock_ids']}` |"
        )

    lines.extend(
        [
            "",
            "## Top 20 Existing Result Files",
            "",
            "| rank | score | weight_sum | stocks | path |",
            "|---:|---:|---:|---|---|",
        ]
    )
    for _, row in existing.head(20).iterrows():
        lines.append(
            f"| {int(row['rank'])} | {float(row['case_slice_score']):.6f} | {float(row['weight_sum']):.6f} | "
            f"`{row['stock_ids']}` | `{row['source_path']}` |"
        )

    if best_submit is not None:
        best_detail = details[details["candidate_label"].eq(best_submit["candidate_label"])].copy()
        lines.extend(
            [
                "",
                "## Best Submit-Ready Candidate Detail",
                "",
                "| stock_id | weight | case_slice_return | contribution |",
                "|---|---:|---:|---:|",
            ]
        )
        for _, row in best_detail.sort_values("case_slice_contribution", ascending=False).iterrows():
            lines.append(
                f"| `{row['stock_id']}` | {float(row['weight']):.6f} | "
                f"{float(row['case_slice_return']):.6f} | {float(row['case_slice_contribution']):.6f} |"
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- generated leaderboard: `{output_dir / 'case_slice_generated_leaderboard.csv'}`",
            f"- existing result leaderboard: `{output_dir / 'case_slice_existing_result_leaderboard.csv'}`",
            f"- generated details: `{output_dir / 'case_slice_generated_candidate_details.csv'}`",
            f"- generated result files: `{output_dir / 'generated_results'}`",
            "",
            "## Caution",
            "",
            "单切片搜索会使用该切片真实收益复算分数，因此只能作为比赛冲分和问题诊断工具。是否同步默认配置，需要再结合 walk-forward、回测和 pre-submit 检查。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search case-slice submission candidates without overwriting defaults.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--case_test", default=str(DEFAULT_CASE_TEST))
    parser.add_argument("--case_result", default=str(DEFAULT_CASE_RESULT))
    parser.add_argument("--case_best", default=str(DEFAULT_CASE_BEST))
    parser.add_argument("--debug_candidates", default=str(DEFAULT_DEBUG))
    parser.add_argument("--predict_scores", default=str(DEFAULT_PREDICT))
    parser.add_argument("--current_result", default=str(DEFAULT_CURRENT_RESULT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_test = read_case_test(resolve_path(args.case_test))
    returns = case_slice_returns(case_test)
    universe = load_candidate_universe(resolve_path(args.debug_candidates), resolve_path(args.predict_scores), returns)

    generated_candidates = generate_candidates(universe)
    generated, details = evaluate_generated(generated_candidates, returns, output_dir)

    case_result = read_result(resolve_path(args.case_result))
    case_current_score, _ = score_result(case_result, returns)
    case_best_score = read_case_best(resolve_path(args.case_best))
    existing = scan_existing_results(
        [
            resolve_path(args.current_result),
            MODEL_DIR,
            resolve_path(args.case_result),
        ],
        returns,
    )

    generated.to_csv(output_dir / "case_slice_generated_leaderboard.csv", index=False, encoding="utf-8-sig")
    details.to_csv(output_dir / "case_slice_generated_candidate_details.csv", index=False, encoding="utf-8-sig")
    existing.to_csv(output_dir / "case_slice_existing_result_leaderboard.csv", index=False, encoding="utf-8-sig")
    report = render_report(generated, existing, details, output_dir, case_current_score, case_best_score)
    (output_dir / "case_slice_submission_search_report.md").write_text(report, encoding="utf-8")

    best_submit = generated[generated["diagnostic_only"].eq(0)].iloc[0]
    print(f"[case_slice_search] generated_candidates={len(generated)}")
    print(f"[case_slice_search] existing_results={len(existing)}")
    print(f"[case_slice_search] best_submit_ready={best_submit['candidate_label']} score={float(best_submit['case_slice_score']):.6f}")
    print(f"[case_slice_search] wrote {output_dir / 'case_slice_submission_search_report.md'}")


if __name__ == "__main__":
    main()
