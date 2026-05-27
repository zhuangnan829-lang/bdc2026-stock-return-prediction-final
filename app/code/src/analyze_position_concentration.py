import argparse
from pathlib import Path

import pandas as pd


def _split_csv_values(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def max_contribution_share(weights: pd.Series, returns: pd.Series) -> float:
    contribution = (weights.astype(float) * returns.astype(float)).abs()
    total = float(contribution.sum())
    if total <= 1e-12:
        return 0.0
    return float(contribution.max() / total)


def summarize_weight_return_frame(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "positions": 0,
            "weight_sum": 0.0,
            "max_single_weight": 0.0,
            "top2_weight_sum": 0.0,
            "max_single_contribution_share": 0.0,
            "portfolio_return": 0.0,
            "max_weight_stock_id": "",
            "max_contribution_stock_id": "",
        }

    work = df.copy()
    work["stock_id"] = work["stock_id"].astype(str).str.zfill(6)
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce").fillna(0.0)
    if "return" not in work.columns:
        work["return"] = 0.0
    work["return"] = pd.to_numeric(work["return"], errors="coerce").fillna(0.0)
    work["abs_contribution"] = (work["weight"] * work["return"]).abs()

    max_weight_idx = work["weight"].idxmax()
    max_contribution_idx = work["abs_contribution"].idxmax()
    return {
        "positions": int(len(work)),
        "weight_sum": float(work["weight"].sum()),
        "max_single_weight": float(work["weight"].max()),
        "top2_weight_sum": float(work["weight"].nlargest(2).sum()),
        "max_single_contribution_share": max_contribution_share(work["weight"], work["return"]),
        "portfolio_return": float((work["weight"] * work["return"]).sum()),
        "max_weight_stock_id": str(work.loc[max_weight_idx, "stock_id"]),
        "max_contribution_stock_id": str(work.loc[max_contribution_idx, "stock_id"]),
    }


def analyze_result_slice(result_path: Path, price_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = pd.read_csv(result_path, encoding="utf-8-sig", dtype={"stock_id": str})
    if list(result.columns) != ["stock_id", "weight"]:
        raise ValueError(f"{result_path} must have exactly columns: stock_id, weight")
    result["stock_id"] = result["stock_id"].astype(str).str.zfill(6)
    result["weight"] = pd.to_numeric(result["weight"], errors="coerce").fillna(0.0)

    detail = result.rename(columns={"weight": "weight"}).copy()
    detail["return"] = 0.0
    if price_path is not None and price_path.exists():
        price_df = pd.read_csv(price_path, encoding="utf-8-sig", dtype={"stock_id": str})
        rename_map = {}
        if "股票代码" in price_df.columns and "stock_id" not in price_df.columns:
            rename_map["股票代码"] = "stock_id"
        if "日期" in price_df.columns and "date" not in price_df.columns:
            rename_map["日期"] = "date"
        if "收盘" in price_df.columns and "close" not in price_df.columns:
            rename_map["收盘"] = "close"
        if rename_map:
            price_df = price_df.rename(columns=rename_map)
        missing_columns = [column for column in ["stock_id", "date"] if column not in price_df.columns]
        if missing_columns:
            raise ValueError(f"{price_path} missing required columns for slice analysis: {missing_columns}")
        price_df["stock_id"] = price_df["stock_id"].astype(str).str.zfill(6)
        price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
        close_col = "close" if "close" in price_df.columns else None
        if close_col is None:
            raise ValueError(f"{price_path} must contain close/收盘 column for slice contribution analysis")
        returns = (
            price_df.sort_values(["stock_id", "date"])
            .groupby("stock_id")[close_col]
            .agg(slice_start="first", slice_end="last")
            .reset_index()
        )
        returns["return"] = (returns["slice_end"] - returns["slice_start"]) / returns["slice_start"]
        detail = result.merge(returns[["stock_id", "return"]], on="stock_id", how="left")
        detail["return"] = pd.to_numeric(detail["return"], errors="coerce").fillna(0.0)

    detail["contribution"] = detail["weight"] * detail["return"]
    detail["abs_contribution"] = detail["contribution"].abs()
    summary = pd.DataFrame([{"scope": "single_slice", **summarize_weight_return_frame(detail)}])
    return summary, detail


def analyze_backtest_daily(daily_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_df = pd.read_csv(daily_path, encoding="utf-8-sig")
    rows = []
    detail_rows = []
    for _, row in daily_df.iterrows():
        stock_ids = _split_csv_values(row.get("selected_stock_ids"))
        weights = [float(value) for value in _split_csv_values(row.get("selected_weights"))]
        returns = [float(value) for value in _split_csv_values(row.get("selected_target_returns"))]
        n = min(len(stock_ids), len(weights), len(returns))
        detail = pd.DataFrame(
            {
                "profile_name": row.get("profile_name", ""),
                "date": row.get("date", ""),
                "stock_id": stock_ids[:n],
                "weight": weights[:n],
                "return": returns[:n],
            }
        )
        if not detail.empty:
            detail["contribution"] = detail["weight"] * detail["return"]
            detail["abs_contribution"] = detail["contribution"].abs()
            detail_rows.append(detail)
        rows.append(
            {
                "profile_name": row.get("profile_name", ""),
                "date": row.get("date", ""),
                **summarize_weight_return_frame(detail),
            }
        )

    by_date = pd.DataFrame(rows)
    detail_all = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    summary = (
        by_date.groupby("profile_name", dropna=False)
        .agg(
            periods=("date", "count"),
            avg_max_single_weight=("max_single_weight", "mean"),
            max_single_weight=("max_single_weight", "max"),
            avg_top2_weight_sum=("top2_weight_sum", "mean"),
            avg_max_single_contribution_share=("max_single_contribution_share", "mean"),
            max_single_contribution_share=("max_single_contribution_share", "max"),
        )
        .reset_index()
    )
    return summary, by_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze portfolio concentration for a result slice or backtest daily file.")
    parser.add_argument("--result_path")
    parser.add_argument("--price_path")
    parser.add_argument("--daily_path")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wrote = []
    if args.result_path:
        summary, detail = analyze_result_slice(Path(args.result_path), Path(args.price_path) if args.price_path else None)
        summary_path = output_dir / "slice_concentration_summary.csv"
        detail_path = output_dir / "slice_concentration_detail.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
        wrote.extend([summary_path, detail_path])
        row = summary.iloc[0]
        print(
            "[concentration] slice "
            f"max_single_weight={row['max_single_weight']:.6f} "
            f"top2_weight_sum={row['top2_weight_sum']:.6f} "
            f"max_single_contribution_share={row['max_single_contribution_share']:.6f}"
        )

    if args.daily_path:
        summary, by_date = analyze_backtest_daily(Path(args.daily_path))
        summary_path = output_dir / "backtest_concentration_summary.csv"
        by_date_path = output_dir / "backtest_concentration_by_date.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        by_date.to_csv(by_date_path, index=False, encoding="utf-8-sig")
        wrote.extend([summary_path, by_date_path])

    for path in wrote:
        print(f"[concentration] wrote {path}")


if __name__ == "__main__":
    main()
