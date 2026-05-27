import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split raw stock_data.csv into train.csv and test.csv for local development."
    )
    parser.add_argument(
        "--input",
        default="app/data/stock_data.csv",
        help="Path to the raw full stock data CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="app/data",
        help="Directory where train.csv and test.csv will be written.",
    )
    parser.add_argument(
        "--train-start",
        default="2024-01-02",
        help="Training start date, inclusive.",
    )
    parser.add_argument(
        "--train-end",
        default="2026-03-06",
        help="Training end date, inclusive.",
    )
    parser.add_argument(
        "--test-start",
        default="2026-03-09",
        help="Local validation start date, inclusive.",
    )
    parser.add_argument(
        "--test-end",
        default="2026-03-13",
        help="Local validation end date, inclusive.",
    )
    return parser.parse_args()


def _to_ts(date_str: str, arg_name: str) -> pd.Timestamp:
    ts = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date for {arg_name}: {date_str}")
    return ts.normalize()


def _filter(df: pd.DataFrame, code_col: str, date_col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if start > end:
        raise ValueError(f"Start date is after end date: {start.date()} > {end.date()}")
    mask = (df[date_col] >= start) & (df[date_col] <= end)
    out = df.loc[mask].copy()
    out = out.sort_values([code_col, date_col]).reset_index(drop=True)
    out[date_col] = out[date_col].dt.strftime("%Y-%m-%d")
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    code_col = df.columns[0]
    date_col = df.columns[1]

    df[code_col] = df[code_col].astype(str).str.zfill(6)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        bad_rows = int(df[date_col].isna().sum())
        raise ValueError(f"Found {bad_rows} rows with invalid dates in the raw input.")

    train_df = _filter(
        df,
        code_col,
        date_col,
        _to_ts(args.train_start, "--train-start"),
        _to_ts(args.train_end, "--train-end"),
    )
    test_df = _filter(
        df,
        code_col,
        date_col,
        _to_ts(args.test_start, "--test-start"),
        _to_ts(args.test_end, "--test-end"),
    )

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(
        f"WROTE {train_path} rows={len(train_df)} stocks={train_df[code_col].nunique()} "
        f"range={train_df[date_col].min()}~{train_df[date_col].max()}"
    )
    print(
        f"WROTE {test_path} rows={len(test_df)} stocks={test_df[code_col].nunique()} "
        f"range={test_df[date_col].min()}~{test_df[date_col].max()}"
    )


if __name__ == "__main__":
    main()
