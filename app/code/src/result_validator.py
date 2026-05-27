import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate competition result.csv format.")
    parser.add_argument("--result_path", required=True)
    return parser.parse_args()


def validate_result_file(result_path: Path) -> dict:
    raw_bytes = result_path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
        encoding_status = "utf-8"
    except UnicodeDecodeError as exc:
        raise ValueError(f"Result file is not valid UTF-8: {exc}") from exc

    df = pd.read_csv(result_path, encoding="utf-8", dtype={"stock_id": str})
    if list(df.columns) != ["stock_id", "weight"]:
        raise ValueError("result.csv must have exactly two columns: stock_id, weight")
    if len(df) > 5:
        raise ValueError("result.csv must contain at most 5 rows")

    df["stock_id"] = df["stock_id"].astype(str).str.zfill(6)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    if df["weight"].isna().any():
        raise ValueError("All weight values must be numeric")
    if not df["stock_id"].str.fullmatch(r"\d{6}").all():
        raise ValueError("All stock_id values must be 6-digit strings")
    if not df["stock_id"].is_unique:
        raise ValueError("stock_id values must be unique")
    if not (df["weight"] >= 0).all():
        raise ValueError("All weights must be non-negative")
    weight_sum = float(df["weight"].sum())
    if weight_sum > 1.0 + 1e-9:
        raise ValueError("Weight sum must be <= 1")

    return {
        "rows": int(len(df)),
        "weight_sum": weight_sum,
        "encoding": encoding_status,
        "stock_ids": df["stock_id"].tolist(),
        "weights": [float(x) for x in df["weight"].tolist()],
        "trailing_newline": text.endswith("\n"),
    }


def main() -> None:
    args = parse_args()
    result_path = Path(args.result_path)
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result file: {result_path}")

    summary = validate_result_file(result_path)
    print(f"[result_validator] path={result_path}")
    print(f"[result_validator] encoding={summary['encoding']} rows={summary['rows']} weight_sum={summary['weight_sum']:.6f}")
    print(f"[result_validator] stock_ids={summary['stock_ids']}")
    print(f"[result_validator] weights={summary['weights']}")
    print("[result_validator] validation passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[result_validator][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
