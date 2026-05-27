import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


LABEL_COLUMN_PATTERNS = (
    "target",
    "label",
    "future",
    "train_target",
    "sample_weight",
    "pred_return",
    "prediction",
)

ALLOWED_NON_FEATURE_COLUMNS = {
    "prediction_date",
}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    default_app_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Programmatic data leakage checks.")
    parser.add_argument("--app-root", default=default_app_root)
    parser.add_argument("--train-features", default=None)
    parser.add_argument("--predict-features", default=None)
    parser.add_argument("--model-meta", default=None)
    parser.add_argument("--walk-forward-metrics", default=None)
    parser.add_argument("--walk-forward-predictions", default=None)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def _resolve_path(app_root: Path, value: str | None, default_relative: str) -> Path:
    if value is None:
        return app_root / default_relative
    path = Path(value)
    return path if path.is_absolute() else app_root / path


def _display_path(app_root: Path, path: Path) -> str:
    try:
        return Path("app", path.resolve().relative_to(app_root)).as_posix()
    except ValueError:
        return path.as_posix()


def _result(name: str, ok: bool, detail: str) -> CheckResult:
    return CheckResult(name=name, status="PASS" if ok else "FAIL", detail=detail)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _leakage_like_columns(columns: Iterable[str]) -> list[str]:
    suspicious = []
    for column in columns:
        lowered = column.lower()
        if column in ALLOWED_NON_FEATURE_COLUMNS:
            continue
        if any(pattern in lowered for pattern in LABEL_COLUMN_PATTERNS):
            suspicious.append(column)
    return sorted(set(suspicious))


def check_required_inputs(app_root: Path, paths: dict[str, Path]) -> list[CheckResult]:
    results = []
    for name, path in paths.items():
        ok = path.is_file()
        detail = f"{_display_path(app_root, path)} exists" if ok else f"Missing {_display_path(app_root, path)}"
        results.append(_result(f"required file: {name}", ok, detail))
    return results


def check_prediction_dates(app_root: Path, predict_features_path: Path) -> CheckResult:
    df = _read_csv(predict_features_path, dtype={"stock_id": str})
    required = {"date", "prediction_date"}
    missing = sorted(required - set(df.columns))
    if missing:
        return _result("feature date <= prediction date", False, f"Missing columns: {missing}")

    feature_dates = _date_series(df["date"])
    prediction_dates = _date_series(df["prediction_date"])
    invalid_dates = int(feature_dates.isna().sum() + prediction_dates.isna().sum())
    if invalid_dates:
        return _result("feature date <= prediction date", False, f"Invalid date cells: {invalid_dates}")

    leaking_rows = int((feature_dates > prediction_dates).sum())
    if leaking_rows:
        examples = df.loc[feature_dates > prediction_dates, ["stock_id", "date", "prediction_date"]].head(5)
        return _result(
            "feature date <= prediction date",
            False,
            f"{leaking_rows} rows have feature date after prediction date. Examples: {examples.to_dict('records')}",
        )

    return _result(
        "feature date <= prediction date",
        True,
        f"{len(df)} prediction rows checked in {_display_path(app_root, predict_features_path)}",
    )


def check_train_label_horizon(app_root: Path, train_features_path: Path) -> CheckResult:
    df = _read_csv(train_features_path, dtype={"stock_id": str}, usecols=["stock_id", "date", "target_return"])
    df["date"] = _date_series(df["date"])
    if df["date"].isna().any():
        return _result("training label horizon is forward only", False, "Invalid dates found in train features")

    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    grouped = df.groupby("stock_id", group_keys=False)
    label_end_date = grouped["date"].shift(-5)
    usable = df["target_return"].notna()
    invalid_horizon = usable & (label_end_date <= df["date"])
    missing_horizon = usable & label_end_date.isna()
    if invalid_horizon.any() or missing_horizon.any():
        bad_rows = int(invalid_horizon.sum() + missing_horizon.sum())
        return _result(
            "training label horizon is forward only",
            False,
            f"{bad_rows} labeled rows do not have a strictly later 5-trading-day label horizon",
        )

    return _result(
        "training label horizon is forward only",
        True,
        f"{int(usable.sum())} labeled rows have strictly later 5-trading-day label horizons",
    )


def check_label_columns_excluded(app_root: Path, model_meta_path: Path, predict_features_path: Path) -> CheckResult:
    meta = _load_json(model_meta_path)
    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        return _result("label fields excluded from model features", False, "model_meta.json has no feature_columns")

    suspicious_features = _leakage_like_columns(feature_columns)
    if suspicious_features:
        return _result(
            "label fields excluded from model features",
            False,
            f"Suspicious feature_columns in {_display_path(app_root, model_meta_path)}: {suspicious_features}",
        )

    predict_columns = list(_read_csv(predict_features_path, nrows=0).columns)
    suspicious_predict_columns = [
        column for column in _leakage_like_columns(predict_columns) if column not in ALLOWED_NON_FEATURE_COLUMNS
    ]
    if suspicious_predict_columns:
        return _result(
            "label fields excluded from prediction feature file",
            False,
            f"Prediction feature file contains label-like columns: {suspicious_predict_columns}",
        )

    return _result(
        "label fields excluded from model features",
        True,
        f"{len(feature_columns)} model feature columns checked; no target/future/prediction label fields included",
    )


def _folds_from_metrics(metrics_df: pd.DataFrame) -> list[dict]:
    folds = []
    for _, row in metrics_df.sort_values("fold_id").iterrows():
        folds.append(
            {
                "fold_id": int(row["fold_id"]),
                "train_date_start": row["train_date_start"],
                "train_date_end": row["train_date_end"],
                "valid_date_start": row["valid_date_start"],
                "valid_date_end": row["valid_date_end"],
            }
        )
    return folds


def _normalise_fold_dates(fold: dict) -> dict:
    out = dict(fold)
    for key in ("train_date_start", "train_date_end", "valid_date_start", "valid_date_end"):
        out[key] = pd.Timestamp(out[key])
    return out


def check_walk_forward_order(
    app_root: Path,
    model_meta_path: Path,
    walk_forward_metrics_path: Path,
    walk_forward_predictions_path: Path,
) -> CheckResult:
    meta = _load_json(model_meta_path)
    metric_folds = _folds_from_metrics(_read_csv(walk_forward_metrics_path))
    meta_folds = meta.get("walk_forward_folds", [])
    if not meta_folds:
        return _result("walk-forward train/validation order", False, "model_meta.json has no walk_forward_folds")

    failures = []
    for source_name, folds in (("model_meta", meta_folds), ("walk_forward_metrics", metric_folds)):
        previous_valid_end = None
        for raw_fold in sorted(folds, key=lambda item: int(item["fold_id"])):
            fold = _normalise_fold_dates(raw_fold)
            if fold["train_date_start"] > fold["train_date_end"]:
                failures.append(f"{source_name} fold {fold['fold_id']}: train start after train end")
            if fold["valid_date_start"] > fold["valid_date_end"]:
                failures.append(f"{source_name} fold {fold['fold_id']}: valid start after valid end")
            if fold["train_date_end"] >= fold["valid_date_start"]:
                failures.append(f"{source_name} fold {fold['fold_id']}: train end is not before valid start")
            if previous_valid_end is not None and previous_valid_end >= fold["valid_date_start"]:
                failures.append(f"{source_name} fold {fold['fold_id']}: validation windows overlap or go backward")
            previous_valid_end = fold["valid_date_end"]

    predictions = _read_csv(walk_forward_predictions_path, dtype={"stock_id": str}, usecols=["date", "fold_id"])
    predictions["date"] = _date_series(predictions["date"])
    metric_fold_by_id = {int(fold["fold_id"]): _normalise_fold_dates(fold) for fold in metric_folds}
    for fold_id, fold_predictions in predictions.groupby("fold_id"):
        fold = metric_fold_by_id.get(int(fold_id))
        if fold is None:
            failures.append(f"predictions fold {fold_id}: no matching metrics fold")
            continue
        outside = (
            (fold_predictions["date"] < fold["valid_date_start"])
            | (fold_predictions["date"] > fold["valid_date_end"])
        )
        if outside.any():
            failures.append(f"predictions fold {fold_id}: {int(outside.sum())} rows outside validation window")

    if failures:
        return _result("walk-forward train/validation order", False, "; ".join(failures[:10]))

    return _result(
        "walk-forward train/validation order",
        True,
        f"{len(metric_folds)} folds checked; all train windows end before validation windows",
    )


def check_rebuilt_walk_forward_windows(app_root: Path, train_features_path: Path, model_meta_path: Path) -> CheckResult:
    meta = _load_json(model_meta_path)
    valid_dates = int(meta.get("valid_dates", 0))
    num_folds = int(meta.get("num_folds", 0))
    if valid_dates <= 0 or num_folds <= 0:
        return _result("walk-forward windows reproducible from feature dates", False, "Invalid valid_dates/num_folds in metadata")

    df = _read_csv(train_features_path, usecols=["date", "target_return"])
    df["date"] = _date_series(df["date"])
    unique_dates = sorted(df.loc[df["target_return"].notna(), "date"].drop_duplicates())
    start_index = len(unique_dates) - valid_dates * num_folds
    if start_index <= 0:
        return _result("walk-forward windows reproducible from feature dates", False, "Not enough dates to rebuild folds")

    expected = []
    for fold_index in range(num_folds):
        valid_start = start_index + fold_index * valid_dates
        valid_end = valid_start + valid_dates
        train_block = unique_dates[:valid_start]
        valid_block = unique_dates[valid_start:valid_end]
        expected.append(
            {
                "fold_id": fold_index + 1,
                "train_date_start": train_block[0].date().isoformat(),
                "train_date_end": train_block[-1].date().isoformat(),
                "valid_date_start": valid_block[0].date().isoformat(),
                "valid_date_end": valid_block[-1].date().isoformat(),
            }
        )

    actual = [
        {
            "fold_id": int(fold["fold_id"]),
            "train_date_start": str(pd.Timestamp(fold["train_date_start"]).date()),
            "train_date_end": str(pd.Timestamp(fold["train_date_end"]).date()),
            "valid_date_start": str(pd.Timestamp(fold["valid_date_start"]).date()),
            "valid_date_end": str(pd.Timestamp(fold["valid_date_end"]).date()),
        }
        for fold in sorted(meta.get("walk_forward_folds", []), key=lambda item: int(item["fold_id"]))
    ]

    ok = expected == actual
    detail = (
        f"Rebuilt {len(expected)} folds from train feature dates and matched model_meta.json"
        if ok
        else f"Expected {expected}, actual {actual}"
    )
    return _result("walk-forward windows reproducible from feature dates", ok, detail)


def write_report(report_path: Path, results: list[CheckResult]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    overall = "PASS" if all(result.status == "PASS" for result in results) else "FAIL"
    lines = [
        "# Data Leakage Check Report",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Overall status: **{overall}**",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for result in results:
        detail = result.detail.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {result.name} | {result.status} | {detail} |")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    app_root = Path(args.app_root).resolve()
    train_features_path = _resolve_path(app_root, args.train_features, "temp/train_features.csv")
    predict_features_path = _resolve_path(app_root, args.predict_features, "temp/predict_features.csv")
    model_meta_path = _resolve_path(app_root, args.model_meta, "model/model_meta.json")
    walk_forward_metrics_path = _resolve_path(app_root, args.walk_forward_metrics, "model/walk_forward_metrics.csv")
    walk_forward_predictions_path = _resolve_path(
        app_root, args.walk_forward_predictions, "model/walk_forward_predictions.csv"
    )
    report_path = _resolve_path(app_root, args.report_path, "model/data_leakage_check_report.md")

    paths = {
        "train_features": train_features_path,
        "predict_features": predict_features_path,
        "model_meta": model_meta_path,
        "walk_forward_metrics": walk_forward_metrics_path,
        "walk_forward_predictions": walk_forward_predictions_path,
    }
    results = check_required_inputs(app_root, paths)
    if all(result.status == "PASS" for result in results):
        results.extend(
            [
                check_prediction_dates(app_root, predict_features_path),
                check_train_label_horizon(app_root, train_features_path),
                check_label_columns_excluded(app_root, model_meta_path, predict_features_path),
                check_walk_forward_order(
                    app_root,
                    model_meta_path,
                    walk_forward_metrics_path,
                    walk_forward_predictions_path,
                ),
                check_rebuilt_walk_forward_windows(app_root, train_features_path, model_meta_path),
            ]
        )

    write_report(report_path, results)
    overall = "PASS" if all(result.status == "PASS" for result in results) else "FAIL"
    print(f"[check_data_leakage] overall={overall} report={report_path}")
    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")
    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
