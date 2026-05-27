from __future__ import annotations

import contextlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, TextIO


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT_ROOT = ROOT_DIR / "app" / "model" / "experiments"
DEFAULT_LEADERBOARD_PATH = ROOT_DIR / "app" / "model" / "experiment_leaderboard.csv"

REQUIRED_EXPERIMENT_FILES = [
    "config.json",
    "metrics.csv",
    "fold_results.csv",
    "backtest_summary.csv",
    "result.csv",
    "run.log",
    "report.md",
]

LEADERBOARD_COLUMNS = [
    "experiment_id",
    "model",
    "feature_set",
    "sequence_length",
    "rank_ic_mean",
    "worst_fold_rank_ic",
    "top5_return_mean",
    "cost_after_return",
    "sharpe",
    "max_drawdown",
    "avg_turnover",
    "single_slice_score",
    "max_single_contribution_ratio",
    "adopted",
    "notes",
]


def sanitize_experiment_component(value: object, default: str = "na") -> str:
    text = str(value).strip().lower() if value is not None else ""
    text = re.sub(r"[^0-9a-zA-Z]+", "-", text).strip("-").lower()
    return text or default


def sanitize_experiment_id(value: object, default: str = "experiment") -> str:
    text = str(value).strip().lower() if value is not None else ""
    text = re.sub(r"[^0-9a-zA-Z_-]+", "_", text).strip("_-").lower()
    return text or default


def build_experiment_id(
    *,
    model: object,
    feature: object | None = None,
    feature_set: object | None = None,
    sequence_length: object = None,
    strategy: object | None = None,
    sort_strategy: object | None = None,
    weighting_scheme: object | None = None,
    remark: object = None,
    now: datetime | None = None,
) -> str:
    timestamp = (now or datetime.now()).strftime("%Y%m%d%H%M%S")
    resolved_feature = feature_set if feature_set is not None else feature
    sl_value = "na" if sequence_length in (None, "", "na") else str(sequence_length).removeprefix("sl")
    parts = [
        timestamp,
        sanitize_experiment_component(model),
        sanitize_experiment_component(resolved_feature),
        f"sl{sanitize_experiment_component(sl_value)}",
    ]
    if strategy not in (None, "", "na"):
        parts.append(sanitize_experiment_component(strategy, default="strategy"))
    else:
        strategy_parts = [part for part in [sort_strategy, weighting_scheme] if part not in (None, "", "na")]
        if strategy_parts:
            parts.extend(sanitize_experiment_component(part, default="strategy") for part in strategy_parts)
        else:
            parts.append("strategy")
    parts.append(sanitize_experiment_component(remark, default="exp"))
    return "_".join(parts)


def ensure_unique_experiment_dir(experiment_root: Path, experiment_id: str) -> Path:
    experiment_dir = experiment_root / experiment_id
    if not experiment_dir.exists():
        return experiment_dir
    suffix = 2
    while True:
        candidate = experiment_root / f"{experiment_id}_v{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def init_experiment_dir(experiment_dir: Path) -> Path:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "figures").mkdir(exist_ok=True)
    for filename in REQUIRED_EXPERIMENT_FILES:
        path = experiment_dir / filename
        if not path.exists():
            path.touch()
    return experiment_dir


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def create_experiment_dir(
    *,
    model: object,
    feature_set: object,
    sequence_length: object,
    strategy: object,
    remark: object = "exp",
    experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT,
    experiment_id: str | None = None,
    now: datetime | None = None,
) -> Path:
    root = resolve_path(experiment_root)
    root.mkdir(parents=True, exist_ok=True)
    resolved_id = experiment_id or build_experiment_id(
        model=model,
        feature_set=feature_set,
        sequence_length=sequence_length,
        strategy=strategy,
        remark=remark,
        now=now,
    )
    experiment_dir = ensure_unique_experiment_dir(root, sanitize_experiment_id(resolved_id))
    return init_experiment_dir(experiment_dir)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_config(experiment_dir: str | Path, config: Mapping[str, Any]) -> Path:
    path = resolve_path(experiment_dir) / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(config), ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return path


def _write_csv(path: Path, data: Any) -> Path:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        if not path.exists():
            path.touch()
        return path
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, Mapping):
        df = pd.DataFrame([dict(data)])
    else:
        df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_metrics(
    experiment_dir: str | Path,
    metrics: Any = None,
    fold_results: Any = None,
    backtest_summary: Any = None,
) -> dict[str, Path]:
    target_dir = resolve_path(experiment_dir)
    return {
        "metrics": _write_csv(target_dir / "metrics.csv", metrics),
        "fold_results": _write_csv(target_dir / "fold_results.csv", fold_results),
        "backtest_summary": _write_csv(target_dir / "backtest_summary.csv", backtest_summary),
    }


def save_report(experiment_dir: str | Path, report: str | Iterable[str]) -> Path:
    path = resolve_path(experiment_dir) / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(report, str):
        content = report
    else:
        content = "\n".join(str(line) for line in report)
    path.write_text(content, encoding="utf-8")
    return path


def load_latest_experiment(experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT) -> Path | None:
    root = resolve_path(experiment_root)
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _read_first_row(path: Path) -> dict[str, Any]:
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path, encoding="utf-8-sig").to_dict("records")


def _first_present(source: Mapping[str, Any], names: Iterable[str], default: Any = "") -> Any:
    for name in names:
        value = source.get(name)
        if value is not None and str(value) != "nan" and str(value) != "":
            return value
    return default


def _load_config(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / "config.json"
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _build_leaderboard_row(
    experiment_dir: Path,
    *,
    adopted: bool = False,
    notes: str = "",
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = _load_config(experiment_dir)
    metrics = _read_first_row(experiment_dir / "metrics.csv")
    fold_rows = _read_csv_rows(experiment_dir / "fold_results.csv")
    backtest = _read_first_row(experiment_dir / "backtest_summary.csv")
    overrides = dict(overrides or {})

    rank_ic_values = [
        float(row[name])
        for row in fold_rows
        for name in ["rank_ic", "rank_ic_mean"]
        if row.get(name) not in (None, "", "nan")
    ]
    worst_fold = min(rank_ic_values) if rank_ic_values else _first_present(metrics, ["worst_fold_rank_ic", "worst_fold", "min_rank_ic"])

    row = {
        "experiment_id": experiment_dir.name,
        "model": _first_present(config, ["model", "model_family"], ""),
        "feature_set": _first_present(config, ["feature_set", "feature"], ""),
        "sequence_length": _first_present(config, ["sequence_length"], ""),
        "rank_ic_mean": _first_present(metrics, ["rank_ic_mean", "rank_ic"], ""),
        "worst_fold_rank_ic": worst_fold,
        "top5_return_mean": _first_present(metrics, ["top5_return_mean", "top5_mean_return_mean", "top5_mean_return"], ""),
        "cost_after_return": _first_present(
            backtest,
            ["cost_after_return", "cumulative_return_after_cost", "return_after_cost", "backtest_return"],
            "",
        ),
        "sharpe": _first_present(backtest, ["sharpe", "sharpe_after_cost"], ""),
        "max_drawdown": _first_present(backtest, ["max_drawdown", "max_drawdown_after_cost"], ""),
        "avg_turnover": _first_present(backtest, ["avg_turnover", "turnover_mean", "turnover"], ""),
        "single_slice_score": _first_present(backtest, ["single_slice_score", "slice_score", "case_slice_score"], ""),
        "max_single_contribution_ratio": _first_present(
            backtest,
            ["max_single_contribution_ratio", "max_contribution_ratio", "max_weight"],
            "",
        ),
        "adopted": bool(adopted),
        "notes": notes,
    }
    row.update({key: value for key, value in overrides.items() if key in LEADERBOARD_COLUMNS})
    return row


def register_experiment_result(
    experiment_dir: str | Path,
    *,
    leaderboard_path: str | Path = DEFAULT_LEADERBOARD_PATH,
    result_path: str | Path | None = None,
    adopted: bool = False,
    notes: str = "",
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    import pandas as pd

    target_dir = resolve_path(experiment_dir)
    if result_path is not None:
        source = resolve_path(result_path)
        if source.exists():
            shutil.copy2(source, target_dir / "result.csv")

    row = _build_leaderboard_row(target_dir, adopted=adopted, notes=notes, overrides=metrics)
    output_path = resolve_path(leaderboard_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        leaderboard = pd.read_csv(output_path, encoding="utf-8-sig")
    else:
        leaderboard = pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    for column in LEADERBOARD_COLUMNS:
        if column not in leaderboard.columns:
            leaderboard[column] = ""

    leaderboard = leaderboard[leaderboard["experiment_id"].astype(str) != str(row["experiment_id"])]
    leaderboard = pd.concat([leaderboard, pd.DataFrame([row])], ignore_index=True)
    leaderboard = leaderboard[LEADERBOARD_COLUMNS + [c for c in leaderboard.columns if c not in LEADERBOARD_COLUMNS]]
    leaderboard.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def resolve_training_output_dir(
    *,
    requested_model_dir: Path,
    experiment_root: str | Path | None,
    experiment_id: str | None,
    model: object,
    feature: object,
    sequence_length: object = None,
    sort_strategy: object = None,
    weighting_scheme: object = None,
    remark: object = None,
) -> tuple[Path, str | None]:
    if not experiment_root:
        requested_model_dir.mkdir(parents=True, exist_ok=True)
        return requested_model_dir, None

    strategy_parts = [part for part in [sort_strategy, weighting_scheme] if part not in (None, "", "na")]
    strategy = "_".join(str(part) for part in strategy_parts) if strategy_parts else "strategy"
    experiment_dir = create_experiment_dir(
        model=model,
        feature_set=feature,
        sequence_length=sequence_length,
        strategy=strategy,
        remark=remark,
        experiment_root=experiment_root,
        experiment_id=experiment_id,
    )
    return experiment_dir, experiment_dir.name


def write_experiment_config(experiment_dir: Path, config: dict) -> Path:
    return save_config(experiment_dir, config)


def write_training_metric_exports(
    *,
    experiment_dir: Path,
    fold_metrics: Any,
    metric_summary: dict,
) -> None:
    save_metrics(experiment_dir, metrics=metric_summary, fold_results=fold_metrics)


def maybe_copy_result_to_experiment(result_path: str | Path, experiment_dir: Path) -> Path | None:
    source = Path(result_path)
    if not source.exists():
        return None
    destination = experiment_dir / "result.csv"
    shutil.copy2(source, destination)
    return destination


class _Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_run_log(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = _Tee(old_stdout, log_file)
        sys.stderr = _Tee(old_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
