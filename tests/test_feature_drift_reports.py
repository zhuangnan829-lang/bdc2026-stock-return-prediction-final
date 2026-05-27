import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "app" / "code" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_feature_drift import build_drift_summary, build_fold_feature_stats  # noqa: E402
from feature_importance_report import compute_feature_ic_by_fold  # noqa: E402


def make_fold_context() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    for day_idx, date in enumerate(dates):
        fold_id = 1 + day_idx // 2
        for stock_idx in range(6):
            rows.append(
                {
                    "stock_id": f"{stock_idx:06d}",
                    "date": date,
                    "fold_id": fold_id,
                    "target_return": stock_idx * 0.01 + day_idx * 0.001,
                    "stable_feature": stock_idx * 0.1,
                    "drifty_feature": stock_idx * 0.1 + fold_id * 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_feature_ic_by_fold_has_stage_columns() -> None:
    context = make_fold_context()
    ic_df = compute_feature_ic_by_fold(context, ["stable_feature", "drifty_feature"])

    assert set(ic_df["feature"]) == {"stable_feature", "drifty_feature"}
    assert set(ic_df["fold_id"]) == {1, 2, 3}
    assert ic_df["stage_ic_days"].min() == 2
    assert np.isfinite(ic_df["stage_ic"]).all()


def test_drift_summary_flags_larger_distribution_shift() -> None:
    context = make_fold_context()
    stats = build_fold_feature_stats(context, ["stable_feature", "drifty_feature"])
    importance = pd.DataFrame(
        {
            "feature": ["stable_feature", "drifty_feature"],
            "importance_rank": [2, 1],
            "gain_importance": [1.0, 2.0],
            "gain_importance_pct": [0.33, 0.67],
            "split_importance": [1.0, 2.0],
            "split_importance_pct": [0.33, 0.67],
        }
    )
    summary = build_drift_summary(stats, importance).set_index("feature")

    assert summary.loc["drifty_feature", "drift_score"] > summary.loc["stable_feature", "drift_score"]
    assert {"mean_range", "stage_ic_range", "importance_rank"}.issubset(summary.columns)
