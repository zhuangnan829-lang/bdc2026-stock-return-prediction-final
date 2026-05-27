import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
FEATUREWORK = ROOT_DIR / "app" / "code" / "src" / "featurework.py"


RAW_COLUMNS = [
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
]


def make_raw_frame(days: int = 32) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    for stock_index, stock_id in enumerate(["1", "600000"]):
        base = 10.0 + stock_index
        for i, date in enumerate(dates):
            close = base + i * 0.1 + stock_index * 0.05
            open_price = close - 0.02
            high = close + 0.08
            low = close - 0.10
            rows.append(
                [
                    stock_id,
                    date.strftime("%Y-%m-%d"),
                    open_price,
                    close,
                    high,
                    low,
                    100000 + i * 100 + stock_index * 10,
                    1000000 + i * 1000 + stock_index * 100,
                    1.0 + i * 0.01,
                    0.1,
                    0.5 + i * 0.01,
                    0.2,
                ]
            )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_featurework_predict_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    temp_dir = tmp_path / "temp"
    data_dir.mkdir()
    raw = make_raw_frame()
    raw.to_csv(data_dir / "stock_data.csv", index=False, encoding="utf-8-sig")
    raw.groupby("股票代码", group_keys=False).tail(5).to_csv(
        data_dir / "test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    subprocess.run(
        [
            sys.executable,
            str(FEATUREWORK),
            "--mode",
            "predict",
            "--data_dir",
            str(data_dir),
            "--temp_dir",
            str(temp_dir),
        ],
        check=True,
        cwd=ROOT_DIR,
    )

    output = pd.read_csv(temp_dir / "predict_features.csv", encoding="utf-8-sig", dtype={"stock_id": str})
    assert len(output) == 10
    assert output["stock_id"].str.len().eq(6).all()
    assert {"ret_1d", "volatility_20d", "turnover_spike_5d", "prediction_date"}.issubset(output.columns)
    assert not output[["ret_1d", "volatility_20d", "turnover_spike_5d"]].isna().any().any()


def test_featurework_train_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    temp_dir = tmp_path / "temp"
    data_dir.mkdir()
    make_raw_frame().to_csv(data_dir / "train.csv", index=False, encoding="utf-8-sig")

    subprocess.run(
        [
            sys.executable,
            str(FEATUREWORK),
            "--mode",
            "train",
            "--data_dir",
            str(data_dir),
            "--temp_dir",
            str(temp_dir),
        ],
        check=True,
        cwd=ROOT_DIR,
    )

    output = pd.read_csv(temp_dir / "train_features.csv", encoding="utf-8-sig", dtype={"stock_id": str})
    assert "target_return" in output.columns
    assert output["target_return"].notna().sum() > 0
