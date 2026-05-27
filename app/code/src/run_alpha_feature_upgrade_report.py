import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT_DIR / "app" / "model" / "alpha_feature_upgrade"
CASE_DIR = ROOT_DIR / "_case_zip" / "THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score_result_against_case_test(result_path: Path) -> float:
    test_df = pd.read_csv(CASE_DIR / "data" / "test.csv")
    code_col = test_df.columns[0]
    open_col = test_df.columns[2]

    result_df = pd.read_csv(result_path, dtype={"stock_id": str})
    if "stock_id" in result_df.columns:
        result_df = result_df.rename(columns={"stock_id": "code", "weight": "weight"})
    else:
        result_df.columns = ["code", "weight"]
    result_df["code"] = result_df["code"].astype(str).str.zfill(6)

    filtered = test_df[test_df[code_col].astype(str).str.zfill(6).isin(result_df["code"])].copy()
    filtered[code_col] = filtered[code_col].astype(str).str.zfill(6)
    filtered = filtered.groupby(code_col).tail(5)
    returns = filtered.groupby(code_col).apply(
        lambda g: (g.iloc[-1][open_col] - g.iloc[0][open_col]) / g.iloc[0][open_col]
    ).reset_index()
    returns.columns = ["code", "ret"]

    merged = returns.merge(result_df, on="code", how="inner")
    return float((merged["ret"] * merged["weight"]).sum())


def collect_row(
    label: str,
    model_dir: Path | None,
    backtest_summary_path: Path | None,
    result_path: Path | None,
    notes: str,
) -> dict:
    row = {
        "label": label,
        "feature_set": "",
        "feature_count": "",
        "rank_ic_mean": "",
        "top5_mean_return_mean": "",
        "cumulative_return_after_cost": "",
        "sharpe_after_cost": "",
        "max_drawdown_after_cost": "",
        "avg_turnover": "",
        "score_self_case_slice": "",
        "case_best_final_score": "",
        "notes": notes,
    }

    if model_dir is not None:
        meta = load_json(model_dir / "model_meta.json")
        row["feature_set"] = meta.get("feature_set", "")
        row["feature_count"] = len(meta.get("feature_columns", []))
        row["rank_ic_mean"] = float(meta["walk_forward_summary"]["rank_ic_mean"])
        row["top5_mean_return_mean"] = float(meta["walk_forward_summary"]["top5_mean_return_mean"])

    if backtest_summary_path is not None:
        summary = pd.read_csv(backtest_summary_path).iloc[0]
        row["cumulative_return_after_cost"] = float(summary["cumulative_return_after_cost"])
        row["sharpe_after_cost"] = float(summary["sharpe_after_cost"])
        row["max_drawdown_after_cost"] = float(summary["max_drawdown_after_cost"])
        row["avg_turnover"] = float(summary["avg_turnover"])

    if result_path is not None:
        row["score_self_case_slice"] = score_result_against_case_test(result_path)

    if label == "case_zip_program":
        row["case_best_final_score"] = 0.037838

    return row


def format_value(value) -> str:
    if value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Alpha 特征升级对比报告",
        "",
        "## 对比对象",
        "",
        "- `current_refined_default`：当前正式默认方案，`LSTM + base + refined 执行参数`",
        "- `lstm_alpha_v3_base`：`base + alpha_v3`",
        "- `lstm_alpha_v3_full`：`base_technical_risk_alpha_v3`",
        "- `case_zip_program`：压缩包内自带程序的可见结果口径",
        "",
        "## 核心结论",
        "",
    ]

    current = df[df["label"] == "current_refined_default"].iloc[0]
    alpha_base = df[df["label"] == "lstm_alpha_v3_base"].iloc[0]
    alpha_full = df[df["label"] == "lstm_alpha_v3_full"].iloc[0]
    case_zip = df[df["label"] == "case_zip_program"].iloc[0]

    lines.extend(
        [
            f"- 当前 refined 默认方案的本地成本后累计收益最高：`{current['cumulative_return_after_cost']:.6f}`，夏普为 `{current['sharpe_after_cost']:.6f}`。",
            f"- `alpha_v3_base` 的单切片自测分数从当前默认版的 `{current['score_self_case_slice']:.6f}` 提升到 `{alpha_base['score_self_case_slice']:.6f}`，但本地多期回测下降到 `{alpha_base['cumulative_return_after_cost']:.6f}`。",
            f"- `alpha_v3_full` 的单切片自测分数也高于当前默认版，为 `{alpha_full['score_self_case_slice']:.6f}`，但本地多期回测进一步降到 `{alpha_full['cumulative_return_after_cost']:.6f}`。",
            f"- 压缩包程序当前可见单切片自测分数为 `{case_zip['score_self_case_slice']:.6f}`，仍高于我方当前默认版和两组 alpha 候选。",
            "- 这说明新 alpha 特征在某个局部测试切片上有帮助，但还没有转化为更稳定的多期收益优势；当前默认方案暂时不应被替换。",
            "",
            "## 正式对比表",
            "",
            "| 方案 | 特征集 | 特征数 | rank_ic_mean | top5_mean_return_mean | cumulative_return_after_cost | sharpe_after_cost | max_drawdown_after_cost | avg_turnover | 压缩包单切片自测分数 | 压缩包公布最佳分数 | 说明 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for _, row in df.iterrows():
        lines.append(
            f"| {row['label']} | {format_value(row['feature_set'])} | {format_value(row['feature_count'])} | "
            f"{format_value(row['rank_ic_mean'])} | {format_value(row['top5_mean_return_mean'])} | "
            f"{format_value(row['cumulative_return_after_cost'])} | {format_value(row['sharpe_after_cost'])} | "
            f"{format_value(row['max_drawdown_after_cost'])} | {format_value(row['avg_turnover'])} | "
            f"{format_value(row['score_self_case_slice'])} | {format_value(row['case_best_final_score'])} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## 如何解读",
            "",
            "- `cumulative_return_after_cost / sharpe_after_cost / max_drawdown_after_cost / avg_turnover` 是我方统一本地回测口径，最能反映策略是否真的更赚钱、更稳。",
            "- `压缩包单切片自测分数` 来自压缩包自带 `score_self.py` 的单次测试切片，只能反映一个局部窗口。",
            "- 当前最好的下一步不是直接替换默认方案，而是保留 refined 默认版，继续沿着 alpha 方向做更小步的筛选和组合适配。",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        collect_row(
            label="current_refined_default",
            model_dir=ROOT_DIR / "app" / "model",
            backtest_summary_path=ROOT_DIR / "app" / "model" / "current_refined_backtest" / "backtest_summary.csv",
            result_path=ROOT_DIR / "app" / "output" / "result.csv",
            notes="当前正式默认方案，作为所有新实验的基线。",
        ),
        collect_row(
            label="lstm_alpha_v3_base",
            model_dir=ROOT_DIR / "app" / "model" / "lstm_alpha_v3_base",
            backtest_summary_path=ROOT_DIR / "app" / "model" / "lstm_alpha_v3_base" / "backtest" / "backtest_summary.csv",
            result_path=ROOT_DIR / "app" / "output" / "result_lstm_alpha_v3_base.csv",
            notes="base 特征上新增 alpha_v3，单切片改善，但本地多期回测明显弱于当前默认版。",
        ),
        collect_row(
            label="lstm_alpha_v3_full",
            model_dir=ROOT_DIR / "app" / "model" / "lstm_alpha_v3_full",
            backtest_summary_path=ROOT_DIR / "app" / "model" / "lstm_alpha_v3_full" / "backtest" / "backtest_summary.csv",
            result_path=ROOT_DIR / "app" / "output" / "result_lstm_alpha_v3_full.csv",
            notes="全量技术/风险/alpha_v3 特征，单切片优于当前默认版，但本地回测仍弱。",
        ),
        collect_row(
            label="case_zip_program",
            model_dir=None,
            backtest_summary_path=None,
            result_path=CASE_DIR / "output" / "result.csv",
            notes="压缩包程序可见结果。未提供完整滚动预测或统一回测轨迹，因此本地多期指标记为 N/A。",
        ),
    ]

    summary_df = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "alpha_feature_upgrade_summary.csv"
    report_path = OUTPUT_DIR / "alpha_feature_upgrade_report.md"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary_df, report_path)

    print(f"[alpha_feature_upgrade] wrote {summary_path}")
    print(f"[alpha_feature_upgrade] wrote {report_path}")


if __name__ == "__main__":
    main()
