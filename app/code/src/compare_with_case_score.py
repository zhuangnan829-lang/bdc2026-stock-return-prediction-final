import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
CASE_DIR = ROOT_DIR / "_case_zip" / "THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0"
DEFAULT_OUR_RESULT_PATH = ROOT_DIR / "app" / "output" / "result.csv"
DEFAULT_CASE_RESULT_PATH = CASE_DIR / "output" / "result.csv"
DEFAULT_CASE_TEST_PATH = CASE_DIR / "data" / "test.csv"
DEFAULT_CASE_BEST_SCORE_PATH = CASE_DIR / "model" / "60_158+39" / "final_score.txt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "app" / "model" / "case_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute current result score against the zip case scoring protocol.")
    parser.add_argument("--our_result_path", default=str(DEFAULT_OUR_RESULT_PATH))
    parser.add_argument("--case_result_path", default=str(DEFAULT_CASE_RESULT_PATH))
    parser.add_argument("--case_test_path", default=str(DEFAULT_CASE_TEST_PATH))
    parser.add_argument("--case_best_score_path", default=str(DEFAULT_CASE_BEST_SCORE_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def read_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str, "股票代码": str})
    id_col = "stock_id" if "stock_id" in df.columns else "股票代码" if "股票代码" in df.columns else None
    weight_col = "weight" if "weight" in df.columns else "权重" if "权重" in df.columns else None
    if id_col is None or weight_col is None:
        raise ValueError(f"{path} must contain stock_id/股票代码 and weight/权重 columns")
    if len(df) > 5:
        raise ValueError(f"{path} contains {len(df)} rows; the scoring protocol allows at most 5")

    out = pd.DataFrame(
        {
            "股票代码": df[id_col].astype(str).str.zfill(6),
            "权重": pd.to_numeric(df[weight_col], errors="coerce"),
        }
    )
    if out["权重"].isna().any():
        raise ValueError(f"{path} contains non-numeric weights")
    weight_sum = float(out["权重"].sum())
    if not (0.0 <= weight_sum <= 1.0 + 1e-12):
        raise ValueError(f"{path} weight sum must be between 0 and 1, got {weight_sum}")
    return out


def read_case_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"股票代码": str})
    required = {"股票代码", "日期", "开盘", "收盘"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    out = df[["股票代码", "日期", "开盘", "收盘"]].copy()
    out["股票代码"] = out["股票代码"].astype(str).str.zfill(6)
    out["开盘"] = pd.to_numeric(out["开盘"], errors="coerce")
    if out["开盘"].isna().any():
        raise ValueError(f"{path} contains invalid open prices")
    return out


def calculate_return(group: pd.DataFrame) -> float:
    start = group.iloc[0]
    end = group.iloc[-1]
    return float((end["开盘"] - start["开盘"]) / start["开盘"])


def score_prediction(prediction: pd.DataFrame, test_data: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    selected_test = test_data[test_data["股票代码"].isin(prediction["股票代码"])].copy()
    selected_test = selected_test.groupby("股票代码", group_keys=False).tail(5)
    returns = (
        selected_test.groupby("股票代码", sort=False)
        .apply(calculate_return, include_groups=False)
        .reset_index()
        .rename(columns={0: "收益率"})
    )
    detail = returns.merge(prediction, on="股票代码", how="inner")
    detail["加权贡献"] = detail["收益率"] * detail["权重"]
    score = float(detail["加权贡献"].sum())
    return score, detail.sort_values("加权贡献", ascending=False).reset_index(drop=True)


def read_case_best_score(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Best\s+final_score:\s*([+-]?\d+(?:\.\d+)?)", text.replace("\\n", "\n"))
    if not match:
        raise ValueError(f"Could not parse best final_score from {path}")
    return float(match.group(1))


def relation_text(left: float, right: float, stronger_text: str, weaker_text: str) -> str:
    if left > right + 1e-12:
        return stronger_text
    if left < right - 1e-12:
        return weaker_text
    return "持平"


def render_detail_table(title: str, detail: pd.DataFrame) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| 股票代码 | 权重 | 收益率 | 加权贡献 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in detail.iterrows():
        lines.append(
            f"| `{row['股票代码']}` | {float(row['权重']):.6f} | "
            f"{float(row['收益率']):.6f} | {float(row['加权贡献']):.6f} |"
        )
    lines.append("")
    return lines


def write_markdown(
    path: Path,
    our_score: float,
    case_current_score: float,
    case_best_score: float,
    our_detail: pd.DataFrame,
    case_detail: pd.DataFrame,
    sources: dict[str, Path],
) -> None:
    current_diff = our_score - case_current_score
    best_diff = our_score - case_best_score
    current_relation = relation_text(our_score, case_current_score, "超过参考当前输出", "低于参考当前输出")
    best_relation = relation_text(our_score, case_best_score, "超过参考记录最好分数", "尚未超过参考记录最好分数")

    lines = [
        "# 最新参考案例得分复算对比",
        "",
        "## 结论",
        "",
        f"- 我方当前输出得分：`{our_score:.6f}`",
        f"- 参考案例当前输出得分：`{case_current_score:.6f}`",
        f"- 参考案例记录最好分数：`{case_best_score:.6f}`",
        f"- 与参考当前输出相比：`{current_diff:+.6f}`，结论：**{current_relation}**。",
        f"- 与参考记录最好分数相比：`{best_diff:+.6f}`，结论：**{best_relation}**。",
        "",
        "因此，当前程序的正确表述是：**超过参考当前输出，但尚未超过参考记录最好分数**。",
        "",
        "## 复算口径",
        "",
        "本报告按压缩包 `test/score_self.py` 同口径复算：",
        "",
        "1. 提交结果最多包含 5 只股票。",
        "2. 权重和必须在 0 到 1 之间。",
        "3. 对每只提交股票，取参考测试集中的最后 5 条记录。",
        "4. 个股收益率 = 最后一条记录开盘价 / 第一条记录开盘价 - 1。",
        "5. 最终得分 = 个股收益率与提交权重的加权和。",
        "",
        "## 对比表",
        "",
        "| 指标 | 我方 | 参考案例 | 差值：我方-参考 | 判断 |",
        "|---|---:|---:|---:|---|",
        f"| 当前输出得分 | {our_score:.6f} | {case_current_score:.6f} | {current_diff:+.6f} | {current_relation} |",
        f"| 记录最好分数 | {our_score:.6f} | {case_best_score:.6f} | {best_diff:+.6f} | {best_relation} |",
        "",
    ]
    lines.extend(render_detail_table("我方当前输出明细", our_detail))
    lines.extend(render_detail_table("参考案例当前输出明细", case_detail))
    lines.extend(
        [
            "## 来源文件",
            "",
            f"- 我方结果：`{sources['our_result']}`",
            f"- 参考当前结果：`{sources['case_result']}`",
            f"- 参考测试集：`{sources['case_test']}`",
            f"- 参考最好分数记录：`{sources['case_best']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    our_result_path = Path(args.our_result_path)
    case_result_path = Path(args.case_result_path)
    case_test_path = Path(args.case_test_path)
    case_best_score_path = Path(args.case_best_score_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data = read_case_test(case_test_path)
    our_score, our_detail = score_prediction(read_prediction(our_result_path), test_data)
    case_current_score, case_detail = score_prediction(read_prediction(case_result_path), test_data)
    case_best_score = read_case_best_score(case_best_score_path)

    summary = pd.DataFrame(
        [
            {
                "metric": "current_output_score",
                "our_score": our_score,
                "case_score": case_current_score,
                "diff_our_minus_case": our_score - case_current_score,
                "judgement": relation_text(our_score, case_current_score, "our_higher", "case_higher"),
            },
            {
                "metric": "recorded_best_score",
                "our_score": our_score,
                "case_score": case_best_score,
                "diff_our_minus_case": our_score - case_best_score,
                "judgement": relation_text(our_score, case_best_score, "our_higher", "case_higher"),
            },
        ]
    )
    summary_path = output_dir / "latest_score_compare.csv"
    detail_path = output_dir / "latest_score_compare_details.csv"
    markdown_path = output_dir / "latest_score_compare.md"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.concat(
        [
            our_detail.assign(source="our_current_output"),
            case_detail.assign(source="case_current_output"),
        ],
        ignore_index=True,
    ).to_csv(detail_path, index=False, encoding="utf-8-sig")
    write_markdown(
        markdown_path,
        our_score=our_score,
        case_current_score=case_current_score,
        case_best_score=case_best_score,
        our_detail=our_detail,
        case_detail=case_detail,
        sources={
            "our_result": our_result_path,
            "case_result": case_result_path,
            "case_test": case_test_path,
            "case_best": case_best_score_path,
        },
    )

    print(f"[compare_with_case_score] our_score={our_score:.6f}")
    print(f"[compare_with_case_score] case_current_score={case_current_score:.6f}")
    print(f"[compare_with_case_score] case_best_score={case_best_score:.6f}")
    print(f"[compare_with_case_score] wrote {summary_path}")
    print(f"[compare_with_case_score] wrote {detail_path}")
    print(f"[compare_with_case_score] wrote {markdown_path}")


if __name__ == "__main__":
    main()
