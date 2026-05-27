from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT_DIR = Path(__file__).resolve().parents[3]
DOCS_DIR = ROOT_DIR / "app" / "docs"
ASSET_DIR = DOCS_DIR / "report_supplement_assets"
FORMAL_PATH = ROOT_DIR / "app" / "model" / "formal_model_comparison" / "formal_model_comparison.csv"
FOLD_STAGE_PATH = ROOT_DIR / "app" / "model" / "market_regime_analysis" / "fold_stage_performance.csv"
RULE_STAGE_PATH = ROOT_DIR / "app" / "model" / "market_regime_analysis" / "rule_stage_performance.csv"
BACKTEST_PATH = ROOT_DIR / "app" / "model" / "backtest_summary.csv"
TABLE_MD_PATH = DOCS_DIR / "report_supplement_tables.md"


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BG = "#f4f7fb"
PANEL = "#ffffff"
TEXT = "#102a43"
SUBTEXT = "#486581"
BORDER = "#d9e2ec"
PRIMARY = "#225ea8"
PRIMARY_SOFT = "#dce8f8"
GREEN = "#2f855a"
RED = "#d64545"
GOLD = "#c2872f"


def ensure_dir() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def format_seq(value) -> str:
    if pd.isna(value):
        return "-"
    try:
        return str(int(value))
    except Exception:
        return str(value)


def display_model_name(name: str) -> str:
    mapping = {
        "LSTM sl20": "LSTM sl20",
        "Momentum (mom_5d)": "动量基线",
        "XGBoost": "XGBoost",
        "LightGBM": "LightGBM",
        "Linear Regression": "线性回归",
    }
    return mapping.get(name, name)


def write_tables_md() -> None:
    formal_df = pd.read_csv(FORMAL_PATH)
    fold_df = pd.read_csv(FOLD_STAGE_PATH)
    regime_df = pd.read_csv(RULE_STAGE_PATH)
    backtest_df = pd.read_csv(BACKTEST_PATH)

    selected_regime = regime_df[regime_df["阶段名"].isin(["低波动", "高波动", "趋势", "震荡", "高波动-震荡"])]

    lines = [
        "# 中期报告补充表格",
        "",
        "下面这些表格可直接复制到 Word 中使用。",
        "",
        "## 表1 正式模型对比表",
        "",
        "| 模型 | 特征集 | sequence_length | MAE | RMSE | RankIC | NDCG@5 | NDCG@10 | NDCG@20 | HitRate@5 | HitRate@10 | HitRate@20 | Top5平均收益 | 回测累计收益 | Sharpe | 最大回撤 | 是否正式候选 | 备注 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in formal_df.iterrows():
        lines.append(
            f"| {row['模型名']} | {row['特征集']} | {format_seq(row['sequence_length'])} | "
            f"{row['MAE']:.6f} | {row['RMSE']:.6f} | {row['RankIC']:.6f} | "
            f"{row['NDCG@5']:.6f} | {row['NDCG@10']:.6f} | {row['NDCG@20']:.6f} | "
            f"{row['HitRate@5']:.6f} | {row['HitRate@10']:.6f} | {row['HitRate@20']:.6f} | "
            f"{row['Top5平均收益']:.6f} | {row['回测累计收益']:.6f} | {row['Sharpe']:.6f} | {row['最大回撤']:.6f} | "
            f"{row['是否正式候选']} | {row['说明']} |"
        )

    lines.extend(
        [
            "",
            "## 表2 Walk-Forward 分折表现表",
            "",
            "| 折次 | 样本天数 | RankIC | Top5平均收益 | 阶段回测收益 | 阶段判断 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in fold_df.iterrows():
        lines.append(
            f"| {row['阶段名']} | {int(row['样本天数'])} | {row['RankIC']:.6f} | "
            f"{row['Top5平均收益']:.6f} | {row['回测收益']:.6f} | {row['结论']} |"
        )

    lines.extend(
        [
            "",
            "## 表3 市场阶段分析表",
            "",
            "| 阶段名 | 样本天数 | RankIC | Top5平均收益 | 阶段回测收益 | 结论 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in selected_regime.iterrows():
        lines.append(
            f"| {row['阶段名']} | {int(row['样本天数'])} | {row['RankIC']:.6f} | "
            f"{row['Top5平均收益']:.6f} | {row['回测收益']:.6f} | {row['结论']} |"
        )

    lines.extend(
        [
            "",
            "## 表4 回测配置对比表",
            "",
            "| 回测配置 | 成本后累计收益 | 成本后最大回撤 | 成本后Sharpe | 成本后胜率 | 平均换手率 | 备注 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in backtest_df.iterrows():
        note = "低换手版本综合表现更优" if str(row["profile_name"]) == "looser_risk_low_turnover" else ""
        lines.append(
            f"| {row['profile_name']} | {row['cumulative_return_after_cost']:.6f} | "
            f"{row['max_drawdown_after_cost']:.6f} | {row['sharpe_after_cost']:.6f} | "
            f"{row['win_rate_after_cost']:.6f} | {row['avg_turnover']:.6f} | {note} |"
        )

    lines.extend(
        [
            "",
            "## 表5 Demo 主流程表",
            "",
            "| 步骤 | 操作命令 | 预期看到的结果 |",
            "|---|---|---|",
            "| 数据与研究入口 | `bash /app/run_research_pipeline.sh` | 训练、Walk-Forward、回测、诊断形成闭环 |",
            "| 正式推理入口 | `bash /app/run_submission.sh` | 生成 `app/output/result.csv` |",
            "| 冻结提交流程 | `bash /app/freeze_submission.sh` | 完成结果校验、自检与冻结 |",
            "| Docker 入口 | `docker build -t bdc2026 .` / `docker compose up` | 容器内正式提交流程可复现 |",
            "",
            "## 表6 关键结果文件说明表",
            "",
            "| 文件/目录 | 用途 | 建议放置位置 |",
            "|---|---|---|",
            "| `app/temp/train_features.csv` | 统一输入特征表 | 第三部分数据处理与特征工程 |",
            "| `app/output/result.csv` | 最终正式提交文件 | 第七部分系统实现与演示进展 |",
            "| `app/model/formal_model_comparison/formal_model_comparison.csv` | 正式模型统一对比结果 | 第五部分阶段实验结果与诊断分析 |",
            "| `app/model/market_regime_analysis/market_regime_analysis.md` | 市场阶段分析说明 | 第五部分稳定性分析后 |",
            "| `app/model/submission_artifacts/` | 冻结提交配置与模型产物 | 第七部分系统实现与演示进展 |",
        ]
    )

    TABLE_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def _make_card(ax, x, y, w, h, title, body, fc="#f8fbff", ec="#8db6d9", title_fc="#d9eaf7", body_size=11):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02", fc=fc, ec=ec, lw=1.6)
    ax.add_patch(patch)
    title_patch = FancyBboxPatch((x, y + h - 0.12), w, 0.12, boxstyle="round,pad=0.01,rounding_size=0.02", fc=title_fc, ec=ec, lw=1.2)
    ax.add_patch(title_patch)
    ax.text(x + 0.02, y + h - 0.06, title, va="center", ha="left", fontsize=12, fontweight="bold", color="#12344d")
    ax.text(
        x + 0.02,
        y + h - 0.15,
        _wrap_for_box(body, w, body_size),
        va="top",
        ha="left",
        fontsize=body_size,
        color="#243b53",
        linespacing=1.35,
        clip_on=True,
    )


def _setup_canvas(fig):
    fig.patch.set_facecolor(BG)


def _rounded_panel(ax, x, y, w, h, fc=PANEL, ec=BORDER, lw=1.2, radius=0.02):
    panel = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={radius}", fc=fc, ec=ec, lw=lw)
    ax.add_patch(panel)
    return panel


def _title(ax, title, subtitle=""):
    ax.text(0.04, 0.975, title, ha="left", va="top", fontsize=20, fontweight="bold", color=TEXT)
    if subtitle:
        ax.text(0.04, 0.935, subtitle, ha="left", va="top", fontsize=10.5, color=SUBTEXT)


def _metric_badge(ax, x, y, w, h, title, value, note="", color=PRIMARY):
    _rounded_panel(ax, x, y, w, h, fc=PANEL, ec=BORDER, lw=1.1)
    ax.add_patch(Rectangle((x + 0.01, y + h - 0.020), w - 0.02, 0.012, color=color, ec="none"))
    ax.text(x + 0.018, y + h - 0.054, title, ha="left", va="top", fontsize=10, color=SUBTEXT)
    ax.text(x + 0.018, y + 0.060, value, ha="left", va="bottom", fontsize=16.5, fontweight="bold", color=TEXT)
    if note:
        ax.text(
            x + 0.018,
            y + 0.020,
            _wrap_for_box(note, w - 0.03, 9.0),
            ha="left",
            va="bottom",
            fontsize=8.2,
            color=SUBTEXT,
            linespacing=1.2,
            clip_on=True,
        )


def _wrap_for_box(text: str, width: float, fontsize: float) -> str:
    char_cap = max(8, int(width * 90 / max(fontsize / 10.0, 0.85)))
    wrapped_lines = []
    for raw in str(text).split("\n"):
        if raw.strip() == "":
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(
            textwrap.wrap(raw, width=char_cap, break_long_words=False, break_on_hyphens=False) or [raw]
        )
    return "\n".join(wrapped_lines)


def save_demo_flowchart_png() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    _setup_canvas(fig)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _title(ax, "系统主流程图", "从数据输入、模型训练、策略验证到冻结提交与容器复现的完整流程")

    left_x, gap = 0.06, 0.045
    widths = [0.16, 0.16, 0.16, 0.16, 0.16]
    labels = [
        ("1 数据输入", "原始行情\n特征构建"),
        ("2 研究主流程", "训练\nWalk-Forward\n诊断"),
        ("3 策略验证", "回测\n阶段分析\n结果对比"),
        ("4 冻结提交", "正式推理\n结果校验\n配置冻结"),
        ("5 Docker 入口", "容器复现\n提交彩排"),
    ]
    colors = ["#e9f2ff", "#edf8f2", "#fff8e8", "#fff0f0", "#f3efff"]
    border_colors = ["#90b4e8", "#8ec9a6", "#dec06f", "#e5a5a5", "#b8a2e8"]
    y = 0.50
    h = 0.22
    for i, ((title, body), w, fc, ec) in enumerate(zip(labels, widths, colors, border_colors)):
        x = left_x + i * (w + gap)
        _rounded_panel(ax, x, y, w, h, fc=fc, ec=ec, lw=1.5, radius=0.03)
        ax.text(x + 0.02, y + h - 0.04, title, ha="left", va="top", fontsize=13, fontweight="bold", color=TEXT)
        ax.text(x + w / 2, y + 0.075, body, ha="center", va="center", fontsize=12, color=TEXT, linespacing=1.6)
        if i < len(labels) - 1:
            ax.annotate("", xy=(x + w + gap - 0.01, y + h / 2), xytext=(x + w + 0.01, y + h / 2),
                        arrowprops=dict(arrowstyle="->", lw=2.0, color="#7b8794"))

    detail_cards = [
        (0.07, 0.16, 0.26, 0.20, "研究流程产物", "训练结果、Walk-Forward 验证结果、回测结果与诊断分析文件。"),
        (0.37, 0.16, 0.26, 0.20, "正式提交流程产物", "冻结模型、正式推理结果 app/output/result.csv 与自检结果。"),
        (0.67, 0.16, 0.26, 0.20, "流程特征", "研究链路、提交链路与 Docker 链路口径统一，可复现性较强。"),
    ]
    for x, y, w, h, title, body in detail_cards:
        _make_card(ax, x, y, w, h, title, body, fc=PANEL, ec=BORDER, title_fc=PRIMARY_SOFT, body_size=10.0)

    plt.tight_layout(pad=1.4)
    fig.savefig(ASSET_DIR / "demo_flowchart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_formal_model_comparison_png() -> None:
    df = pd.read_csv(FORMAL_PATH)
    df["综合得分"] = (
        df["RankIC"].rank(pct=True) * 0.30
        + df["Top5平均收益"].rank(pct=True) * 0.25
        + df["回测累计收益"].rank(pct=True) * 0.25
        + df["Sharpe"].rank(pct=True) * 0.20
    )
    df = df.sort_values("综合得分", ascending=True).reset_index(drop=True)
    display_names = [display_model_name(x) for x in df["模型名"]]

    fig = plt.figure(figsize=(16, 10))
    _setup_canvas(fig)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    _title(canvas, "正式模型对比图", "基于排序能力、推荐收益与回测表现的统一比较")

    best = df.iloc[-1]
    _metric_badge(canvas, 0.04, 0.80, 0.22, 0.105, "当前正式候选模型", "LSTM sl20", "同特征集主线方案", PRIMARY)
    _metric_badge(canvas, 0.29, 0.80, 0.17, 0.105, "最高 RankIC", f"{best['RankIC']:.6f}", "排序能力最强", GREEN)
    _metric_badge(canvas, 0.49, 0.80, 0.17, 0.105, "最高 Top5平均收益", f"{best['Top5平均收益']:.6f}", "推荐收益最好", GOLD)
    _metric_badge(canvas, 0.69, 0.80, 0.17, 0.105, "最高 Sharpe", f"{best['Sharpe']:.3f}", "收益风险比最优", "#805ad5")

    bar_ax = fig.add_axes([0.08, 0.18, 0.35, 0.53], facecolor=PANEL)
    for spine in bar_ax.spines.values():
        spine.set_color(BORDER)
    colors = [PRIMARY if name == "LSTM sl20" else "#9fb3c8" for name in df["模型名"]]
    bar_ax.barh(display_names, df["综合得分"], color=colors, edgecolor="none", height=0.58)
    bar_ax.set_title("综合表现排序", fontsize=14, fontweight="bold", color=TEXT, pad=12)
    bar_ax.set_xlabel("综合得分（按 RankIC、Top5平均收益、回测累计收益、Sharpe 加权）", fontsize=10, color=SUBTEXT)
    bar_ax.grid(axis="x", alpha=0.18)
    bar_ax.tick_params(axis="y", labelsize=10, colors=TEXT)
    bar_ax.tick_params(axis="x", colors=SUBTEXT)
    for y, v in enumerate(df["综合得分"]):
        bar_ax.text(v + 0.01, y, f"{v:.2f}", va="center", ha="left", fontsize=10, color=TEXT)

    heat_ax = fig.add_axes([0.50, 0.27, 0.45, 0.44], facecolor=PANEL)
    metric_cols = ["RankIC", "NDCG@20", "HitRate@20", "Top5平均收益", "回测累计收益", "Sharpe"]
    metric_labels = ["RankIC", "NDCG@20", "命中率@20", "Top5平均收益", "回测累计收益", "Sharpe"]
    norm_df = df[metric_cols].copy()
    for col in metric_cols:
        col_min = norm_df[col].min()
        col_max = norm_df[col].max()
        if abs(col_max - col_min) < 1e-12:
            norm_df[col] = 0.5
        else:
            norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
    cmap = LinearSegmentedColormap.from_list("custom_cn", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5"])
    heat_ax.imshow(norm_df.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    heat_ax.set_title("关键指标热力对比", fontsize=14, fontweight="bold", color=TEXT, pad=12)
    heat_ax.set_xticks(np.arange(len(metric_labels)))
    heat_ax.set_xticklabels(metric_labels, fontsize=10, color=TEXT)
    heat_ax.set_yticks(np.arange(len(display_names)))
    heat_ax.set_yticklabels(display_names, fontsize=10.5, color=TEXT)
    for i in range(len(df)):
        for j, col in enumerate(metric_cols):
            heat_ax.text(j, i, f"{df.iloc[i][col]:.3f}", ha="center", va="center", fontsize=9, color=TEXT)
    for spine in heat_ax.spines.values():
        spine.set_color(BORDER)

    note_ax = fig.add_axes([0.50, 0.10, 0.45, 0.11])
    note_ax.axis("off")
    _rounded_panel(note_ax, 0.0, 0.0, 1.0, 1.0, fc=PANEL, ec=BORDER, lw=1.1, radius=0.03)
    note_text = (
        "LSTM sl20 在 RankIC、Top5平均收益、回测累计收益和 Sharpe 上同时占优，"
        "说明该方案能够较稳定地将排序能力转化为组合收益。动量基线在 NDCG 和命中率上更高，"
        "但收益转化能力偏弱，因此未作为正式主线方案。"
    )
    note_ax.text(0.03, 0.62, "结论说明", ha="left", va="center", fontsize=12, fontweight="bold", color=TEXT)
    note_ax.text(0.03, 0.28, _wrap_for_box(note_text, 0.92, 9.6), ha="left", va="center", fontsize=9.6, color=SUBTEXT, linespacing=1.3, clip_on=True)

    fig.savefig(ASSET_DIR / "formal_model_comparison_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_market_regime_png() -> None:
    df = pd.read_csv(RULE_STAGE_PATH)
    stage_order = ["低波动", "高波动", "趋势", "震荡", "高波动-震荡"]
    use_df = df[df["阶段名"].isin(stage_order)].copy()
    use_df["阶段名"] = pd.Categorical(use_df["阶段名"], categories=stage_order, ordered=True)
    use_df = use_df.sort_values("阶段名")

    fig = plt.figure(figsize=(15, 9.5))
    _setup_canvas(fig)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    _title(canvas, "市场阶段表现图", "不同市场环境下模型排序能力与推荐收益表现")

    _metric_badge(canvas, 0.04, 0.845, 0.20, 0.075, "最优规则阶段", "低波动", "排序稳定，收益转化较强", GREEN)
    _metric_badge(canvas, 0.27, 0.845, 0.20, 0.075, "最弱规则阶段", "高波动-震荡", "噪声较高，趋势较弱", RED)
    _metric_badge(canvas, 0.50, 0.845, 0.18, 0.075, "低波动 RankIC", f"{float(use_df[use_df['阶段名']=='低波动']['RankIC'].iloc[0]):.6f}", "高于高波动阶段", PRIMARY)
    _metric_badge(canvas, 0.71, 0.845, 0.18, 0.075, "趋势 Top5平均收益", f"{float(use_df[use_df['阶段名']=='趋势']['Top5平均收益'].iloc[0]):.6f}", "优于震荡阶段", GOLD)

    colors = ["#4c9f70" if s in ["低波动", "趋势"] else "#d96c6c" if s == "高波动-震荡" else "#7aa6d1" for s in use_df["阶段名"]]
    ax1 = fig.add_axes([0.08, 0.20, 0.36, 0.54], facecolor=PANEL)
    ax2 = fig.add_axes([0.56, 0.20, 0.36, 0.54], facecolor=PANEL)
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.grid(axis="x", alpha=0.16)
        ax.tick_params(colors=TEXT, labelsize=10)

    ax1.barh(use_df["阶段名"], use_df["RankIC"], color=colors, edgecolor="none", height=0.6)
    ax1.set_title("不同市场阶段的 RankIC", fontsize=13.5, fontweight="bold", color=TEXT, pad=10)
    for y, v in enumerate(use_df["RankIC"]):
        ax1.text(v + (0.0015 if v >= 0 else -0.006), y, f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=10, color=TEXT)

    ax2.barh(use_df["阶段名"], use_df["Top5平均收益"], color=colors, edgecolor="none", height=0.6)
    ax2.set_title("不同市场阶段的 Top5平均收益", fontsize=13.5, fontweight="bold", color=TEXT, pad=10)
    for y, v in enumerate(use_df["Top5平均收益"]):
        ax2.text(v + (0.0005 if v >= 0 else -0.0015), y, f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=10, color=TEXT)

    summary_ax = fig.add_axes([0.06, 0.05, 0.88, 0.09])
    summary_ax.axis("off")
    _rounded_panel(summary_ax, 0.0, 0.0, 1.0, 1.0, fc=PANEL, ec=BORDER, lw=1.0, radius=0.02)
    summary_ax.text(
        0.02,
        0.5,
        _wrap_for_box("阶段分析表明：模型在低波动、趋势较明确的环境中更容易将排序信号转化为组合收益；进入高波动且偏震荡阶段后，短期信号联动更容易放大噪声，从而削弱排序稳定性。", 0.95, 9.8),
        ha="left",
        va="center",
        fontsize=9.8,
        color=SUBTEXT,
        linespacing=1.25,
    )
    fig.savefig(ASSET_DIR / "market_regime_analysis_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_key_commands_png() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    _setup_canvas(fig)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _title(ax, "关键命令与流程入口", "研究链路、正式推理链路、冻结提交流程与 Docker 复现入口")
    cards = [
        ("1 研究主流程", "命令：\nbash /app/run_research_pipeline.sh\n\n用途：\n训练模型、Walk-Forward 验证、冻结推理快照、本地回测与诊断。"),
        ("2 正式推理入口", "命令：\nbash /app/run_submission.sh\n\n用途：\n默认使用冻结模型，直接生成 app/output/result.csv。"),
        ("3 冻结提交流程", "命令：\nbash /app/freeze_submission.sh\n\n用途：\n同步正式配置、校验结果、执行 pre_submit_check。"),
        ("4 Docker 复现", "命令：\ndocker build -t bdc2026 .\ndocker compose up\n\n用途：\n本地流程与容器入口保持一致，可复现提交。"),
    ]
    positions = [(0.05, 0.50), (0.53, 0.50), (0.05, 0.13), (0.53, 0.13)]
    for (title, body), (x, y) in zip(cards, positions):
        _make_card(ax, x, y, 0.40, 0.30, title, body, fc=PANEL, ec=BORDER, title_fc=PRIMARY_SOFT, body_size=9.8)
    _rounded_panel(ax, 0.05, 0.04, 0.88, 0.05, fc="#eaf3ff", ec="#c7d7ea", lw=1.0, radius=0.015)
    ax.text(0.49, 0.065, "四类入口分别对应研究实验、正式推理、提交冻结与容器复现，可共同构成完整工程闭环。", ha="center", va="center", fontsize=10.0, color=TEXT, clip_on=True)
    fig.savefig(ASSET_DIR / "demo_key_commands.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_result_files_png() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    _setup_canvas(fig)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _title(ax, "关键结果文件说明", "中期阶段的主要输入、输出与冻结产物")
    cards = [
        ("输入特征表", "app/temp/train_features.csv\n训练与验证统一使用的结构化特征表，说明数据输入已标准化。"),
        ("正式提交结果", "app/output/result.csv\n最终正式提交文件，记录推荐股票及权重。"),
        ("模型对比结果", "app/model/formal_model_comparison/formal_model_comparison.csv\n统一展示主线模型与各类基线。"),
        ("市场阶段分析", "app/model/market_regime_analysis/market_regime_analysis.md\n解释不同折次和市场状态下的表现差异。"),
        ("冻结提交产物", "app/model/submission_artifacts/\n保存正式冻结模型、配置快照和提交最小产物。"),
    ]
    positions = [(0.06, 0.60, 0.39, 0.23), (0.53, 0.60, 0.39, 0.23), (0.06, 0.32, 0.39, 0.23), (0.53, 0.32, 0.39, 0.23), (0.295, 0.04, 0.41, 0.20)]
    for (title, body), (x, y, w, h) in zip(cards, positions):
        _make_card(ax, x, y, w, h, title, body, fc=PANEL, ec=BORDER, title_fc=PRIMARY_SOFT, body_size=10.0)
    fig.savefig(ASSET_DIR / "demo_result_files.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_report_insert_guide_png() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    _setup_canvas(fig)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _title(ax, "新增图件目录", "报告中新增图件及其对应位置")
    body = (
        "图7 Demo 主流程图\n"
        "位置：第七部分 7.2 可演示主流程\n\n"
        "图8 正式模型对比图\n"
        "位置：第五部分 5.1 模型总体对比后\n\n"
        "图9 市场阶段表现图\n"
        "位置：第五部分 5.2 Walk-Forward 分折表现后\n\n"
        "图10 关键命令截图页\n"
        "位置：第七部分或附录\n\n"
        "图11 结果文件说明页\n"
        "位置：第七部分系统实现与演示进展后"
    )
    _make_card(ax, 0.10, 0.12, 0.80, 0.72, "图件位置说明", body, fc=PANEL, ec=BORDER, title_fc=PRIMARY_SOFT, body_size=11.5)
    fig.savefig(ASSET_DIR / "report_image_insert_guide.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir()
    write_tables_md()
    save_demo_flowchart_png()
    save_formal_model_comparison_png()
    save_market_regime_png()
    save_key_commands_png()
    save_result_files_png()
    save_report_insert_guide_png()
    print(f"[report_supplement_assets] wrote {TABLE_MD_PATH}")
    print(f"[report_supplement_assets] wrote assets to {ASSET_DIR}")


if __name__ == "__main__":
    main()
