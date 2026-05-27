from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
TEMP_DIR = ROOT / "temp"
OUTPUT_DIR = ROOT / "docs" / "figures" / "midterm"


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#fcfcfd"
    plt.rcParams["axes.edgecolor"] = "#d0d7de"
    plt.rcParams["grid.color"] = "#e5e7eb"
    plt.rcParams["grid.linewidth"] = 0.8
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_current(fig_name: str) -> Path:
    out = OUTPUT_DIR / fig_name
    plt.tight_layout()
    plt.savefig(out, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


def plot_label_distribution() -> Path:
    df = pd.read_csv(TEMP_DIR / "train_features.csv", usecols=["target_return"])
    series = df["target_return"].dropna()
    plt.figure(figsize=(9.2, 5.4))
    ax = sns.histplot(series, bins=90, kde=True, color="#2563eb", edgecolor="white", alpha=0.92)
    mean_value = series.mean()
    ax.axvline(mean_value, color="#dc2626", linestyle="--", linewidth=1.8, label=f"均值 {mean_value:.4f}")
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    plt.title("图1 未来5日收益标签分布", pad=12, fontsize=16, fontweight="bold")
    plt.xlabel("未来5日收益率标签")
    plt.ylabel("样本数")
    return save_current("fig1_label_distribution.png")


def plot_fold_rankic() -> Path:
    df = pd.read_csv(MODEL_DIR / "walk_forward_metrics.csv")
    plt.figure(figsize=(8.2, 5.4))
    df["折次"] = df["fold_id"].map(lambda value: f"第{value}折")
    df["颜色"] = df["rank_ic"].map(lambda value: "负值" if value < 0 else "正值")
    ax = sns.barplot(
        data=df,
        x="折次",
        y="rank_ic",
        hue="颜色",
        dodge=False,
        palette={"负值": "#dc2626", "正值": "#16a34a"},
        legend=False,
    )
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_ylim(min(-0.08, df["rank_ic"].min() - 0.02), max(0.14, df["rank_ic"].max() + 0.03))
    for idx, row in df.reset_index(drop=True).iterrows():
        y_pos = row["rank_ic"] + (0.004 if row["rank_ic"] >= 0 else -0.012)
        ax.text(idx, y_pos, f"{row['rank_ic']:.4f}", ha="center", fontsize=10)
    plt.title("图2 当前正式方案 Walk-Forward 分折 RankIC", pad=12, fontsize=16, fontweight="bold")
    plt.xlabel("验证折次")
    plt.ylabel("RankIC")
    return save_current("fig2_fold_rankic.png")


def plot_equity_curve() -> Path:
    df = pd.read_csv(MODEL_DIR / "backtest_daily_results.csv")
    df = df[df["profile_name"] == "default_risk_adjusted"].copy()
    df["date"] = pd.to_datetime(df["date"])
    plt.figure(figsize=(9.4, 5.4))
    ax = sns.lineplot(data=df, x="date", y="net_value_after_cost", linewidth=2.5, color="#2563eb")
    ax.fill_between(df["date"], df["net_value_after_cost"], df["net_value_after_cost"].min(), color="#93c5fd", alpha=0.18)
    plt.title("图3 默认回测配置净值曲线", pad=12, fontsize=16, fontweight="bold")
    plt.xlabel("日期")
    plt.ylabel("成本后净值")
    return save_current("fig3_equity_curve.png")


def plot_drawdown_curve() -> Path:
    df = pd.read_csv(MODEL_DIR / "backtest_daily_results.csv")
    df = df[df["profile_name"] == "default_risk_adjusted"].copy()
    df["date"] = pd.to_datetime(df["date"])
    plt.figure(figsize=(9.4, 5.4))
    sns.lineplot(data=df, x="date", y="drawdown_after_cost", linewidth=2.5, color="#dc2626")
    plt.fill_between(df["date"], df["drawdown_after_cost"], 0, color="#fca5a5", alpha=0.26)
    plt.title("图4 默认回测配置回撤曲线", pad=12, fontsize=16, fontweight="bold")
    plt.xlabel("日期")
    plt.ylabel("成本后回撤")
    return save_current("fig4_drawdown_curve.png")


def plot_ablation_comparison() -> Path:
    df = pd.read_csv(MODEL_DIR / "fine_short_term_ablation_comparison.csv")
    label_map = {
        "lstm_sl20_current": "当前sl20",
        "drop_ret_1d": "去ret_1d",
        "drop_ret_3d": "去ret_3d",
        "drop_intraday_return": "去日内收益",
        "drop_ret_1d_intraday_return": "去ret_1d+日内收益",
    }
    metric_map = {
        "rank_ic_mean": "整体RankIC均值",
        "fold_1_rank_ic": "第一折RankIC",
    }
    df["方案"] = df["experiment"].map(label_map).fillna(df["experiment"])
    melted = df.melt(
        id_vars=["方案"],
        value_vars=["rank_ic_mean", "fold_1_rank_ic"],
        var_name="指标",
        value_name="数值",
    )
    melted["指标"] = melted["指标"].map(metric_map)
    plt.figure(figsize=(10.5, 5.8))
    sns.barplot(data=melted, x="方案", y="数值", hue="指标", palette=["#2563eb", "#f59e0b"])
    plt.axhline(0.0, color="#4a5568", linewidth=1.0)
    plt.title("图5 短期特征消融实验对比", pad=12, fontsize=16, fontweight="bold")
    plt.xlabel("实验方案")
    plt.ylabel("RankIC")
    plt.xticks(rotation=10, ha="right")
    plt.legend(title="", frameon=False)
    return save_current("fig5_short_term_ablation.png")


def plot_ticket_diagnostics() -> Path:
    df = pd.read_csv(MODEL_DIR / "fold1_short_term_ticket_diagnostics.csv", dtype={"stock_id": str})
    stock_order = ["300502", "000807", "601377", "603019"]
    sub = df[df["stock_id"].isin(stock_order)].copy()
    sub["trade_date"] = pd.to_datetime(sub["trade_date"])
    sub["短期特征合力"] = sub["ret_1d"] + sub["ret_3d"] + sub["intraday_return"]
    sub["排名前移幅度"] = -sub["rank_pred_pct_delta"]
    sub["误差改善幅度"] = sub["rank_error_improvement"]
    stock_labels = {
        "300502": "300502（改善型）",
        "000807": "000807（改善型）",
        "601377": "601377（恶化型）",
        "603019": "603019（恶化型）",
    }
    metric_order = ["短期特征合力", "排名前移幅度", "误差改善幅度"]
    sub["股票标签"] = sub["stock_id"].map(stock_labels)
    sub["交易日期"] = sub["trade_date"].dt.strftime("%m-%d")
    melted = sub.melt(
        id_vars=["股票标签", "交易日期"],
        value_vars=metric_order,
        var_name="指标",
        value_name="数值",
    )
    pivot = melted.pivot_table(index=["股票标签", "指标"], columns="交易日期", values="数值")
    plt.figure(figsize=(15.8, 7.6))
    ax = sns.heatmap(
        pivot,
        cmap=sns.color_palette("vlag", as_cmap=True),
        center=0.0,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "相对强弱"},
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.title("图6 第1折代表性股票短期特征与排序诊断热力图", pad=14, fontsize=16, fontweight="bold")
    plt.xlabel("交易日期")
    plt.ylabel("")
    return save_current("fig6_ticket_diagnostics.png")


def main() -> None:
    setup_style()
    outputs = [
        plot_label_distribution(),
        plot_fold_rankic(),
        plot_equity_curve(),
        plot_drawdown_curve(),
        plot_ablation_comparison(),
        plot_ticket_diagnostics(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
