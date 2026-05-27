from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "model"
OUTPUT_DIR = APP_DIR / "output"

PACKAGE_ZIP_PATH = Path(r"D:\Desktop\THU-BDC2026-aggressive-score-entry-fixed-clean-docker-ok_20260526_chinese.zip")
DEFAULT_CONFIG_PATH = MODEL_DIR / "default_submission_config.json"
MODEL_META_PATH = MODEL_DIR / "model_meta.json"
PACKAGE_VARIANT_PATH = MODEL_DIR / "package_variant.json"
AGGRESSIVE_CONFIG_PATH = MODEL_DIR / "aggressive_score_submission_candidate" / "submission_aggressive_score_candidate.json"
SCORE_COMPARE_PATH = MODEL_DIR / "aggressive_score_submission_candidate" / "case_score_recheck" / "latest_score_compare.csv"
RESULT_PATH = OUTPUT_DIR / "result.csv"
PREDICT_SCORES_PATH = OUTPUT_DIR / "predict_scores.csv"
DEBUG_CANDIDATES_PATH = OUTPUT_DIR / "debug_candidates.csv"
LEADERBOARD_PATH = MODEL_DIR / "experiment_leaderboard.csv"
BACKTEST_SUMMARY_PATH = MODEL_DIR / "backtest_summary.csv"
STABILITY_SUMMARY_PATH = MODEL_DIR / "stability_eval" / "stability_summary.csv"
BACKTEST_STRESS_PATH = MODEL_DIR / "backtest_stress_test" / "backtest_stress_summary.csv"
WALK_FORWARD_METRICS_PATH = MODEL_DIR / "walk_forward_metrics.csv"
WALK_FORWARD_PREDICTIONS_PATH = MODEL_DIR / "walk_forward_predictions.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


st.set_page_config(
    page_title="沪深300收益预测演示系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --panel-bg: rgba(244, 251, 255, 0.78);
            --panel-border: rgba(255, 255, 255, 0.94);
            --active-pink: #f7c9cc;
            --accent-rose: #b8847d;
            --text-main: #263444;
            --text-soft: #667788;
        }

        .stApp {
            color: var(--text-main);
            background:
                radial-gradient(circle at 9% 11%, rgba(255, 255, 255, 0.86), transparent 18%),
                radial-gradient(circle at 19% 17%, rgba(129, 210, 255, 0.52), transparent 25%),
                radial-gradient(circle at 78% 18%, rgba(255, 255, 255, 0.64), transparent 24%),
                linear-gradient(120deg, #dbf1ff 0%, #eff9ff 47%, #f5f9fd 100%);
            background-attachment: fixed;
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background:
                linear-gradient(90deg, rgba(255,255,255,0.18), transparent 22% 78%, rgba(210,232,246,0.24)),
                radial-gradient(circle at 38% 92%, rgba(247,201,204,0.24), transparent 28%);
        }

        [data-testid="stHeader"] { background: rgba(232, 245, 255, 0.72); }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu {
            display: none;
        }
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 18% 7%, rgba(255,255,255,0.95), transparent 16%),
                linear-gradient(180deg, rgba(218, 240, 253, 0.88), rgba(246, 251, 255, 0.78));
            border-right: 1px solid rgba(255,255,255,0.78);
            box-shadow: 24px 0 64px rgba(94, 144, 174, 0.15);
            backdrop-filter: blur(22px) saturate(130%);
            -webkit-backdrop-filter: blur(22px) saturate(130%);
        }
        [data-testid="stSidebar"] * { color: var(--text-main); }

        .sidebar-title {
            color: #a77a72;
            font-size: 1.65rem;
            line-height: 1.1;
            font-weight: 850;
            margin: 14px 0 28px;
        }

        [data-testid="stSidebar"] [role="radiogroup"] {
            display: grid;
            gap: 14px;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            min-height: 58px;
            padding: 0 18px;
            border-radius: 999px;
            border: 1px solid transparent;
            background: transparent;
            transition: background 160ms ease, box-shadow 160ms ease, transform 160ms ease;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.48);
            transform: translateX(2px);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(90deg, rgba(255,255,255,0.86), var(--active-pink));
            border-color: rgba(255,255,255,0.96);
            box-shadow:
                inset 1px 1px 0 rgba(255,255,255,0.95),
                0 16px 34px rgba(184, 132, 125, 0.18);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label p {
            color: var(--text-main);
            font-size: 1.06rem;
            font-weight: 850;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
            display: none;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label > div:last-child {
            width: 100%;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1480px;
        }

        h1, h2, h3 {
            color: var(--text-main);
            letter-spacing: 0;
        }

        .hero-panel,
        .soft-panel,
        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"],
        div[data-testid="stAlert"] {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            box-shadow:
                inset 1px 1px 0 rgba(255,255,255,0.92),
                inset -1px -1px 0 rgba(176,205,222,0.18),
                0 22px 58px rgba(111, 158, 184, 0.18);
            backdrop-filter: blur(20px) saturate(128%);
            -webkit-backdrop-filter: blur(20px) saturate(128%);
            border-radius: 8px;
        }

        .hero-panel {
            position: relative;
            overflow: hidden;
            padding: 30px 34px 26px;
            margin-bottom: 22px;
        }

        .hero-panel::after {
            content: "";
            position: absolute;
            inset: 16px 18px auto auto;
            width: 42%;
            height: 118px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(255,255,255,0.22), rgba(247,201,204,0.40));
            filter: blur(2px);
        }

        .hero-title {
            position: relative;
            z-index: 1;
            margin: 0 0 18px;
            font-size: clamp(2rem, 4.2vw, 3.8rem);
            line-height: 1.04;
            font-weight: 850;
            color: var(--text-main);
        }

        .hero-tags {
            position: relative;
            z-index: 1;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .hero-tags span,
        .pill {
            display: inline-flex;
            align-items: center;
            min-height: 32px;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.78);
            background: rgba(255,255,255,0.58);
            box-shadow: inset 1px 1px 0 rgba(255,255,255,0.86), 0 10px 24px rgba(111,158,184,0.10);
            color: #445365;
            font-size: 0.86rem;
            white-space: nowrap;
        }

        .soft-panel {
            padding: 18px;
            margin: 0 0 18px;
        }

        .soft-panel.hot {
            background: linear-gradient(90deg, rgba(255,255,255,0.76), rgba(247,201,204,0.58));
            border-color: rgba(255,255,255,0.94);
            box-shadow:
                inset 1px 1px 0 rgba(255,255,255,0.96),
                0 18px 44px rgba(184, 132, 125, 0.16);
        }

        .panel-kicker {
            color: var(--accent-rose);
            font-size: 0.82rem;
            font-weight: 800;
        }

        .panel-title {
            margin: 4px 0 8px;
            font-size: 1.35rem;
            font-weight: 780;
            color: var(--text-main);
        }

        .panel-copy {
            margin: 0;
            color: var(--text-soft);
            line-height: 1.7;
        }

        .config-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin-top: 16px;
        }

        .config-chip {
            padding: 13px 14px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.82);
            background: rgba(255,255,255,0.54);
            box-shadow: inset 1px 1px 0 rgba(255,255,255,0.8);
        }

        .config-chip b {
            display: block;
            color: var(--accent-rose);
            font-size: 0.76rem;
            margin-bottom: 7px;
        }

        .config-chip span {
            color: var(--text-main);
            font-weight: 720;
            overflow-wrap: anywhere;
        }

        div[data-testid="stMetric"] {
            padding: 16px 18px;
            background: rgba(255,255,255,0.64);
            border-color: rgba(255,255,255,0.88);
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
            color: #6d7d8e !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text-main);
            font-size: 1.52rem;
            line-height: 1.18;
            text-shadow: 0 1px 0 rgba(255,255,255,0.82);
        }

        div[data-testid="stMetricValue"] > div {
            overflow: visible;
            text-overflow: clip;
            white-space: normal;
        }

        div[data-testid="stDataFrame"] {
            overflow: hidden;
            padding: 4px;
        }

        @media (max-width: 900px) {
            .config-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .hero-panel { padding: 22px 18px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_or_none(path: Path, label: str, show_warning: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if show_warning:
            st.warning(f"缺少{label}，请先运行对应流程。")
        return None
    try:
        return load_csv(path)
    except Exception as exc:
        st.error(f"{label}读取失败：{exc}")
        return None


def read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def fmt_number(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_percent(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def translate_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    text = str(value)
    mapping = {
        "lstm": "LSTM",
        "torch_lstm": "LSTM",
        "base_alpha_v3_rs_crowding_mini4": "基础特征加相对强弱与拥挤度小组合",
        "cross_section_rank": "横截面排序",
        "risk_adjusted": "风险调整排序",
        "pred": "预测值加权",
        "equal": "等权配置",
        "hv_rerank": "高波动再排序",
        "regime_rerank_switch/hv_close_position_20d_m005": "高波动区间收盘位置再排序方案",
        "lstm_sl20_base_alpha_v3_rs_crowding_mini4__hv_close_position_rerank": "二十日序列高波动再排序方案",
        "close_position_20d": "二十日收盘位置",
        "clipped_return__topk_weighted_rank": "截尾收益目标与前排加权排序",
        "original_return__topk_weighted_rank": "原始收益目标与前排加权排序",
        "residual_return__topk_weighted_rank": "残差收益目标与前排加权排序",
        "aggressive_score_submission": "正式等权提交方案",
        "aggressive_score_recent_strength_top6_take5_cap020": "近期强势候选前六取五等权方案",
        "single_slice_score_chase": "单切片评分优化",
        "recent_strength_pred": "近期强势预测候选",
        "top6_take5_2": "前六候选取五",
        "equal_full_cap0.20": "五只股票等权配置",
        "aggressive_package_ok; default_sync_requires_manual_confirmation": "正式提交包校验通过",
        "training_dir": "训练实验目录",
        "replay_walk_forward_predictions": "Walk-forward 回放预测",
        "default_risk_adjusted": "默认风险调整",
        "looser_risk": "宽松风险约束",
        "stricter_risk": "严格风险约束",
        "looser_risk_low_turnover": "宽松风险低换手",
        "risk_adjusted_sort": "风险调整排序",
        "pred_weight": "预测值加权",
        "adopt": "采用",
        "unknown": "未记录",
        "True": "是",
        "False": "否",
        "true": "是",
        "false": "否",
    }
    return mapping.get(text, text.replace("_", " "))


def translate_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "rank": "排名",
        "stable_alpha_score": "稳定综合分",
        "composite_score": "综合分",
        "decision": "决策",
        "risk_flags": "风险规则",
        "candidate_id": "候选编号",
        "candidate_label": "候选名称",
        "source_kind": "来源类型",
        "rank_ic_mean": "平均排序相关系数",
        "worst_fold": "最弱分折表现",
        "top5_mean_return": "前五平均收益",
        "cumulative_return_after_cost": "成本后累计收益",
        "sharpe_after_cost": "成本后夏普比率",
        "max_drawdown_after_cost": "成本后最大回撤",
        "avg_turnover": "平均换手率",
        "profile_name": "方案名称",
        "prediction_source": "预测来源",
        "weighting_scheme": "权重策略",
        "sort_strategy": "排序策略",
        "top_k": "Top K",
        "enable_risk_filters": "启用风险过滤",
        "allow_cash_fallback": "允许现金兜底",
        "max_volatility_20d_pct": "二十日波动率上限分位",
        "max_volatility_5d_pct": "五日波动率上限分位",
        "turnover_rate_lower_pct": "换手率下限分位",
        "turnover_rate_upper_pct": "换手率上限分位",
        "turnover_ratio_upper_pct": "换手率倍数上限分位",
        "risk_penalty_weight": "风险惩罚权重",
        "transaction_cost": "交易成本",
        "max_turnover": "最大换手约束",
        "periods": "回测期数",
        "cumulative_return_before_cost": "成本前累计收益",
        "mean_period_return_before_cost": "成本前平均单期收益",
        "mean_period_return_after_cost": "成本后平均单期收益",
        "return_volatility_before_cost": "成本前收益波动率",
        "return_volatility_after_cost": "成本后收益波动率",
        "max_drawdown_before_cost": "成本前最大回撤",
        "sharpe_before_cost": "成本前夏普比率",
        "win_rate_before_cost": "成本前胜率",
        "win_rate_after_cost": "成本后胜率",
        "avg_selected_count": "平均入选数量",
        "avg_signal_count": "平均信号数量",
        "avg_cash_weight": "平均现金权重",
        "avg_desired_turnover": "平均目标换手率",
        "avg_execution_strength": "平均执行强度",
        "total_transaction_cost": "总交易成本",
        "weight_blend_alpha": "权重融合系数",
        "max_single_weight_param": "单票权重参数",
        "rerank_signal_column": "再排序信号",
        "rerank_signal_weight": "再排序权重",
        "secondary_candidate_size": "二次候选池规模",
        "secondary_screen_mode": "二次筛选模式",
        "secondary_screen_weight": "二次筛选权重",
        "local_tiebreak_start_rank": "局部择优起始名次",
        "local_tiebreak_end_rank": "局部择优结束名次",
        "avg_max_single_weight": "平均最大单票权重",
        "max_single_weight_observed": "实际最大单票权重",
        "avg_top2_weight_sum": "前二权重平均合计",
        "avg_max_single_contribution_share": "最大单票贡献平均占比",
        "max_single_contribution_share": "最大单票贡献占比",
        "grid_index": "网格编号",
        "weight_cap": "单票权重上限",
        "is_positive_after_cost": "成本后是否为正",
        "cost_environment_positive_rate": "成本环境正收益比例",
        "stress_rank": "压力测试排名",
        "stock_id": "股票代码",
        "weight": "配置权重",
        "fold_id": "分折编号",
        "rmse": "均方根误差",
        "mae": "平均绝对误差",
        "rank_ic": "排序相关系数",
        "top5_mean_return": "前五平均收益",
        "train_rows": "训练样本数",
        "valid_rows": "验证样本数",
        "train_date_start": "训练开始日期",
        "train_date_end": "训练结束日期",
        "valid_date_start": "验证开始日期",
        "valid_date_end": "验证结束日期",
    }
    out = df.rename(columns={c: rename_map.get(c, c) for c in df.columns}).copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda x: translate_value(x) if isinstance(x, str) else x)
    return out


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def panel(kicker: str, title: str, copy: str = "", hot: bool = False) -> None:
    hot_class = " hot" if hot else ""
    st.markdown(
        f"""
        <div class="soft-panel{hot_class}">
            <div class="panel-kicker">{kicker}</div>
            <div class="panel-title">{title}</div>
            <p class="panel-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def package_summary() -> pd.DataFrame:
    result_df = pd.read_csv(RESULT_PATH) if RESULT_PATH.exists() else pd.DataFrame()
    weight_sum = result_df["weight"].sum() if "weight" in result_df else None
    stock_count = len(result_df) if not result_df.empty else 0
    is_equal_weight = bool("weight" in result_df and result_df["weight"].round(10).nunique() == 1)
    rows = [
        ("提交压缩包", PACKAGE_ZIP_PATH.exists()),
        ("最终权重文件", RESULT_PATH.exists()),
        ("最终股票数量为五只", stock_count == 5),
        ("最终权重合计为一", bool(weight_sum is not None and abs(weight_sum - 1.0) < 1e-8)),
        ("最终采用五只等权", is_equal_weight),
        ("冻结配置", DEFAULT_CONFIG_PATH.exists()),
        ("正式变体说明", PACKAGE_VARIANT_PATH.exists()),
        ("高分候选配置", AGGRESSIVE_CONFIG_PATH.exists()),
        ("模型元信息", MODEL_META_PATH.exists()),
        ("稳定性摘要", STABILITY_SUMMARY_PATH.exists()),
        ("提交模型文件", (MODEL_DIR / "submission_artifacts" / "lstm_model.pt").exists()),
    ]
    if PACKAGE_ZIP_PATH.exists():
        with zipfile.ZipFile(PACKAGE_ZIP_PATH) as zf:
            names = zf.namelist()
        rows.extend(
            [
                ("压缩包内文件数", len(names)),
                ("压缩包内表格数", sum(name.endswith(".csv") for name in names)),
                ("压缩包内配置数", sum(name.endswith(".json") for name in names)),
            ]
        )
    return pd.DataFrame(rows, columns=["检查项", "结果"]).assign(
        结果=lambda df: df["结果"].map(lambda x: "通过" if x is True else "缺失" if x is False else x)
    )


def render_hero(
    config: dict[str, Any] | None,
    leaderboard_df: pd.DataFrame | None,
    package_variant: dict[str, Any] | None,
    aggressive_config: dict[str, Any] | None,
) -> None:
    aggressive_selection = (aggressive_config or {}).get("selection_logic", {})
    selection = aggressive_selection or (config.get("selection_logic", {}) if config else {})
    profile = translate_value(
        (package_variant or {}).get("variant")
        or (aggressive_config or {}).get("profile_name")
        or (config.get("profile_name") if config else None)
    )
    best_label = "-"
    best_score = "-"
    rank_ic = "-"
    if leaderboard_df is not None and "rank" in leaderboard_df:
        best = coerce_numeric(leaderboard_df, ["rank", "composite_score", "rank_ic_mean"]).sort_values("rank").iloc[0]
        best_label = translate_value(best.get("candidate_label"))
        best_score = fmt_number(best.get("composite_score"), 4)
        rank_ic = fmt_number(best.get("rank_ic_mean"), 4)

    st.markdown(
        f"""
        <section class="hero-panel">
            <div class="hero-title">沪深300收益预测演示系统</div>
            <div class="hero-tags">
                <span>当前方案：{profile}</span>
                <span>榜首候选：{best_label}</span>
                <span>综合分：{best_score}</span>
                <span>排序相关系数：{rank_ic}</span>
                <span>入选数量：{translate_value(selection.get("top_k"))}</span>
                <span>权重方式：五只股票等权配置</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_strategy_snapshot(
    config: dict[str, Any] | None,
    result_df: pd.DataFrame | None,
    aggressive_config: dict[str, Any] | None,
    package_variant: dict[str, Any] | None,
) -> None:
    selection = (aggressive_config or {}).get("selection_logic", {}) or (config.get("selection_logic", {}) if config else {})
    rerank = config.get("regime_rerank", {}) if config else {}
    risk = config.get("risk_filter_thresholds", {}) if config else {}
    evidence = config.get("evidence", {}) if config else {}

    total_weight = result_df["weight"].sum() if result_df is not None and "weight" in result_df else None
    max_weight = result_df["weight"].max() if result_df is not None and "weight" in result_df else None

    cols = st.columns(5)
    cols[0].metric("提交方案", "正式等权")
    cols[1].metric("特征数量", translate_value(config.get("feature_count") if config else None))
    cols[2].metric("入选数量", translate_value(selection.get("top_k")))
    cols[3].metric("权重合计", fmt_number(total_weight, 4))
    cols[4].metric("实际最大权重", fmt_percent(max_weight, 2))

    st.markdown(
        f"""
        <div class="soft-panel hot">
            <div class="panel-kicker">最终提交口径</div>
            <div class="panel-title">正式等权提交方案</div>
            <p class="panel-copy">
                当前正式提交以高分候选变体同步后的 <b>app/output/result.csv</b> 为准：
                选取 <b>{translate_value(selection.get("top_k"))}</b> 只股票，采用 <b>{translate_value(selection.get("weighting_scheme"))}</b>，
                每只股票实际权重 <b>{fmt_percent(max_weight, 2)}</b>，权重合计 <b>{fmt_number(total_weight, 4)}</b>。
                默认 LSTM 配置仍保留在包内，用于复现推理链路；最终提交结果以正式等权组合为准。
            </p>
            <div class="config-grid">
                <div class="config-chip"><b>提交变体</b><span>{translate_value((package_variant or {}).get("variant"))}</span></div>
                <div class="config-chip"><b>候选家族</b><span>{translate_value(selection.get("candidate_family"))}</span></div>
                <div class="config-chip"><b>选取规则</b><span>{translate_value(selection.get("take_rule"))}</span></div>
                <div class="config-chip"><b>权重方式</b><span>{translate_value(selection.get("weighting_scheme"))}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="sidebar-title">沪深300预测演示</div>', unsafe_allow_html=True)
        return st.radio(
            "演示导航",
            ["项目总览", "正式提交", "模型排行", "回测表现", "稳定性验证", "压力测试"],
            index=0,
            label_visibility="collapsed",
        )


def render_overview(
    train_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    meta: dict[str, Any] | None,
    result_df: pd.DataFrame | None,
    package_variant: dict[str, Any] | None,
    score_compare_df: pd.DataFrame | None,
) -> None:
    panel("项目总览", "数据与模型概览", "展示训练数据规模、验证区间、模型配置和正式提交组合摘要。")
    cols = st.columns(4)
    cols[0].metric("训练样本行数", f"{0 if train_df is None else len(train_df):,}")
    cols[1].metric("测试样本行数", f"{0 if test_df is None else len(test_df):,}")
    cols[2].metric("正式持仓数量", f"{0 if result_df is None else len(result_df):,}")
    cols[3].metric("训练轮数", translate_value((meta or {}).get("epochs")))

    if meta:
        wf = meta.get("walk_forward_summary", {})
        summary = pd.DataFrame(
            [
                ("平均排序相关系数", fmt_number(wf.get("rank_ic_mean"), 4)),
                ("前五平均收益", fmt_percent(wf.get("top5_mean_return_mean"), 2)),
                ("平均绝对误差", fmt_number(wf.get("mae_mean"), 4)),
                ("均方根误差", fmt_number(wf.get("rmse_mean"), 4)),
                ("训练日期范围", " 至 ".join(meta.get("train_date_range_full", []))),
            ],
            columns=["指标", "数值"],
        )
        st.markdown("#### 训练验证摘要")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    if train_df is not None and {"日期", "涨跌幅"}.issubset(train_df.columns):
        st.markdown("#### 近阶段市场平均涨跌幅")
        daily = train_df.assign(日期=pd.to_datetime(train_df["日期"], errors="coerce"))
        daily = daily.groupby("日期", as_index=False)["涨跌幅"].mean().tail(120)
        st.line_chart(daily, x="日期", y="涨跌幅")

    render_score_comparison(score_compare_df, package_variant)


def render_leaderboard(leaderboard_df: pd.DataFrame | None) -> None:
    panel("模型排行", "候选模型评分表", "候选模型按综合表现排序，核心关注排序相关性、前五收益、成本后收益、回撤与换手。", hot=True)
    if leaderboard_df is None:
        st.info("当前提交包不包含完整候选排行表，页面将展示已有正式候选信息。")
        return

    numeric_cols = [
        "rank",
        "composite_score",
        "rank_ic_mean",
        "top5_mean_return",
        "cumulative_return_after_cost",
        "sharpe_after_cost",
        "max_drawdown_after_cost",
        "avg_turnover",
    ]
    table = coerce_numeric(leaderboard_df, numeric_cols).sort_values("rank").head(30)
    best = table.iloc[0]
    cols = st.columns(5)
    cols[0].metric("候选总数", f"{len(leaderboard_df):,}")
    cols[1].metric("榜首综合分", fmt_number(best.get("composite_score")))
    cols[2].metric("排序相关系数", fmt_number(best.get("rank_ic_mean")))
    cols[3].metric("成本后收益", fmt_percent(best.get("cumulative_return_after_cost")))
    cols[4].metric("最大回撤", fmt_percent(best.get("max_drawdown_after_cost")))

    show_cols = [
        "rank",
        "decision",
        "risk_flags",
        "composite_score",
        "candidate_label",
        "source_kind",
        "rank_ic_mean",
        "top5_mean_return",
        "cumulative_return_after_cost",
        "sharpe_after_cost",
        "max_drawdown_after_cost",
        "avg_turnover",
    ]
    view = table[[c for c in show_cols if c in table.columns]]
    st.dataframe(translate_columns(view), use_container_width=True, hide_index=True)
    st.line_chart(table.rename(columns={"rank": "排名", "composite_score": "综合分"}), x="排名", y="综合分")


def render_stability(
    metrics_df: pd.DataFrame | None,
    predictions_df: pd.DataFrame | None,
    stability_df: pd.DataFrame | None,
    meta: dict[str, Any] | None,
) -> None:
    panel("稳定性验证", "滚动验证与分折表现", "验证模型在不同时间窗口中的排序能力和前五收益稳定性。")

    if meta and meta.get("walk_forward_folds"):
        folds = pd.DataFrame(meta["walk_forward_folds"])
        st.markdown("#### 分折验证结果")
        st.dataframe(translate_columns(folds), use_container_width=True, hide_index=True)
        chart = folds.rename(
            columns={"fold_id": "分折编号", "rank_ic": "排序相关系数", "top5_mean_return": "前五平均收益"}
        )
        st.line_chart(chart, x="分折编号", y=["排序相关系数", "前五平均收益"])

    source_df = metrics_df if metrics_df is not None else stability_df
    if source_df is not None:
        st.markdown("#### 稳定性摘要表")
        st.dataframe(translate_columns(source_df), use_container_width=True, hide_index=True)

    if predictions_df is not None:
        st.markdown("#### 预测样本预览")
        st.dataframe(translate_columns(predictions_df.head(200)), use_container_width=True, hide_index=True)


def render_backtest(backtest_df: pd.DataFrame | None, stress_df: pd.DataFrame | None, stability_df: pd.DataFrame | None) -> None:
    panel("回测表现", "成本后收益与风险表现", "展示收益、回撤、换手和交易成本，辅助说明最终方案不是单指标选择。", hot=True)
    if backtest_df is not None:
        bt = coerce_numeric(
            backtest_df,
            [
                "cumulative_return_after_cost",
                "sharpe_after_cost",
                "max_drawdown_after_cost",
                "avg_turnover",
                "total_transaction_cost",
            ],
        )
        default = bt.iloc[0]
        cols = st.columns(5)
        cols[0].metric("成本后累计收益", fmt_percent(default.get("cumulative_return_after_cost")))
        cols[1].metric("成本后夏普比率", fmt_number(default.get("sharpe_after_cost"), 3))
        cols[2].metric("最大回撤", fmt_percent(default.get("max_drawdown_after_cost")))
        cols[3].metric("平均换手率", fmt_number(default.get("avg_turnover"), 3))
        cols[4].metric("交易成本", fmt_percent(default.get("total_transaction_cost")))
        st.dataframe(translate_columns(bt), use_container_width=True, hide_index=True)
    else:
        st.info("当前正式提交包未包含完整回测摘要表。")

    if stress_df is not None:
        st.markdown("#### 压力测试摘要")
        st.dataframe(translate_columns(stress_df.head(80)), use_container_width=True, hide_index=True)
    elif stability_df is not None:
        st.markdown("#### 稳定性补充")
        st.dataframe(translate_columns(stability_df), use_container_width=True, hide_index=True)


def build_selected_reason_table(
    result_df: pd.DataFrame,
    predict_scores_df: pd.DataFrame | None,
    aggressive_config: dict[str, Any] | None,
) -> pd.DataFrame:
    result = result_df.copy()
    result["股票代码"] = result["stock_id"].astype(str).str.zfill(6)
    result["配置权重"] = pd.to_numeric(result["weight"], errors="coerce")

    if predict_scores_df is not None and {"stock_id", "pred_return"}.issubset(predict_scores_df.columns):
        scores = predict_scores_df.copy()
        scores["股票代码"] = scores["stock_id"].astype(str).str.zfill(6)
        scores["预测得分"] = pd.to_numeric(scores["pred_return"], errors="coerce")
        scores = scores.sort_values("预测得分", ascending=False)
        scores["预测得分排名"] = range(1, len(scores) + 1)
        result = result.merge(scores[["股票代码", "预测得分", "预测得分排名"]], on="股票代码", how="left")

    selected = [str(x).zfill(6) for x in (aggressive_config or {}).get("candidate_stocks", [])]
    if selected:
        order_map = {stock: index + 1 for index, stock in enumerate(selected)}
        result["候选顺序"] = result["股票代码"].map(order_map)
    else:
        result["候选顺序"] = range(1, len(result) + 1)

    result["入选理由"] = "近期强势候选，预测得分靠前，最终采用五只等权配置"
    result = result.sort_values("候选顺序")
    cols = ["股票代码", "配置权重", "预测得分", "预测得分排名", "候选顺序", "入选理由"]
    return result[[c for c in cols if c in result.columns]]


def render_score_comparison(score_compare_df: pd.DataFrame | None, package_variant: dict[str, Any] | None) -> None:
    rows: list[tuple[str, str, str, str]] = []
    if score_compare_df is not None:
        for _, row in score_compare_df.iterrows():
            metric = str(row.get("metric", ""))
            name = "当前输出得分" if metric == "current_output_score" else "记录最佳得分" if metric == "recorded_best_score" else metric
            rows.append(
                (
                    name,
                    fmt_number(row.get("our_score"), 6),
                    fmt_number(row.get("case_score"), 6),
                    fmt_number(row.get("diff_our_minus_case"), 6),
                )
            )
    elif package_variant:
        rows.append(("单切片评分", fmt_number(package_variant.get("case_slice_score"), 6), "-", "-"))

    if rows:
        st.markdown("#### 单切片评分对比")
        st.dataframe(
            pd.DataFrame(rows, columns=["对比项", "当前提交得分", "基准得分", "得分差值"]),
            use_container_width=True,
            hide_index=True,
        )


def render_submission(
    config: dict[str, Any] | None,
    result_df: pd.DataFrame | None,
    predict_scores_df: pd.DataFrame | None,
    debug_candidates_df: pd.DataFrame | None,
    aggressive_config: dict[str, Any] | None,
    package_variant: dict[str, Any] | None,
    score_compare_df: pd.DataFrame | None,
) -> None:
    panel("正式提交", "最终提交组合", "展示正式提交文件中的股票与权重，并补充候选打分和调试明细。")
    if result_df is None:
        return

    result = coerce_numeric(result_df, ["stock_id", "weight"])
    cols = st.columns(4)
    cols[0].metric("提交股票数", f"{len(result):,}")
    cols[1].metric("权重合计", fmt_number(result["weight"].sum()))
    cols[2].metric("最大单票权重", fmt_percent(result["weight"].max()))
    cols[3].metric("最小单票权重", fmt_percent(result["weight"].min()))

    view = result.rename(columns={"stock_id": "股票代码", "weight": "配置权重"})
    st.dataframe(view, use_container_width=True, hide_index=True)
    chart_df = view.assign(股票代码=view["股票代码"].astype(str)).set_index("股票代码")
    st.bar_chart(chart_df["配置权重"])

    st.markdown("#### 入选理由表")
    st.dataframe(
        build_selected_reason_table(result_df, predict_scores_df, aggressive_config),
        use_container_width=True,
        hide_index=True,
    )

    render_score_comparison(score_compare_df, package_variant)

    if config:
        selection = config.get("selection_logic", {})
        config_table = pd.DataFrame(
            [
                ("模型主线", translate_value(config.get("model_family"))),
                ("特征方案", translate_value(config.get("feature_set"))),
                ("训练目标", translate_value(config.get("target_mode"))),
                ("默认排序策略", translate_value(selection.get("sort_strategy"))),
                ("默认权重策略", translate_value(selection.get("weighting_scheme"))),
                ("默认候选池规模", translate_value(selection.get("primary_candidate_size"))),
                ("最终提交口径", "高分候选变体同步后的五只等权组合"),
            ],
            columns=["配置项", "取值"],
        )
        st.markdown("#### 默认模型配置摘要")
        st.dataframe(config_table, use_container_width=True, hide_index=True)

    if predict_scores_df is not None:
        st.markdown("#### 最终预测分数")
        st.dataframe(translate_columns(predict_scores_df.head(200)), use_container_width=True, hide_index=True)


def render_pressure(stress_df: pd.DataFrame | None, stability_df: pd.DataFrame | None) -> None:
    panel("压力测试", "交易成本与约束敏感性", "展示不同成本、换手和权重约束下的稳定性表现。", hot=True)
    if stress_df is not None:
        st.dataframe(translate_columns(stress_df.head(100)), use_container_width=True, hide_index=True)
    elif stability_df is not None:
        st.info("当前提交包未包含压力测试网格，展示稳定性摘要作为补充。")
        st.dataframe(translate_columns(stability_df), use_container_width=True, hide_index=True)
    else:
        st.info("当前提交包未包含压力测试结果。")


def main() -> None:
    inject_style()
    config = read_json_or_none(DEFAULT_CONFIG_PATH)
    meta = read_json_or_none(MODEL_META_PATH)
    package_variant = read_json_or_none(PACKAGE_VARIANT_PATH)
    aggressive_config = read_json_or_none(AGGRESSIVE_CONFIG_PATH)
    page = render_sidebar()

    train_df = read_csv_or_none(TRAIN_PATH, "训练数据")
    test_df = read_csv_or_none(TEST_PATH, "测试数据")
    leaderboard_df = read_csv_or_none(LEADERBOARD_PATH, "模型排行表", show_warning=False)
    backtest_df = read_csv_or_none(BACKTEST_SUMMARY_PATH, "回测摘要", show_warning=False)
    stability_df = read_csv_or_none(STABILITY_SUMMARY_PATH, "稳定性摘要", show_warning=False)
    result_df = read_csv_or_none(RESULT_PATH, "最终提交")
    stress_df = read_csv_or_none(BACKTEST_STRESS_PATH, "压力测试", show_warning=False)
    metrics_df = read_csv_or_none(WALK_FORWARD_METRICS_PATH, "滚动验证", show_warning=False)
    predictions_df = read_csv_or_none(WALK_FORWARD_PREDICTIONS_PATH, "预测样本", show_warning=False)
    predict_scores_df = read_csv_or_none(PREDICT_SCORES_PATH, "预测分数", show_warning=False)
    debug_candidates_df = read_csv_or_none(DEBUG_CANDIDATES_PATH, "调试候选", show_warning=False)
    score_compare_df = read_csv_or_none(SCORE_COMPARE_PATH, "单切片评分对比", show_warning=False)

    render_hero(config, leaderboard_df, package_variant, aggressive_config)
    render_strategy_snapshot(config, result_df, aggressive_config, package_variant)

    if page == "项目总览":
        render_overview(train_df, test_df, meta, result_df, package_variant, score_compare_df)
    elif page == "模型排行":
        render_leaderboard(leaderboard_df)
    elif page == "稳定性验证":
        render_stability(metrics_df, predictions_df, stability_df, meta)
    elif page == "回测表现":
        render_backtest(backtest_df, stress_df, stability_df)
    elif page == "压力测试":
        render_pressure(stress_df, stability_df)
    elif page == "正式提交":
        render_submission(
            config,
            result_df,
            predict_scores_df,
            debug_candidates_df,
            aggressive_config,
            package_variant,
            score_compare_df,
        )


if __name__ == "__main__":
    main()
