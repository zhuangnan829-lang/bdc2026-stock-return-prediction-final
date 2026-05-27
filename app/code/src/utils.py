from pathlib import Path

import pandas as pd


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_feature_frame(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_id": str})
    df["stock_id"] = df["stock_id"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad_rows = int(df["date"].isna().sum())
        raise ValueError(f"Found {bad_rows} rows with invalid dates in {path}")
    return df


def apply_candidate_filters(
    latest_df: pd.DataFrame,
    top_k: int,
    primary_candidate_size: int,
    max_volatility_20d_pct: float,
    max_volatility_5d_pct: float,
    turnover_rate_lower_pct: float,
    turnover_rate_upper_pct: float,
    turnover_ratio_upper_pct: float,
    enable_risk_filters: bool = True,
    allow_cash_fallback: bool = False,
) -> tuple[pd.DataFrame, dict]:
    working = latest_df.copy()
    diagnostics = {"initial_candidates": int(len(working))}

    working = working.sort_values(["pred_return", "stock_id"], ascending=[False, True]).head(
        min(primary_candidate_size, len(working))
    )
    diagnostics["after_primary_screen"] = int(len(working))

    if not enable_risk_filters:
        diagnostics["after_risk_filters"] = int(len(working))
        diagnostics["fallback_used"] = False
        diagnostics["risk_filters_enabled"] = False
        return working, diagnostics

    vol20_threshold = latest_df["volatility_20d"].quantile(max_volatility_20d_pct)
    vol5_threshold = latest_df["volatility_5d"].quantile(max_volatility_5d_pct)
    turnover_low = latest_df["turnover_rate"].quantile(turnover_rate_lower_pct)
    turnover_high = latest_df["turnover_rate"].quantile(turnover_rate_upper_pct)
    turnover_ratio_high = latest_df["turnover_ratio_10d"].quantile(turnover_ratio_upper_pct)

    filtered = working[
        (working["volatility_20d"] <= vol20_threshold)
        & (working["volatility_5d"] <= vol5_threshold)
        & (working["turnover_rate"] >= turnover_low)
        & (working["turnover_rate"] <= turnover_high)
        & (working["turnover_ratio_10d"] <= turnover_ratio_high)
    ].copy()

    diagnostics["after_risk_filters"] = int(len(filtered))
    diagnostics["vol20_threshold"] = float(vol20_threshold)
    diagnostics["vol5_threshold"] = float(vol5_threshold)
    diagnostics["turnover_low"] = float(turnover_low)
    diagnostics["turnover_high"] = float(turnover_high)
    diagnostics["turnover_ratio_high"] = float(turnover_ratio_high)
    diagnostics["risk_filters_enabled"] = True

    if len(filtered) < top_k and not allow_cash_fallback:
        diagnostics["fallback_used"] = True
        filtered = working.copy()
    else:
        diagnostics["fallback_used"] = False

    return filtered, diagnostics


def rerank_with_risk_controls(candidate_df: pd.DataFrame, risk_penalty_weight: float) -> pd.DataFrame:
    scored = candidate_df.copy()
    scored["pred_rank_pct"] = scored["pred_return"].rank(pct=True)
    scored["risk_vol20_pct"] = scored["volatility_20d"].rank(pct=True)
    scored["risk_vol5_pct"] = scored["volatility_5d"].rank(pct=True)
    scored["risk_turnover_pct"] = scored["turnover_ratio_10d"].rank(pct=True)
    scored["risk_amplitude_pct"] = scored["amplitude_ratio_5d"].rank(pct=True)

    scored["risk_penalty"] = (
        0.4 * scored["risk_vol20_pct"]
        + 0.25 * scored["risk_vol5_pct"]
        + 0.2 * scored["risk_turnover_pct"]
        + 0.15 * scored["risk_amplitude_pct"]
    )
    scored["selection_score"] = scored["pred_rank_pct"] - risk_penalty_weight * scored["risk_penalty"]
    return scored


def apply_additional_rerank_signal(
    scored_df: pd.DataFrame,
    rerank_signal_column: str | None = None,
    rerank_signal_weight: float = 0.0,
) -> pd.DataFrame:
    scored = scored_df.copy()
    if not rerank_signal_column or abs(rerank_signal_weight) <= 1e-12:
        scored["rerank_signal_score"] = 0.0
        scored["selection_score_final"] = scored["selection_score"]
        return scored

    if rerank_signal_column not in scored.columns:
        raise ValueError(f"Missing rerank signal column: {rerank_signal_column}")

    signal_rank_pct = scored[rerank_signal_column].rank(pct=True)
    scored["rerank_signal_score"] = signal_rank_pct - 0.5
    scored["selection_score_final"] = scored["selection_score"] + rerank_signal_weight * scored["rerank_signal_score"]
    return scored


def apply_secondary_screen(
    scored_df: pd.DataFrame,
    secondary_candidate_size: int | None = None,
    secondary_screen_mode: str = "none",
    secondary_screen_weight: float = 0.0,
    local_tiebreak_start_rank: int = 8,
    local_tiebreak_end_rank: int = 15,
) -> tuple[pd.DataFrame, dict]:
    scored = scored_df.copy()
    diagnostics = {
        "secondary_screen_mode": secondary_screen_mode or "none",
        "secondary_quality_filter_count": int(len(scored)),
        "secondary_screen_size": int(secondary_candidate_size) if secondary_candidate_size else 0,
        "secondary_screen_weight": float(secondary_screen_weight),
        "local_tiebreak_start_rank": int(local_tiebreak_start_rank),
        "local_tiebreak_end_rank": int(local_tiebreak_end_rank),
    }
    scored["secondary_signal_score"] = 0.0
    scored["secondary_quality_penalty"] = 0.0
    scored["secondary_alpha_score"] = 0.0

    mode = (secondary_screen_mode or "none").strip().lower()
    if mode == "none":
        if secondary_candidate_size is not None and secondary_candidate_size > 0:
            secondary_size = min(int(secondary_candidate_size), len(scored))
            scored = scored.sort_values(
                ["selection_score_final", "selection_score", "pred_return", "stock_id"],
                ascending=[False, False, False, True],
            ).head(secondary_size)
        diagnostics["secondary_quality_filter_count"] = int(len(scored))
        return scored, diagnostics

    if mode in {"alpha_combo", "alpha_blend", "alpha_local_tiebreak"}:
        required = ["rel_strength_accel_5d_v2", "trend_persistence_score_10d_v2", "crowding_reversal_risk_5d"]
        missing = [column for column in required if column not in scored.columns]
        if missing:
            raise ValueError(f"Missing columns for alpha secondary screen: {missing}")

        rs_rank = scored["rel_strength_accel_5d_v2"].rank(pct=True)
        trend_rank = scored["trend_persistence_score_10d_v2"].rank(pct=True)
        crowding_safe_rank = 1.0 - scored["crowding_reversal_risk_5d"].rank(pct=True)
        scored["secondary_alpha_score"] = (
            0.45 * (rs_rank - 0.5)
            + 0.35 * (trend_rank - 0.5)
            + 0.20 * (crowding_safe_rank - 0.5)
        )
        scored["secondary_signal_score"] = scored["selection_score_final"] + float(secondary_screen_weight) * scored["secondary_alpha_score"]
        if mode == "alpha_blend":
            diagnostics["secondary_quality_filter_count"] = int(len(scored))
            return scored, diagnostics
        if mode == "alpha_local_tiebreak":
            base_sorted = scored.sort_values(
                ["selection_score_final", "selection_score", "pred_return", "stock_id"],
                ascending=[False, False, False, True],
            ).reset_index(drop=True)
            base_sorted["base_rank"] = range(1, len(base_sorted) + 1)
            local_start = max(1, int(local_tiebreak_start_rank))
            local_end = max(local_start, int(local_tiebreak_end_rank))

            head = base_sorted[base_sorted["base_rank"] < local_start].copy()
            middle = base_sorted[
                (base_sorted["base_rank"] >= local_start) & (base_sorted["base_rank"] <= local_end)
            ].copy()
            tail = base_sorted[base_sorted["base_rank"] > local_end].copy()

            if not middle.empty:
                middle = middle.sort_values(
                    ["secondary_signal_score", "selection_score_final", "selection_score", "pred_return", "stock_id"],
                    ascending=[False, False, False, False, True],
                )
            scored = pd.concat([head, middle, tail], ignore_index=True)
            diagnostics["secondary_quality_filter_count"] = int(len(scored))
            return scored, diagnostics

    elif mode == "quality_layer":
        required = ["volatility_20d", "volatility_5d", "crowding_reversal_risk_5d", "turnover_spike_5d"]
        missing = [column for column in required if column not in scored.columns]
        if missing:
            raise ValueError(f"Missing columns for quality-layer secondary screen: {missing}")

        scored["secondary_quality_penalty"] = (
            0.45 * scored["volatility_20d"].rank(pct=True)
            + 0.20 * scored["volatility_5d"].rank(pct=True)
            + 0.25 * scored["crowding_reversal_risk_5d"].rank(pct=True)
            + 0.10 * scored["turnover_spike_5d"].rank(pct=True)
        )
        scored["secondary_signal_score"] = 0.5 - scored["secondary_quality_penalty"]
        threshold = scored["secondary_quality_penalty"].quantile(0.75)
        filtered = scored[scored["secondary_quality_penalty"] <= threshold].copy()
        if not filtered.empty:
            scored = filtered
        diagnostics["secondary_quality_filter_count"] = int(len(scored))
    else:
        raise ValueError(f"Unsupported secondary_screen_mode: {secondary_screen_mode}")

    if secondary_candidate_size is not None and secondary_candidate_size > 0:
        secondary_size = min(int(secondary_candidate_size), len(scored))
        scored = scored.sort_values(
            ["secondary_signal_score", "selection_score_final", "selection_score", "pred_return", "stock_id"],
            ascending=[False, False, False, False, True],
        ).head(secondary_size)
    else:
        scored = scored.sort_values(
            ["secondary_signal_score", "selection_score_final", "selection_score", "pred_return", "stock_id"],
            ascending=[False, False, False, False, True],
        )

    diagnostics["secondary_quality_filter_count"] = int(len(scored))
    return scored, diagnostics


def select_top_candidates(
    latest_df: pd.DataFrame,
    top_k: int,
    primary_candidate_size: int,
    max_volatility_20d_pct: float,
    max_volatility_5d_pct: float,
    turnover_rate_lower_pct: float,
    turnover_rate_upper_pct: float,
    turnover_ratio_upper_pct: float,
    risk_penalty_weight: float,
    sort_strategy: str = "risk_adjusted",
    rerank_signal_column: str | None = None,
    rerank_signal_weight: float = 0.0,
    secondary_candidate_size: int | None = None,
    secondary_screen_mode: str = "none",
    secondary_screen_weight: float = 0.0,
    local_tiebreak_start_rank: int = 8,
    local_tiebreak_end_rank: int = 15,
    enable_risk_filters: bool = True,
    allow_cash_fallback: bool = False,
) -> tuple[pd.DataFrame, dict]:
    candidate_df, diagnostics = apply_candidate_filters(
        latest_df=latest_df,
        top_k=top_k,
        primary_candidate_size=primary_candidate_size,
        max_volatility_20d_pct=max_volatility_20d_pct,
        max_volatility_5d_pct=max_volatility_5d_pct,
        turnover_rate_lower_pct=turnover_rate_lower_pct,
        turnover_rate_upper_pct=turnover_rate_upper_pct,
        turnover_ratio_upper_pct=turnover_ratio_upper_pct,
        enable_risk_filters=enable_risk_filters,
        allow_cash_fallback=allow_cash_fallback,
    )
    scored_candidates = rerank_with_risk_controls(candidate_df, risk_penalty_weight)
    scored_candidates = apply_additional_rerank_signal(
        scored_candidates,
        rerank_signal_column=rerank_signal_column,
        rerank_signal_weight=rerank_signal_weight,
    )
    scored_candidates, secondary_diagnostics = apply_secondary_screen(
        scored_candidates,
        secondary_candidate_size=secondary_candidate_size,
        secondary_screen_mode=secondary_screen_mode,
        secondary_screen_weight=secondary_screen_weight,
        local_tiebreak_start_rank=local_tiebreak_start_rank,
        local_tiebreak_end_rank=local_tiebreak_end_rank,
    )
    diagnostics["after_secondary_screen"] = int(len(scored_candidates))
    diagnostics["secondary_quality_filter_count"] = int(secondary_diagnostics.get("secondary_quality_filter_count", len(scored_candidates)))
    if sort_strategy == "pure_prediction":
        selected = scored_candidates.sort_values(
            ["pred_return", "stock_id"], ascending=[False, True]
        ).head(top_k)
    elif sort_strategy == "risk_adjusted":
        selected = scored_candidates.sort_values(
            ["selection_score_final", "selection_score", "pred_return", "stock_id"], ascending=[False, False, False, True]
        ).head(top_k)
    else:
        raise ValueError(f"Unsupported sort_strategy: {sort_strategy}")
    diagnostics["selected_count"] = int(len(selected))
    diagnostics["sort_strategy"] = sort_strategy
    diagnostics["rerank_signal_column"] = rerank_signal_column or ""
    diagnostics["rerank_signal_weight"] = float(rerank_signal_weight)
    diagnostics["secondary_candidate_size"] = int(secondary_candidate_size) if secondary_candidate_size else 0
    diagnostics["secondary_screen_mode"] = secondary_screen_mode or "none"
    diagnostics["secondary_screen_weight"] = float(secondary_screen_weight)
    diagnostics["local_tiebreak_start_rank"] = int(local_tiebreak_start_rank)
    diagnostics["local_tiebreak_end_rank"] = int(local_tiebreak_end_rank)
    return selected, diagnostics


def build_candidate_debug_frame(
    latest_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    weighted_df: pd.DataFrame,
    executed_weights: dict[str, float],
    diagnostics: dict,
    top_k: int,
) -> pd.DataFrame:
    debug_df = latest_df.copy()
    selected_cols = [
        "stock_id",
        "pred_return",
        "pred_rank_pct",
        "risk_penalty",
        "selection_score",
        "rerank_signal_score",
        "selection_score_final",
        "secondary_signal_score",
        "secondary_quality_penalty",
        "secondary_alpha_score",
    ]
    available_selected_cols = [column for column in selected_cols if column in selected_df.columns]
    if available_selected_cols:
        debug_df = debug_df.merge(
            selected_df[available_selected_cols].drop_duplicates("stock_id"),
            on="stock_id",
            how="left",
            suffixes=("", "_selected"),
        )

    weighted_cols = ["stock_id", "weight"]
    debug_df = debug_df.merge(
        weighted_df[weighted_cols].rename(columns={"weight": "target_weight"}),
        on="stock_id",
        how="left",
    )
    executed_df = pd.DataFrame(
        [{"stock_id": stock_id, "executed_weight": float(weight)} for stock_id, weight in executed_weights.items()]
    )
    if not executed_df.empty:
        debug_df = debug_df.merge(executed_df, on="stock_id", how="left")
    else:
        debug_df["executed_weight"] = 0.0

    selected_rank_map = {
        stock_id: rank
        for rank, stock_id in enumerate(selected_df["stock_id"].astype(str).tolist(), start=1)
    }
    debug_df["selected_rank"] = debug_df["stock_id"].astype(str).map(selected_rank_map)
    debug_df["is_selected"] = debug_df["selected_rank"].notna().astype(int)
    debug_df["target_weight"] = pd.to_numeric(debug_df["target_weight"], errors="coerce").fillna(0.0)
    debug_df["executed_weight"] = pd.to_numeric(debug_df["executed_weight"], errors="coerce").fillna(0.0)
    debug_df["selected_rank"] = pd.to_numeric(debug_df["selected_rank"], errors="coerce")
    debug_df["top_k"] = int(top_k)
    debug_df["primary_candidate_size"] = int(diagnostics.get("after_primary_screen", 0))
    debug_df["after_risk_filters"] = int(diagnostics.get("after_risk_filters", 0))
    debug_df["after_secondary_screen"] = int(diagnostics.get("after_secondary_screen", 0))
    debug_df["risk_filters_enabled"] = int(bool(diagnostics.get("risk_filters_enabled", True)))
    debug_df["fallback_used"] = int(bool(diagnostics.get("fallback_used", False)))
    debug_df["sort_strategy"] = diagnostics.get("sort_strategy", "")
    debug_df["secondary_screen_mode"] = diagnostics.get("secondary_screen_mode", "none")
    debug_df = debug_df.sort_values(
        ["is_selected", "selection_score_final", "selection_score", "pred_return", "stock_id"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    return debug_df


def build_portfolio_weights(
    selected_df: pd.DataFrame,
    top_k: int,
    weighting_scheme: str,
    max_single_weight: float | None = None,
    weight_blend_alpha: float = 1.0,
) -> pd.DataFrame:
    if selected_df.empty:
        out = selected_df.copy()
        out["weight"] = pd.Series(dtype=float)
        return out

    out = selected_df.copy()
    invested_ratio = min(len(out), top_k) / float(top_k)
    equal_weight = pd.Series(invested_ratio / float(len(out)), index=out.index, dtype=float)

    if weighting_scheme == "equal":
        out["weight"] = equal_weight
        out["weight"] = apply_single_weight_cap(out["weight"], max_single_weight)
        return out

    if weighting_scheme in {"pred", "pred_equal_blend"}:
        raw = out["pred_return"].clip(lower=0.0)
        if raw.sum() <= 1e-12:
            raw = pd.Series(1.0, index=out.index)
        pred_weight = invested_ratio * raw / raw.sum()
        if weighting_scheme == "pred_equal_blend":
            alpha = max(0.0, min(1.0, float(weight_blend_alpha)))
            out["weight"] = alpha * pred_weight + (1.0 - alpha) * equal_weight
        else:
            out["weight"] = pred_weight
        out["weight"] = apply_single_weight_cap(out["weight"], max_single_weight)
        return out

    if weighting_scheme == "risk_adjusted":
        strength = out["pred_rank_pct"].clip(lower=0.0)
        adjusted = strength / (1.0 + out["risk_penalty"].clip(lower=0.0))
        if adjusted.sum() <= 1e-12:
            adjusted = pd.Series(1.0, index=out.index)
        out["weight"] = invested_ratio * adjusted / adjusted.sum()
        out["weight"] = apply_single_weight_cap(out["weight"], max_single_weight)
        return out

    raise ValueError(f"Unsupported weighting_scheme: {weighting_scheme}")


def apply_single_weight_cap(weights: pd.Series, max_single_weight: float | None = None) -> pd.Series:
    if max_single_weight is None:
        return weights.astype(float)

    cap = float(max_single_weight)
    if cap <= 0.0:
        return pd.Series(0.0, index=weights.index, dtype=float)
    if cap >= 1.0 or weights.empty:
        return weights.astype(float)

    capped = weights.astype(float).clip(lower=0.0).copy()
    target_total = float(capped.sum())
    if target_total <= 1e-12:
        return capped

    capped = capped.clip(upper=cap)
    for _ in range(len(capped) + 1):
        current_total = float(capped.sum())
        shortfall = target_total - current_total
        if shortfall <= 1e-12:
            break

        room = (cap - capped).clip(lower=0.0)
        total_room = float(room.sum())
        if total_room <= 1e-12:
            break

        add = room / total_room * min(shortfall, total_room)
        capped = (capped + add).clip(upper=cap)

    return capped


def load_result_portfolio(path: str | Path) -> dict[str, float]:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_id": str})
    if list(df.columns) != ["stock_id", "weight"]:
        raise ValueError(f"Invalid result portfolio format in {path}")
    df["stock_id"] = df["stock_id"].astype(str).str.zfill(6)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    return {
        stock_id: float(weight)
        for stock_id, weight in zip(df["stock_id"], df["weight"])
        if abs(float(weight)) > 1e-12
    }


def calculate_turnover(previous_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    universe = set(previous_weights) | set(new_weights)
    return float(sum(abs(new_weights.get(stock_id, 0.0) - previous_weights.get(stock_id, 0.0)) for stock_id in universe))


def apply_turnover_cap(
    previous_weights: dict[str, float],
    target_weights: dict[str, float],
    max_turnover: float,
) -> tuple[dict[str, float], float, float]:
    desired_turnover = calculate_turnover(previous_weights, target_weights)
    if desired_turnover <= max_turnover + 1e-12:
        return target_weights, desired_turnover, 1.0

    if desired_turnover <= 1e-12:
        return target_weights, desired_turnover, 1.0

    strength = max(0.0, min(1.0, max_turnover / desired_turnover))
    universe = set(previous_weights) | set(target_weights)
    blended: dict[str, float] = {}
    for stock_id in universe:
        prev = previous_weights.get(stock_id, 0.0)
        target = target_weights.get(stock_id, 0.0)
        weight = prev + strength * (target - prev)
        if abs(weight) > 1e-12:
            blended[stock_id] = float(weight)

    capped_turnover = calculate_turnover(previous_weights, blended)
    return blended, desired_turnover, strength
