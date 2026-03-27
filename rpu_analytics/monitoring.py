"""Module A: Weekly Monitoring -- 7 sub-modules for RPU analytics."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG
from .utils import shapley_decomposition, z_score
from .data_layer import EnrichedData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> float:
    """Safe division returning 0.0 when denominator is zero or NaN."""
    if den is None or den == 0 or (isinstance(den, float) and np.isnan(den)):
        return 0.0
    return float(num / den)


def _pct_change(curr: float, prev: float) -> float:
    """Percentage change from prev to curr.  Returns 0 if prev is zero."""
    if prev is None or prev == 0:
        return 0.0
    return float((curr - prev) / abs(prev) * 100.0)


def _trend_label(wow: float, vs4w: float) -> str:
    """Classify trend as improving / declining / stable."""
    if wow > 1 and vs4w > 1:
        return "improving"
    if wow < -1 and vs4w < -1:
        return "declining"
    return "stable"


def _to_native(val: Any) -> Any:
    """Convert numpy / pandas scalars to JSON-serializable Python types."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val) if not np.isnan(val) else None
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _serialize(obj: Any) -> Any:
    """Recursively ensure an object tree is JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return _to_native(obj)


# Click-cohort constants used in the enriched DataFrame
_W1 = "1. W1"
_W2W4 = "2. W2-W4"
_M3 = "3. M3"
_ENROLLED = "Enrolled"
_NON_ENROLLED = "Non-Enrolled"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MonitoringModule:
    """Weekly monitoring dashboard with 7 sub-modules."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config if config is not None else DEFAULT_CONFIG["monitoring"]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, data: EnrichedData) -> dict:
        """Execute all sub-modules and return a combined results dict."""
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "kpi_summary": self._kpi_summary(data),
            "waterfall": self._waterfall(data),
            "segment_tracker": self._segment_tracker(data),
            "touchpoint_matrix": self._touchpoint_matrix(data),
            "affiliate_scorecard": self._affiliate_scorecard(data),
            "mix_analysis": self._mix_analysis(data),
            "alerts": self._anomaly_detect(data),
        }
        return _serialize(result)

    # ------------------------------------------------------------------
    # Internal helpers for cohort slicing
    # ------------------------------------------------------------------

    def _cohort_weeks(self, df: pd.DataFrame) -> List:
        """Return sorted list of unique feo_cohort values."""
        return sorted(df["feo_cohort"].dropna().unique().tolist())

    def _latest_two(self, data: EnrichedData):
        """Return (latest_cohort, prior_cohort) timestamps."""
        weeks = self._cohort_weeks(data.enriched)
        latest = data.latest_cohort
        prior = weeks[-2] if len(weeks) >= 2 else latest
        return latest, prior

    def _week_agg(self, df: pd.DataFrame, cohort) -> pd.DataFrame:
        """Return rows for a single feo_cohort."""
        return df[df["feo_cohort"] == cohort]

    def _trailing_weeks(self, df: pd.DataFrame, latest, n: int = 4):
        """Return the last *n* cohort timestamps up to (but not including) latest."""
        weeks = sorted(df["feo_cohort"].dropna().unique().tolist())
        idx = weeks.index(latest) if latest in weeks else len(weeks)
        start = max(0, idx - n)
        return weeks[start:idx]

    # ------------------------------------------------------------------
    # A1. KPI Summary
    # ------------------------------------------------------------------

    def _kpi_summary(self, data: EnrichedData) -> dict:
        df = data.enriched
        latest, prior = self._latest_two(data)
        trailing = self._trailing_weeks(df, latest, n=4)

        def _compute_kpis(subset: pd.DataFrame) -> dict:
            total_payout = subset["payout"].sum()
            total_enrolls = subset["ENROLLS"].sum()
            total_rpu = _safe_div(total_payout, total_enrolls)

            # Per click-cohort RPU
            w1 = subset[subset["click_cohort"] == _W1] if "click_cohort" in subset.columns else pd.DataFrame()
            w2w4 = subset[subset["click_cohort"] == _W2W4] if "click_cohort" in subset.columns else pd.DataFrame()
            m3 = subset[subset["click_cohort"] == _M3] if "click_cohort" in subset.columns else pd.DataFrame()

            w1_rpu = _safe_div(w1["payout"].sum(), total_enrolls) if not w1.empty else 0.0
            w2w4_rpu = _safe_div(w2w4["payout"].sum(), total_enrolls) if not w2w4.empty else 0.0
            m3_rpu = _safe_div(m3["payout"].sum(), total_enrolls) if not m3.empty else 0.0

            # Blended CTR, RPC, Imp%
            total_impressions = subset["impression_count"].sum()
            total_clicks = subset["click_count"].sum()
            blended_ctr = _safe_div(total_clicks, total_impressions)
            blended_rpc = _safe_div(total_payout, total_clicks)
            blended_imp_pct = _safe_div(total_impressions, total_enrolls)

            # Non-click RPU: rows with TOUCHPOINT == "Non-click"
            non_click = subset[subset["TOUCHPOINT"] == "Non-click"]
            non_click_rpu = _safe_div(non_click["payout"].sum(), total_enrolls)

            # Non-enrolled RPU
            if "enrol_status" in subset.columns:
                non_enrolled = subset[subset["enrol_status"] == _NON_ENROLLED]
            else:
                non_enrolled = pd.DataFrame()
            non_enrolled_rpu = _safe_div(non_enrolled["payout"].sum(), total_enrolls) if not non_enrolled.empty else 0.0

            return {
                "total_rpu": total_rpu,
                "w1_rpu": w1_rpu,
                "w2w4_rpu": w2w4_rpu,
                "m3_rpu": m3_rpu,
                "blended_ctr": blended_ctr,
                "blended_rpc": blended_rpc,
                "blended_imp_pct": blended_imp_pct,
                "total_enrollments": total_enrolls,
                "total_payout": total_payout,
                "non_click_rpu": non_click_rpu,
                "non_enrolled_rpu": non_enrolled_rpu,
            }

        curr_kpis = _compute_kpis(self._week_agg(df, latest))
        prior_kpis = _compute_kpis(self._week_agg(df, prior))

        # 4-week average
        trailing_kpi_list = [_compute_kpis(self._week_agg(df, w)) for w in trailing]
        avg_4w: Dict[str, float] = {}
        if trailing_kpi_list:
            for key in curr_kpis:
                avg_4w[key] = float(np.mean([t[key] for t in trailing_kpi_list]))
        else:
            avg_4w = {k: 0.0 for k in curr_kpis}

        # Build per-metric detail
        metrics = {}
        for key in curr_kpis:
            c = curr_kpis[key]
            p = prior_kpis[key]
            a = avg_4w[key]
            wow = _pct_change(c, p)
            vs4w = _pct_change(c, a)
            metrics[key] = {
                "current": c,
                "prior_week": p,
                "wow_change_pct": wow,
                "4w_avg": a,
                "vs_4w_pct": vs4w,
                "trend": _trend_label(wow, vs4w),
            }

        return {
            "latest_cohort": str(latest),
            "prior_cohort": str(prior),
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # A2. Waterfall
    # ------------------------------------------------------------------

    def _waterfall(self, data: EnrichedData) -> dict:
        df = data.enriched
        latest, prior = self._latest_two(data)
        curr_df = self._week_agg(df, latest)
        prev_df = self._week_agg(df, prior)

        total_enrolls_curr = curr_df["ENROLLS"].sum()
        total_enrolls_prev = prev_df["ENROLLS"].sum()

        def _cohort_rpu(sub: pd.DataFrame, enrolls: float) -> float:
            return _safe_div(sub["payout"].sum(), enrolls)

        # RPU by click-cohort bucket
        buckets = {
            "W1": _W1,
            "W2-W4": _W2W4,
            "M3": _M3,
        }

        deltas: Dict[str, float] = {}
        for label, ck in buckets.items():
            curr_sub = curr_df[curr_df["click_cohort"] == ck] if "click_cohort" in curr_df.columns else pd.DataFrame()
            prev_sub = prev_df[prev_df["click_cohort"] == ck] if "click_cohort" in prev_df.columns else pd.DataFrame()
            c_rpu = _cohort_rpu(curr_sub, total_enrolls_curr)
            p_rpu = _cohort_rpu(prev_sub, total_enrolls_prev)
            deltas[label] = c_rpu - p_rpu

        # Non-click
        nc_curr = curr_df[curr_df["TOUCHPOINT"] == "Non-click"]
        nc_prev = prev_df[prev_df["TOUCHPOINT"] == "Non-click"]
        deltas["non_click"] = _cohort_rpu(nc_curr, total_enrolls_curr) - _cohort_rpu(nc_prev, total_enrolls_prev)

        # Non-enrolled
        if "enrol_status" in curr_df.columns:
            ne_curr = curr_df[curr_df["enrol_status"] == _NON_ENROLLED]
            ne_prev = prev_df[prev_df["enrol_status"] == _NON_ENROLLED]
        else:
            ne_curr = pd.DataFrame()
            ne_prev = pd.DataFrame()
        deltas["non_enroll"] = (
            _cohort_rpu(ne_curr, total_enrolls_curr) - _cohort_rpu(ne_prev, total_enrolls_prev)
        )

        total_delta = sum(deltas.values())

        # Shapley decomposition for W1 per (touchpoint, lender, segment) combo
        w1_contributors = []
        if "click_cohort" in df.columns:
            w1_curr = curr_df[curr_df["click_cohort"] == _W1]
            w1_prev = prev_df[prev_df["click_cohort"] == _W1]

            # Group by touchpoint x lender x segment
            group_cols = ["TOUCHPOINT", "LENDER", "SEGMENT"]
            available_cols = [c for c in group_cols if c in df.columns]
            if available_cols:
                combos = set()
                if not w1_curr.empty:
                    for _, row in w1_curr[available_cols].drop_duplicates().iterrows():
                        combos.add(tuple(row[c] for c in available_cols))
                if not w1_prev.empty:
                    for _, row in w1_prev[available_cols].drop_duplicates().iterrows():
                        combos.add(tuple(row[c] for c in available_cols))

                for combo in combos:
                    mask_curr = pd.Series(True, index=w1_curr.index)
                    mask_prev = pd.Series(True, index=w1_prev.index)
                    combo_label = {}
                    for i, col in enumerate(available_cols):
                        mask_curr = mask_curr & (w1_curr[col] == combo[i])
                        mask_prev = mask_prev & (w1_prev[col] == combo[i])
                        combo_label[col] = combo[i]

                    c_sub = w1_curr[mask_curr]
                    p_sub = w1_prev[mask_prev]

                    c_imp = _safe_div(c_sub["impression_count"].sum(), total_enrolls_curr)
                    c_ctr = _safe_div(c_sub["click_count"].sum(), c_sub["impression_count"].sum())
                    c_rpc = _safe_div(c_sub["payout"].sum(), c_sub["click_count"].sum())

                    p_imp = _safe_div(p_sub["impression_count"].sum(), total_enrolls_prev)
                    p_ctr = _safe_div(p_sub["click_count"].sum(), p_sub["impression_count"].sum())
                    p_rpc = _safe_div(p_sub["payout"].sum(), p_sub["click_count"].sum())

                    prev_factors = {"imp": p_imp, "ctr": p_ctr, "rpc": p_rpc}
                    curr_factors = {"imp": c_imp, "ctr": c_ctr, "rpc": c_rpc}
                    decomp = shapley_decomposition(prev_factors, curr_factors, ["imp", "ctr", "rpc"])

                    total_contribution = sum(decomp.values())
                    w1_contributors.append({
                        "combo": combo_label,
                        "contribution": total_contribution,
                        "decomposition": decomp,
                    })

            # Sort by absolute contribution
            w1_contributors.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "total_delta_rpu": total_delta,
            "bucket_deltas": deltas,
            "w1_top_contributors": w1_contributors[:20],
        }

    # ------------------------------------------------------------------
    # A3. Segment Tracker
    # ------------------------------------------------------------------

    def _segment_tracker(self, data: EnrichedData) -> dict:
        df = data.enriched
        latest, prior = self._latest_two(data)
        curr_df = self._week_agg(df, latest)
        total_enrolls = curr_df["ENROLLS"].sum()

        segments = {}
        for seg in data.segments:
            seg_df = curr_df[curr_df["SEGMENT"] == seg]
            seg_prev = self._week_agg(df, prior)
            seg_prev = seg_prev[seg_prev["SEGMENT"] == seg]

            seg_enrolls = seg_df["ENROLLS"].sum()
            seg_payout = seg_df["payout"].sum()
            seg_impressions = seg_df["impression_count"].sum()
            seg_clicks = seg_df["click_count"].sum()

            prev_enrolls = seg_prev["ENROLLS"].sum()

            # Top 3 touchpoint x lender by payout
            top_combos = []
            if not seg_df.empty and "TOUCHPOINT" in seg_df.columns and "LENDER" in seg_df.columns:
                grouped = seg_df.groupby(["TOUCHPOINT", "LENDER"])["payout"].sum().reset_index()
                grouped = grouped.sort_values("payout", ascending=False).head(3)
                for _, row in grouped.iterrows():
                    top_combos.append({
                        "touchpoint": row["TOUCHPOINT"],
                        "lender": row["LENDER"],
                        "payout": float(row["payout"]),
                    })

            segments[str(seg)] = {
                "rpu": _safe_div(seg_payout, seg_enrolls),
                "imp_pct": _safe_div(seg_impressions, seg_enrolls),
                "ctr": _safe_div(seg_clicks, seg_impressions),
                "rpc": _safe_div(seg_payout, seg_clicks),
                "enrollment_volume": float(seg_enrolls),
                "mix_share": _safe_div(seg_enrolls, total_enrolls),
                "mix_shift_pct": _pct_change(
                    _safe_div(seg_enrolls, total_enrolls),
                    _safe_div(prev_enrolls, self._week_agg(df, prior)["ENROLLS"].sum()),
                ),
                "top_3_combos": top_combos,
            }

        return segments

    # ------------------------------------------------------------------
    # A4. Touchpoint Matrix
    # ------------------------------------------------------------------

    def _touchpoint_matrix(self, data: EnrichedData) -> dict:
        df = data.enriched
        latest, prior = self._latest_two(data)
        curr_df = self._week_agg(df, latest)
        prev_df = self._week_agg(df, prior)
        total_enrolls_curr = curr_df["ENROLLS"].sum()
        total_enrolls_prev = prev_df["ENROLLS"].sum()

        matrix = {}
        for tp in data.touchpoints:
            tp_curr = curr_df[curr_df["TOUCHPOINT"] == tp]
            tp_prev = prev_df[prev_df["TOUCHPOINT"] == tp]

            c_imp = tp_curr["impression_count"].sum()
            c_clicks = tp_curr["click_count"].sum()
            c_payout = tp_curr["payout"].sum()
            c_conversions = tp_curr["conversion_count"].sum()

            p_imp = tp_prev["impression_count"].sum()
            p_clicks = tp_prev["click_count"].sum()
            p_payout = tp_prev["payout"].sum()

            imp_pct = _safe_div(c_imp, total_enrolls_curr)
            ctr = _safe_div(c_clicks, c_imp)
            rpc = _safe_div(c_payout, c_clicks)
            rpu_contrib = _safe_div(c_payout, total_enrolls_curr)
            conv_rate = _safe_div(c_conversions, c_clicks)

            p_imp_pct = _safe_div(p_imp, total_enrolls_prev)
            p_ctr = _safe_div(p_clicks, p_imp)
            p_rpc = _safe_div(p_payout, p_clicks)
            p_rpu_contrib = _safe_div(p_payout, total_enrolls_prev)

            # Top lenders within this touchpoint
            top_lenders = []
            if not tp_curr.empty:
                lender_grp = tp_curr.groupby("LENDER").agg(
                    payout=("payout", "sum"),
                    clicks=("click_count", "sum"),
                ).reset_index().sort_values("payout", ascending=False).head(5)
                for _, row in lender_grp.iterrows():
                    top_lenders.append({
                        "lender": row["LENDER"],
                        "payout": float(row["payout"]),
                        "clicks": float(row["clicks"]),
                    })

            matrix[str(tp)] = {
                "imp_pct": imp_pct,
                "ctr": ctr,
                "rpc": rpc,
                "rpu_contribution": rpu_contrib,
                "conversion_rate": conv_rate,
                "wow_changes": {
                    "imp_pct": _pct_change(imp_pct, p_imp_pct),
                    "ctr": _pct_change(ctr, p_ctr),
                    "rpc": _pct_change(rpc, p_rpc),
                    "rpu_contribution": _pct_change(rpu_contrib, p_rpu_contrib),
                },
                "top_lenders": top_lenders,
            }

        return matrix

    # ------------------------------------------------------------------
    # A5. Affiliate Scorecard
    # ------------------------------------------------------------------

    def _affiliate_scorecard(self, data: EnrichedData) -> dict:
        df = data.enriched
        latest, prior = self._latest_two(data)
        curr_df = self._week_agg(df, latest)
        total_enrolls = curr_df["ENROLLS"].sum()
        total_payout = curr_df["payout"].sum()

        scorecards = {}
        for lender in data.lenders:
            l_curr = curr_df[curr_df["LENDER"] == lender]
            l_prev = self._week_agg(df, prior)
            l_prev = l_prev[l_prev["LENDER"] == lender]

            l_payout = l_curr["payout"].sum()
            l_clicks = l_curr["click_count"].sum()
            l_impressions = l_curr["impression_count"].sum()
            l_conversions = l_curr["conversion_count"].sum()

            p_clicks = l_prev["click_count"].sum()
            p_payout = l_prev["payout"].sum()

            # CTR by touchpoint
            ctr_by_tp = {}
            if not l_curr.empty:
                for tp in l_curr["TOUCHPOINT"].unique():
                    tp_sub = l_curr[l_curr["TOUCHPOINT"] == tp]
                    ctr_by_tp[str(tp)] = _safe_div(
                        tp_sub["click_count"].sum(), tp_sub["impression_count"].sum()
                    )

            # Revenue timing split
            w1_pay = l_curr[l_curr["click_cohort"] == _W1]["payout"].sum() if "click_cohort" in l_curr.columns else 0.0
            w2w4_pay = l_curr[l_curr["click_cohort"] == _W2W4]["payout"].sum() if "click_cohort" in l_curr.columns else 0.0
            m3_pay = l_curr[l_curr["click_cohort"] == _M3]["payout"].sum() if "click_cohort" in l_curr.columns else 0.0

            scorecards[str(lender)] = {
                "total_payout": float(l_payout),
                "rpu_contribution": _safe_div(l_payout, total_enrolls),
                "share_of_total": _safe_div(l_payout, total_payout) * 100.0,
                "clicks": float(l_clicks),
                "ctr_by_touchpoint": ctr_by_tp,
                "rpc_current": _safe_div(l_payout, l_clicks),
                "rpc_prior": _safe_div(p_payout, p_clicks),
                "rpc_trend": _pct_change(
                    _safe_div(l_payout, l_clicks),
                    _safe_div(p_payout, p_clicks),
                ),
                "conversion_rate": _safe_div(l_conversions, l_clicks),
                "ecpm": _safe_div(l_payout, l_impressions) * 1000.0 if l_impressions > 0 else 0.0,
                "revenue_timing": {
                    "w1_pct": _safe_div(w1_pay, l_payout) * 100.0,
                    "w2w4_pct": _safe_div(w2w4_pay, l_payout) * 100.0,
                    "m3_pct": _safe_div(m3_pay, l_payout) * 100.0,
                },
            }

        return scorecards

    # ------------------------------------------------------------------
    # A6. Mix Analysis
    # ------------------------------------------------------------------

    def _mix_analysis(self, data: EnrichedData) -> dict:
        df = data.enriched
        latest, prior = self._latest_two(data)
        curr_df = self._week_agg(df, latest)
        prev_df = self._week_agg(df, prior)

        # Decompose by segment: rate_effect vs mix_effect
        total_enrolls_curr = curr_df["ENROLLS"].sum()
        total_enrolls_prev = prev_df["ENROLLS"].sum()

        segments = data.segments
        rate_effect_total = 0.0
        mix_effect_total = 0.0
        segment_details = {}

        for seg in segments:
            c_seg = curr_df[curr_df["SEGMENT"] == seg]
            p_seg = prev_df[prev_df["SEGMENT"] == seg]

            c_rate = _safe_div(c_seg["payout"].sum(), c_seg["ENROLLS"].sum())  # RPU_i curr
            p_rate = _safe_div(p_seg["payout"].sum(), p_seg["ENROLLS"].sum())  # RPU_i prev

            c_mix = _safe_div(c_seg["ENROLLS"].sum(), total_enrolls_curr)
            p_mix = _safe_div(p_seg["ENROLLS"].sum(), total_enrolls_prev)

            rate_contrib = (c_rate - p_rate) * p_mix
            mix_contrib = p_rate * (c_mix - p_mix)

            rate_effect_total += rate_contrib
            mix_effect_total += mix_contrib

            segment_details[str(seg)] = {
                "curr_rate": c_rate,
                "prev_rate": p_rate,
                "curr_mix": c_mix,
                "prev_mix": p_mix,
                "rate_effect": rate_contrib,
                "mix_effect": mix_contrib,
            }

        interaction = (
            _safe_div(curr_df["payout"].sum(), total_enrolls_curr)
            - _safe_div(prev_df["payout"].sum(), total_enrolls_prev)
            - rate_effect_total
            - mix_effect_total
        )

        return {
            "rate_effect": rate_effect_total,
            "mix_effect": mix_effect_total,
            "interaction": interaction,
            "total_delta_rpu": rate_effect_total + mix_effect_total + interaction,
            "segments": segment_details,
        }

    # ------------------------------------------------------------------
    # A7. Anomaly Detection
    # ------------------------------------------------------------------

    def _anomaly_detect(self, data: EnrichedData) -> list:
        df = data.enriched
        latest, prior = self._latest_two(data)
        thresholds = self.config.get("alert_thresholds", {})
        lookback = self.config.get("lookback_weeks", 8)

        weeks = self._cohort_weeks(df)
        latest_idx = weeks.index(latest) if latest in weeks else len(weeks) - 1
        trail_start = max(0, latest_idx - lookback)
        trailing_weeks = weeks[trail_start:latest_idx]

        alerts: List[dict] = []

        # Pre-compute weekly aggregate series
        def _weekly_metric(metric_fn) -> pd.Series:
            vals = {}
            for w in trailing_weeks:
                w_df = self._week_agg(df, w)
                vals[w] = metric_fn(w_df)
            return pd.Series(vals)

        curr_df = self._week_agg(df, latest)
        prev_df = self._week_agg(df, prior)
        total_enrolls_curr = curr_df["ENROLLS"].sum()
        total_enrolls_prev = prev_df["ENROLLS"].sum()

        # 1. RPU spike/drop
        curr_rpu = _safe_div(curr_df["payout"].sum(), total_enrolls_curr)
        rpu_history = _weekly_metric(lambda w: _safe_div(w["payout"].sum(), w["ENROLLS"].sum()))
        rpu_z = z_score(curr_rpu, rpu_history)
        z_thresh = thresholds.get("rpu_z_score", 2.0)
        if abs(rpu_z) > z_thresh:
            alerts.append({
                "type": "rpu_spike" if rpu_z > 0 else "rpu_drop",
                "z_score": rpu_z,
                "current_value": curr_rpu,
                "threshold": z_thresh,
                "severity": "high",
            })

        # 2. CTR collapse (>30% drop)
        curr_ctr = _safe_div(curr_df["click_count"].sum(), curr_df["impression_count"].sum())
        prev_ctr = _safe_div(prev_df["click_count"].sum(), prev_df["impression_count"].sum())
        ctr_drop = thresholds.get("ctr_drop_pct", 30)
        if prev_ctr > 0:
            ctr_change = (curr_ctr - prev_ctr) / prev_ctr * 100
            if ctr_change < -ctr_drop:
                alerts.append({
                    "type": "ctr_collapse",
                    "wow_change_pct": ctr_change,
                    "current_value": curr_ctr,
                    "prior_value": prev_ctr,
                    "threshold_pct": -ctr_drop,
                    "severity": "high",
                })

        # 3. Affiliate gone (>5% share to 0)
        min_share = thresholds.get("affiliate_gone_min_share", 5)
        prev_total_payout = prev_df["payout"].sum()
        for lender in data.lenders:
            prev_lender_pay = prev_df[prev_df["LENDER"] == lender]["payout"].sum()
            prev_share = _safe_div(prev_lender_pay, prev_total_payout) * 100
            curr_lender_pay = curr_df[curr_df["LENDER"] == lender]["payout"].sum()
            if prev_share >= min_share and curr_lender_pay == 0:
                alerts.append({
                    "type": "affiliate_gone",
                    "lender": str(lender),
                    "prior_share_pct": prev_share,
                    "threshold_pct": min_share,
                    "severity": "critical",
                })

        # 4. Impression flood/drought (z>2)
        curr_imp = curr_df["impression_count"].sum()
        imp_history = _weekly_metric(lambda w: w["impression_count"].sum())
        imp_z = z_score(float(curr_imp), imp_history)
        if abs(imp_z) > z_thresh:
            alerts.append({
                "type": "impression_flood" if imp_z > 0 else "impression_drought",
                "z_score": imp_z,
                "current_value": float(curr_imp),
                "threshold": z_thresh,
                "severity": "medium",
            })

        # 5. Segment divergence (>10% opposite)
        seg_div_thresh = thresholds.get("segment_divergence_pct", 10)
        seg_changes = {}
        for seg in data.segments:
            c_seg = curr_df[curr_df["SEGMENT"] == seg]
            p_seg = prev_df[prev_df["SEGMENT"] == seg]
            c_rpu = _safe_div(c_seg["payout"].sum(), c_seg["ENROLLS"].sum())
            p_rpu = _safe_div(p_seg["payout"].sum(), p_seg["ENROLLS"].sum())
            seg_changes[seg] = _pct_change(c_rpu, p_rpu)

        if len(seg_changes) >= 2:
            vals = list(seg_changes.values())
            max_change = max(vals)
            min_change = min(vals)
            if max_change > seg_div_thresh and min_change < -seg_div_thresh:
                improving = [s for s, v in seg_changes.items() if v > seg_div_thresh]
                declining = [s for s, v in seg_changes.items() if v < -seg_div_thresh]
                alerts.append({
                    "type": "segment_divergence",
                    "improving_segments": [str(s) for s in improving],
                    "declining_segments": [str(s) for s in declining],
                    "threshold_pct": seg_div_thresh,
                    "severity": "medium",
                })

        # 6. Enrollment shock (>20% WoW)
        enroll_shock = thresholds.get("enrollment_shock_pct", 20)
        enroll_change = _pct_change(total_enrolls_curr, total_enrolls_prev)
        if abs(enroll_change) > enroll_shock:
            alerts.append({
                "type": "enrollment_shock",
                "wow_change_pct": enroll_change,
                "current_value": float(total_enrolls_curr),
                "prior_value": float(total_enrolls_prev),
                "threshold_pct": enroll_shock,
                "severity": "high",
            })

        # 7. Conversion rate drop (>40%)
        conv_drop = thresholds.get("conversion_rate_drop_pct", 40)
        curr_conv = _safe_div(curr_df["conversion_count"].sum(), curr_df["click_count"].sum())
        prev_conv = _safe_div(prev_df["conversion_count"].sum(), prev_df["click_count"].sum())
        if prev_conv > 0:
            conv_change = (curr_conv - prev_conv) / prev_conv * 100
            if conv_change < -conv_drop:
                alerts.append({
                    "type": "conversion_rate_drop",
                    "wow_change_pct": conv_change,
                    "current_value": curr_conv,
                    "prior_value": prev_conv,
                    "threshold_pct": -conv_drop,
                    "severity": "critical",
                })

        return alerts
