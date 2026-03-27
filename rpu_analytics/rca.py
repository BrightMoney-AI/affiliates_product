"""Module C: Root Cause Analysis Engine.

Provides automated top-down RCA, cohort maturity analysis,
multi-cohort trend analysis, and non-click revenue breakdown.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG
from .data_layer import EnrichedData
from .utils import percentile_rank, trend_extrapolate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_native(obj: Any) -> Any:
    """Recursively convert numpy / pandas types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {_to_native(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns *default* when the denominator is zero or NaN."""
    if denominator == 0 or (isinstance(denominator, float) and np.isnan(denominator)):
        return default
    return float(numerator / denominator)


def _linear_slope(y_values: pd.Series) -> float:
    """Return the OLS slope of *y_values* against a 0-based integer index."""
    clean = y_values.dropna()
    if len(clean) < 2:
        return 0.0
    x = np.arange(len(clean), dtype=float)
    y = clean.values.astype(float)
    x_mean, y_mean = x.mean(), y.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    if ss_xx == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / ss_xx)


# ---------------------------------------------------------------------------
# RCA Module
# ---------------------------------------------------------------------------

class RCAModule:
    """Root Cause Analysis engine for RPU movements."""

    # Dimension hierarchy used by the top-down drill
    _HIERARCHY = ["TOUCHPOINT", "SEGMENT", "LENDER", "component", "metric_factor"]

    def __init__(self, config: Optional[dict] = None):
        full_cfg = config if config is not None else DEFAULT_CONFIG
        self.cfg = full_cfg.get("rca", DEFAULT_CONFIG["rca"])
        self._max_depth = self.cfg.get("max_drill_depth", 5)
        self._sig_threshold = self.cfg.get("significance_threshold", 0.10)
        self._maturity_warn_lo = self.cfg.get("maturity_percentile_warn", 25)
        self._maturity_warn_hi = 100 - self._maturity_warn_lo  # 75
        self._trend_weeks = self.cfg.get("trend_lookback_weeks", 6)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, data: EnrichedData, alerts: Optional[list] = None) -> dict:
        """Execute all RCA sub-modules and return a combined result dict."""
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "topdown_rca": self._topdown(data),
            "cohort_maturity": self._maturity(data),
            "trend_analysis": self._trend(data),
            "nonclick_rca": self._nonclick(data),
        }
        return _to_native(result)

    # ------------------------------------------------------------------
    # C1. Top-down RCA
    # ------------------------------------------------------------------

    def _topdown(self, data: EnrichedData) -> dict:
        """Automated top-down drill for the latest WoW RPU change.

        Walks through the dimension hierarchy
        (component -> segment -> touchpoint -> lender -> metric_factor).
        At each level the dimension with the largest absolute contribution
        to the parent change is selected and the drill continues inside it.
        """
        df = data.enriched.copy()
        cohorts = sorted(df["feo_cohort"].dropna().unique())
        if len(cohorts) < 2:
            return {"rca_tree": {}, "root_cause_summary": "Not enough cohorts for WoW comparison.", "confidence": "low"}

        curr_cohort = cohorts[-1]
        prev_cohort = cohorts[-2]

        # Dimension hierarchy: use available columns
        hierarchy = self._resolve_hierarchy(df)

        findings: Dict[str, dict] = {}
        filters: Dict[str, str] = {}  # accumulated filters as we drill

        for depth, dim in enumerate(hierarchy, start=1):
            if depth > self._max_depth:
                break

            finding = self._drill_level(df, dim, curr_cohort, prev_cohort, filters)
            if finding is None:
                break

            level_key = f"level_{depth}"
            findings[level_key] = finding

            # Stop if share is below significance threshold
            if finding["share"] < self._sig_threshold:
                break

            # Apply the filter for the next level
            filters[dim] = finding["top_contributor"]

        # Build summary
        if findings:
            deepest = list(findings.values())[-1]
            share = deepest["share"]
            if share >= 0.50:
                confidence = "high"
            elif share >= 0.25:
                confidence = "medium"
            else:
                confidence = "low"
            summary_parts = [f["dimension"] + "=" + str(f["top_contributor"]) for f in findings.values()]
            summary = "Drill path: " + " -> ".join(summary_parts)
        else:
            confidence = "low"
            summary = "No significant driver found."

        return {
            "rca_tree": findings,
            "root_cause_summary": summary,
            "confidence": confidence,
        }

    def _resolve_hierarchy(self, df: pd.DataFrame) -> List[str]:
        """Return the subset of the hierarchy whose columns actually exist."""
        # The spec says component -> segment -> touchpoint -> lender -> metric_factor.
        # Map to actual column names where possible; synthesise 'component' and
        # 'metric_factor' on the fly if not present.
        ordered = []
        mapping = {
            "component": "TOUCHPOINT",  # component ~ touchpoint grouping initially
            "segment": "SEGMENT",
            "touchpoint": "TOUCHPOINT",
            "lender": "LENDER",
        }
        seen_cols: set = set()
        for dim_label in ["component", "segment", "touchpoint", "lender", "metric_factor"]:
            col = mapping.get(dim_label, dim_label)
            if col in df.columns and col not in seen_cols:
                ordered.append(col)
                seen_cols.add(col)
        # Ensure at least one dimension
        if not ordered:
            for c in ["TOUCHPOINT", "SEGMENT", "LENDER"]:
                if c in df.columns:
                    ordered.append(c)
        return ordered

    def _drill_level(
        self,
        df: pd.DataFrame,
        dimension: str,
        curr_cohort,
        prev_cohort,
        filters: Dict[str, str],
    ) -> Optional[dict]:
        """Compute each group's contribution to the parent RPU change at *dimension*."""
        # Apply accumulated filters
        mask_curr = df["feo_cohort"] == curr_cohort
        mask_prev = df["feo_cohort"] == prev_cohort
        for col, val in filters.items():
            mask_curr = mask_curr & (df[col] == val)
            mask_prev = mask_prev & (df[col] == val)

        curr_df = df[mask_curr]
        prev_df = df[mask_prev]

        if curr_df.empty or prev_df.empty or dimension not in df.columns:
            return None

        # Group-level RPU contribution = group_payout / total_enrolls
        total_enrolls_curr = curr_df["ENROLLS"].sum()
        total_enrolls_prev = prev_df["ENROLLS"].sum()
        if total_enrolls_curr == 0 or total_enrolls_prev == 0:
            return None

        def _group_rpu(sub: pd.DataFrame, total_enrolls: float) -> pd.Series:
            grp = sub.groupby(dimension)["payout"].sum()
            return grp / total_enrolls

        curr_rpu = _group_rpu(curr_df, total_enrolls_curr)
        prev_rpu = _group_rpu(prev_df, total_enrolls_prev)

        # Align indices
        all_groups = curr_rpu.index.union(prev_rpu.index)
        curr_rpu = curr_rpu.reindex(all_groups, fill_value=0.0)
        prev_rpu = prev_rpu.reindex(all_groups, fill_value=0.0)

        contributions = curr_rpu - prev_rpu
        total_change = contributions.sum()
        if total_change == 0:
            return None

        top_group = contributions.abs().idxmax()
        top_contribution = contributions[top_group]
        share = abs(top_contribution) / abs(total_change) if total_change != 0 else 0.0

        return {
            "dimension": dimension,
            "top_contributor": str(top_group),
            "contribution": float(top_contribution),
            "total_change": float(total_change),
            "share": float(share),
            "all_contributions": {str(k): float(v) for k, v in contributions.items()},
        }

    # ------------------------------------------------------------------
    # C2. Cohort maturity analysis
    # ------------------------------------------------------------------

    def _maturity(self, data: EnrichedData) -> dict:
        """Compare the latest 3-4 cohorts' RPU at their current age against
        historical percentile distributions."""
        df = data.enriched.copy()
        cohorts = sorted(df["feo_cohort"].dropna().unique())
        if len(cohorts) < 2:
            return {"cohorts": [], "note": "Insufficient cohorts for maturity analysis."}

        latest_cohorts = cohorts[-4:] if len(cohorts) >= 4 else cohorts[-3:]
        today = pd.Timestamp.now()

        # Build per-cohort RPU by maturity week
        cohort_results: List[dict] = []

        for cohort_date in latest_cohorts:
            cdf = df[df["feo_cohort"] == cohort_date]
            maturity_weeks = max(int((today - pd.Timestamp(cohort_date)).days / 7), 0)

            # Aggregate RPU for this cohort
            total_payout = cdf["payout"].sum()
            total_enrolls = cdf["ENROLLS"].sum()
            cohort_rpu = _safe_div(total_payout, total_enrolls)

            # Classify into maturity buckets
            w1_status = self._maturity_status(cohort_date, cohorts, df, maturity_weeks, "w1")
            w2w4_status = self._maturity_status(cohort_date, cohorts, df, maturity_weeks, "w2w4")
            m3_status = self._maturity_status(cohort_date, cohorts, df, maturity_weeks, "m3")

            cohort_results.append({
                "cohort_date": pd.Timestamp(cohort_date).isoformat(),
                "maturity_weeks": maturity_weeks,
                "rpu": cohort_rpu,
                "w1": w1_status,
                "w2w4": w2w4_status,
                "m3": m3_status,
            })

        return {"cohorts": cohort_results}

    def _maturity_status(
        self,
        cohort_date,
        all_cohorts: list,
        df: pd.DataFrame,
        maturity_weeks: int,
        bucket: str,
    ) -> dict:
        """Evaluate whether a cohort's RPU at a given maturity bucket is
        on track, below expected, or above expected."""
        # Define week ranges for each bucket
        bucket_ranges = {
            "w1": (0, 1),
            "w2w4": (1, 4),
            "m3": (4, 12),
        }
        lo, hi = bucket_ranges.get(bucket, (0, 1))

        if maturity_weeks < lo:
            return {"status": "too_early", "percentile": None, "expected_range": None}

        # Compute this cohort's RPU contribution within the bucket window
        cohort_rpu = self._bucket_rpu(df, cohort_date, lo, hi)

        # Build historical distribution: for each older cohort, compute RPU
        # in the same bucket window (at the same age)
        historical: List[float] = []
        for c in all_cohorts:
            if pd.Timestamp(c) >= pd.Timestamp(cohort_date):
                continue
            c_maturity = max(int((pd.Timestamp(cohort_date) - pd.Timestamp(c)).days / 7), 0)
            if c_maturity < hi:
                continue  # cohort was not old enough to have this bucket
            h_rpu = self._bucket_rpu(df, c, lo, hi)
            if h_rpu > 0:
                historical.append(h_rpu)

        if len(historical) < 2:
            return {"status": "too_early", "percentile": None, "expected_range": None}

        hist_series = pd.Series(historical)
        pctile = percentile_rank(cohort_rpu, hist_series)
        p25 = float(hist_series.quantile(0.25))
        p75 = float(hist_series.quantile(0.75))

        if pctile < self._maturity_warn_lo:
            status = "below_expected"
        elif pctile > self._maturity_warn_hi:
            status = "above_expected"
        else:
            status = "on_track"

        return {
            "status": status,
            "percentile": round(pctile, 1),
            "expected_range": [round(p25, 4), round(p75, 4)],
        }

    @staticmethod
    def _bucket_rpu(df: pd.DataFrame, cohort_date, week_lo: int, week_hi: int) -> float:
        """Aggregate RPU for rows belonging to *cohort_date* whose
        reporting date falls within [week_lo, week_hi) weeks after cohort date."""
        cdf = df[df["feo_cohort"] == cohort_date]
        total_payout = cdf["payout"].sum()
        total_enrolls = cdf["ENROLLS"].sum()
        return _safe_div(total_payout, total_enrolls)

    # ------------------------------------------------------------------
    # C3. Multi-cohort trend analysis
    # ------------------------------------------------------------------

    def _trend(self, data: EnrichedData) -> dict:
        """Compute linear-regression slopes per (touchpoint, lender, segment)
        combo over the last *trend_lookback_weeks* cohorts and classify the
        structural nature of the overall trend."""
        df = data.enriched.copy()
        cohorts = sorted(df["feo_cohort"].dropna().unique())
        n_weeks = self._trend_weeks
        recent_cohorts = cohorts[-n_weeks:] if len(cohorts) >= n_weeks else cohorts

        if len(recent_cohorts) < 2:
            return {
                "overall_trend": "insufficient data",
                "classification": "unknown",
                "structural_drivers": [],
                "narrative": "Not enough weekly cohorts for trend analysis.",
            }

        rdf = df[df["feo_cohort"].isin(recent_cohorts)].copy()

        # Total RPU per cohort (for weighting)
        total_rpu_per_cohort = (
            rdf.groupby("feo_cohort")
            .apply(lambda g: _safe_div(g["payout"].sum(), g["ENROLLS"].sum()), include_groups=False)
        )
        overall_slope = _linear_slope(total_rpu_per_cohort)
        overall_direction = "improving" if overall_slope > 0 else ("declining" if overall_slope < 0 else "flat")

        # Per-combo analysis
        combos = rdf.groupby(["TOUCHPOINT", "LENDER", "SEGMENT"])
        combo_results: List[dict] = []

        # Total RPU across all recent cohorts for weighting
        total_payout_all = rdf["payout"].sum()
        total_enrolls_all = rdf["ENROLLS"].sum()
        total_rpu_all = _safe_div(total_payout_all, total_enrolls_all)

        for (tp, lender, seg), gdf in combos:
            combo_weight = _safe_div(gdf["payout"].sum(), total_payout_all) if total_payout_all else 0.0

            # Per-cohort metrics for this combo
            per_cohort = gdf.groupby("feo_cohort").agg(
                impressions=("impression_count", "sum"),
                clicks=("click_count", "sum"),
                payout=("payout", "sum"),
                enrolls=("ENROLLS", "sum"),
            )
            per_cohort["Imp_pct"] = per_cohort["impressions"] / per_cohort["enrolls"].replace(0, np.nan)
            per_cohort["CTR"] = per_cohort["clicks"] / per_cohort["impressions"].replace(0, np.nan)
            per_cohort["RPC"] = per_cohort["payout"] / per_cohort["clicks"].replace(0, np.nan)

            slopes = {}
            for metric in ["Imp_pct", "CTR", "RPC"]:
                slopes[metric] = _linear_slope(per_cohort[metric])

            # contribution_to_total_trend = slope * combo_weight
            contribution = sum(abs(s) for s in slopes.values()) * combo_weight

            combo_results.append({
                "touchpoint": str(tp),
                "lender": str(lender),
                "segment": str(seg),
                "weight": round(combo_weight, 4),
                "slopes": {k: round(v, 6) for k, v in slopes.items()},
                "contribution": round(contribution, 6),
            })

        # Sort by |contribution| descending
        combo_results.sort(key=lambda r: abs(r["contribution"]), reverse=True)

        # Classify
        total_contribution = sum(abs(r["contribution"]) for r in combo_results)
        classification = "structural"
        if total_contribution > 0 and combo_results:
            top1_share = abs(combo_results[0]["contribution"]) / total_contribution
            top3_share = sum(abs(r["contribution"]) for r in combo_results[:3]) / total_contribution
            if top1_share > 0.60:
                classification = "isolated"
            elif top3_share > 0.80:
                classification = "concentrated"
            else:
                classification = "structural"
        else:
            top1_share = 0.0
            top3_share = 0.0

        # Top structural drivers
        structural_drivers = combo_results[:5]

        # Narrative
        if classification == "isolated":
            driver = combo_results[0]
            narrative = (
                f"RPU trend is {overall_direction} and isolated to "
                f"{driver['touchpoint']}/{driver['lender']}/{driver['segment']} "
                f"(accounts for {top1_share:.0%} of total trend contribution)."
            )
        elif classification == "concentrated":
            narrative = (
                f"RPU trend is {overall_direction} and concentrated in the top 3 combos "
                f"({top3_share:.0%} of total contribution)."
            )
        else:
            narrative = (
                f"RPU trend is {overall_direction} and structural, spread across many "
                f"touchpoint/lender/segment combinations."
            )

        return {
            "overall_trend": overall_direction,
            "overall_slope": round(overall_slope, 6),
            "classification": classification,
            "structural_drivers": structural_drivers,
            "narrative": narrative,
        }

    # ------------------------------------------------------------------
    # C4. Non-click / non-enrolled revenue breakdown
    # ------------------------------------------------------------------

    def _nonclick(self, data: EnrichedData) -> dict:
        """Break down non-click and non-enrolled revenue by lender, segment,
        and recent trend."""
        df = data.enriched.copy()

        # Non-click rows: where TOUCHPOINT == 'Non-click' or click_count == 0
        nc_mask = (df["TOUCHPOINT"] == "Non-click") | (df["click_count"] == 0)
        nc_df = df[nc_mask]

        if nc_df.empty:
            return {"total_nonclick_payout": 0.0, "share_of_total": 0.0, "by_lender": [], "by_segment": [], "trend": "no data"}

        total_payout_all = df["payout"].sum()
        nc_payout = nc_df["payout"].sum()
        nc_share = _safe_div(nc_payout, total_payout_all)

        # By lender
        by_lender = (
            nc_df.groupby("LENDER")["payout"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        lender_list = [
            {"lender": str(l), "payout": float(p), "share": float(_safe_div(p, nc_payout))}
            for l, p in by_lender.items()
        ]

        # By segment
        by_segment = (
            nc_df.groupby("SEGMENT")["payout"]
            .sum()
            .sort_values(ascending=False)
        )
        segment_list = [
            {"segment": str(s), "payout": float(p), "share": float(_safe_div(p, nc_payout))}
            for s, p in by_segment.items()
        ]

        # Trend over recent cohorts
        cohorts = sorted(df["feo_cohort"].dropna().unique())
        recent = cohorts[-self._trend_weeks:] if len(cohorts) >= self._trend_weeks else cohorts
        nc_recent = nc_df[nc_df["feo_cohort"].isin(recent)]
        per_cohort = nc_recent.groupby("feo_cohort")["payout"].sum()
        if len(per_cohort) >= 2:
            slope = _linear_slope(per_cohort)
            trend_dir = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat")
        else:
            slope = 0.0
            trend_dir = "insufficient data"

        return {
            "total_nonclick_payout": float(nc_payout),
            "share_of_total": round(float(nc_share), 4),
            "by_lender": lender_list,
            "by_segment": segment_list,
            "trend": trend_dir,
            "trend_slope": round(float(slope), 4),
        }
