"""Module D: Insights Engine.

Provides structured analytical insights across affiliate rankings,
touchpoint effectiveness, affinity matrices, concentration risk,
funnel leakage, and revenue timing.
"""

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG
from .utils import hhi, trend_arrow, trend_extrapolate
from .data_layer import EnrichedData


class InsightsModule:
    """Generates a comprehensive insights dictionary from enriched RPU data."""

    def __init__(self, config: Optional[dict] = None):
        full_config = config if config is not None else DEFAULT_CONFIG
        self._cfg = full_config.get("insights", DEFAULT_CONFIG["insights"])

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, data: EnrichedData) -> dict:
        """Run all insight sub-modules and return a JSON-serializable dict."""
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "affiliate_rankings": self._affiliate_rank(data),
            "touchpoint_effectiveness": self._touchpoint_eff(data),
            "affinity_matrix": self._affinity_matrix(data),
            "concentration": self._concentration(data),
            "funnel_leakage": self._funnel_leakage(data),
            "revenue_timing": self._revenue_timing(data),
        }

    # ------------------------------------------------------------------
    # D1. Affiliate Ranking
    # ------------------------------------------------------------------

    def _affiliate_rank(self, data: EnrichedData) -> dict:
        df = data.enriched.copy()
        lookback = self._cfg.get("tier_lookback_weeks", 8)

        # Determine the most recent N weeks of cohort dates
        cohort_dates = sorted(df["feo_cohort"].dropna().unique())
        recent_dates = cohort_dates[-lookback:] if len(cohort_dates) >= lookback else cohort_dates
        recent_4w = cohort_dates[-4:] if len(cohort_dates) >= 4 else cohort_dates
        recent_8w = cohort_dates[-8:] if len(cohort_dates) >= 8 else cohort_dates

        df_4w = df[df["feo_cohort"].isin(recent_4w)]
        df_8w = df[df["feo_cohort"].isin(recent_8w)]

        rankings = []

        for lender in data.lenders:
            lender_4w = df_4w[df_4w["LENDER"] == lender]
            lender_8w = df_8w[df_8w["LENDER"] == lender]

            total_payout_4w = float(lender_4w["payout"].sum())
            total_enrolls_4w = float(lender_4w["ENROLLS"].sum())
            rpu_contribution = total_payout_4w / total_enrolls_4w if total_enrolls_4w > 0 else 0.0

            # Growth slope: linear regression of weekly payout over 8 weeks
            weekly_payout = (
                lender_8w.groupby("feo_cohort")["payout"]
                .sum()
                .sort_index()
            )
            growth_slope = 0.0
            if len(weekly_payout) >= 2:
                x = np.arange(len(weekly_payout), dtype=float)
                y = weekly_payout.values.astype(float)
                x_mean, y_mean = x.mean(), y.mean()
                ss_xx = ((x - x_mean) ** 2).sum()
                if ss_xx > 0:
                    growth_slope = float(((x - x_mean) * (y - y_mean)).sum() / ss_xx)

            # RPC stability: coefficient of variation of weekly RPC
            weekly_clicks = lender_8w.groupby("feo_cohort")["click_count"].sum()
            weekly_pay = lender_8w.groupby("feo_cohort")["payout"].sum()
            weekly_rpc = (weekly_pay / weekly_clicks.replace(0, np.nan)).dropna()
            rpc_stability = float(weekly_rpc.std() / weekly_rpc.mean()) if len(weekly_rpc) > 1 and weekly_rpc.mean() != 0 else 0.0

            # Conversion rate (4w)
            total_clicks_4w = float(lender_4w["click_count"].sum())
            total_conversions_4w = float(lender_4w["conversion_count"].sum())
            conversion_rate = total_conversions_4w / total_clicks_4w if total_clicks_4w > 0 else 0.0

            # Best touchpoint and segment
            tp_rpu = lender_4w.groupby("TOUCHPOINT").apply(
                lambda g: g["payout"].sum() / g["ENROLLS"].sum() if g["ENROLLS"].sum() > 0 else 0.0
            )
            best_touchpoint = str(tp_rpu.idxmax()) if len(tp_rpu) > 0 and tp_rpu.max() > 0 else None

            seg_rpu = lender_4w.groupby("SEGMENT").apply(
                lambda g: g["payout"].sum() / g["ENROLLS"].sum() if g["ENROLLS"].sum() > 0 else 0.0
            )
            best_segment = str(seg_rpu.idxmax()) if len(seg_rpu) > 0 and seg_rpu.max() > 0 else None

            rankings.append({
                "lender": lender,
                "rpu_contribution": round(rpu_contribution, 4),
                "growth_slope": round(growth_slope, 4),
                "rpc_stability_cv": round(rpc_stability, 4),
                "conversion_rate": round(conversion_rate, 4),
                "total_payout_4w": round(total_payout_4w, 2),
                "best_touchpoint": best_touchpoint,
                "best_segment": best_segment,
            })

        # Tier assignment
        if rankings:
            rpus = [r["rpu_contribution"] for r in rankings]
            q75 = float(np.percentile(rpus, 75)) if rpus else 0.0
            q50 = float(np.percentile(rpus, 50)) if rpus else 0.0
            q25 = float(np.percentile(rpus, 25)) if rpus else 0.0

            for r in rankings:
                rpu_val = r["rpu_contribution"]
                slope = r["growth_slope"]
                positive_growth = slope > 0

                if rpu_val >= q75 and positive_growth:
                    tier = "Tier 1 Star"
                elif rpu_val >= q75:
                    tier = "Tier 2 Stable"
                elif rpu_val < q50 and positive_growth:
                    tier = "Tier 3 Growing"
                elif rpu_val >= q50 and rpu_val < q75:
                    # Was top-half but now below median threshold -> Declining
                    # Interpret as: in top half but with negative growth
                    tier = "Tier 4 Declining" if not positive_growth else "Tier 3 Growing"
                elif rpu_val <= q25 and not positive_growth:
                    tier = "Tier 5 Long tail"
                else:
                    # Below median, no positive growth, but above q25
                    tier = "Tier 5 Long tail"

                r["tier"] = tier

        # Sort by RPU descending
        rankings.sort(key=lambda r: r["rpu_contribution"], reverse=True)

        return {"lenders": rankings}

    # ------------------------------------------------------------------
    # D2. Touchpoint Effectiveness
    # ------------------------------------------------------------------

    def _touchpoint_eff(self, data: EnrichedData) -> dict:
        df = data.enriched.copy()
        cohort_dates = sorted(df["feo_cohort"].dropna().unique())
        latest = cohort_dates[-1] if cohort_dates else None
        prev = cohort_dates[-2] if len(cohort_dates) >= 2 else None

        results = []

        for tp in data.touchpoints:
            tp_df = df[df["TOUCHPOINT"] == tp]
            tp_curr = tp_df[tp_df["feo_cohort"] == latest] if latest is not None else tp_df.iloc[0:0]
            tp_prev = tp_df[tp_df["feo_cohort"] == prev] if prev is not None else tp_df.iloc[0:0]

            total_enrolls = float(tp_curr["ENROLLS"].sum())
            total_impressions = float(tp_curr["impression_count"].sum())
            total_clicks = float(tp_curr["click_count"].sum())
            total_payout = float(tp_curr["payout"].sum())
            total_conversions = float(tp_curr["conversion_count"].sum())

            imp_pct = total_impressions / total_enrolls if total_enrolls > 0 else 0.0
            ctr = total_clicks / total_impressions if total_impressions > 0 else 0.0
            rpc = total_payout / total_clicks if total_clicks > 0 else 0.0
            rpu_contribution = total_payout / total_enrolls if total_enrolls > 0 else 0.0
            conversion_rate = total_conversions / total_clicks if total_clicks > 0 else 0.0

            # WoW changes
            prev_enrolls = float(tp_prev["ENROLLS"].sum())
            prev_impressions = float(tp_prev["impression_count"].sum())
            prev_clicks = float(tp_prev["click_count"].sum())
            prev_payout = float(tp_prev["payout"].sum())

            prev_imp_pct = prev_impressions / prev_enrolls if prev_enrolls > 0 else 0.0
            prev_ctr = prev_clicks / prev_impressions if prev_impressions > 0 else 0.0
            prev_rpc = prev_payout / prev_clicks if prev_clicks > 0 else 0.0
            prev_rpu = prev_payout / prev_enrolls if prev_enrolls > 0 else 0.0

            wow = {
                "imp_pct": _safe_pct_change(imp_pct, prev_imp_pct),
                "ctr": _safe_pct_change(ctr, prev_ctr),
                "rpc": _safe_pct_change(rpc, prev_rpc),
                "rpu_contribution": _safe_pct_change(rpu_contribution, prev_rpu),
            }

            # Revenue timing: W1 / W2-W4 / M3 split
            timing = _revenue_timing_split(tp_df, cohort_dates)

            results.append({
                "touchpoint": tp,
                "imp_pct": round(imp_pct, 4),
                "ctr": round(ctr, 4),
                "rpc": round(rpc, 4),
                "rpu_contribution": round(rpu_contribution, 4),
                "conversion_rate": round(conversion_rate, 4),
                "wow_changes": {k: round(v, 4) for k, v in wow.items()},
                "revenue_timing": timing,
            })

        # Cross analysis: best touchpoint per segment
        best_tp_per_segment = {}
        for seg in data.segments:
            seg_df = df[df["SEGMENT"] == seg]
            if seg_df.empty:
                continue
            tp_rpu = seg_df.groupby("TOUCHPOINT").apply(
                lambda g: g["payout"].sum() / g["ENROLLS"].sum() if g["ENROLLS"].sum() > 0 else 0.0
            )
            if len(tp_rpu) > 0 and tp_rpu.max() > 0:
                best_tp_per_segment[seg] = str(tp_rpu.idxmax())

        # Best touchpoint per lender
        best_tp_per_lender = {}
        for lender in data.lenders:
            l_df = df[df["LENDER"] == lender]
            if l_df.empty:
                continue
            tp_rpu = l_df.groupby("TOUCHPOINT").apply(
                lambda g: g["payout"].sum() / g["ENROLLS"].sum() if g["ENROLLS"].sum() > 0 else 0.0
            )
            if len(tp_rpu) > 0 and tp_rpu.max() > 0:
                best_tp_per_lender[lender] = str(tp_rpu.idxmax())

        return {
            "touchpoints": results,
            "best_touchpoint_per_segment": best_tp_per_segment,
            "best_touchpoint_per_lender": best_tp_per_lender,
        }

    # ------------------------------------------------------------------
    # D3. Affinity Matrix
    # ------------------------------------------------------------------

    def _affinity_matrix(self, data: EnrichedData) -> dict:
        df = data.enriched.copy()
        cohort_dates = sorted(df["feo_cohort"].dropna().unique())
        latest = cohort_dates[-1] if cohort_dates else None
        prev = cohort_dates[-2] if len(cohort_dates) >= 2 else None

        matrix = []

        for seg in data.segments:
            row = {"segment": seg, "lenders": {}}
            for lender in data.lenders:
                mask = (df["SEGMENT"] == seg) & (df["LENDER"] == lender)
                pair_df = df[mask]

                curr_df = pair_df[pair_df["feo_cohort"] == latest] if latest is not None else pair_df.iloc[0:0]
                prev_df = pair_df[pair_df["feo_cohort"] == prev] if prev is not None else pair_df.iloc[0:0]

                curr_enrolls = float(curr_df["ENROLLS"].sum())
                curr_payout = float(curr_df["payout"].sum())
                curr_rpu = curr_payout / curr_enrolls if curr_enrolls > 0 else 0.0

                prev_enrolls = float(prev_df["ENROLLS"].sum())
                prev_payout = float(prev_df["payout"].sum())
                prev_rpu = prev_payout / prev_enrolls if prev_enrolls > 0 else 0.0

                arrow = trend_arrow(curr_rpu, prev_rpu)

                row["lenders"][lender] = {
                    "rpu_contribution": round(curr_rpu, 4),
                    "trend": arrow,
                }
            matrix.append(row)

        return {"matrix": matrix}

    # ------------------------------------------------------------------
    # D4. Concentration
    # ------------------------------------------------------------------

    def _concentration(self, data: EnrichedData) -> dict:
        df = data.enriched.copy()
        cohort_dates = sorted(df["feo_cohort"].dropna().unique())
        latest = cohort_dates[-1] if cohort_dates else None
        prev = cohort_dates[-2] if len(cohort_dates) >= 2 else None
        warning_hhi = self._cfg.get("concentration_warning_hhi", 0.25)

        results = {}

        for dim_name, col in [("lender", "LENDER"), ("touchpoint", "TOUCHPOINT")]:
            curr_df = df[df["feo_cohort"] == latest] if latest is not None else df.iloc[0:0]
            prev_df = df[df["feo_cohort"] == prev] if prev is not None else df.iloc[0:0]

            curr_payout = curr_df.groupby(col)["payout"].sum()
            prev_payout = prev_df.groupby(col)["payout"].sum()

            curr_total = curr_payout.sum()
            prev_total = prev_payout.sum()

            curr_shares = (curr_payout / curr_total).tolist() if curr_total > 0 else []
            prev_shares = (prev_payout / prev_total).tolist() if prev_total > 0 else []

            curr_hhi = hhi(curr_shares)
            prev_hhi = hhi(prev_shares)

            # Top 3 share
            top_3_share = float(curr_payout.nlargest(3).sum() / curr_total) if curr_total > 0 else 0.0

            # SPOF risk: entity with max share
            if curr_total > 0 and len(curr_payout) > 0:
                max_entity = str(curr_payout.idxmax())
                max_share = float(curr_payout.max() / curr_total)
            else:
                max_entity = None
                max_share = 0.0

            hhi_change = curr_hhi - prev_hhi
            direction = "concentrating" if hhi_change > 0 else "diversifying" if hhi_change < 0 else "stable"

            results[dim_name] = {
                "hhi": round(curr_hhi, 4),
                "top_3_share": round(top_3_share, 4),
                "spof_risk": {
                    "entity": max_entity,
                    "share": round(max_share, 4),
                },
                "wow_hhi_change": round(hhi_change, 4),
                "direction": direction,
                "warning": curr_hhi >= warning_hhi,
            }

        return results

    # ------------------------------------------------------------------
    # D5. Funnel Leakage
    # ------------------------------------------------------------------

    def _funnel_leakage(self, data: EnrichedData) -> dict:
        df = data.enriched.copy()
        threshold = self._cfg.get("leakage_threshold", 0.50)

        # Per (touchpoint, lender, segment) compute funnel metrics
        groups = df.groupby(["TOUCHPOINT", "LENDER", "SEGMENT"])

        records = []
        for (tp, lender, seg), gdf in groups:
            enrolls = float(gdf["ENROLLS"].sum())
            impressions = float(gdf["impression_count"].sum())
            clicks = float(gdf["click_count"].sum())
            payout = float(gdf["payout"].sum())
            conversions = float(gdf["conversion_count"].sum())

            imp_pct = impressions / enrolls if enrolls > 0 else 0.0
            ctr = clicks / impressions if impressions > 0 else 0.0
            conversion_rate = conversions / clicks if clicks > 0 else 0.0
            rev_per_conv = payout / conversions if conversions > 0 else 0.0

            records.append({
                "touchpoint": str(tp),
                "lender": str(lender),
                "segment": str(seg),
                "imp_pct": imp_pct,
                "ctr": ctr,
                "conversion_rate": conversion_rate,
                "revenue_per_conversion": rev_per_conv,
            })

        if not records:
            return {"leakages": [], "total_rpu_upside": 0.0}

        metrics_df = pd.DataFrame(records)

        # Cross-dimensional medians
        stage_keys = ["imp_pct", "ctr", "conversion_rate", "revenue_per_conversion"]
        medians = {}
        for key in stage_keys:
            vals = metrics_df[key]
            nonzero = vals[vals > 0]
            medians[key] = float(nonzero.median()) if len(nonzero) > 0 else 0.0

        # Flag combos significantly below median
        leakages = []
        total_upside = 0.0

        for rec in records:
            flagged_stages = []
            upside = 0.0

            for key in stage_keys:
                median_val = medians[key]
                if median_val > 0 and rec[key] < median_val * threshold:
                    gap = median_val - rec[key]
                    flagged_stages.append({
                        "stage": key,
                        "actual": round(rec[key], 4),
                        "median": round(median_val, 4),
                        "gap": round(gap, 4),
                    })

            if flagged_stages:
                # Estimate RPU upside: if each flagged stage were at median
                # RPU = imp_pct * ctr * rpc  where rpc = conversion_rate * rev_per_conv
                current_rpu = rec["imp_pct"] * rec["ctr"] * rec["conversion_rate"] * rec["revenue_per_conversion"]
                fixed = {k: rec[k] for k in stage_keys}
                for fs in flagged_stages:
                    fixed[fs["stage"]] = fs["median"]
                fixed_rpu = fixed["imp_pct"] * fixed["ctr"] * fixed["conversion_rate"] * fixed["revenue_per_conversion"]
                est_upside = max(0.0, fixed_rpu - current_rpu)
                total_upside += est_upside

                leakages.append({
                    "touchpoint": rec["touchpoint"],
                    "lender": rec["lender"],
                    "segment": rec["segment"],
                    "flagged_stages": flagged_stages,
                    "estimated_rpu_upside": round(est_upside, 4),
                })

        leakages.sort(key=lambda x: x["estimated_rpu_upside"], reverse=True)

        return {
            "leakages": leakages,
            "total_rpu_upside": round(total_upside, 4),
            "medians": {k: round(v, 4) for k, v in medians.items()},
        }

    # ------------------------------------------------------------------
    # D6. Revenue Timing
    # ------------------------------------------------------------------

    def _revenue_timing(self, data: EnrichedData) -> dict:
        df = data.enriched.copy()
        cohort_dates = sorted(df["feo_cohort"].dropna().unique())

        results = []

        for lender in data.lenders:
            lender_df = df[df["LENDER"] == lender]

            timing = _revenue_timing_split(lender_df, cohort_dates)

            w1_pct = timing.get("w1_pct", 0.0)
            if w1_pct >= 60:
                classification = "fast"
            elif w1_pct >= 30:
                classification = "moderate"
            else:
                classification = "slow"

            # Maturity projection for recent cohorts
            # Use historical completion rates to project final RPU
            maturity = _maturity_projection(lender_df, cohort_dates)

            results.append({
                "lender": lender,
                "w1_pct": timing.get("w1_pct", 0.0),
                "w2_w4_pct": timing.get("w2_w4_pct", 0.0),
                "m3_pct": timing.get("m3_pct", 0.0),
                "classification": classification,
                "maturity_projection": maturity,
            })

        return {"lenders": results}


# ======================================================================
# Private helpers
# ======================================================================

def _safe_pct_change(current: float, previous: float) -> float:
    """Percentage change, returning 0.0 when previous is zero."""
    if previous == 0:
        return 0.0
    return (current - previous) / abs(previous)


def _revenue_timing_split(sub_df: pd.DataFrame, cohort_dates: list) -> dict:
    """Compute W1 / W2-W4 / M3 payout percentage split.

    W1 = first cohort week, W2-W4 = next 3 weeks, M3 = remaining through
    week 12. Splits are relative to total payout across all available weeks.
    """
    if sub_df.empty or not cohort_dates:
        return {"w1_pct": 0.0, "w2_w4_pct": 0.0, "m3_pct": 0.0}

    weekly = sub_df.groupby("feo_cohort")["payout"].sum().reindex(cohort_dates, fill_value=0.0)
    total = float(weekly.sum())
    if total == 0:
        return {"w1_pct": 0.0, "w2_w4_pct": 0.0, "m3_pct": 0.0}

    # W1 = first week in this sub-dataset that has data
    ordered = weekly[weekly > 0]
    if ordered.empty:
        return {"w1_pct": 0.0, "w2_w4_pct": 0.0, "m3_pct": 0.0}

    # Use positional indexing relative to the first active week
    active_weeks = ordered.values
    w1 = float(active_weeks[0]) if len(active_weeks) >= 1 else 0.0
    w2_w4 = float(active_weeks[1:4].sum()) if len(active_weeks) > 1 else 0.0
    m3 = float(active_weeks[4:12].sum()) if len(active_weeks) > 4 else 0.0

    return {
        "w1_pct": round(w1 / total * 100, 2),
        "w2_w4_pct": round(w2_w4 / total * 100, 2),
        "m3_pct": round(m3 / total * 100, 2),
    }


def _maturity_projection(lender_df: pd.DataFrame, cohort_dates: list) -> dict:
    """Project final RPU for recent cohorts based on historical completion rates.

    Uses mature cohorts (those with 8+ weeks of data) to compute the typical
    fraction of final payout realized at each week. Then applies those rates
    to recent (immature) cohorts to estimate their final RPU.
    """
    if lender_df.empty or len(cohort_dates) < 4:
        return {"current_rpu": 0.0, "projected_rpu": 0.0}

    # Weekly payout and enrolls per cohort
    weekly = lender_df.groupby("feo_cohort").agg(
        payout=("payout", "sum"),
        enrolls=("ENROLLS", "sum"),
    ).reindex(cohort_dates, fill_value=0)

    cumulative_payout = weekly["payout"].cumsum()
    total_payout = float(weekly["payout"].sum())
    total_enrolls = float(weekly["enrolls"].max()) if float(weekly["enrolls"].max()) > 0 else float(weekly["enrolls"].sum())

    if total_enrolls == 0 or total_payout == 0:
        return {"current_rpu": 0.0, "projected_rpu": 0.0}

    # Completion rates: fraction of final payout at each week index
    n_weeks = len(cohort_dates)
    completion_rates = []
    for i in range(n_weeks):
        cum_at_i = float(cumulative_payout.iloc[i])
        rate = cum_at_i / total_payout if total_payout > 0 else 0.0
        completion_rates.append(rate)

    # For the most recent cohort: estimate projected RPU
    recent_idx = min(3, n_weeks - 1)  # look at a recent (partially mature) cohort
    recent_payout = float(weekly["payout"].iloc[-1])
    recent_enrolls = float(weekly["enrolls"].iloc[-1])

    current_rpu = recent_payout / recent_enrolls if recent_enrolls > 0 else 0.0

    # Use completion rate at week 1 (the earliest) to project
    # How much of the total is typically earned by this point?
    # The latest cohort is at position 0 in its lifecycle
    early_rate = completion_rates[0] if completion_rates else 1.0
    projected_rpu = current_rpu / early_rate if early_rate > 0 else current_rpu

    return {
        "current_rpu": round(current_rpu, 4),
        "projected_rpu": round(projected_rpu, 4),
    }
