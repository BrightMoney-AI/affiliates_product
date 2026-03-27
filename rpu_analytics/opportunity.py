"""Module E: Opportunity Sizing -- what-if scenarios and impression optimisation."""

import copy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG
from .data_layer import EnrichedData


class OpportunityModule:
    """Sizes revenue opportunities via what-if scenarios and impression allocation."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config if config is not None else DEFAULT_CONFIG
        self._cfg = cfg.get("opportunity", {})

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, data: EnrichedData) -> dict:
        """Run all opportunity sub-modules and return a JSON-serialisable dict."""
        scenarios = self._cfg.get("default_scenarios", [])
        scenario_results = [self._whatif(data, s) for s in scenarios]
        optimizer_result = self._optimizer(data)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scenarios": scenario_results,
            "impression_optimizer": optimizer_result,
        }

    # ------------------------------------------------------------------
    # What-if scenario engine
    # ------------------------------------------------------------------

    def _whatif(self, data: EnrichedData, scenario: dict) -> dict:
        """Evaluate a single what-if scenario.

        Parameters
        ----------
        scenario : dict
            ``{"name": "...", "changes": [{"dimension": "touchpoint"|"lender",
            "value": "Discover Tab"|"__TOP_1__", "metric": "imp_pct"|"ctr"|"rpc",
            "change_pct": 10}  # or "change_abs": 5 }]}``

        Returns a dict with current/projected RPU, delta, and revenue impact.
        """
        df = data.enriched.copy()
        latest = data.latest_cohort
        if latest is None:
            return {"scenario": scenario.get("name", ""), "error": "no cohort data"}

        base = df[df["feo_cohort"] == latest].copy()
        modified = base.copy()

        for change in scenario.get("changes", []):
            modified = self._apply_change(modified, change, data)

        current_total_rpu = self._compute_total_rpu(base)
        projected_total_rpu = self._compute_total_rpu(modified)
        delta = projected_total_rpu - current_total_rpu
        delta_pct = float(np.where(current_total_rpu != 0,
                                   delta / current_total_rpu * 100, 0.0))
        revenue_impact_per_1000 = delta * 1000

        return {
            "scenario": scenario.get("name", ""),
            "current_total_rpu": round(float(current_total_rpu), 4),
            "projected_total_rpu": round(float(projected_total_rpu), 4),
            "delta": round(float(delta), 4),
            "delta_pct": round(delta_pct, 2),
            "revenue_impact_per_1000_enrolls": round(float(revenue_impact_per_1000), 2),
        }

    def _apply_change(self, df: pd.DataFrame, change: dict,
                      data: EnrichedData) -> pd.DataFrame:
        """Apply a single change entry to *df* (in-place safe copy)."""
        dimension = change.get("dimension", "")
        value = change.get("value", "")
        metric = change.get("metric", "")

        # Resolve __TOP_1__ to the top lender by payout
        if value == "__TOP_1__":
            if dimension == "lender":
                top = df.groupby("LENDER")["payout"].sum().idxmax()
                value = top
            elif dimension == "touchpoint":
                top = df.groupby("TOUCHPOINT")["payout"].sum().idxmax()
                value = top

        # Map dimension to column
        col_map = {"touchpoint": "TOUCHPOINT", "lender": "LENDER", "segment": "SEGMENT"}
        col = col_map.get(dimension)
        if col is None:
            return df

        # Map metric to column
        metric_col_map = {
            "imp_pct": "Imp_pct",
            "ctr": "CTR",
            "rpc": "RPC",
            "rpu": "RPU",
            "conversion_rate": "Conversion_rate",
        }
        target_col = metric_col_map.get(metric)
        if target_col is None or target_col not in df.columns:
            return df

        mask = df[col] == value

        if "change_pct" in change:
            factor = 1 + change["change_pct"] / 100.0
            df.loc[mask, target_col] = df.loc[mask, target_col] * factor
        elif "change_abs" in change:
            df.loc[mask, target_col] = df.loc[mask, target_col] + change["change_abs"]

        # Recompute payout bottom-up: payout = ENROLLS * Imp_pct * CTR * RPC
        df["payout"] = df["ENROLLS"] * df["Imp_pct"] * df["CTR"] * df["RPC"]
        # Recompute RPU
        df["RPU"] = np.where(df["ENROLLS"] != 0, df["payout"] / df["ENROLLS"], 0.0)

        return df

    @staticmethod
    def _compute_total_rpu(df: pd.DataFrame) -> float:
        """Total RPU = total payout / total unique enrollments for the cohort."""
        total_payout = df["payout"].sum()
        # ENROLLS is per-row (repeated per touchpoint/lender); use unique cohort enrolls
        enrolls_series = df.groupby("UID")["ENROLLS"].first()
        total_enrolls = enrolls_series.sum()
        if total_enrolls == 0:
            return 0.0
        return float(total_payout / total_enrolls)

    # ------------------------------------------------------------------
    # Impression allocation optimizer
    # ------------------------------------------------------------------

    def _optimizer(self, data: EnrichedData) -> dict:
        """Greedy marginal-RPU impression optimizer.

        For each touchpoint compute marginal_rpu = CTR * RPC.  Iteratively
        shift 1 pp of impression share from the lowest-marginal to the
        highest-marginal touchpoint, recomputing after each step, until no
        improvement or 20 steps.
        """
        df = data.enriched.copy()
        latest = data.latest_cohort
        if latest is None:
            return {"error": "no cohort data"}

        base = df[df["feo_cohort"] == latest].copy()

        # Aggregate per touchpoint
        tp_agg = (
            base.groupby("TOUCHPOINT")
            .agg(
                impressions=("impression_count", "sum"),
                clicks=("click_count", "sum"),
                payout=("payout", "sum"),
            )
            .reset_index()
        )
        total_imp = tp_agg["impressions"].sum()
        if total_imp == 0:
            return {"error": "no impressions"}

        tp_agg["imp_share"] = tp_agg["impressions"] / total_imp
        tp_agg["ctr"] = np.where(tp_agg["impressions"] != 0,
                                 tp_agg["clicks"] / tp_agg["impressions"], 0.0)
        tp_agg["rpc"] = np.where(tp_agg["clicks"] != 0,
                                 tp_agg["payout"] / tp_agg["clicks"], 0.0)
        tp_agg["marginal_rpu"] = tp_agg["ctr"] * tp_agg["rpc"]

        current_allocation = dict(zip(tp_agg["TOUCHPOINT"], tp_agg["imp_share"].round(4)))
        current_rpu = self._weighted_rpu(tp_agg)

        # Greedy reallocation loop
        alloc = tp_agg.copy()
        step_size = 0.01  # 1 pp
        max_steps = 20
        for _ in range(max_steps):
            alloc["marginal_rpu"] = alloc["ctr"] * alloc["rpc"]
            if len(alloc) < 2:
                break
            best_idx = alloc["marginal_rpu"].idxmax()
            worst_idx = alloc["marginal_rpu"].idxmin()
            if best_idx == worst_idx:
                break
            if alloc.loc[worst_idx, "imp_share"] < step_size:
                break

            alloc.loc[worst_idx, "imp_share"] -= step_size
            alloc.loc[best_idx, "imp_share"] += step_size

            new_rpu = self._weighted_rpu(alloc)
            if new_rpu <= current_rpu:
                # revert
                alloc.loc[worst_idx, "imp_share"] += step_size
                alloc.loc[best_idx, "imp_share"] -= step_size
                break
            current_rpu = new_rpu

        optimal_allocation = dict(
            zip(alloc["TOUCHPOINT"], alloc["imp_share"].round(4))
        )
        original_rpu = self._weighted_rpu(tp_agg)
        improvement = current_rpu - original_rpu

        return {
            "current_allocation": {k: float(v) for k, v in current_allocation.items()},
            "optimal_allocation": {k: float(v) for k, v in optimal_allocation.items()},
            "projected_rpu_improvement": round(float(improvement), 4),
        }

    @staticmethod
    def _weighted_rpu(tp_df: pd.DataFrame) -> float:
        """Weighted RPU across touchpoints: sum(share * CTR * RPC)."""
        return float((tp_df["imp_share"] * tp_df["ctr"] * tp_df["rpc"]).sum())
