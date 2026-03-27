"""Module F: Reporting -- executive summaries and stakeholder views."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from .config import DEFAULT_CONFIG
from .utils import format_pct


class ReportingModule:
    """Generates executive summaries and role-specific stakeholder views."""

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        monitoring: dict,
        projections: dict,
        rca: dict,
        insights: dict,
        opportunity: dict,
    ) -> dict:
        """Compile all module outputs into a human-readable report."""
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "executive_summary": self._executive(monitoring, projections, rca),
            "stakeholder_views": self._stakeholder_views(
                monitoring, projections, rca, insights, opportunity
            ),
        }

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------

    def _executive(
        self, monitoring: dict, projections: dict, rca: dict
    ) -> List[str]:
        """Generate a 5-bullet executive summary.

        Extracts:
        1. Latest total RPU + WoW change
        2. Top driver from waterfall / Shapley decomposition
        3. Top risk from alerts
        4. Top opportunity from funnel leakage
        5. Enrollment trend
        """
        bullets: List[str] = []

        # 1. Latest RPU + WoW change
        bullets.append(self._bullet_rpu(monitoring))

        # 2. Top waterfall driver
        bullets.append(self._bullet_top_driver(rca))

        # 3. Top risk from alerts
        bullets.append(self._bullet_top_risk(monitoring))

        # 4. Top opportunity / funnel leakage
        bullets.append(self._bullet_leakage(rca))

        # 5. Enrollment trend
        bullets.append(self._bullet_enrollment_trend(monitoring, projections))

        return bullets

    # -- individual bullet helpers ------------------------------------

    @staticmethod
    def _bullet_rpu(monitoring: dict) -> str:
        total_rpu = monitoring.get("total_rpu")
        wow = monitoring.get("wow_change_pct")
        if total_rpu is not None and wow is not None:
            direction = "up" if wow > 0 else "down" if wow < 0 else "flat"
            return (
                f"Total RPU is ${total_rpu:.2f}, "
                f"{direction} {abs(wow):.1f}% week-over-week."
            )
        if total_rpu is not None:
            return f"Total RPU is ${total_rpu:.2f} (WoW change unavailable)."
        return "RPU data unavailable for the latest cohort."

    @staticmethod
    def _bullet_top_driver(rca: dict) -> str:
        waterfall = rca.get("waterfall", [])
        if waterfall:
            top = waterfall[0]
            name = top.get("factor", top.get("name", "unknown"))
            contrib = top.get("contribution", 0)
            return (
                f"Top RPU driver: {name} contributed "
                f"{'+'if contrib >= 0 else ''}{contrib:.4f} to the change."
            )
        return "No waterfall decomposition available."

    @staticmethod
    def _bullet_top_risk(monitoring: dict) -> str:
        alerts = monitoring.get("alerts", [])
        if alerts:
            a = alerts[0]
            return f"Top risk: {a.get('message', a.get('type', 'alert'))}."
        return "No active alerts."

    @staticmethod
    def _bullet_leakage(rca: dict) -> str:
        leakage = rca.get("funnel_leakage", rca.get("leakage", []))
        if leakage:
            top = leakage[0] if isinstance(leakage, list) else leakage
            if isinstance(top, dict):
                stage = top.get("stage", "unknown")
                drop = top.get("drop_pct", top.get("leakage_pct", 0))
                return (
                    f"Largest funnel leakage at {stage} stage "
                    f"({drop:.1f}% drop) -- opportunity for improvement."
                )
        return "No significant funnel leakage detected."

    @staticmethod
    def _bullet_enrollment_trend(monitoring: dict, projections: dict) -> str:
        trend = projections.get("enrollment_trend")
        forecast = projections.get("enrollment_forecast")
        if trend is not None:
            direction = "growing" if trend > 0 else "declining" if trend < 0 else "stable"
            msg = f"Enrollment trend is {direction}"
            if forecast is not None:
                msg += f"; next-week forecast: {forecast:,.0f} enrollments"
            return msg + "."
        return "Enrollment trend data unavailable."

    # ------------------------------------------------------------------
    # Stakeholder views
    # ------------------------------------------------------------------

    def _stakeholder_views(
        self,
        monitoring: dict,
        projections: dict,
        rca: dict,
        insights: dict,
        opportunity: dict,
    ) -> Dict[str, List[str]]:
        """Role-specific summaries for product, partnerships, finance, leadership."""
        return {
            "product": self._view_product(monitoring, rca, insights, opportunity),
            "partnerships": self._view_partnerships(monitoring, rca, insights),
            "finance": self._view_finance(monitoring, projections, opportunity),
            "leadership": self._view_leadership(
                monitoring, projections, rca, insights, opportunity
            ),
        }

    # -- Product -------------------------------------------------------

    @staticmethod
    def _view_product(
        monitoring: dict, rca: dict, insights: dict, opportunity: dict
    ) -> List[str]:
        bullets: List[str] = []

        # Touchpoint performance
        tp_perf = insights.get("touchpoint_performance", [])
        if tp_perf:
            top_tp = tp_perf[0] if isinstance(tp_perf, list) else {}
            if isinstance(top_tp, dict):
                bullets.append(
                    f"Top touchpoint: {top_tp.get('touchpoint', 'N/A')} "
                    f"(RPU contribution {top_tp.get('rpu_contribution', 'N/A')})."
                )

        # Funnel leakage
        leakage = rca.get("funnel_leakage", rca.get("leakage", []))
        if leakage and isinstance(leakage, list) and leakage:
            stages = [l.get("stage", "") for l in leakage[:2] if isinstance(l, dict)]
            if stages:
                bullets.append(
                    f"Focus funnel optimisation on: {', '.join(stages)}."
                )

        # Impression optimizer
        optimizer = opportunity.get("impression_optimizer", {})
        improvement = optimizer.get("projected_rpu_improvement", 0)
        if improvement > 0:
            bullets.append(
                f"Impression reallocation could lift RPU by ${improvement:.4f}."
            )

        # CTR alert
        alerts = monitoring.get("alerts", [])
        ctr_alerts = [a for a in alerts if "ctr" in str(a).lower()]
        if ctr_alerts:
            bullets.append(f"CTR alert: {ctr_alerts[0].get('message', '')}.")

        if not bullets:
            bullets.append("No actionable product insights this period.")
        return bullets[:5]

    # -- Partnerships --------------------------------------------------

    @staticmethod
    def _view_partnerships(
        monitoring: dict, rca: dict, insights: dict
    ) -> List[str]:
        bullets: List[str] = []

        # Lender concentration
        concentration = insights.get("lender_concentration", {})
        hhi_val = concentration.get("hhi")
        if hhi_val is not None:
            bullets.append(
                f"Lender HHI is {hhi_val:.3f} "
                f"({'concentrated' if hhi_val > 0.25 else 'diversified'})."
            )

        # Top lender
        lender_tiers = insights.get("lender_tiers", [])
        if lender_tiers and isinstance(lender_tiers, list):
            top = lender_tiers[0] if isinstance(lender_tiers[0], dict) else {}
            bullets.append(
                f"Top lender: {top.get('lender', 'N/A')} "
                f"(payout share {top.get('payout_share', 'N/A')})."
            )

        # Affiliate gone alert
        alerts = monitoring.get("alerts", [])
        gone = [a for a in alerts if "affiliate" in str(a).lower() or "gone" in str(a).lower()]
        if gone:
            bullets.append(f"Partner risk: {gone[0].get('message', '')}.")

        # RPC changes
        rpc_drivers = [
            w for w in rca.get("waterfall", [])
            if "rpc" in str(w.get("factor", "")).lower()
        ]
        if rpc_drivers:
            c = rpc_drivers[0].get("contribution", 0)
            bullets.append(
                f"RPC {'improved' if c > 0 else 'declined'} "
                f"(contribution: {'+' if c >= 0 else ''}{c:.4f})."
            )

        if not bullets:
            bullets.append("No partnership-specific insights this period.")
        return bullets[:5]

    # -- Finance -------------------------------------------------------

    @staticmethod
    def _view_finance(
        monitoring: dict, projections: dict, opportunity: dict
    ) -> List[str]:
        bullets: List[str] = []

        # Revenue projection
        rpu_forecast = projections.get("rpu_forecast")
        enroll_forecast = projections.get("enrollment_forecast")
        if rpu_forecast is not None and enroll_forecast is not None:
            projected_rev = rpu_forecast * enroll_forecast
            bullets.append(
                f"Projected next-week revenue: ${projected_rev:,.0f} "
                f"(RPU ${rpu_forecast:.2f} x {enroll_forecast:,.0f} enrollments)."
            )
        elif rpu_forecast is not None:
            bullets.append(f"RPU forecast: ${rpu_forecast:.2f}.")

        # Confidence intervals
        ci = projections.get("confidence_interval", {})
        if ci:
            bullets.append(
                f"RPU 95% CI: ${ci.get('lower', 0):.2f} - ${ci.get('upper', 0):.2f}."
            )

        # Scenario impacts
        scenarios = opportunity.get("scenarios", [])
        for s in scenarios[:2]:
            impact = s.get("revenue_impact_per_1000_enrolls", 0)
            if impact != 0:
                bullets.append(
                    f"Scenario '{s.get('scenario', '')}': "
                    f"{'+'if impact > 0 else ''}${impact:,.2f} per 1,000 enrollments."
                )

        if not bullets:
            bullets.append("No finance-specific insights this period.")
        return bullets[:5]

    # -- Leadership ----------------------------------------------------

    @staticmethod
    def _view_leadership(
        monitoring: dict,
        projections: dict,
        rca: dict,
        insights: dict,
        opportunity: dict,
    ) -> List[str]:
        bullets: List[str] = []

        # Headline RPU
        total_rpu = monitoring.get("total_rpu")
        wow = monitoring.get("wow_change_pct")
        if total_rpu is not None:
            direction = ""
            if wow is not None:
                direction = f" ({'+'if wow > 0 else ''}{wow:.1f}% WoW)"
            bullets.append(f"RPU: ${total_rpu:.2f}{direction}.")

        # Alert count
        alerts = monitoring.get("alerts", [])
        if alerts:
            bullets.append(f"{len(alerts)} active alert(s) requiring attention.")
        else:
            bullets.append("No active alerts -- system healthy.")

        # Top opportunity
        scenarios = opportunity.get("scenarios", [])
        if scenarios:
            best = max(scenarios, key=lambda s: s.get("delta", 0))
            if best.get("delta", 0) > 0:
                bullets.append(
                    f"Top opportunity: '{best.get('scenario', '')}' "
                    f"could add ${best.get('delta', 0):.4f} to RPU."
                )

        # Enrollment outlook
        enroll_forecast = projections.get("enrollment_forecast")
        if enroll_forecast is not None:
            bullets.append(f"Next-week enrollment forecast: {enroll_forecast:,.0f}.")

        if not bullets:
            bullets.append("No leadership-specific insights this period.")
        return bullets[:5]
