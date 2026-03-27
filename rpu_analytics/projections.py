"""Module B: Projections Engine.

Produces backward-compatible RPU projections (matching rpu_output.json schema)
plus enhanced per-combo projections, gap diagnosis, and backtesting metrics.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG
from .utils import ewma, trend_extrapolate
from .data_layer import EnrichedData


# Column names that the existing Vercel dashboard expects
_ROW_COLS = {
    "feo_cohort": "feo_cohort",
    "w1_predicted": "W1 RPU Predicted",
    "w1_actual": "W1 RPU Actual",
    "w2w4_predicted": "W2-4 RPU Predicted",
    "w2w4_actual": "W2-4 RPU Actual",
    "m3_predicted": "M3 RPU Predicted",
    "m3_actual": "M3 RPU Actual",
    "nonclick_predicted": "Non-click Predicted",
    "nonclick_actual": "Non-click Actual",
    "nonenroll_predicted": "Non-Enroll Predicted",
    "nonenroll_actual": "Non-Enroll Actual",
    "total_predicted": "Total Predicted",
    "total_actual": "Total Actual",
    "delta": "Delta (Actual - Predicted)",
    "delta_pct": "Delta %",
}


class ProjectionsModule:
    """Projections engine for RPU analytics."""

    def __init__(self, config: Optional[dict] = None):
        full_config = config if config is not None else DEFAULT_CONFIG
        self.cfg = full_config.get("projections", DEFAULT_CONFIG["projections"])
        self._decay = self.cfg.get("ewma_decay", 0.7)
        self._rolling_window = self.cfg.get("rolling_window", 4)
        self._trend_min_r2 = self.cfg.get("trend_min_r2", 0.5)
        self._backtest_weeks = self.cfg.get("backtest_weeks", 8)
        self._confidence_level = self.cfg.get("confidence_level", 0.95)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, data: EnrichedData) -> dict:
        """Run all projection sub-modules and return a JSON-serializable dict."""
        df = data.enriched.copy()
        enroll = data.enroll.copy()

        # Ensure feo_cohort is datetime
        df["feo_cohort"] = pd.to_datetime(df["feo_cohort"], errors="coerce")
        if "feo_cohort" in enroll.columns:
            enroll["feo_cohort"] = pd.to_datetime(enroll["feo_cohort"], errors="coerce")

        # B1: W1 projection
        w1_result = self._w1_projection(df, enroll)

        # B2: Ratio projections for W2-W4 and M3
        w2w4_result = self._ratio_projection(df, enroll, "w2w4", ["2. W2-W4"])
        m3_result = self._ratio_projection(df, enroll, "m3", ["3. M3"])

        # Non-click and non-enroll actuals (predicted = 0)
        nc_result = self._nonclick_rpu(df, enroll)
        ne_result = self._nonenroll_rpu(df, enroll)

        # B3: Composite backward-compatible rows
        rows = self._composite(enroll, w1_result, w2w4_result, m3_result, nc_result, ne_result)

        # B4: Gap diagnosis
        gap_diagnosis = self._gap_diagnosis(df, enroll, w1_result, w2w4_result, m3_result)

        # B5: Backtest
        backtest = self._backtest(df, enroll)

        # Timestamps
        ist = timezone(timedelta(hours=5, minutes=30))
        generated_utc = datetime.now(timezone.utc)
        generated_ist = generated_utc.astimezone(ist)

        return {
            "generated_at": generated_utc.isoformat(),
            "generated_at_ist": generated_ist.strftime("%d %b %Y, %I:%M %p IST"),
            "rows": rows,
            "enhanced_projections": {
                "w1": w1_result,
                "w2w4": w2w4_result,
                "m3": m3_result,
                "nonclick": nc_result,
                "nonenroll": ne_result,
            },
            "gap_diagnosis": gap_diagnosis,
            "backtest": backtest,
        }

    # ------------------------------------------------------------------
    # B1: Enhanced W1 RPU Projection
    # ------------------------------------------------------------------

    def _w1_projection(self, df: pd.DataFrame, enroll: pd.DataFrame) -> dict:
        """Enhanced W1 RPU projection per (touchpoint, lender, segment) combo."""
        enrolled_w1 = df[
            (df["click_cohort"] == "1. W1") & (df["enrol_status"] == "Enrolled")
        ]
        if enrolled_w1.empty:
            return {"cohort_projections": [], "combo_details": []}

        # Aggregate to combo level
        combo = (
            enrolled_w1.groupby(["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"])
            .agg(
                imp_pct=("Imp_pct", "sum"),
                ctr=("CTR", "mean"),
                total_payout=("payout", "sum"),
                total_clicks=("click_count", "sum"),
            )
            .reset_index()
        )
        combo["rpc"] = np.where(
            combo["total_clicks"] > 0,
            combo["total_payout"] / combo["total_clicks"],
            0.0,
        )
        combo = combo.sort_values(
            ["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"]
        )

        combo_details = []
        projected_rows = []

        for (tp, lender, seg), grp in combo.groupby(
            ["TOUCHPOINT", "LENDER", "SEGMENT"]
        ):
            grp = grp.sort_values("feo_cohort").reset_index(drop=True)
            cohorts = grp["feo_cohort"].values

            # Project each metric: RPC, Imp%, CTR
            proj_rpc = self._project_metric(grp["rpc"])
            proj_imp = self._project_metric(grp["imp_pct"])
            proj_ctr = self._project_metric(grp["ctr"])

            for i, cohort in enumerate(cohorts):
                p_rpc = proj_rpc["values"][i] if i < len(proj_rpc["values"]) else np.nan
                p_imp = proj_imp["values"][i] if i < len(proj_imp["values"]) else np.nan
                p_ctr = proj_ctr["values"][i] if i < len(proj_ctr["values"]) else np.nan

                projected_w1_rpu = (
                    p_imp * p_ctr * p_rpc
                    if not (np.isnan(p_imp) or np.isnan(p_ctr) or np.isnan(p_rpc))
                    else np.nan
                )
                actual_w1_rpu = (
                    grp["imp_pct"].iloc[i] * grp["ctr"].iloc[i] * grp["rpc"].iloc[i]
                )

                projected_rows.append(
                    {
                        "feo_cohort": cohort,
                        "TOUCHPOINT": tp,
                        "LENDER": lender,
                        "SEGMENT": seg,
                        "projected_rpc": _safe_float(p_rpc),
                        "projected_imp_pct": _safe_float(p_imp),
                        "projected_ctr": _safe_float(p_ctr),
                        "projected_w1_rpu": _safe_float(projected_w1_rpu),
                        "actual_w1_rpu": _safe_float(actual_w1_rpu),
                    }
                )

                combo_details.append(
                    {
                        "touchpoint": tp,
                        "lender": lender,
                        "segment": seg,
                        "feo_cohort": _ts_to_str(cohort),
                        "projected_rpc": _safe_float(p_rpc),
                        "projected_imp_pct": _safe_float(p_imp),
                        "projected_ctr": _safe_float(p_ctr),
                        "projected_w1_rpu": _safe_float(projected_w1_rpu),
                        "actual_w1_rpu": _safe_float(actual_w1_rpu),
                        "rpc_r2": proj_rpc.get("r_squared", 0.0),
                        "imp_r2": proj_imp.get("r_squared", 0.0),
                        "ctr_r2": proj_ctr.get("r_squared", 0.0),
                    }
                )

        # Aggregate to cohort level using enrollment-weighted average
        proj_df = pd.DataFrame(projected_rows)
        if proj_df.empty:
            return {"cohort_projections": [], "combo_details": combo_details}

        # Merge enrollments
        enroll_agg = self._get_segment_enrollments(enroll)
        proj_df = proj_df.merge(
            enroll_agg, on=["feo_cohort", "SEGMENT"], how="left"
        )
        proj_df["enrols"] = proj_df["enrols"].fillna(0)

        cohort_projections = []
        for cohort, cgrp in proj_df.groupby("feo_cohort"):
            total_enrols = cgrp["enrols"].sum()
            if total_enrols > 0:
                w1_pred = np.average(cgrp["projected_w1_rpu"], weights=cgrp["enrols"])
                w1_act = np.average(cgrp["actual_w1_rpu"], weights=cgrp["enrols"])
            else:
                w1_pred = cgrp["projected_w1_rpu"].mean()
                w1_act = cgrp["actual_w1_rpu"].mean()

            # Confidence interval from trailing prediction errors
            ci = self._prediction_error_ci(cgrp, cohort, proj_df)

            cohort_projections.append(
                {
                    "feo_cohort": _ts_to_str(cohort),
                    "w1_predicted": _safe_float(w1_pred),
                    "w1_actual": _safe_float(w1_act),
                    "ci_lower": _safe_float(w1_pred - ci),
                    "ci_upper": _safe_float(w1_pred + ci),
                }
            )

        return {
            "cohort_projections": cohort_projections,
            "combo_details": combo_details,
        }

    def _project_metric(self, series: pd.Series) -> dict:
        """Project a single metric using EWMA + trend blending.

        Excludes current week (shift=1 on EWMA), blends with trend
        if r2 > threshold.
        """
        if series is None or len(series) == 0:
            return {"values": [], "r_squared": 0.0}

        # EWMA with shift=1 (exclude current week)
        shifted = series.shift(1)
        ewma_vals = ewma(shifted.dropna(), decay=self._decay)

        # Pad ewma_vals back to original length: first value is NaN (no history)
        ewma_full = pd.Series(np.nan, index=series.index)
        if len(ewma_vals) > 0:
            ewma_full.iloc[1: 1 + len(ewma_vals)] = ewma_vals.values

        # Trend extrapolation on the shifted series
        shifted_clean = shifted.dropna()
        _, r_squared = trend_extrapolate(shifted_clean, periods_ahead=1)

        # Build per-period projected values
        projected_values = []
        for i in range(len(series)):
            e_val = ewma_full.iloc[i]
            if np.isnan(e_val):
                projected_values.append(np.nan)
                continue

            if r_squared > self._trend_min_r2:
                # Use the history up to this point for trend
                history_up_to = series.iloc[:i]
                t_val, _ = trend_extrapolate(history_up_to, periods_ahead=1)
                if np.isnan(t_val):
                    projected_values.append(_safe_float(e_val))
                else:
                    blended = 0.5 * e_val + 0.5 * t_val
                    projected_values.append(_safe_float(blended))
            else:
                projected_values.append(_safe_float(e_val))

        return {"values": projected_values, "r_squared": _safe_float(r_squared)}

    def _prediction_error_ci(
        self, combo_grp: pd.DataFrame, current_cohort, full_proj_df: pd.DataFrame
    ) -> float:
        """Compute confidence interval half-width from trailing 4-week prediction errors."""
        # Get all cohorts before current
        all_cohorts = sorted(full_proj_df["feo_cohort"].unique())
        current_idx = list(all_cohorts).index(current_cohort) if current_cohort in all_cohorts else -1
        if current_idx < 1:
            return 0.0

        lookback = min(self._rolling_window, current_idx)
        trailing_cohorts = all_cohorts[current_idx - lookback: current_idx]

        errors = []
        for tc in trailing_cohorts:
            tc_data = full_proj_df[full_proj_df["feo_cohort"] == tc]
            if tc_data.empty:
                continue
            total_enrols = tc_data["enrols"].sum()
            if total_enrols > 0:
                pred = np.average(tc_data["projected_w1_rpu"], weights=tc_data["enrols"])
                actual = np.average(tc_data["actual_w1_rpu"], weights=tc_data["enrols"])
            else:
                pred = tc_data["projected_w1_rpu"].mean()
                actual = tc_data["actual_w1_rpu"].mean()
            errors.append(actual - pred)

        if len(errors) < 2:
            return 0.0

        std_err = float(np.std(errors, ddof=1))
        return 1.96 * std_err

    # ------------------------------------------------------------------
    # B2: Ratio Projection (W2-W4, M3)
    # ------------------------------------------------------------------

    def _ratio_projection(
        self,
        df: pd.DataFrame,
        enroll: pd.DataFrame,
        label: str,
        target_cohorts: List[str],
    ) -> dict:
        """Ratio-based projection for W2-W4 or M3 components."""
        enrolled = df[df["enrol_status"] == "Enrolled"]
        w1 = enrolled[enrolled["click_cohort"] == "1. W1"]
        target = enrolled[enrolled["click_cohort"].isin(target_cohorts)]

        if w1.empty:
            return {"cohort_projections": [], "combo_details": []}

        # W1 base metrics per combo
        w1_agg = (
            w1.groupby(["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"])
            .agg(
                w1_imp=("Imp_pct", "sum"),
                w1_ctr=("CTR", "mean"),
                w1_payout=("payout", "sum"),
                w1_clicks=("click_count", "sum"),
            )
            .reset_index()
        )
        w1_agg["w1_rpc"] = np.where(
            w1_agg["w1_clicks"] > 0,
            w1_agg["w1_payout"] / w1_agg["w1_clicks"],
            0.0,
        )

        # Target metrics per combo
        if target.empty:
            t_agg = pd.DataFrame(
                columns=[
                    "TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort",
                    "t_imp", "t_ctr", "t_payout", "t_clicks",
                ]
            )
        else:
            t_agg = (
                target.groupby(["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"])
                .agg(
                    t_imp=("Imp_pct", "sum"),
                    t_ctr=("CTR", "mean"),
                    t_payout=("payout", "sum"),
                    t_clicks=("click_count", "sum"),
                )
                .reset_index()
            )
        t_agg["t_rpc"] = np.where(
            t_agg["t_clicks"] > 0,
            t_agg["t_payout"] / t_agg["t_clicks"],
            0.0,
        )

        merged = w1_agg.merge(
            t_agg,
            on=["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"],
            how="left",
        )
        for c in ["t_imp", "t_ctr", "t_rpc", "t_payout"]:
            if c in merged.columns:
                merged[c] = merged[c].fillna(0)
            else:
                merged[c] = 0.0

        # Compute ratios
        merged["imp_ratio"] = np.where(
            merged["w1_imp"] > 0, merged["t_imp"] / merged["w1_imp"], 0.0
        )
        merged["ctr_ratio"] = np.where(
            merged["w1_ctr"] > 0, merged["t_ctr"] / merged["w1_ctr"], 1.0
        )
        merged["rpc_ratio"] = np.where(
            merged["w1_rpc"] > 0, merged["t_rpc"] / merged["w1_rpc"], 1.0
        )

        merged = merged.sort_values(
            ["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"]
        )

        # EWMA of ratios (shifted, decay)
        for rc in ["imp_ratio", "ctr_ratio", "rpc_ratio"]:
            merged[f"{rc}_ewma"] = merged.groupby(
                ["TOUCHPOINT", "LENDER", "SEGMENT"]
            )[rc].transform(
                lambda x: ewma(x.shift(1).dropna(), decay=self._decay)
                .reindex(x.index[1:])
                .reindex(x.index)
                .values
                if len(x) > 1
                else pd.Series(np.nan, index=x.index)
            )
            merged[f"{rc}_ewma"] = merged[f"{rc}_ewma"].fillna(0)

        # Projected target RPU = W1 metrics * EWMA ratios
        merged["projected_rpu"] = (
            merged["w1_imp"]
            * merged["w1_ctr"]
            * merged["w1_rpc"]
            * merged["imp_ratio_ewma"]
            * merged["ctr_ratio_ewma"]
            * merged["rpc_ratio_ewma"]
        )

        # Actual target RPU per combo
        enroll_seg = self._get_segment_enrollments(enroll)
        merged = merged.merge(
            enroll_seg, on=["feo_cohort", "SEGMENT"], how="left"
        )
        merged["enrols"] = merged["enrols"].fillna(0)
        merged["actual_rpu"] = np.where(
            merged["enrols"] > 0,
            merged["t_payout"] / merged["enrols"],
            0.0,
        )

        # Aggregate to cohort level
        combo_details = []
        cohort_projections = []

        for cohort, cgrp in merged.groupby("feo_cohort"):
            total_enrols = cgrp["enrols"].sum()
            if total_enrols > 0:
                pred = float(np.average(cgrp["projected_rpu"], weights=cgrp["enrols"]))
                actual = float(np.average(cgrp["actual_rpu"], weights=cgrp["enrols"]))
            else:
                pred = float(cgrp["projected_rpu"].mean())
                actual = float(cgrp["actual_rpu"].mean())

            cohort_projections.append(
                {
                    "feo_cohort": _ts_to_str(cohort),
                    f"{label}_predicted": _safe_float(pred),
                    f"{label}_actual": _safe_float(actual),
                }
            )

        for _, row in merged.iterrows():
            combo_details.append(
                {
                    "touchpoint": row["TOUCHPOINT"],
                    "lender": row["LENDER"],
                    "segment": row["SEGMENT"],
                    "feo_cohort": _ts_to_str(row["feo_cohort"]),
                    "imp_ratio_ewma": _safe_float(row["imp_ratio_ewma"]),
                    "ctr_ratio_ewma": _safe_float(row["ctr_ratio_ewma"]),
                    "rpc_ratio_ewma": _safe_float(row["rpc_ratio_ewma"]),
                    "projected_rpu": _safe_float(row["projected_rpu"]),
                    "actual_rpu": _safe_float(row["actual_rpu"]),
                }
            )

        return {
            "cohort_projections": cohort_projections,
            "combo_details": combo_details,
        }

    # ------------------------------------------------------------------
    # Non-click and Non-enroll helpers
    # ------------------------------------------------------------------

    def _nonclick_rpu(self, df: pd.DataFrame, enroll: pd.DataFrame) -> dict:
        """Non-click RPU (predicted = 0, actual from data)."""
        nc = (
            df[df["TOUCHPOINT"] == "Non-click"]
            .groupby("feo_cohort")
            .agg(total_payout=("payout", "sum"))
            .reset_index()
        )
        total_enroll = (
            enroll.groupby("feo_cohort")
            .agg(enrols=("enrols", "sum"))
            .reset_index()
        )
        nc = nc.merge(total_enroll, on="feo_cohort", how="left")
        nc["enrols"] = nc["enrols"].fillna(0)
        nc["nonclick_actual"] = np.where(
            nc["enrols"] > 0, nc["total_payout"] / nc["enrols"], 0.0
        )
        nc["nonclick_predicted"] = 0.0

        return {
            "cohort_projections": [
                {
                    "feo_cohort": _ts_to_str(row["feo_cohort"]),
                    "nonclick_predicted": 0.0,
                    "nonclick_actual": _safe_float(row["nonclick_actual"]),
                }
                for _, row in nc.iterrows()
            ]
        }

    def _nonenroll_rpu(self, df: pd.DataFrame, enroll: pd.DataFrame) -> dict:
        """Non-enrolled RPU (predicted = 0, actual from data)."""
        ne = (
            df[df["enrol_status"] == "Non-Enrolled"]
            .groupby("feo_cohort")
            .agg(total_payout=("payout", "sum"))
            .reset_index()
        )
        total_enroll = (
            enroll.groupby("feo_cohort")
            .agg(enrols=("enrols", "sum"))
            .reset_index()
        )
        ne = ne.merge(total_enroll, on="feo_cohort", how="left")
        ne["enrols"] = ne["enrols"].fillna(0)
        ne["nonenroll_actual"] = np.where(
            ne["enrols"] > 0, ne["total_payout"] / ne["enrols"], 0.0
        )
        ne["nonenroll_predicted"] = 0.0

        return {
            "cohort_projections": [
                {
                    "feo_cohort": _ts_to_str(row["feo_cohort"]),
                    "nonenroll_predicted": 0.0,
                    "nonenroll_actual": _safe_float(row["nonenroll_actual"]),
                }
                for _, row in ne.iterrows()
            ]
        }

    # ------------------------------------------------------------------
    # B3: Composite (backward-compatible rows)
    # ------------------------------------------------------------------

    def _composite(
        self,
        enroll: pd.DataFrame,
        w1_result: dict,
        w2w4_result: dict,
        m3_result: dict,
        nc_result: dict,
        ne_result: dict,
    ) -> list:
        """Combine all components into backward-compatible rows format."""
        # Build a base dataframe from unique cohort dates in enroll
        if "feo_cohort" in enroll.columns:
            base = (
                enroll[["feo_cohort"]]
                .drop_duplicates()
                .sort_values("feo_cohort", ascending=True)
            )
            base["feo_cohort"] = pd.to_datetime(base["feo_cohort"], errors="coerce")
        else:
            base = pd.DataFrame(columns=["feo_cohort"])

        # Convert each result's cohort_projections into a DataFrame and merge
        for result, prefix in [
            (w1_result, "w1"),
            (w2w4_result, "w2w4"),
            (m3_result, "m3"),
            (nc_result, "nonclick"),
            (ne_result, "nonenroll"),
        ]:
            cp = result.get("cohort_projections", [])
            if cp:
                rdf = pd.DataFrame(cp)
                rdf["feo_cohort"] = pd.to_datetime(rdf["feo_cohort"], errors="coerce")
                # Keep only the columns we need
                keep_cols = ["feo_cohort"] + [
                    c for c in rdf.columns if c.startswith(prefix)
                ]
                rdf = rdf[[c for c in keep_cols if c in rdf.columns]]
                base = base.merge(rdf, on="feo_cohort", how="left")

        base = base.fillna(0)

        # Ensure all expected columns exist
        for col in [
            "w1_predicted", "w1_actual",
            "w2w4_predicted", "w2w4_actual",
            "m3_predicted", "m3_actual",
            "nonclick_predicted", "nonclick_actual",
            "nonenroll_predicted", "nonenroll_actual",
        ]:
            if col not in base.columns:
                base[col] = 0.0

        # Totals
        base["total_predicted"] = (
            base["w1_predicted"]
            + base["w2w4_predicted"]
            + base["m3_predicted"]
            + base["nonclick_predicted"]
            + base["nonenroll_predicted"]
        )
        base["total_actual"] = (
            base["w1_actual"]
            + base["w2w4_actual"]
            + base["m3_actual"]
            + base["nonclick_actual"]
            + base["nonenroll_actual"]
        )
        base["delta"] = base["total_actual"] - base["total_predicted"]
        base["delta_pct"] = np.where(
            base["total_actual"] != 0,
            base["delta"] / base["total_actual"],
            0.0,
        )

        # Format cohort as string
        base["feo_cohort"] = base["feo_cohort"].dt.strftime("%Y-%m-%d")

        # Rename to dashboard column names
        base = base.rename(columns=_ROW_COLS)

        # Return only the dashboard columns, in order
        output_cols = list(_ROW_COLS.values())
        for c in output_cols:
            if c not in base.columns:
                base[c] = 0.0

        rows = base[output_cols].to_dict(orient="records")

        # Ensure all values are JSON-serializable floats
        for row in rows:
            for k, v in row.items():
                if k == "feo_cohort":
                    continue
                row[k] = _safe_float(v)

        return rows

    # ------------------------------------------------------------------
    # B4: Gap Diagnosis
    # ------------------------------------------------------------------

    def _gap_diagnosis(
        self,
        df: pd.DataFrame,
        enroll: pd.DataFrame,
        w1_result: dict,
        w2w4_result: dict,
        m3_result: dict,
    ) -> list:
        """For each cohort, decompose gap into Imp%, CTR, RPC contributions per combo.

        Returns top-5 gap drivers per cohort.
        """
        diagnosis = []

        for component_result, component_name in [
            (w1_result, "w1"),
            (w2w4_result, "w2w4"),
            (m3_result, "m3"),
        ]:
            combo_details = component_result.get("combo_details", [])
            if not combo_details:
                continue

            cdf = pd.DataFrame(combo_details)
            if cdf.empty:
                continue

            # Group by cohort and find gaps
            for cohort, cgrp in cdf.groupby("feo_cohort"):
                drivers = []
                for _, row in cgrp.iterrows():
                    if component_name == "w1":
                        actual = row.get("actual_w1_rpu", 0.0)
                        projected = row.get("projected_w1_rpu", 0.0)
                    else:
                        actual = row.get("actual_rpu", 0.0)
                        projected = row.get("projected_rpu", 0.0)

                    gap = actual - projected
                    if abs(gap) < 1e-10:
                        continue

                    drivers.append(
                        {
                            "touchpoint": row.get("touchpoint", ""),
                            "lender": row.get("lender", ""),
                            "segment": row.get("segment", ""),
                            "gap": _safe_float(gap),
                            "component": component_name,
                        }
                    )

                # Sort by absolute gap descending, take top 5
                drivers.sort(key=lambda x: abs(x["gap"]), reverse=True)
                if drivers:
                    diagnosis.append(
                        {
                            "feo_cohort": cohort,
                            "component": component_name,
                            "top_drivers": drivers[:5],
                        }
                    )

        return diagnosis

    # ------------------------------------------------------------------
    # B5: Backtest
    # ------------------------------------------------------------------

    def _backtest(self, df: pd.DataFrame, enroll: pd.DataFrame) -> dict:
        """Compute MAPE and bias over last N weeks, per component and per segment."""
        enrolled = df[df["enrol_status"] == "Enrolled"]
        cohorts = sorted(df["feo_cohort"].dropna().unique())

        if len(cohorts) < 3:
            return {"mape": {}, "bias": {}, "by_segment": []}

        # Determine backtest cohorts (last N, excluding the very latest)
        n_bt = min(self._backtest_weeks, len(cohorts) - 1)
        backtest_cohorts = cohorts[-(n_bt + 1): -1]

        results_by_component = {}
        segment_results = []

        for component, target_click_cohorts in [
            ("w1", ["1. W1"]),
            ("w2w4", ["2. W2-W4"]),
            ("m3", ["3. M3"]),
        ]:
            errors = []
            pct_errors = []

            for bt_cohort in backtest_cohorts:
                # History = everything before bt_cohort
                history_df = enrolled[enrolled["feo_cohort"] < bt_cohort]
                target_df = enrolled[
                    (enrolled["feo_cohort"] == bt_cohort)
                    & (enrolled["click_cohort"].isin(target_click_cohorts))
                ]

                if history_df.empty or target_df.empty:
                    continue

                # Actual RPU for this cohort
                total_payout = target_df["payout"].sum()
                cohort_enroll = enroll[enroll["feo_cohort"] == bt_cohort]
                total_enrols = (
                    cohort_enroll["enrols"].sum()
                    if not cohort_enroll.empty
                    else 0
                )
                actual_rpu = total_payout / total_enrols if total_enrols > 0 else 0.0

                # Simple projection: EWMA of trailing RPU
                trailing = []
                for tc in sorted(enrolled["feo_cohort"].unique()):
                    if tc >= bt_cohort:
                        break
                    t_data = enrolled[
                        (enrolled["feo_cohort"] == tc)
                        & (enrolled["click_cohort"].isin(target_click_cohorts))
                    ]
                    t_enroll = enroll[enroll["feo_cohort"] == tc]
                    t_enrols = t_enroll["enrols"].sum() if not t_enroll.empty else 0
                    t_rpu = (
                        t_data["payout"].sum() / t_enrols if t_enrols > 0 else 0.0
                    )
                    trailing.append(t_rpu)

                if not trailing:
                    continue

                trailing_series = pd.Series(trailing)
                ewma_pred = ewma(trailing_series, decay=self._decay).iloc[-1]
                predicted_rpu = float(ewma_pred)

                error = actual_rpu - predicted_rpu
                errors.append(error)
                if actual_rpu != 0:
                    pct_errors.append(abs(error / actual_rpu))

            mape = float(np.mean(pct_errors)) if pct_errors else 0.0
            bias = float(np.mean(errors)) if errors else 0.0
            results_by_component[component] = {
                "mape": _safe_float(mape),
                "bias": _safe_float(bias),
                "n_weeks": len(errors),
            }

        # By-segment backtest for W1
        segments = sorted(df["SEGMENT"].dropna().unique())
        for seg in segments:
            seg_enrolled = enrolled[enrolled["SEGMENT"] == seg]
            seg_enroll = enroll[enroll["segment"] == seg] if "segment" in enroll.columns else pd.DataFrame()
            errors = []
            pct_errors = []

            for bt_cohort in backtest_cohorts:
                target_data = seg_enrolled[
                    (seg_enrolled["feo_cohort"] == bt_cohort)
                    & (seg_enrolled["click_cohort"] == "1. W1")
                ]
                if target_data.empty:
                    continue

                total_payout = target_data["payout"].sum()
                ce = seg_enroll[seg_enroll["feo_cohort"] == bt_cohort] if not seg_enroll.empty else pd.DataFrame()
                total_enrols = ce["enrols"].sum() if not ce.empty else 0
                actual_rpu = total_payout / total_enrols if total_enrols > 0 else 0.0

                trailing = []
                for tc in sorted(seg_enrolled["feo_cohort"].unique()):
                    if tc >= bt_cohort:
                        break
                    t_data = seg_enrolled[
                        (seg_enrolled["feo_cohort"] == tc)
                        & (seg_enrolled["click_cohort"] == "1. W1")
                    ]
                    te = seg_enroll[seg_enroll["feo_cohort"] == tc] if not seg_enroll.empty else pd.DataFrame()
                    t_enrols = te["enrols"].sum() if not te.empty else 0
                    t_rpu = t_data["payout"].sum() / t_enrols if t_enrols > 0 else 0.0
                    trailing.append(t_rpu)

                if not trailing:
                    continue

                trailing_series = pd.Series(trailing)
                ewma_pred = ewma(trailing_series, decay=self._decay).iloc[-1]
                error = actual_rpu - float(ewma_pred)
                errors.append(error)
                if actual_rpu != 0:
                    pct_errors.append(abs(error / actual_rpu))

            segment_results.append(
                {
                    "segment": seg,
                    "w1_mape": _safe_float(np.mean(pct_errors)) if pct_errors else 0.0,
                    "w1_bias": _safe_float(np.mean(errors)) if errors else 0.0,
                    "n_weeks": len(errors),
                }
            )

        return {
            "mape": {k: v["mape"] for k, v in results_by_component.items()},
            "bias": {k: v["bias"] for k, v in results_by_component.items()},
            "by_component": results_by_component,
            "by_segment": segment_results,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_segment_enrollments(self, enroll: pd.DataFrame) -> pd.DataFrame:
        """Return a (feo_cohort, SEGMENT, enrols) DataFrame from enrollment data."""
        if "segment" not in enroll.columns or "feo_cohort" not in enroll.columns:
            return pd.DataFrame(columns=["feo_cohort", "SEGMENT", "enrols"])

        result = (
            enroll.groupby(["feo_cohort", "segment"])
            .agg(enrols=("enrols", "sum"))
            .reset_index()
            .rename(columns={"segment": "SEGMENT"})
        )
        return result


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _safe_float(val) -> float:
    """Convert to a JSON-safe float (NaN -> 0.0)."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return 0.0
        return round(f, 6)
    except (TypeError, ValueError):
        return 0.0


def _ts_to_str(ts) -> str:
    """Convert a pandas Timestamp or datetime-like to YYYY-MM-DD string."""
    if isinstance(ts, str):
        return ts
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)
