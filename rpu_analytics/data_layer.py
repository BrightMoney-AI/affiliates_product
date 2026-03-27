"""Data foundation module: loading, enrichment, and structured output."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG


EXCEL_EPOCH = pd.Timestamp("1899-12-30")


@dataclass(frozen=True)
class EnrichedData:
    """Immutable container for all enriched analytics data."""

    enroll: pd.DataFrame
    non_api: pd.DataFrame
    enriched: pd.DataFrame
    cohort_dates: list
    latest_cohort: Any
    segments: list
    touchpoints: list
    lenders: list


class DataLayer:
    """Loads, validates, and enriches RPU analytics data."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config if config is not None else DEFAULT_CONFIG
        # Support both full config (with "data_layer" key) and sub-config
        if "data_layer" in cfg:
            self._dl_config = cfg["data_layer"]
        elif "imp_source_to_touchpoint" in cfg:
            self._dl_config = cfg
        else:
            self._dl_config = DEFAULT_CONFIG.get("data_layer", {})
        self.config = cfg
        self._tp_map = self._dl_config.get("imp_source_to_touchpoint", {})

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def from_csv(self, enroll_path: str, non_api_path: str) -> EnrichedData:
        """Load two CSV files and return enriched data."""
        enroll = pd.read_csv(enroll_path)
        non_api = pd.read_csv(non_api_path)
        return self.enrich(enroll, non_api)

    def from_dataframes(self, enroll: pd.DataFrame, non_api: pd.DataFrame) -> EnrichedData:
        """Accept DataFrames directly and return enriched data."""
        return self.enrich(enroll.copy(), non_api.copy())

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def enrich(self, enroll: pd.DataFrame, non_api: pd.DataFrame) -> EnrichedData:
        """Apply all computed columns and return an ``EnrichedData`` bundle."""
        df = non_api.copy()

        # -- Parse cohort dates ----------------------------------------
        df["feo_cohort"] = pd.to_datetime(df["feo_cohort"], errors="coerce")

        # -- Ensure numeric columns ------------------------------------
        numeric_cols = [
            "enrols",
            "impression_count",
            "click_count",
            "payout",
            "conversion_count",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # -- Touchpoint mapping ----------------------------------------
        df["TOUCHPOINT"] = df["imp_source"].map(self._tp_map).fillna("Non-click")

        # -- Lender & segment ------------------------------------------
        df["LENDER"] = df["product_name"]
        df["SEGMENT"] = df["segment"]

        # -- UID (Excel serial date of cohort + segment) ---------------
        excel_serial = (df["feo_cohort"] - EXCEL_EPOCH).dt.days
        df["UID"] = excel_serial.astype(str) + df["segment"].astype(str)

        # -- ENROLLS via vlookup into enroll ---------------------------
        enroll = enroll.copy()
        if "feo_cohort" in enroll.columns:
            enroll["feo_cohort"] = pd.to_datetime(enroll["feo_cohort"], errors="coerce")
        if "enrols" in enroll.columns:
            enroll["enrols"] = pd.to_numeric(enroll["enrols"], errors="coerce").fillna(0)

        if "segment" in enroll.columns and "feo_cohort" in enroll.columns:
            enroll_serial = (enroll["feo_cohort"] - EXCEL_EPOCH).dt.days
            enroll["UID"] = enroll_serial.astype(str) + enroll["segment"].astype(str)
            enroll_lookup = enroll.drop_duplicates(subset="UID").set_index("UID")["enrols"]
            df["ENROLLS"] = df["UID"].map(enroll_lookup).fillna(0)
        else:
            df["ENROLLS"] = 0

        # -- Derived metrics (safe division via numpy) -----------------
        enrolls = df["ENROLLS"].values
        impressions = df["impression_count"].values
        clicks = df["click_count"].values
        payout = df["payout"].values
        conversions = df["conversion_count"].values

        df["Imp_pct"] = np.where(enrolls != 0, impressions / enrolls, 0.0)
        df["CTR"] = np.where(impressions != 0, clicks / impressions, 0.0)
        df["RPC"] = np.where(clicks != 0, payout / clicks, 0.0)
        df["RPU"] = np.where(enrolls != 0, payout / enrolls, 0.0)
        df["Conversion_rate"] = np.where(clicks != 0, conversions / clicks, 0.0)
        df["eCPM"] = np.where(impressions != 0, (payout / impressions) * 1000, 0.0)

        # -- Metadata lists --------------------------------------------
        cohort_dates = sorted(df["feo_cohort"].dropna().unique().tolist())
        latest_cohort = cohort_dates[-1] if cohort_dates else None
        segments = sorted(df["SEGMENT"].dropna().unique().tolist())
        touchpoints = sorted(df["TOUCHPOINT"].dropna().unique().tolist())
        lenders = sorted(df["LENDER"].dropna().unique().tolist())

        return EnrichedData(
            enroll=enroll,
            non_api=non_api,
            enriched=df,
            cohort_dates=cohort_dates,
            latest_cohort=latest_cohort,
            segments=segments,
            touchpoints=touchpoints,
            lenders=lenders,
        )
