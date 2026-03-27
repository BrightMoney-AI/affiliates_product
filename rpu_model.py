"""
RPU Projection Model — Python replacement for the Google Sheet model
=====================================================================
INPUTS (2 CSVs or DataFrames):
  1. R_Enroll:   feo_cohort, segment, enrols
  2. R_Non_API:  feo_cohort, segment, enrol_status, click_cohort, imp_source,
                 product_name, enrols, impression_count, click_count, payout, conversion_count

OUTPUT:
  Main tab — feo_cohort-level predicted vs actual RPU breakdown
"""

import pandas as pd
import numpy as np
import warnings, sys, os
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. MAPPINGS (hardcoded — same as Mappings tab)
# ─────────────────────────────────────────────────────────────
IMP_SOURCE_TO_TOUCHPOINT = {
    "5. DISCOVER_TAB_V2": "Discover Tab",
    "5. DISCOVER_TAB_V3": "Discover Tab",
    "2. FUNNEL_TWO": "F2",
    "1. FUNNEL": "F1",
    "7. AI_ASSISTANT": "AI_assistant",
    "4. DASHBOARD": "Dashboard",
    "3. FUNNEL_THREE": "F3",
    "8. LOAN_AGENT": "Loan agent",
    "9. COMMS": "Comms",
}

SEGMENT_MAP = {
    "NS1/3 & all CS": "NS1/3 & all CS",
    "NS2 & all CS": "NS2 & all CS",
}


def load_inputs(enroll_path: str, non_api_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two input files (CSV or Excel)."""
    read = lambda p: pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
    enroll = read(enroll_path)
    non_api = read(non_api_path)

    # Normalise dates
    enroll["feo_cohort"] = pd.to_datetime(enroll["feo_cohort"])
    non_api["feo_cohort"] = pd.to_datetime(non_api["feo_cohort"])

    # Ensure numeric
    for col in ["enrols"]:
        enroll[col] = pd.to_numeric(enroll[col], errors="coerce").fillna(0)
    for col in ["enrols", "impression_count", "click_count", "payout", "conversion_count"]:
        non_api[col] = pd.to_numeric(non_api[col], errors="coerce").fillna(0)

    return enroll, non_api


def load_inputs_from_workbook(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both inputs from the existing workbook (R Enroll & R Non API tabs)."""
    enroll = pd.read_excel(path, sheet_name="R Enroll", header=0)
    # Columns: UID, feo_cohort, segment, enrols  (UID is col A — derived, we skip it)
    enroll = enroll[["feo_cohort", "segment", "enrols"]].copy()
    enroll["feo_cohort"] = pd.to_datetime(enroll["feo_cohort"])
    enroll["enrols"] = pd.to_numeric(enroll["enrols"], errors="coerce").fillna(0)

    non_api = pd.read_excel(path, sheet_name="R Non API", header=1)
    # Row 1 is "Input/Formulas" labels, Row 2 is actual headers
    input_cols = ["feo_cohort", "segment", "enrol_status", "click_cohort", "imp_source",
                  "product_name", "enrols", "impression_count", "click_count", "payout", "conversion_count"]
    non_api = non_api[input_cols].copy()
    non_api["feo_cohort"] = pd.to_datetime(non_api["feo_cohort"])
    for col in ["enrols", "impression_count", "click_count", "payout", "conversion_count"]:
        non_api[col] = pd.to_numeric(non_api[col], errors="coerce").fillna(0)

    return enroll, non_api


# ─────────────────────────────────────────────────────────────
# 2. ENRICH R Non API (replicate the formula columns)
# ─────────────────────────────────────────────────────────────
def enrich_non_api(non_api: pd.DataFrame, enroll: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns: TOUCHPOINT, LENDER, UID, ENROLLS, Imp%, CTR, RPC, RPU."""
    df = non_api.copy()

    # SEGMENT — pass through
    df["SEGMENT"] = df["segment"]

    # TOUCHPOINT — vlookup on imp_source → Mappings
    df["TOUCHPOINT"] = df["imp_source"].map(IMP_SOURCE_TO_TOUCHPOINT).fillna("Non-click")

    # LENDER — same as product_name
    df["LENDER"] = df["product_name"]

    # UID — CONCATENATE(feo_cohort serial, segment)
    # The sheet uses Excel date serial number. We replicate:
    excel_epoch = pd.Timestamp("1899-12-30")
    df["feo_serial"] = ((df["feo_cohort"] - excel_epoch).dt.days).astype(str)
    df["UID"] = df["feo_serial"] + df["SEGMENT"]

    # ENROLLS — vlookup on UID into R Enroll
    enroll_lookup = enroll.copy()
    enroll_lookup["feo_serial"] = ((enroll_lookup["feo_cohort"] - excel_epoch).dt.days).astype(str)
    enroll_lookup["UID"] = enroll_lookup["feo_serial"] + enroll_lookup["segment"]
    enroll_map = enroll_lookup.drop_duplicates("UID").set_index("UID")["enrols"]
    df["ENROLLS"] = df["UID"].map(enroll_map).fillna(0)

    # Imp % = impression_count / ENROLLS
    df["Imp_pct"] = np.where(df["ENROLLS"] > 0, df["impression_count"] / df["ENROLLS"], 0)

    # CTR = click_count / impression_count
    df["CTR"] = np.where(df["impression_count"] > 0, df["click_count"] / df["impression_count"], 0)

    # RPC = payout / click_count
    df["RPC"] = np.where(df["click_count"] > 0, df["payout"] / df["click_count"], 0)

    # RPU = payout / ENROLLS
    df["RPU"] = np.where(df["ENROLLS"] > 0, df["payout"] / df["ENROLLS"], 0)

    return df


# ─────────────────────────────────────────────────────────────
# 3. BUILD ACTUAL RPC + 4-WEEK ROLLING AVERAGE
# ─────────────────────────────────────────────────────────────
def build_actual_rpc(enriched: pd.DataFrame) -> pd.DataFrame:
    """
    For W1 Enrolled data, compute RPC per (TOUCHPOINT, LENDER, SEGMENT, feo_cohort)
    and a rolling 4-week average RPC.
    """
    w1 = enriched[
        (enriched["click_cohort"] == "1. W1") & (enriched["enrol_status"] == "Enrolled")
    ].copy()

    rpc_agg = (
        w1.groupby(["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"])
        .agg(total_payout=("payout", "sum"), total_clicks=("click_count", "sum"))
        .reset_index()
    )
    rpc_agg["RPC"] = np.where(rpc_agg["total_clicks"] > 0, rpc_agg["total_payout"] / rpc_agg["total_clicks"], 0)

    # Rolling 4-week average per (TOUCHPOINT, LENDER, SEGMENT)
    rpc_agg = rpc_agg.sort_values(["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"])
    rpc_agg["rolling_4w_avg_RPC"] = (
        rpc_agg.groupby(["TOUCHPOINT", "LENDER", "SEGMENT"])["RPC"]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    ).fillna(0)

    rpc_agg["combo_key"] = rpc_agg["TOUCHPOINT"] + rpc_agg["LENDER"] + rpc_agg["SEGMENT"]
    rpc_agg["full_uid"] = rpc_agg["combo_key"] + rpc_agg["feo_cohort"].dt.strftime("%Y%m%d")

    return rpc_agg


# ─────────────────────────────────────────────────────────────
# 4. W1 RPU PROJECTIONS
# ─────────────────────────────────────────────────────────────
def compute_w1_rpu(enriched: pd.DataFrame, actual_rpc: pd.DataFrame, enroll: pd.DataFrame) -> pd.DataFrame:
    """
    W1 Projected RPU = Imp% × CTR × rolling_4w_avg_RPC  (per touchpoint/lender/segment/cohort)
    W1 Actual RPU    = Imp% × CTR × actual_RPC
    Then aggregate to (feo_cohort, segment) level.
    """
    w1 = enriched[
        (enriched["click_cohort"] == "1. W1") & (enriched["enrol_status"] == "Enrolled")
    ].copy()

    # Aggregate Imp% and CTR per combo
    w1_agg = (
        w1.groupby(["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"])
        .agg(Imp_pct=("Imp_pct", "sum"), CTR=("CTR", "mean"),
             total_payout=("payout", "sum"), total_clicks=("click_count", "sum"))
        .reset_index()
    )
    w1_agg["actual_RPC"] = np.where(w1_agg["total_clicks"] > 0, w1_agg["total_payout"] / w1_agg["total_clicks"], 0)

    # Merge rolling avg RPC
    rpc_lookup = actual_rpc[["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort", "rolling_4w_avg_RPC"]].copy()
    w1_agg = w1_agg.merge(rpc_lookup, on=["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"], how="left")
    w1_agg["rolling_4w_avg_RPC"] = w1_agg["rolling_4w_avg_RPC"].fillna(0)

    # Per-row RPU
    w1_agg["w1_actual_rpu"] = w1_agg["Imp_pct"] * w1_agg["CTR"] * w1_agg["actual_RPC"]
    w1_agg["w1_projected_rpu"] = w1_agg["Imp_pct"] * w1_agg["CTR"] * w1_agg["rolling_4w_avg_RPC"]

    # Aggregate to segment level
    seg_rpu = (
        w1_agg.groupby(["feo_cohort", "SEGMENT"])
        .agg(w1_actual_rpu=("w1_actual_rpu", "sum"), w1_projected_rpu=("w1_projected_rpu", "sum"))
        .reset_index()
    )

    # Weighted average across segments using enrolls
    enroll_by = enroll.groupby("feo_cohort").agg(total_enrols=("enrols", "sum")).reset_index()
    seg_rpu = seg_rpu.merge(
        enroll[["feo_cohort", "segment", "enrols"]].rename(columns={"segment": "SEGMENT"}),
        on=["feo_cohort", "SEGMENT"], how="left"
    )
    seg_rpu["enrols"] = seg_rpu["enrols"].fillna(0)

    # Weighted average to get blended RPU per feo_cohort
    cohort_rpu = seg_rpu.groupby("feo_cohort").apply(
        lambda g: pd.Series({
            "w1_predicted": np.average(g["w1_projected_rpu"], weights=g["enrols"]) if g["enrols"].sum() > 0 else 0,
            "w1_actual": np.average(g["w1_actual_rpu"], weights=g["enrols"]) if g["enrols"].sum() > 0 else 0,
        })
    ).reset_index()

    return cohort_rpu


# ─────────────────────────────────────────────────────────────
# 5. RATIO-BASED PROJECTIONS (W2-W4 and M3)
# ─────────────────────────────────────────────────────────────
def compute_ratio_projections(enriched: pd.DataFrame, enroll: pd.DataFrame, 
                               cohort_label: str, target_click_cohorts: list) -> pd.DataFrame:
    """
    For W2-W4 or M3:
    - Compute Imp ratio, CTR ratio, RPC ratio (target period / W1) per combo
    - Use rolling 4-week average of ratios
    - Projected RPU = W1_CTR × W1_Imp% × W1_RPC × imp_ratio × ctr_ratio × rpc_ratio
    - Actual RPU = actual payout / enrolls
    """
    enrolled = enriched[enriched["enrol_status"] == "Enrolled"].copy()

    # W1 base metrics
    w1 = enrolled[enrolled["click_cohort"] == "1. W1"].copy()
    w1_base = (
        w1.groupby(["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"])
        .agg(w1_imp=("Imp_pct", "sum"), w1_ctr=("CTR", "mean"),
             w1_payout=("payout", "sum"), w1_clicks=("click_count", "sum"))
        .reset_index()
    )
    w1_base["w1_rpc"] = np.where(w1_base["w1_clicks"] > 0, w1_base["w1_payout"] / w1_base["w1_clicks"], 0)

    # Target period metrics
    target = enrolled[enrolled["click_cohort"].isin(target_click_cohorts)].copy()
    target_agg = (
        target.groupby(["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"])
        .agg(t_imp=("Imp_pct", "sum"), t_ctr=("CTR", "mean"),
             t_payout=("payout", "sum"), t_clicks=("click_count", "sum"))
        .reset_index()
    )
    target_agg["t_rpc"] = np.where(target_agg["t_clicks"] > 0, target_agg["t_payout"] / target_agg["t_clicks"], 0)

    # Merge and compute ratios
    merged = w1_base.merge(target_agg, on=["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"], how="left")
    for col in ["t_imp", "t_ctr", "t_rpc", "t_payout"]:
        merged[col] = merged[col].fillna(0)

    merged["imp_ratio"] = np.where(merged["w1_imp"] > 0, merged["t_imp"] / merged["w1_imp"], 0)
    merged["ctr_ratio"] = np.where(merged["w1_ctr"] > 0, merged["t_ctr"] / merged["w1_ctr"], 1)
    merged["rpc_ratio"] = np.where(merged["w1_rpc"] > 0, merged["t_rpc"] / merged["w1_rpc"], 1)

    # Rolling 4-week average of ratios
    merged = merged.sort_values(["TOUCHPOINT", "LENDER", "SEGMENT", "feo_cohort"])
    for ratio_col in ["imp_ratio", "ctr_ratio", "rpc_ratio"]:
        merged[f"{ratio_col}_4w"] = (
            merged.groupby(["TOUCHPOINT", "LENDER", "SEGMENT"])[ratio_col]
            .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        ).fillna(0)

    # Projected RPU per combo
    merged["projected_rpu"] = (
        merged["w1_ctr"] * merged["w1_imp"] * merged["w1_rpc"]
        * merged["imp_ratio_4w"] * merged["ctr_ratio_4w"] * merged["rpc_ratio_4w"]
    )

    # Actual RPU per combo (actual payout / enrolls at segment-cohort level)
    merged["actual_rpu_combo"] = merged["t_payout"]  # will be divided by enrolls after agg

    # Aggregate to segment level
    seg = (
        merged.groupby(["feo_cohort", "SEGMENT"])
        .agg(projected=("projected_rpu", "sum"), actual_payout=("t_payout", "sum"))
        .reset_index()
    )
    seg = seg.merge(
        enroll[["feo_cohort", "segment", "enrols"]].rename(columns={"segment": "SEGMENT"}),
        on=["feo_cohort", "SEGMENT"], how="left"
    )
    seg["enrols"] = seg["enrols"].fillna(0)
    seg["actual_rpu"] = np.where(seg["enrols"] > 0, seg["actual_payout"] / seg["enrols"], 0)

    # Weighted average across segments
    cohort = seg.groupby("feo_cohort").apply(
        lambda g: pd.Series({
            f"{cohort_label}_predicted": np.average(g["projected"], weights=g["enrols"]) if g["enrols"].sum() > 0 else 0,
            f"{cohort_label}_actual": np.average(g["actual_rpu"], weights=g["enrols"]) if g["enrols"].sum() > 0 else 0,
        })
    ).reset_index()

    return cohort


# ─────────────────────────────────────────────────────────────
# 6. NON-CLICK AND NON-ENROLL RPU
# ─────────────────────────────────────────────────────────────
def compute_nonclick_rpu(enriched: pd.DataFrame, enroll: pd.DataFrame) -> pd.DataFrame:
    """Non-click payout / total enrolls per feo_cohort."""
    nonclick = enriched[enriched["TOUCHPOINT"] == "Non-click"].copy()
    nc_agg = nonclick.groupby("feo_cohort").agg(nc_payout=("payout", "sum")).reset_index()
    total_enroll = enroll.groupby("feo_cohort").agg(total_enrols=("enrols", "sum")).reset_index()
    nc_agg = nc_agg.merge(total_enroll, on="feo_cohort", how="left")
    nc_agg["nonclick_actual"] = np.where(nc_agg["total_enrols"] > 0, nc_agg["nc_payout"] / nc_agg["total_enrols"], 0)
    nc_agg["nonclick_predicted"] = 0.0  # model uses 0 for non-click predicted
    return nc_agg[["feo_cohort", "nonclick_predicted", "nonclick_actual"]]


def compute_nonenroll_rpu(enriched: pd.DataFrame, enroll: pd.DataFrame) -> pd.DataFrame:
    """Non-enrolled payout / total enrolls per feo_cohort."""
    nonenroll = enriched[enriched["enrol_status"] == "Non-Enrolled"].copy()
    ne_agg = nonenroll.groupby("feo_cohort").agg(ne_payout=("payout", "sum")).reset_index()
    total_enroll = enroll.groupby("feo_cohort").agg(total_enrols=("enrols", "sum")).reset_index()
    ne_agg = ne_agg.merge(total_enroll, on="feo_cohort", how="left")
    ne_agg["nonenroll_actual"] = np.where(ne_agg["total_enrols"] > 0, ne_agg["ne_payout"] / ne_agg["total_enrols"], 0)
    ne_agg["nonenroll_predicted"] = 0.0
    return ne_agg[["feo_cohort", "nonenroll_predicted", "nonenroll_actual"]]


# ─────────────────────────────────────────────────────────────
# 7. ASSEMBLE MAIN OUTPUT
# ─────────────────────────────────────────────────────────────
def build_main(enroll: pd.DataFrame, non_api: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: inputs → Main tab output."""
    
    print("  [1/6] Enriching R Non API data...")
    enriched = enrich_non_api(non_api, enroll)
    
    print("  [2/6] Building Actual RPC + rolling averages...")
    actual_rpc = build_actual_rpc(enriched)
    
    print("  [3/6] Computing W1 RPU projections...")
    w1 = compute_w1_rpu(enriched, actual_rpc, enroll)
    
    print("  [4/6] Computing W2-W4 RPU projections...")
    w2w4 = compute_ratio_projections(enriched, enroll, "w2w4", ["2. W2-W4"])
    
    print("  [5/6] Computing M3 RPU projections...")
    m3 = compute_ratio_projections(enriched, enroll, "m3", ["3. M3"])
    
    print("  [6/6] Computing Non-click and Non-enroll RPU...")
    nc = compute_nonclick_rpu(enriched, enroll)
    ne = compute_nonenroll_rpu(enriched, enroll)

    # Get cohort list from enroll
    cohorts = enroll.groupby("feo_cohort").agg(total_enrols=("enrols", "sum")).reset_index()
    
    # Merge everything
    main = cohorts[["feo_cohort"]].drop_duplicates().sort_values("feo_cohort", ascending=False)
    main = main.merge(w1, on="feo_cohort", how="left")
    main = main.merge(w2w4, on="feo_cohort", how="left")
    main = main.merge(m3, on="feo_cohort", how="left")
    main = main.merge(nc, on="feo_cohort", how="left")
    main = main.merge(ne, on="feo_cohort", how="left")
    main = main.fillna(0)

    # Total predicted and actual
    main["total_predicted"] = (
        main["w1_predicted"] + main["w2w4_predicted"] + main["m3_predicted"]
        + main["nonclick_predicted"] + main["nonenroll_predicted"]
    )
    main["total_actual"] = (
        main["w1_actual"] + main["w2w4_actual"] + main["m3_actual"]
        + main["nonclick_actual"] + main["nonenroll_actual"]
    )

    # Delta
    main["delta"] = main["total_actual"] - main["total_predicted"]
    main["delta_pct"] = np.where(main["total_actual"] != 0, main["delta"] / main["total_actual"], 0)

    # Rename for clarity
    main = main.rename(columns={
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
    })

    main["feo_cohort"] = main["feo_cohort"].dt.date
    return main


# ─────────────────────────────────────────────────────────────
# 8. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RPU Projection Model — Python Engine")
    print("=" * 60)

    # Determine input mode
    if len(sys.argv) >= 3:
        # Mode 1: Two separate CSV files
        enroll_path, non_api_path = sys.argv[1], sys.argv[2]
        print(f"\nLoading inputs from CSVs:\n  Enroll:  {enroll_path}\n  Non-API: {non_api_path}")
        enroll, non_api = load_inputs(enroll_path, non_api_path)
    elif len(sys.argv) == 2:
        # Mode 2: Single workbook with both tabs
        wb_path = sys.argv[1]
        print(f"\nLoading inputs from workbook: {wb_path}")
        enroll, non_api = load_inputs_from_workbook(wb_path)
    else:
        # Default: use the uploaded workbook
        wb_path = "/mnt/user-data/uploads/V2_RV__Non-API_RPU_Projections_.xlsx"
        print(f"\nLoading inputs from workbook: {wb_path}")
        enroll, non_api = load_inputs_from_workbook(wb_path)

    print(f"\n  Enroll rows:  {len(enroll):,}")
    print(f"  Non-API rows: {len(non_api):,}")
    print(f"  Date range:   {enroll['feo_cohort'].min().date()} → {enroll['feo_cohort'].max().date()}")
    print()

    main = build_main(enroll, non_api)

    # Save output
    out_path = "/home/claude/rpu_output.csv"
    main.to_csv(out_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Output saved to: {out_path}")
    print(f"Rows: {len(main)}")
    print(f"{'=' * 60}")
    print("\nFirst 10 rows:\n")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.6f}".format)
    print(main.head(10).to_string(index=False))
