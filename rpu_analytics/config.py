DEFAULT_CONFIG = {
    "data_layer": {
        "imp_source_to_touchpoint": {
            "5. DISCOVER_TAB_V2": "Discover Tab", "5. DISCOVER_TAB_V3": "Discover Tab",
            "2. FUNNEL_TWO": "F2", "1. FUNNEL": "F1", "7. AI_ASSISTANT": "AI_assistant",
            "4. DASHBOARD": "Dashboard", "3. FUNNEL_THREE": "F3",
            "8. LOAN_AGENT": "Loan agent", "9. COMMS": "Comms",
        },
    },
    "monitoring": {
        "lookback_weeks": 8,
        "alert_thresholds": {
            "rpu_z_score": 2.0, "ctr_drop_pct": 30, "enrollment_shock_pct": 20,
            "affiliate_gone_min_share": 5, "conversion_rate_drop_pct": 40, "segment_divergence_pct": 10,
        },
    },
    "projections": {"ewma_decay": 0.7, "rolling_window": 4, "trend_min_r2": 0.5, "backtest_weeks": 8, "confidence_level": 0.95},
    "rca": {"max_drill_depth": 5, "significance_threshold": 0.10, "maturity_percentile_warn": 25, "trend_lookback_weeks": 6},
    "insights": {"tier_lookback_weeks": 8, "leakage_threshold": 0.50, "concentration_warning_hhi": 0.25},
    "opportunity": {
        "default_scenarios": [
            {"name": "Discover Tab +10% Imp%", "changes": [{"dimension": "touchpoint", "value": "Discover Tab", "metric": "imp_pct", "change_pct": 10}]},
            {"name": "Top lender RPC +5", "changes": [{"dimension": "lender", "value": "__TOP_1__", "metric": "rpc", "change_abs": 5}]},
        ],
    },
}
