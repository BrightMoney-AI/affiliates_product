"""
RPU Analytics Runner — Orchestrates all modules and produces output JSON files.
"""

import json
import sys
import os
from datetime import datetime, timezone, timedelta

# Add parent to path so rpu_analytics is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rpu_analytics.config import DEFAULT_CONFIG
from rpu_analytics.data_layer import DataLayer
from rpu_analytics.monitoring import MonitoringModule
from rpu_analytics.projections import ProjectionsModule
from rpu_analytics.rca import RCAModule
from rpu_analytics.insights import InsightsModule
from rpu_analytics.opportunity import OpportunityModule
from rpu_analytics.reporting import ReportingModule


class RPUAnalyticsRunner:
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.data_layer = DataLayer(self.config.get("data_layer"))
        self.monitoring = MonitoringModule(self.config.get("monitoring"))
        self.projections = ProjectionsModule(self.config.get("projections"))
        self.rca = RCAModule(self.config.get("rca"))
        self.insights = InsightsModule(self.config.get("insights"))
        self.opportunity = OpportunityModule(self.config.get("opportunity"))
        self.reporting = ReportingModule()

    def run_all(self, enroll_path, non_api_path, output_dir):
        """Run full analytics suite from CSV files. Writes JSON outputs to output_dir."""
        ist = timezone(timedelta(hours=5, minutes=30))
        generated_utc = datetime.now(timezone.utc)
        generated_ist = generated_utc.astimezone(ist).strftime("%d %b %Y, %I:%M %p IST")

        print("=" * 60)
        print("RPU Analytics Platform — Full Run")
        print(f"Generated: {generated_ist}")
        print("=" * 60)

        # Step 1: Load and enrich data
        print("\n[1/8] Loading and enriching data...")
        data = self.data_layer.from_csv(enroll_path, non_api_path)
        print(f"  Enroll rows: {len(data.enroll):,}")
        print(f"  Non-API rows: {len(data.non_api):,}")
        print(f"  Enriched rows: {len(data.enriched):,}")
        print(f"  Cohorts: {len(data.cohort_dates)}")
        print(f"  Latest cohort: {data.latest_cohort}")
        print(f"  Segments: {data.segments}")
        print(f"  Touchpoints: {data.touchpoints}")
        print(f"  Lenders: {len(data.lenders)}")

        # Step 2: Run modules
        outputs = {}

        print("\n[2/8] Running Monitoring module...")
        try:
            outputs["monitoring"] = self.monitoring.run(data)
            print("  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
            outputs["monitoring"] = {"error": str(e), "generated_at": generated_utc.isoformat()}

        print("[3/8] Running Projections module...")
        try:
            outputs["rpu_output"] = self.projections.run(data)
            print("  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
            outputs["rpu_output"] = {"error": str(e), "generated_at": generated_utc.isoformat()}

        print("[4/8] Running RCA module...")
        try:
            alerts = outputs.get("monitoring", {}).get("alerts", [])
            outputs["rca"] = self.rca.run(data, alerts=alerts)
            print("  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
            outputs["rca"] = {"error": str(e), "generated_at": generated_utc.isoformat()}

        print("[5/8] Running Insights module...")
        try:
            outputs["insights"] = self.insights.run(data)
            print("  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
            outputs["insights"] = {"error": str(e), "generated_at": generated_utc.isoformat()}

        print("[6/8] Running Opportunity module...")
        try:
            outputs["opportunity"] = self.opportunity.run(data)
            print("  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
            outputs["opportunity"] = {"error": str(e), "generated_at": generated_utc.isoformat()}

        print("[7/8] Running Reporting module...")
        try:
            outputs["report"] = self.reporting.run(
                outputs.get("monitoring", {}),
                outputs.get("rpu_output", {}),
                outputs.get("rca", {}),
                outputs.get("insights", {}),
                outputs.get("opportunity", {}),
            )
            print("  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
            outputs["report"] = {"error": str(e), "generated_at": generated_utc.isoformat()}

        # Step 3: Manifest
        print("[8/8] Generating manifest...")
        outputs["manifest"] = {
            "generated_at": generated_utc.isoformat(),
            "generated_at_ist": generated_ist,
            "modules": list(outputs.keys()),
            "cohort_range": {
                "start": str(data.cohort_dates[0].date()) if data.cohort_dates else None,
                "end": str(data.cohort_dates[-1].date()) if data.cohort_dates else None,
            },
            "latest_cohort": str(data.latest_cohort.date()) if data.latest_cohort else None,
        }

        # Step 4: Write output files
        os.makedirs(output_dir, exist_ok=True)
        file_map = {
            "monitoring": "monitoring.json",
            "rpu_output": "rpu_output.json",
            "rca": "rca.json",
            "insights": "insights.json",
            "opportunity": "opportunity.json",
            "report": "report.json",
            "manifest": "manifest.json",
        }

        print(f"\nWriting output files to {output_dir}/")
        for key, filename in file_map.items():
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(outputs[key], f, indent=2, default=str)
            size = os.path.getsize(filepath)
            print(f"  {filename:25s} {size:>8,} bytes")

        print(f"\n{'=' * 60}")
        print("All modules complete.")
        print(f"{'=' * 60}")

        return outputs


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    enroll_path = os.path.join(base, "..", "1.csv")
    non_api_path = os.path.join(base, "..", "2.csv")
    output_dir = os.path.join(base, "..", "output_files")

    runner = RPUAnalyticsRunner()
    runner.run_all(enroll_path, non_api_path, output_dir)
