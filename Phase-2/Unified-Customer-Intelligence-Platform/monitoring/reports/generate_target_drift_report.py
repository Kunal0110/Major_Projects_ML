import pandas as pd
from pathlib import Path
from evidently import Report
from evidently.metrics import TargetDriftPreset
from utils.load_data import load_reference_data, load_current_data
from utils.alerts import notify

REPORTS_DIR = Path("monitoring/reports")

def generate_target_drift_report():
    reference = load_reference_data()
    current = load_current_data()

    report = Report(metrics=[TargetDriftPreset()])

    report.run(
        reference_data=reference,
        current_data=current
    )

    output = REPORTS_DIR / "target_drift_report.html"
    report.save_html(output)

    drift = report.as_dict()["metrics"][0]["result"]["drift_detected"]
    if drift:
        notify("🚨 Target Drift Detected!")

    print(f"Target drift report saved to: {output}")

if __name__ == "__main__":
    generate_target_drift_report()
