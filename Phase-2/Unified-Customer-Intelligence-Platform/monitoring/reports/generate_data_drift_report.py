from evidently import Report
from evidently.metrics import DataDriftPreset
from utils.load_data import load_reference_data, load_current_data
from utils.alerts import notify
from pathlib import Path

REPORTS_DIR = Path("monitoring/reports")

def generate_data_drift_report():
    reference = load_reference_data()
    current = load_current_data()

    report = Report(metrics=[
        DataDriftPreset()
    ])

    report.run(reference_data=reference, current_data=current)

    output_path = REPORTS_DIR / "data_drift_report.html"
    report.save_html(output_path)

    drift_score = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
    if drift_score:
        notify("🚨 Data Drift Detected!")

    print(f"Data Drift Report saved to: {output_path}")

if __name__ == "__main__":
    generate_data_drift_report()
