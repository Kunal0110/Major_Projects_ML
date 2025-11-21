import pandas as pd
from pathlib import Path

from evidently import Report
from evidently.metrics import PredictionDrift
from utils.load_data import load_reference_data, load_current_data
from utils.load_models import load_churn_model
from utils.alerts import notify


REPORTS_DIR = Path("monitoring/reports")

def generate_prediction_drift_report():
    reference = load_reference_data()
    current = load_current_data()

    model = load_churn_model()

    ref_preds = model.predict_proba(reference.drop(columns=["churn"]))[:,1]
    curr_preds = model.predict_proba(current.drop(columns=["churn"]))[:,1]

    ref_df = pd.DataFrame({"prediction": ref_preds})
    curr_df = pd.DataFrame({"prediction": curr_preds})

    report = Report(metrics=[
        PredictionDrift()
    ])

    report.run(reference_data=ref_df, current_data=curr_df)

    output_path = REPORTS_DIR / "prediction_drift_report.html"
    report.save_html(output_path)

    drift = report.as_dict()["metrics"][0]["result"]["drift_score"]
    if drift > 0.1:  # threshold
        notify("Prediction Drift Detected!")

    print(f"Prediction drift report saved to: {output_path}")

if __name__ == "__main__":
    generate_prediction_drift_report()
