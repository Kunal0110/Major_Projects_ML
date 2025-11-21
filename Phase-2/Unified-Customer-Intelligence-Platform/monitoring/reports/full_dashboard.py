from evidently import Dashboard
from evidently import (
    DataDriftTab,
    CatTargetDriftTab,
    RegressionPerformanceTab,
    ProbClassificationPerformanceTab
)
from utils.load_data import load_reference_data, load_current_data
from utils.load_models import load_churn_model
from pathlib import Path

REPORTS_DIR = Path("monitoring/reports")

def full_dashboard():
    reference = load_reference_data()
    current = load_current_data()
    model = load_churn_model()

    dashboard = Dashboard(tabs=[
        DataDriftTab(),
        CatTargetDriftTab(),
        ProbClassificationPerformanceTab()
    ])

    preds_ref = model.predict_proba(reference.drop(columns=["churn"]))[:,1]
    preds_curr = model.predict_proba(current.drop(columns=["churn"]))[:,1]

    reference_copy = reference.copy()
    current_copy = current.copy()

    reference_copy["prediction"] = preds_ref
    current_copy["prediction"] = preds_curr

    dashboard.calculate(reference_copy, current_copy)

    output = REPORTS_DIR / "full_dashboard.html"
    dashboard.save(output)

    print(f"Full monitoring dashboard saved to {output}")

if __name__ == "__main__":
    full_dashboard()
