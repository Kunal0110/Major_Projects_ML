from prefect import flow, task
from monitoring.generate_data_drift_report import generate_data_drift_report
from monitoring.generate_prediction_drift_report import generate_prediction_drift_report
from monitoring.generate_target_drift_report import generate_target_drift_report
from monitoring.full_dashboard import full_dashboard

@flow(name="Monitoring Pipeline")
def monitoring_flow():
    generate_data_drift_report()
    generate_prediction_drift_report()
    generate_target_drift_report()
    full_dashboard()

if __name__ == "__main__":
    monitoring_flow()
