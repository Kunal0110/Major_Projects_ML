from evidently import Report
from evidently.metrics import ColumnDriftMetric
from monitoring.utils.load_data import load_reference_data, load_current_data
from pathlib import Path
from datetime import datetime

def generate_target_drift_report():
    """Generate target drift report for churn labels"""
    try:
        reference_data = load_reference_data()
        current_data = load_current_data()
        
        if 'churn' not in reference_data.columns or 'churn' not in current_data.columns:
            print("⚠️ Churn column not found, skipping target drift report")
            return False
        
        report = Report(metrics=[ColumnDriftMetric(column_name='churn')])
        report.run(reference_data=reference_data, current_data=current_data)
        
        # Save timestamped version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = Path(f"monitoring/reports/archive/target_drift_{timestamp}.html")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(archive_path))
        
        # Save latest version (overwritten)
        latest_path = Path("monitoring/reports/target_drift_report.html")
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(latest_path))
        
        print(f"✅ Target drift report saved to {latest_path} and {archive_path}")
        return True
    except Exception as e:
        print(f"❌ Target drift report failed: {str(e)}")
        return False
