from evidently import Report
from evidently import DataDriftPreset
from monitoring.utils.load_data import load_reference_data, load_current_data
from pathlib import Path
from datetime import datetime

def generate_data_drift_report():
    """Generate data drift report comparing reference and current data"""
    try:
        reference_data = load_reference_data()
        current_data = load_current_data()
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        
        # Save timestamped version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = Path(f"monitoring/reports/archive/data_drift_{timestamp}.html")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(archive_path))
        
        # Save latest version (overwritten)
        latest_path = Path("monitoring/reports/data_drift_report.html")
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(latest_path))
        
        print(f"✅ Data drift report saved to {latest_path} and {archive_path}")
        return True
    except Exception as e:
        print(f"❌ Data drift report failed: {str(e)}")
        return False
