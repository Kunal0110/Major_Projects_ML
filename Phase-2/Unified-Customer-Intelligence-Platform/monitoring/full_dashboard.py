from evidently import Report
from evidently import DataDriftPreset, DataQualityPreset
from monitoring.utils.load_data import load_reference_data, load_current_data
from monitoring.utils.load_models import load_churn_model
from pathlib import Path
from datetime import datetime

def full_dashboard():
    """Generate comprehensive monitoring dashboard"""
    try:
        reference_data = load_reference_data()
        current_data = load_current_data()
        
        model = load_churn_model()
        
        reference_data['prediction'] = model.predict_proba(reference_data.drop(columns=['churn'], errors='ignore'))[:, 1]
        current_data['prediction'] = model.predict_proba(current_data.drop(columns=['churn'], errors='ignore'))[:, 1]
        
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])
        
        report.run(reference_data=reference_data, current_data=current_data)
        
        # Save timestamped version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = Path(f"monitoring/reports/archive/full_dashboard_{timestamp}.html")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(archive_path))
        
        # Save latest version (overwritten)
        latest_path = Path("monitoring/reports/full_dashboard.html")
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(latest_path))
        
        print(f"✅ Full dashboard saved to {latest_path} and {archive_path}")
        return True
    except Exception as e:
        print(f"❌ Full dashboard failed: {str(e)}")
        return False
