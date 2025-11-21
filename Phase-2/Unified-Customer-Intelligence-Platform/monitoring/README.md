# Monitoring Setup

## What It Does

The monitoring system tracks:
1. **Data Drift** - Detects changes in input feature distributions
2. **Prediction Drift** - Monitors model prediction distributions
3. **Target Drift** - Tracks changes in actual churn rates
4. **Full Dashboard** - Comprehensive view of all metrics

## Setup Instructions

### 1. Install Dependencies
```bash
pip install evidently prefect
```

### 2. Start Prefect Server (Terminal 1)
```bash
prefect server start
```

### 3. Build and Deploy Monitoring Flow (Terminal 2)
```bash
cd monitoring

# Build deployment (runs hourly)
prefect deployment build scheduler_prefect_flow.py:monitoring_flow -n monitor --cron "0 * * * *"

# Apply deployment
prefect deployment apply monitoring_flow-deployment.yaml
```

### 4. Start Prefect Agent (Terminal 3)
```bash
prefect agent start -q default
```

### 5. Manual Run (Optional)
```bash
python scheduler_prefect_flow.py
```

## View Reports

After running, check:
- `monitoring/reports/data_drift_report.html`
- `monitoring/reports/prediction_drift_report.html`
- `monitoring/reports/target_drift_report.html`
- `monitoring/reports/full_dashboard.html`

## Cron Schedule

`"0 * * * *"` = Every hour at minute 0

Change to:
- `"0 0 * * *"` = Daily at midnight
- `"0 */6 * * *"` = Every 6 hours
- `"*/30 * * * *"` = Every 30 minutes
