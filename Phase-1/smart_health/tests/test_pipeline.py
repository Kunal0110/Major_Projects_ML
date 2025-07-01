from pathlib import Path
import pandas as pd, joblib

def test_pipeline_runs():
    root = Path(__file__).resolve().parents[1]
    model = joblib.load(root / "models" / "best_model.pkl")
    df = pd.read_csv(root / "data" / "cleaned_diabetic_data.csv").head(10)
    assert model.predict_proba(df).shape[0] == 10