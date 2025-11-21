import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path("models/churn/xgb_best.pkl")

clf = joblib.load(MODEL_PATH)

def predict_churn(df: pd.DataFrame):
    """
    Returns:
        proba: churn probability
        pred: churn class (0/1)
    """
    proba = clf.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba