import joblib
from pathlib import Path

def save_model(model, path:str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path:str):
    return joblib.load(path)
