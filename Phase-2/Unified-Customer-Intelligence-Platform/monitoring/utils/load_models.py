import joblib
from pathlib import Path

def load_churn_model():
    """Load trained churn model"""
    model_path = Path("models/churn/stacking_model.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    return model

def load_clv_model():
    """Load trained CLV model"""
    model_path = Path("models/clv/clv_model.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    return model

def load_segmentation_model():
    """Load trained segmentation model"""
    model_path = Path("models/segmentation/kmeans_model.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    return model
