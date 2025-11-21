from fastapi import APIRouter, HTTPException
from api.model_loader import registry
from api.schemas import (
    ChurnRequest,
    ChurnResponse,
    ChurnBatchRequest,
    ChurnBatchResponse
)

from api.utils import dict_to_dataframe
import shap
import pandas as pd

router = APIRouter(prefix="/churn", tags=["Churn Prediction"])

@router.post("/predict", response_model=ChurnResponse)
def predict_churn(data: ChurnRequest):
    try:
        model = registry.churn.model
        if model is None:
            raise HTTPException(status_code=500, detail="Churn model not loaded")
        
        X_raw = dict_to_dataframe(data.customer_data)
        preprocessor = model.named_steps["preprocessor"]
        
        print("Received columns:", list(X_raw.columns))
        print("Expected columns:", list(preprocessor.feature_names_in_))
        
        proba = model.predict_proba(X_raw)[0][1]
        pred = int(proba >= 0.5)

        return ChurnResponse(
            churn_probability=float(proba),
            churn_prediction=pred
        )
    except Exception as e:
        print(f"Error in predict_churn: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/explain")
def explain_churn(data: ChurnRequest):
    try:
        import joblib
        from pathlib import Path
        
        xgb_model = joblib.load(Path("models/churn/xgb_best.pkl"))
        
        X = dict_to_dataframe(data.customer_data)
        Xt = xgb_model.named_steps["preprocessor"].transform(X)

        explainer = shap.TreeExplainer(xgb_model.named_steps["model"])
        shap_values = explainer.shap_values(Xt)

        return {
            "shap_values": shap_values.tolist(),
            "base_value": float(explainer.expected_value) if hasattr(explainer.expected_value, '__float__') else explainer.expected_value.tolist()
        }
    except Exception as e:
        print(f"Error in explain_churn: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed: {str(e)}"
        )

@router.post("/batch", response_model=ChurnBatchResponse)
def batch_churn(data: ChurnBatchRequest):
    try:
        model = registry.churn.model
        if model is None:
            raise HTTPException(status_code=500, detail="Churn model not loaded")
        
        X = pd.DataFrame(data.records)
        print(f"Batch - Received {len(X)} records with columns:", list(X.columns))
        
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)

        output = []
        for idx, row in X.iterrows():
            output.append({
                "customer_data": row.to_dict(),
                "prediction": int(preds[idx]),
                "probability": float(proba[idx])
            })

        return ChurnBatchResponse(predictions=output)
    except Exception as e:
        print(f"Error in batch_churn: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )