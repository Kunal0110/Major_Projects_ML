from fastapi import APIRouter, HTTPException
from api.model_loader import registry
from api.schemas import (
    ChurnRequest,
    ChurnResponse,
    ChurnBatchRequest,
    ChurnBatchResponse
)

from api.utils import dict_to_dataframe
from api.feature_utils import prepare_features
import shap
import pandas as pd

router = APIRouter(prefix="/churn", tags=["Churn Prediction"])

@router.post("/predict", response_model=ChurnResponse)
def predict_churn(data: ChurnRequest):
    try:
        import joblib
        from pathlib import Path
        
        model = registry.churn.model
        if model is None:
            raise HTTPException(status_code=500, detail="Churn model not loaded")
        
        X_raw = prepare_features(data.customer_data)
        
        # Check for enhanced model components
        preprocessor_path = Path("models/churn/preprocessor.pkl")
        selector_path = Path("models/churn/feature_selector.pkl")
        
        if preprocessor_path.exists() and selector_path.exists():
            # Enhanced model with separate components
            preprocessor = joblib.load(preprocessor_path)
            selector = joblib.load(selector_path)
            
            # Apply preprocessing pipeline
            X_processed = preprocessor.transform(X_raw)
            X_selected = selector.transform(X_processed)
            
            proba = model.predict_proba(X_selected)[0][1]
        else:
            # Pipeline model or direct prediction
            if hasattr(model, 'named_steps'):
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
        
        # Load components separately for enhanced model
        model_path = Path("models/churn/xgb_best.pkl")
        preprocessor_path = Path("models/churn/preprocessor.pkl")
        selector_path = Path("models/churn/feature_selector.pkl")
        
        if model_path.exists() and preprocessor_path.exists() and selector_path.exists():
            # Enhanced model with separate components
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            selector = joblib.load(selector_path)
            
            X = prepare_features(data.customer_data)
            
            # Apply preprocessing pipeline
            X_processed = preprocessor.transform(X)
            X_selected = selector.transform(X_processed)
            
            # Generate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_selected)
            
            # Get feature names (limited to selected features)
            feature_names = [f"feature_{i}" for i in range(X_selected.shape[1])]
            
            return {
                "shap_values": shap_values[0].tolist() if len(shap_values.shape) > 1 else shap_values.tolist(),
                "base_value": float(explainer.expected_value),
                "feature_names": feature_names,
                "prediction": float(model.predict_proba(X_selected)[0][1])
            }
        
        else:
            # Fallback to pipeline model
            pipeline_path = Path("models/churn/stacking_model.pkl")
            if pipeline_path.exists():
                model = joblib.load(pipeline_path)
                X = prepare_features(data.customer_data)
                
                if hasattr(model, 'named_steps'):
                    Xt = model.named_steps["preprocessor"].transform(X)
                    base_model = model.named_steps["model"]
                    
                    explainer = shap.TreeExplainer(base_model)
                    shap_values = explainer.shap_values(Xt)
                    
                    return {
                        "shap_values": shap_values.tolist(),
                        "base_value": float(explainer.expected_value)
                    }
            
            # No SHAP available
            return {
                "explanation": "SHAP explanation not available - model components not found",
                "note": "Please retrain the model to enable SHAP explanations"
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