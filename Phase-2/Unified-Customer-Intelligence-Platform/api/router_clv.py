from fastapi import APIRouter, HTTPException
from api.model_loader import registry
from api.schemas import CLVRequest, CLVResponse
from api.utils import dict_to_dataframe
from api.feature_utils import prepare_features

router = APIRouter(prefix="/clv", tags=["CLV Prediction"])

@router.post("/predict", response_model=CLVResponse)
def predict_clv(data: CLVRequest):
    try:
        import joblib
        from pathlib import Path
        
        model = registry.clv.model
        if model is None:
            raise HTTPException(status_code=500, detail="CLV model not loaded")
        
        X = prepare_features(data.customer_data)
        print("CLV - Received columns:", list(X.columns))
        
        # Check if we have separate preprocessor and selector
        preprocessor_path = Path("models/clv/clv_preprocessor.pkl")
        selector_path = Path("models/clv/clv_feature_selector.pkl")
        
        if preprocessor_path.exists() and selector_path.exists():
            # Enhanced model with separate components
            preprocessor = joblib.load(preprocessor_path)
            selector = joblib.load(selector_path)
            
            # Apply preprocessing pipeline
            X_processed = preprocessor.transform(X)
            X_selected = selector.transform(X_processed)
            
            pred = model.predict(X_selected)[0]
        else:
            # Try direct prediction (pipeline model)
            pred = model.predict(X)[0]

        return CLVResponse(clv=float(pred))
    except Exception as e:
        print(f"Error in predict_clv: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"CLV prediction failed: {str(e)}"
        )
