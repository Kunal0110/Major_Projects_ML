from fastapi import APIRouter, HTTPException
from api.model_loader import registry
from api.schemas import CLVRequest, CLVResponse
from api.utils import dict_to_dataframe

router = APIRouter(prefix="/clv", tags=["CLV Prediction"])

@router.post("/predict", response_model=CLVResponse)
def predict_clv(data: CLVRequest):
    try:
        model = registry.clv.model

        if model is None:
            raise HTTPException(status_code=500, detail="CLV model not loaded")
        
        X = dict_to_dataframe(data.customer_data)
        print("CLV - Received columns:", list(X.columns))
        
        pred = model.predict(X)[0]

        return CLVResponse(clv=float(pred))
    except Exception as e:
        print(f"Error in predict_clv: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"CLV prediction failed: {str(e)}"
        )
