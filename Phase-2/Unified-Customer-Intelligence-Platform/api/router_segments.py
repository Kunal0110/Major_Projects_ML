from fastapi import APIRouter, HTTPException
from api.model_loader import registry
from api.schemas import SegmentRequest, SegmentResponse
from api.utils import dict_to_dataframe
from sklearn.preprocessing import StandardScaler
import pandas as pd

router = APIRouter(prefix="/segments", tags=["Customer Segmentation"])

FEATURES = [
    "usage_data_used_gb_mean",
    "usage_voice_minutes_mean",
    "billing_billed_amount_mean",
    "billing_payment_delay_days_max",
    "mkt_touch_count",
    "avg_monthly_revenue"
]

@router.post("/predict", response_model=SegmentResponse)
def get_segment(data: SegmentRequest):
    model = registry.segmentation.model

    if model is None:
        raise HTTPException(status_code=500, detail="Segmentation model not loaded")

    X = dict_to_dataframe(data.customer_data)
    X = X[FEATURES]

    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)

    seg = model.predict(X_scaled)[0]

    return SegmentResponse(segment=int(seg))
