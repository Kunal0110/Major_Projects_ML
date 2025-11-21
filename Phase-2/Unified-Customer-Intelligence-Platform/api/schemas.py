from pydantic import BaseModel
from typing import List, Dict, Any

class ChurnRequest(BaseModel):
    customer_data: Dict[str, Any]

class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: int

class ChurnBatchRequest(BaseModel):
    records: List[Dict[str, Any]]

class ChurnBatchResponse(BaseModel):
    predictions: List[Dict[str,Any]]

class CLVRequest(BaseModel):
    customer_data: Dict[str, Any]

class CLVResponse(BaseModel):
    clv: float

class SegmentRequest(BaseModel):
    customer_data: Dict[str, Any]

class SegmentResponse(BaseModel):
    segment: int

class SystemInfo(BaseModel):
    status: str
    model_versions: Dict[str, str]
