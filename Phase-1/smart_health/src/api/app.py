"""
FastAPI service.

python -m uvicorn src.api.app:app --reload
"""

from pathlib import Path
from typing import List

import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "best_model.pkl"

app = FastAPI(
    title="Readmission prediction API",
    description="Predict 30-day readmission probability for EHR rows",
    version="1.0.0"
)

# load pickled pipeline once
model = joblib.load(MODEL_PATH)

#Pydantic payloads

class PatientRecord(BaseModel):
    patient_number: int|str
    class Config:
        extra = "allow"  # accept any additional EHRs

class PredictionRequest(BaseModel):
    patients: List[PatientRecord]

# Helpers

def _json_to_df(payload: PredictionRequest) -> pd.DataFrame:
    df = pd.DataFrame([p.dict(exclude_unset=False) for p in payload.patients])
    return df


# Endpoints

@app.get("/")
def root():
    return {"msg": "alive"}

@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        df = _json_to_df(payload)
        probs = model.predict_proba(df)[:,1]
        return {"risk": probs.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        df = (pd.read_csv(pd.io.common.BytesIO(raw))
              if file.filename.endswith(".csv")
              else pd.read_parquet(pd.io.common.BytesIO(raw)))
        probs = model.predict_proba(df)[:,1]
        return {"risk": probs.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
