from fastapi import APIRouter
from api.model_loader import registry
from api.schemas import SystemInfo

router = APIRouter(prefix="/system", tags=["System"])

@router.get("/health")
def health():
    return {"status":"ok"}

@router.get("/info", response_model=SystemInfo)
def system_info():
    return SystemInfo(
        status="running",
        model_versions={
            "churn": "v1",
            "clv": "v1",
            "segmentation":"v1"
        }
    )

@router.post("/reload-models")
def reload_models():
    registry.reload()
    return {"status": "models reloaded"}