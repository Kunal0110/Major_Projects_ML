from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from api.router_churn import router as churn_router
from api.router_clv import router as clv_router
from api.router_segments import router as segment_router
from api.router_system import router as system_router
from api.middleware import RateLimitMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified Customer Intelligence API",
    description="API for churn prediction, CLV estimation, segmentation, and explanations.",
    version="2.0.0"
)

# Add middlewares
app.add_middleware(RateLimitMiddleware, calls=100, period=60)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

app.include_router(churn_router)
app.include_router(clv_router)
app.include_router(segment_router)
app.include_router(system_router)

@app.get("/")
def root():
    return {
        "message": "Unified Customer Intelligence API v2.0",
        "status": "running",
        "features": [
            "Enhanced churn prediction with SMOTE + Feature Selection",
            "Improved CLV forecasting with tenure decay",
            "Auto-optimized customer segmentation",
            "Real-time model monitoring"
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/models/status")
def model_status():
    """Get status of all loaded models"""
    from api.model_loader import registry
    return registry.get_status()