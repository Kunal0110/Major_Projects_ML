from fastapi import FastAPI
from api.router_churn import router as churn_router
from api.router_clv import router as clv_router
from api.router_segments import router as segment_router
from api.router_system import router as system_router

app = FastAPI(
    title="Unified Customer Intelligence API",
    description="API for churn prediction, CLV estimation, segmentation, and explanations.",
    version="1.0.0"
)


app.include_router(churn_router)
app.include_router(clv_router)
app.include_router(segment_router)
app.include_router(system_router)

@app.get("/")
def root():
    return {"message": "Unified Customer Intelligence API is running."}