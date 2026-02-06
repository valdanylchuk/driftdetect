"""
Loan Triage API - Example with Drift Detection

Demo service showing how to add drift detection to a FastAPI endpoint.

Usage:
    uvicorn examples.lendingclub.api:app --port 8000

Endpoints:
    POST /triage - Score a loan application
    GET  /health - Service health check
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from checkdrift import check_drift

from .models import LoanApplication, TriageResult, LoanTriageModel


BASELINE_PATH = Path(__file__).parent / "baseline.json"

model: LoanTriageModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    model = LoanTriageModel(seed=42)
    yield
    model = None


app = FastAPI(
    title="Loan Triage API",
    description="Demo API with drift detection using checkdrift.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict:
    """Service health check."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/triage", response_model=TriageResult)
@check_drift(baseline=BASELINE_PATH)
async def triage_application(application: LoanApplication) -> TriageResult:
    """
    Score a loan application and assign to review queue.

    Returns risk score and recommended review tier.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return model.predict(application)
