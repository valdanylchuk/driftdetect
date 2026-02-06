"""
Loan Triage System - Shared Models

Pydantic models and mock ML model for the example.
"""

from enum import Enum
from pydantic import BaseModel, Field
import numpy as np


class ReviewTier(str, Enum):
    """Review queue assignment based on risk score."""
    FAST_TRACK = "fast_track"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    MANUAL_REVIEW = "manual_review"


class LoanApplication(BaseModel):
    """Loan application input."""

    annual_inc: float = Field(..., gt=0, description="Annual income in USD")
    dti: float = Field(..., ge=0, le=100, description="Debt-to-income ratio")
    loan_amnt: float = Field(..., gt=0, le=100000, description="Loan amount in USD")
    int_rate: float = Field(..., gt=0, le=35, description="Interest rate")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "annual_inc": 75000,
                    "dti": 18.5,
                    "loan_amnt": 15000,
                    "int_rate": 11.99
                }
            ]
        }
    }


class TriageResult(BaseModel):
    """Triage system output."""

    risk_score: float = Field(..., ge=0, le=1, description="Probability of default")
    review_tier: ReviewTier = Field(..., description="Assigned review queue")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    flags: list[str] = Field(default_factory=list, description="Risk flags")


class LoanTriageModel:
    """
    Mock loan risk model.

    Uses simplified heuristics that produce realistic scores.
    """

    TIER_THRESHOLDS = {
        "fast_track": 0.15,
        "standard": 0.35,
        "enhanced": 0.55,
    }

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def predict(self, application: LoanApplication) -> TriageResult:
        """Score a single loan application."""
        flags = []

        loan_to_income = application.loan_amnt / application.annual_inc

        # Logistic-like risk score
        z = (
            -2.5
            + 0.04 * application.dti
            + 2.0 * loan_to_income
            + 0.08 * application.int_rate
            - 0.000005 * application.annual_inc
        )
        base_risk = 1 / (1 + np.exp(-z))

        noise = self._rng.normal(0, 0.02)
        risk_score = float(np.clip(base_risk + noise, 0.01, 0.99))

        # Flag risk factors
        if application.dti > 35:
            flags.append("HIGH_DTI")
        if loan_to_income > 0.5:
            flags.append("HIGH_LEVERAGE")
        if application.int_rate > 20:
            flags.append("SUBPRIME_RATE")
        if application.annual_inc < 30000:
            flags.append("LOW_INCOME")

        # Assign tier
        if flags and risk_score > 0.25:
            tier = ReviewTier.MANUAL_REVIEW
        elif risk_score <= self.TIER_THRESHOLDS["fast_track"]:
            tier = ReviewTier.FAST_TRACK
        elif risk_score <= self.TIER_THRESHOLDS["standard"]:
            tier = ReviewTier.STANDARD
        elif risk_score <= self.TIER_THRESHOLDS["enhanced"]:
            tier = ReviewTier.ENHANCED
        else:
            tier = ReviewTier.MANUAL_REVIEW

        distances = [abs(risk_score - t) for t in self.TIER_THRESHOLDS.values()]
        confidence = float(np.clip(0.7 + min(distances), 0.5, 0.95))

        return TriageResult(
            risk_score=round(risk_score, 4),
            review_tier=tier,
            confidence=round(confidence, 4),
            flags=flags
        )
