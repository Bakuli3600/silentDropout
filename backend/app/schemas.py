"""Pydantic models for request/response validation in the FastAPI backend.

The FastAPI app in ``backend/main.py`` re-exports these models at module
import time (the ``models`` module is not a FastAPI sub-package), so this file
must be importable without starting the server.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# National Student ID (NSI) policy: do not expose raw identifiers.  Anything
# derived (hashed pseudonym) is acceptable.
class PredictionInput(BaseModel):
    """Input for /api/predict.  Fields map to the early-warning feature set in
    ml/feature_engineering.py.  Ranges are clamped to [-10, 300] to allow
    missing-but-recorded values without producing 0 that would mean "absence"."""

    attendance_rate: float = Field(75.0, ge=0, le=100, description="Attendance %")
    assignment_submission_rate: float = Field(70.0, ge=0, le=100, description="Assignment submission %")
    lms_login_frequency: float = Field(4.0, ge=0, le=14, description="LMS logins per week")
    avg_session_time: float = Field(45.0, ge=0, le=300, description="Average session minutes")
    grades: float = Field(65.0, ge=0, le=100, description="Current grade average")
    semester: int = Field(1, ge=1, le=10, description="Current semester")
    previous_backlog_count: int = Field(0, ge=0, le=20, description="Prior backlogs")
    prediction_date: Optional[str] = Field(None, description="ISO date of snapshot (YYYY-MM-DD)")

    class Config:
        allow_population_by_field_name = True


class PredictionOutput(BaseModel):
    risk: int = Field(..., description="1 = high risk, 0 = low risk")
    probability: float = Field(..., ge=0.0, le=1.0, description="Dropout probability")
    risk_category: str = Field(..., description="Human-readable risk band")
    top_factors: List[str] = Field(default_factory=list, description="Top 5 SHAP contributors")
    directions: List[str] = Field(default_factory=list, description="Contributing factor directions")
    explanation: str = Field(..., description="Human-readable plain-language explanation")
    intervention: List[str] = Field(default_factory=list, description="Recommended actions")
    features_used: List[str] = Field(default_factory=list)
    threshold_applied: float = Field(0.35, description="Threshold applied")


class ExplainInput(BaseModel):
    attendance_rate: float = Field(..., ge=0, le=100)
    assignment_submission_rate: float = Field(..., ge=0, le=100)
    lms_login_frequency: float = Field(..., ge=0, le=14)
    avg_session_time: float = Field(..., ge=0, le=300)
    grades: float = Field(..., ge=0, le=100)
    semester: int = Field(..., ge=1, le=10)
    previous_backlog_count: int = Field(..., ge=0, le=20)


class ExplainOutput(BaseModel):
    predicted_probability: float
    risk_category: str
    top_risk_contributors: List[Dict[str, Any]]
    top_protective_contributors: List[Dict[str, Any]]
    explanation: str


class StudentOutput(BaseModel):
    anonymous_student_id: str = Field(..., alias="student_id", description="Pseudonymised id")
    institution_id: Optional[str] = None
    risk: Optional[int] = None
    probability: Optional[float] = None
    outcome_status: Optional[str] = Field(None, alias="outcome")

    class Config:
        allow_population_by_field_name = True
