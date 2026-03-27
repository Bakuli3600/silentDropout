from pydantic import BaseModel, Field
from typing import List

class PredictionInput(BaseModel):
    attendance_rate: float = Field(..., ge=0, le=100, description="Attendance rate (0-100)")
    assignment_submission_rate: float = Field(..., ge=0, le=100, description="Assignment submission rate (0-100)")
    lms_login_frequency: float = Field(..., ge=0, le=14, description="LMS logins per week (0-14)")
    avg_session_time: float = Field(..., ge=0, le=300, description="Average session time in minutes")
    grades: float = Field(..., ge=0, le=100, description="Current grade average (0-100)")

class PredictionOutput(BaseModel):
    risk: int = Field(..., description="Dropout risk: 1 (High Risk), 0 (Low Risk)")
    probability: float = Field(..., description="Probability of dropout (0.0 to 1.0)")
    explanation: List[str] = Field(default=[], description="Top contributing factors for the prediction")
