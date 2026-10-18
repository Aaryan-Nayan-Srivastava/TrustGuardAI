"""Public API contracts for the TrustGuard AI service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


EXAMPLE_POSTING = {
    "has_company_logo": True,
    "has_questions": False,
    "has_salary_range": True,
    "experience_missing": False,
    "experience_not_applicable": False,
    "education_missing": False,
    "education_unspecified": False,
    "department": "engineering",
    "employment_type": "Full-time",
    "required_experience": "senior",
    "required_education": "bachelors",
}


class JobPostingRequest(BaseModel):
    """Structured metadata accepted by the existing TrustGuard UI."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_POSTING})
    has_company_logo: bool = Field(description="Whether a company logo is present.")
    has_questions: bool = Field(description="Whether screening questions are present.")
    has_salary_range: bool = Field(description="Whether a salary range is provided.")
    experience_missing: bool = Field(description="Whether experience is not specified.")
    experience_not_applicable: bool = Field(description="Whether experience is not applicable.")
    education_missing: bool = Field(description="Whether education is missing.")
    education_unspecified: bool = Field(description="Whether education is marked unspecified.")
    department: str = Field(min_length=1, max_length=200, description="Department category used by the UI.")
    employment_type: str = Field(min_length=1, max_length=200, description="Employment-type category used by the UI.")
    required_experience: str = Field(min_length=1, max_length=200, description="Required-experience category used by the UI.")
    required_education: str = Field(min_length=1, max_length=200, description="Required-education category used by the UI.")


class ModelAPrediction(BaseModel):
    predicted_label: Literal[0, 1] = Field(description="0 means likely real; 1 means potentially fake.")
    classification: Literal["likely_real_job", "potentially_fake_job"]
    fake_probability: float = Field(ge=0, le=1, description="Model A class-1 score (fake-job probability).")
    real_probability: float = Field(ge=0, le=1, description="One minus Model A's fake-job probability.")
    threshold: float = Field(description="Model A classification threshold.")
    entropy: float = Field(ge=0, description="Entropy computed from Model A's fake-job probability.")
    confidence_margin: float = Field(ge=0, description="Absolute distance of Model A's score from 0.50.")


class ModelBRisk(BaseModel):
    needs_human_review: Literal[0, 1] = Field(description="1 means Model B recommends escalation.")
    risk_output: Literal["auto_decision_acceptable", "human_review_recommended"]
    escalation_probability: float = Field(ge=0, le=1, description="Model B class-1 score for escalation.")
    threshold: float = Field(description="Model B escalation threshold.")


class PredictionResponse(BaseModel):
    model_a: ModelAPrediction
    model_b: ModelBRisk
    final_decision: Literal["auto_decision_acceptable", "human_review_recommended"]


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    models_loaded: bool


class ModelInfoResponse(BaseModel):
    model_a: dict[str, Any]
    model_b: dict[str, Any]
    input_fields: list[str]
    ui_category_options: dict[str, list[str]]


class EvaluationResponse(BaseModel):
    model: Literal["model_b_risk_classifier"]
    metrics: dict[str, int | float | str | bool | None]
