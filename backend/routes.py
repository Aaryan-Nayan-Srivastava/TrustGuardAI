"""HTTP routes; all business logic remains in backend.inference."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from .inference import InferenceError, MODEL_A_THRESHOLD, MODEL_B_THRESHOLD, predict
from .preprocessing import INPUT_FIELDS
from .schemas import (
    EvaluationResponse, HealthResponse, JobPostingRequest, ModelInfoResponse,
    ModelAPrediction, ModelBRisk, PredictionResponse, ReadyResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

UI_CATEGORY_OPTIONS = {
    "department": ["sales", "engineering", "marketing", "operations", "development", "it", "product", "other"],
    "employment_type": ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "unknown", "Other"],
    "required_experience": ["internship", "junior", "senior", "unspecified"],
    "required_education": ["high School or equivalent", "bachelors", "masters", "unspecified", "other"],
}


def _artifacts_or_503(request: Request):
    artifacts = getattr(request.app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Inference service is not ready")
    return artifacts


@router.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse, tags=["service"])
def ready(request: Request) -> ReadyResponse:
    if getattr(request.app.state, "artifacts", None) is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Required inference models are unavailable")
    return ReadyResponse(status="ready", models_loaded=True)


@router.post("/api/v1/predict", response_model=PredictionResponse, tags=["inference"])
def make_prediction(payload: JobPostingRequest, request: Request) -> PredictionResponse:
    try:
        result = predict(payload.model_dump(), _artifacts_or_503(request))
    except InferenceError:
        logger.exception("Prediction pipeline failed")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="The posting could not be processed") from None
    classification = "potentially_fake_job" if result.predicted_label else "likely_real_job"
    decision = "human_review_recommended" if result.needs_human_review else "auto_decision_acceptable"
    return PredictionResponse(
        model_a=ModelAPrediction(
            predicted_label=result.predicted_label,
            classification=classification,
            fake_probability=result.fake_probability,
            real_probability=1 - result.fake_probability,
            threshold=MODEL_A_THRESHOLD,
            entropy=result.entropy,
            confidence_margin=result.confidence_margin,
        ),
        model_b=ModelBRisk(
            needs_human_review=result.needs_human_review,
            risk_output=decision,
            escalation_probability=result.escalation_probability,
            threshold=MODEL_B_THRESHOLD,
        ),
        final_decision=decision,
    )


@router.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["metadata"])
def model_info(request: Request) -> ModelInfoResponse:
    artifacts = _artifacts_or_503(request)
    return ModelInfoResponse(
        model_a={
            "purpose": "classify job postings as likely real or potentially fake",
            "labels": {"0": "likely_real_job", "1": "potentially_fake_job"},
            "score": "fake_probability",
            "threshold": MODEL_A_THRESHOLD,
            "feature_count": len(artifacts.features_a),
        },
        model_b={
            "purpose": "recommend auto-decision or human review for Model A output",
            "labels": {"0": "auto_decision_acceptable", "1": "human_review_recommended"},
            "score": "escalation_probability",
            "threshold": MODEL_B_THRESHOLD,
            "feature_count": len(artifacts.features_b),
            "input_features": artifacts.features_b,
        },
        input_fields=list(INPUT_FIELDS),
        ui_category_options=UI_CATEGORY_OPTIONS,
    )


def _json_metrics(value: Any) -> dict[str, int | float | str | bool | None] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, int | float | str | bool | None] = {}
    for key, item in value.items():
        if isinstance(item, (str, bool, int, float)) or item is None:
            result[str(key)] = item
        elif hasattr(item, "item"):
            scalar = item.item()
            if isinstance(scalar, (str, bool, int, float)) or scalar is None:
                result[str(key)] = scalar
    return result


@router.get("/api/v1/evaluation", response_model=EvaluationResponse, tags=["metadata"])
def evaluation(request: Request) -> EvaluationResponse:
    metrics = _json_metrics(_artifacts_or_503(request).evaluation_summary)
    if metrics is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No JSON-compatible evaluation summary is available")
    return EvaluationResponse(model="model_b_risk_classifier", metrics=metrics)
