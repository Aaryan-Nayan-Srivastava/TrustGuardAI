"""Framework-independent two-model prediction and decision pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .model_loader import ModelArtifacts
from .preprocessing import prepare_model_a_input

MODEL_A_THRESHOLD = 0.5
MODEL_B_THRESHOLD = 0.5


class InferenceError(RuntimeError):
    """Raised for input transformation or trusted-model inference failures."""


@dataclass(frozen=True)
class InferenceResult:
    predicted_label: int
    fake_probability: float
    entropy: float
    confidence_margin: float
    needs_human_review: int
    escalation_probability: float


def _class_one_probability(model: object, data: pd.DataFrame) -> float:
    probabilities = model.predict_proba(data)
    return float(probabilities[0, 1])


def predict(posting: Mapping[str, object], artifacts: ModelArtifacts) -> InferenceResult:
    """Run the exact Model A -> uncertainty signals -> Model B pipeline from app.py."""
    try:
        model_a_input = prepare_model_a_input(posting, artifacts.features_a)
        fake_probability = _class_one_probability(artifacts.model_a, model_a_input)
        predicted_label = int(fake_probability >= MODEL_A_THRESHOLD)
        entropy = -(
            fake_probability * math.log(fake_probability + 1e-8)
            + (1 - fake_probability) * math.log(1 - fake_probability + 1e-8)
        )
        confidence_margin = abs(fake_probability - 0.5)
        model_b_input = pd.DataFrame([{
            "predicted_probability": fake_probability,
            "confidence_margin": confidence_margin,
            "entropy": entropy,
            "predicted_label": predicted_label,
        }]).reindex(columns=artifacts.features_b, fill_value=0)
        escalation_probability = _class_one_probability(artifacts.model_b, model_b_input)
        needs_human_review = int(escalation_probability >= MODEL_B_THRESHOLD)
        return InferenceResult(
            predicted_label=predicted_label,
            fake_probability=fake_probability,
            entropy=float(entropy),
            confidence_margin=float(confidence_margin),
            needs_human_review=needs_human_review,
            escalation_probability=escalation_probability,
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise InferenceError("The prediction pipeline could not process this posting") from exc
