"""Loading and validating trusted, local model artifacts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when required local artifacts cannot provide the expected interface."""


@dataclass(frozen=True)
class ModelArtifacts:
    """The two trusted classifiers and their training-time feature definitions."""

    model_a: Any
    features_a: list[str]
    model_b: Any
    features_b: list[str]
    evaluation_summary: dict[str, Any] | None


def _load(path: Path) -> Any:
    if not path.is_file():
        raise ModelLoadError(f"Required model artifact is unavailable: {path.name}")
    # These files are bundled project artifacts. This service never accepts model uploads.
    return joblib.load(path)


def _feature_list(value: Any, artifact_name: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ModelLoadError(f"{artifact_name} has an invalid feature definition")
    return list(value)


def load_artifacts(model_dir: Path) -> ModelArtifacts:
    """Load all artifacts once and verify their inference interfaces."""
    try:
        model_a = _load(model_dir / "xgboost_model.pkl")
        features_a = _feature_list(_load(model_dir / "feature_columns.pkl"), "feature_columns.pkl")
        model_b = _load(model_dir / "model_b_risk_classifier.pkl")
        features_b = _feature_list(_load(model_dir / "model_b_features.pkl"), "model_b_features.pkl")
        raw_evaluation = _load(model_dir / "model_b_eval_summary.pkl")
    except ModelLoadError:
        raise
    except Exception as exc:
        logger.exception("Unable to load trusted model artifacts")
        raise ModelLoadError("Required model artifacts could not be loaded") from exc

    for name, model, features in (("Model A", model_a, features_a), ("Model B", model_b, features_b)):
        if not callable(getattr(model, "predict_proba", None)):
            raise ModelLoadError(f"{name} does not support probability prediction")
        expected_count = getattr(model, "n_features_in_", len(features))
        if expected_count != len(features):
            raise ModelLoadError(f"{name} feature definition does not match the model")

    evaluation_summary = raw_evaluation if isinstance(raw_evaluation, dict) else None
    return ModelArtifacts(model_a, features_a, model_b, features_b, evaluation_summary)
