"""Regression tests for the extracted two-stage inference pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.inference import predict
from backend.main import create_app
from backend.model_loader import load_artifacts
from backend.schemas import EXAMPLE_POSTING


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def artifacts():
    return load_artifacts(ROOT / "models")


def legacy_prediction(posting: dict):
    """The app.py algorithm before extraction, retained here as a regression oracle."""
    model_a = joblib.load(ROOT / "models" / "xgboost_model.pkl")
    features_a = joblib.load(ROOT / "models" / "feature_columns.pkl")
    model_b = joblib.load(ROOT / "models" / "model_b_risk_classifier.pkl")
    features_b = joblib.load(ROOT / "models" / "model_b_features.pkl")
    input_frame = pd.get_dummies(pd.DataFrame([posting])).reindex(columns=features_a, fill_value=0)
    fake_probability = float(model_a.predict_proba(input_frame)[0, 1])
    predicted_label = int(fake_probability >= 0.5)
    entropy = -(fake_probability * np.log(fake_probability + 1e-8) + (1 - fake_probability) * np.log(1 - fake_probability + 1e-8))
    model_b_frame = pd.DataFrame([{
        "predicted_probability": fake_probability,
        "confidence_margin": abs(fake_probability - 0.5),
        "entropy": entropy,
        "predicted_label": predicted_label,
    }]).reindex(columns=features_b, fill_value=0)
    escalation_probability = float(model_b.predict_proba(model_b_frame)[0, 1])
    return fake_probability, predicted_label, entropy, escalation_probability


def test_extracted_pipeline_matches_original_app(artifacts):
    result = predict(EXAMPLE_POSTING, artifacts)
    fake_probability, predicted_label, entropy, escalation_probability = legacy_prediction(EXAMPLE_POSTING)
    assert result.predicted_label == predicted_label
    assert result.needs_human_review == int(escalation_probability >= 0.5)
    assert result.fake_probability == pytest.approx(fake_probability)
    assert result.entropy == pytest.approx(entropy)
    assert result.escalation_probability == pytest.approx(escalation_probability)


def test_ready_and_prediction_contract():
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready", "models_loaded": True}
        response = client.post("/api/v1/predict", json=EXAMPLE_POSTING)
        assert response.status_code == 200
        body = response.json()
        assert body["model_a"]["classification"] in {"likely_real_job", "potentially_fake_job"}
        assert body["model_b"]["risk_output"] == body["final_decision"]
        assert 0 <= body["model_a"]["fake_probability"] <= 1


def test_request_validation_rejects_missing_required_field():
    with TestClient(create_app()) as client:
        incomplete = dict(EXAMPLE_POSTING)
        incomplete.pop("department")
        assert client.post("/api/v1/predict", json=incomplete).status_code == 422


def test_readiness_and_prediction_fail_when_artifacts_are_missing(tmp_path):
    settings = Settings(project_root=ROOT, model_dir=tmp_path, cors_origins=[], log_level="ERROR")
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/api/v1/predict", json=EXAMPLE_POSTING).status_code == 503
