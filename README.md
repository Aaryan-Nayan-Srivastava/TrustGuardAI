# TrustGuard AI

TrustGuard AI is a two-stage job-posting fraud system. The original Streamlit UI remains in `app.py`; its inference logic now lives in `backend/` and is available as a FastAPI service for a separate frontend.

## Inference behavior preserved from the notebook and UI

Model A is an XGBoost binary classifier trained on structured metadata only. A request supplies seven binary indicators plus department, employment type, required experience, and required education. The service performs the same `pandas.get_dummies()` transform as `app.py`, then reindexes to the saved training feature list (26 columns, filling absent columns with `0`).

Model A class `1` is **potentially fake** and class `0` is **likely real**. Its class-1 score is thresholded at `0.50`. The service calculates entropy and the absolute score distance from `0.50`, then feeds Model B, in its saved feature order: `predicted_label`, `predicted_probability`, `confidence_margin`, `entropy`.

Model B class `1` means **human review recommended**; class `0` means **auto-decision acceptable**. Its class-1 escalation score is also thresholded at `0.50`. Scores are model probabilities, not confidence claims.

## Install and run

Use Python 3.12, then install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

The API is served at `http://127.0.0.1:8000`. OpenAPI docs remain available at `http://127.0.0.1:8000/docs` and the OpenAPI document at `/openapi.json`.

The legacy UI can still be started independently:

```powershell
streamlit run app.py
```

## React frontend

The separate React + TypeScript + Vite frontend lives in `frontend/`. It uses the real FastAPI contract only: `GET /health`, `GET /ready`, `POST /api/v1/predict`, `GET /api/v1/model-info`, and `GET /api/v1/evaluation`.

Start the backend first, then in a second terminal:

```powershell
cd frontend
Copy-Item .env.example .env
npm install
npm run dev
```

`VITE_API_BASE_URL` defaults to `http://127.0.0.1:8000`; set it in `frontend/.env` when the backend is served elsewhere. The backend's default CORS allowlist already includes Vite's `http://127.0.0.1:5173` and `http://localhost:5173` development origins. For production, set both the frontend API base URL and backend `CORS_ALLOW_ORIGINS` to the deployed frontend origin.

Build and test the frontend with:

```powershell
cd frontend
npm run test
npm run build
```

## Configuration

Copy `.env.example` values into your process environment (or configure them in your deployment). `CORS_ALLOW_ORIGINS` is a comma-separated allowlist and defaults to explicit local React/Vite origins. `TRUSTGUARD_MODEL_DIR` optionally points to the local trusted artifact directory; model uploads are not supported. `LOG_LEVEL` controls server diagnostics. No secrets are included in this project.

## API contracts

`GET /health` reports process liveness. `GET /ready` returns `200` only after both models and feature definitions load; it returns `503` if they are unavailable. These endpoints intentionally have no `/api/v1` prefix.

`POST /api/v1/predict` accepts one posting. All fields are required. Boolean fields correspond to the UI checkboxes. The four categorical strings are deliberately not enum-restricted because the established dummy-and-reindex pipeline safely handles unknown categories as the saved baseline/zero columns; `/api/v1/model-info` lists the UI’s offered options.

```json
{
  "has_company_logo": true,
  "has_questions": false,
  "has_salary_range": true,
  "experience_missing": false,
  "experience_not_applicable": false,
  "education_missing": false,
  "education_unspecified": false,
  "department": "engineering",
  "employment_type": "Full-time",
  "required_experience": "senior",
  "required_education": "bachelors"
}
```

Its response is typed and has this shape (numeric values vary by model output):

```json
{
  "model_a": {
    "predicted_label": 0,
    "classification": "likely_real_job",
    "fake_probability": 0.12,
    "real_probability": 0.88,
    "threshold": 0.5,
    "entropy": 0.37,
    "confidence_margin": 0.38
  },
  "model_b": {
    "needs_human_review": 0,
    "risk_output": "auto_decision_acceptable",
    "escalation_probability": 0.19,
    "threshold": 0.5
  },
  "final_decision": "auto_decision_acceptable"
}
```

`GET /api/v1/model-info` provides safe frontend metadata: labels, thresholds, feature counts, accepted input names, and the legacy UI category choices. It does not disclose artifact paths or serialized model contents.

`GET /api/v1/evaluation` returns JSON-compatible metrics saved in `model_b_eval_summary.pkl`, explicitly identified as `model_b_risk_classifier` metrics. The project does not implement separate batch, baseline-comparison, or standalone risk endpoints, so none were added.

Invalid bodies receive FastAPI’s `422` validation response. A valid request when models are unavailable receives `503`; preprocessing failures receive a sanitized `422`; internal diagnostic details are logged only on the server.

## Verification

Run:

```powershell
pytest -q
```

The tests compare the extracted pipeline with the pre-extraction `app.py` transform on a representative UI posting, exercise successful API prediction and validation, and verify `503` readiness/prediction behavior when artifacts are missing.
