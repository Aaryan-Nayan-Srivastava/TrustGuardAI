"""Training-compatible input transforms for job posting metadata."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


INPUT_FIELDS = (
    "has_company_logo",
    "has_questions",
    "has_salary_range",
    "experience_missing",
    "experience_not_applicable",
    "education_missing",
    "education_unspecified",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
)


def prepare_model_a_input(posting: Mapping[str, object], feature_columns: list[str]) -> pd.DataFrame:
    """Match the Streamlit inference transform: dummies followed by feature alignment."""
    row = {field: posting[field] for field in INPUT_FIELDS}
    # The Streamlit app explicitly converted each checkbox to int before DataFrame creation.
    for field in INPUT_FIELDS[:7]:
        row[field] = int(row[field])
    # Deliberately no drop_first: the existing UI uses this exact transform at inference;
    # reindexing against the saved, drop_first training columns discards baseline columns.
    encoded = pd.get_dummies(pd.DataFrame([row]))
    return encoded.reindex(columns=feature_columns, fill_value=0)
