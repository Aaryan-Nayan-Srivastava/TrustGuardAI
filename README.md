# TrustGuard AI

## Fake Job Posting Detection with Human-in-the-Loop Decision Layer

This project demonstrates a two-stage machine learning system for detecting fraudulent job postings, enhanced with a decision layer that determines when a model’s prediction should be trusted and when human review is required.

The goal is not only to classify job postings as real or fake, but also to explicitly model uncertainty and reduce the risk of incorrect automated decisions.

---

## Project Overview

The system is composed of two models:

### Model A: Job Posting Fraud Classifier
- Predicts whether a job posting is **real or fake**
- Uses **structured metadata only**
- No NLP or text embeddings are used
- Trained using XGBoost

### Model B: Risk and Escalation Model
- Does **not** predict fraud
- Predicts whether Model A’s output should be:
  - Automatically accepted, or
  - Sent for human review
- Learns from Model A’s confidence and uncertainty signals

This design reflects how real-world ML systems are deployed in safety-critical settings.

---

## Why a Two-Model Design?

Most ML projects stop at accuracy. This project goes further by answering:

- How confident is the model?
- When is the model likely to be wrong?
- When should a human intervene?

Model B acts as a **decision policy**, not a classifier of the original problem.

---

## Features Used (No NLP)

The system uses only structured job posting metadata:

### Binary Indicators
- Company logo present
- Screening questions present
- Salary range provided
- Experience missing
- Experience not applicable
- Education missing
- Education unspecified

### Categorical Features (One-Hot Encoded)
- Department
- Employment type
- Required experience
- Required education

Baseline categories are handled using `drop_first=True`.

---

## Model A Details

- Algorithm: XGBoost (binary classification)
- Output:
  - Fraud probability
  - Predicted label (real / fake)

From these outputs, uncertainty metrics are derived:
- Prediction entropy
- Confidence margin

---

## Model B Details

- Algorithm: XGBoost (binary classification)
- Input features:
  - Model A predicted probability
  - Entropy
  - Predicted label
- Target:
  - `needs_human_review`

A case is escalated if:
- Model A is wrong, or
- Model A is highly uncertain

---

## Evaluation Summary (Model B)

- Recall on risky cases: ~98%
- False negatives are kept minimal
- Threshold sensitivity is low, indicating stable behavior

The system prioritizes safety over raw accuracy.

---

## Live Demo

A Streamlit application is included to demonstrate the full pipeline:

1. User inputs job posting metadata
2. Model A predicts fraud probability
3. Uncertainty is computed
4. Model B decides whether human review is required

This simulates a real deployment scenario.

---



