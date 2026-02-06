
# company logo - 0,1
# has questions- 0,1
# has salary range- 0,1
# experience missing- 0,1
# experience not applicable- 0,1
# education missing- 0,1
# education unspecified- 0,1

# Encoded features:
# Department - dropdown
# employement type- dropdown
# required experience- dropdown
# required education- dropdown

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import time

# # ----------------------------
# # Load models and artifacts
# # ----------------------------
# model_a = joblib.load("models/xgboost_model.pkl")
# features_a = joblib.load("models/feature_columns.pkl")

# model_b = joblib.load("models/model_b_risk_classifier.pkl")
# features_b = joblib.load("models/model_b_features.pkl")

# # threshold_b = joblib.load("models/model_b_threshold.pkl")

# # ----------------------------
# # UI
# # ----------------------------
# st.set_page_config(page_title="Job Fraud Detection", layout="centered")
# st.title(" 🔍 TrustGuard AI")
# st.markdown("### Job Posting Fraud Detection (Human-in-the-Loop)")
# st.markdown("""
# ####  About This System

# This application demonstrates a **two-layer machine learning system** designed for 
# **reliability-aware decision making**, not just prediction accuracy.

# - **Model A** performs the core task: classifying job postings as *real* or *potentially fake* 
#   using structured metadata features.
# - **Model B** acts as a *decision layer*, evaluating the **risk of trusting Model A’s prediction**
#   based on confidence, uncertainty, and prediction behavior.

# Instead of blindly trusting every model output, the system selectively escalates **high-risk or
# uncertain cases for human review**. This *human-in-the-loop* design reflects real-world ML
# deployments where **safety, trust, and accountability** matter as much as accuracy.

# The result is a system that not only predicts — but **knows when it might be wrong**.
# """)


# # st.title("Job Posting Fraud Detection (Human-in-the-Loop)")

# st.markdown("## Fill in the job posting metadata below.")

# with st.form("job_form"):
#     # Binary flags
#     has_company_logo = st.checkbox("Company logo present")
#     has_questions = st.checkbox("Screening questions present")
#     has_salary_range = st.checkbox("Salary range provided")
#     experience_missing = st.checkbox("Experience not specified")
#     experience_not_applicable = st.checkbox("Experience not applicable")
#     education_missing = st.checkbox("Education not specified")
#     education_unspecified = st.checkbox("Education unspecified")

#     # Dropdowns (must match training categories)
#     department = st.selectbox(
#         "Department",
#         ["sales", "engineering", "marketing", "operations",
#          "development", "it", "product", "other"]
#     )

#     employment_type = st.selectbox(
#         "Employment Type",
#         ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "unknown", "Other"]
#     )

#     required_experience = st.selectbox(
#         "Required Experience",
#         ["internship", "junior", "senior",  "unspecified"]
#     )

#     required_education = st.selectbox(
#         "Required Education",
#         ["high School or equivalent", "bachelors", "masters", "unspecified", "other"]
#     )

#     submit = st.form_submit_button("Analyze job posting")

# # ----------------------------
# # Inference
# # ----------------------------
# if submit:
#     # Build single-row dataframe
#     input_dict = {
#         "has_company_logo": int(has_company_logo),
#         "has_questions": int(has_questions),
#         "has_salary_range": int(has_salary_range),
#         "experience_missing": int(experience_missing),
#         "experience_not_applicable": int(experience_not_applicable),
#         "education_missing": int(education_missing),
#         "education_unspecified": int(education_unspecified),
#         "department": department,
#         "employment_type": employment_type,
#         "required_experience": required_experience,
#         "required_education": required_education,
#     }

#     X_input = pd.DataFrame([input_dict])

#     # One-hot encode
#     X_input = pd.get_dummies(X_input)

#     # Align with Model A features
#     X_input = X_input.reindex(columns=features_a, fill_value=0)

#     # ----------------------------
#     # Model A prediction
#     # ----------------------------
#     prob_fake = model_a.predict_proba(X_input)[0, 1]
#     pred_label = int(prob_fake >= 0.5)

#     # Uncertainty (entropy)
#     entropy = -(prob_fake * np.log(prob_fake + 1e-8) +
#                 (1 - prob_fake) * np.log(1 - prob_fake + 1e-8))

#     # ----------------------------
#     # Model B input
#     # ----------------------------
#     X_b = pd.DataFrame([{
#         "predicted_probability": prob_fake,
#         "confidence_margin": np.abs(prob_fake - 0.5),
#         "entropy": entropy,
#         "predicted_label": pred_label
#     }])

#     X_b = X_b.reindex(columns=features_b, fill_value=0)

#     risk_prob = model_b.predict_proba(X_b)[0, 1]
#     needs_human = int(risk_prob >= 0.5)

#     # ----------------------------
#     # Output
#     # ----------------------------
#     st.markdown("---")

#     if pred_label == 1:
#         st.error(f"Prediction: *Potentially Fake Job* (probability: {prob_fake:.2f})")
#     else:
#         st.success(f"Prediction: *Likely Real Job* (probability: {1 - prob_fake:.2f})")

#     if needs_human:
#         st.warning("⚠️ *Human review recommended* due to elevated risk.")
#     else:
#         st.info("✅ *Auto-decision acceptable*. Low risk detected.")

#     st.caption(
#         f"Escalation probability: {risk_prob:.2f} | Threshold: {0.5}"
#     )



# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import time

# # ----------------------------
# # Load models and artifacts
# # ----------------------------
# model_a = joblib.load("models/xgboost_model.pkl")
# features_a = joblib.load("models/feature_columns.pkl")

# model_b = joblib.load("models/model_b_risk_classifier.pkl")
# features_b = joblib.load("models/model_b_features.pkl")

# # ----------------------------
# # UI CONFIG
# # ----------------------------
# st.set_page_config(page_title="TrustGuard AI", layout="centered")

# st.markdown("## 🔍 TrustGuard AI")
# st.markdown("### Job Posting Fraud Detection (Human-in-the-Loop)")

# st.markdown("""
# #### About This System

# This application demonstrates a **two-layer machine learning system** designed for  
# **reliability-aware decision making**, not just prediction accuracy.

# - **Model A** classifies job postings as *real* or *potentially fake* using structured metadata.
# - **Model B** evaluates whether **Model A’s prediction should be trusted**, based on confidence
#   and uncertainty signals.

# Instead of blindly trusting every prediction, the system selectively escalates
# **high-risk or uncertain cases for human review**, reflecting real-world ML deployments
# where **trust, safety, and accountability** matter.
# """)

# st.markdown("---")
# st.markdown("## Fill in the job posting metadata")

# # ----------------------------
# # INPUT FORM
# # ----------------------------
# with st.form("job_form"):
#     # Binary features
#     has_company_logo = st.checkbox("Company logo present")
#     has_questions = st.checkbox("Screening questions present")
#     has_salary_range = st.checkbox("Salary range provided")
#     experience_missing = st.checkbox("Experience not specified")
#     experience_not_applicable = st.checkbox("Experience not applicable")
#     education_missing = st.checkbox("Education not specified")
#     education_unspecified = st.checkbox("Education unspecified")

#     # Categorical features (must match training)
#     department = st.selectbox(
#         "Department",
#         ["sales", "engineering", "marketing", "operations",
#          "development", "it", "product", "other"]
#     )

#     employment_type = st.selectbox(
#         "Employment Type",
#         ["Full-time", "Part-time", "Contract", "Temporary",
#          "Internship", "unknown", "Other"]
#     )

#     required_experience = st.selectbox(
#         "Required Experience",
#         ["internship", "junior", "senior", "unspecified"]
#     )

#     required_education = st.selectbox(
#         "Required Education",
#         ["high School or equivalent", "bachelors",
#          "masters", "unspecified", "other"]
#     )

#     submit = st.form_submit_button("Analyze job posting")

# # ----------------------------
# # INFERENCE
# # ----------------------------
# if submit:
#     # Build input dataframe
#     input_dict = {
#         "has_company_logo": int(has_company_logo),
#         "has_questions": int(has_questions),
#         "has_salary_range": int(has_salary_range),
#         "experience_missing": int(experience_missing),
#         "experience_not_applicable": int(experience_not_applicable),
#         "education_missing": int(education_missing),
#         "education_unspecified": int(education_unspecified),
#         "department": department,
#         "employment_type": employment_type,
#         "required_experience": required_experience,
#         "required_education": required_education,
#     }

#     X_input = pd.DataFrame([input_dict])

#     # One-hot encode + align
#     X_input = pd.get_dummies(X_input)
#     X_input = X_input.reindex(columns=features_a, fill_value=0)

#     # ----------------------------
#     # MODEL A (Prediction)
#     # ----------------------------
#     with st.spinner("🔍 Analyzing job posting metadata..."):
#         time.sleep(1.5)

#         prob_fake = model_a.predict_proba(X_input)[0, 1]
#         pred_label = int(prob_fake >= 0.5)

#         entropy = -(prob_fake * np.log(prob_fake + 1e-8) +
#                     (1 - prob_fake) * np.log(1 - prob_fake + 1e-8))

#     # ----------------------------
#     # MODEL B (Trust / Risk)
#     # ----------------------------
#     with st.spinner("🧠 Evaluating prediction reliability..."):
#         time.sleep(1.2)

#         X_b = pd.DataFrame([{
#             "predicted_probability": prob_fake,
#             "confidence_margin": abs(prob_fake - 0.5),
#             "entropy": entropy,
#             "predicted_label": pred_label
#         }])

#         X_b = X_b.reindex(columns=features_b, fill_value=0)

#         risk_prob = model_b.predict_proba(X_b)[0, 1]
#         needs_human = int(risk_prob >= 0.5)

#     # ----------------------------
#     # OUTPUT
#     # ----------------------------
#     st.markdown("---")

#     if pred_label == 1:
#         st.error(
#             f"Prediction: **Potentially Fake Job**  \n"
#             f"Fake probability: **{prob_fake:.2f}**"
#         )
#     else:
#         st.success(
#             f"Prediction: **Likely Real Job**  \n"
#             f"Real probability: **{1 - prob_fake:.2f}**"
#         )

#     if needs_human:
#         st.warning("⚠️ **Human review recommended** due to elevated risk.")
#     else:
#         st.info("✅ **Auto-decision acceptable**. Low risk detected.")

#     st.caption(
#         f"Escalation probability: {risk_prob:.2f} | Decision threshold: 0.50"
#     )


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ----------------------------
# Load models and artifacts
# ----------------------------
model_a = joblib.load("models/xgboost_model.pkl")
features_a = joblib.load("models/feature_columns.pkl")

model_b = joblib.load("models/model_b_risk_classifier.pkl")
features_b = joblib.load("models/model_b_features.pkl")

# ----------------------------
# UI CONFIG
# ----------------------------
st.set_page_config(page_title="TrustGuard AI", layout="centered")

st.markdown("## 🔍 TrustGuard AI")
st.markdown("### Job Posting Fraud Detection (Human-in-the-Loop)")

st.markdown("""
#### About This System

This application demonstrates a **two-layer machine learning system** designed for  
**reliability-aware decision making**, not just prediction accuracy.

- **Model A** predicts whether a job posting is *real* or *potentially fake*.
- **Model B** evaluates whether that prediction should be **trusted or escalated**.

Rather than blindly trusting automation, the system selectively routes uncertain or
high-risk predictions for **human review**, reflecting real-world ML deployment practices.
""")

st.markdown("---")
st.markdown("## Fill in the job posting metadata")

# ----------------------------
# INPUT FORM
# ----------------------------
with st.form("job_form"):
    has_company_logo = st.checkbox("Company logo present")
    has_questions = st.checkbox("Screening questions present")
    has_salary_range = st.checkbox("Salary range provided")
    experience_missing = st.checkbox("Experience not specified")
    experience_not_applicable = st.checkbox("Experience not applicable")
    education_missing = st.checkbox("Education missing")
    education_unspecified = st.checkbox("Education unspecified")

    department = st.selectbox(
        "Department",
        ["sales", "engineering", "marketing", "operations",
         "development", "it", "product", "other"]
    )

    employment_type = st.selectbox(
        "Employment Type",
        ["Full-time", "Part-time", "Contract", "Temporary",
         "Internship", "unknown", "Other"]
    )

    required_experience = st.selectbox(
        "Required Experience",
        ["internship", "junior", "senior", "unspecified"]
    )

    required_education = st.selectbox(
        "Required Education",
        ["high School or equivalent", "bachelors",
         "masters", "unspecified", "other"]
    )

    submit = st.form_submit_button("Analyze job posting")

# ----------------------------
# INFERENCE
# ----------------------------
if submit:
    # ----------------------------
    # Prepare input
    # ----------------------------
    input_dict = {
        "has_company_logo": int(has_company_logo),
        "has_questions": int(has_questions),
        "has_salary_range": int(has_salary_range),
        "experience_missing": int(experience_missing),
        "experience_not_applicable": int(experience_not_applicable),
        "education_missing": int(education_missing),
        "education_unspecified": int(education_unspecified),
        "department": department,
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
    }

    X_input = pd.DataFrame([input_dict])
    X_input = pd.get_dummies(X_input)
    X_input = X_input.reindex(columns=features_a, fill_value=0)

    # ----------------------------
    # MODEL A — Prediction
    # ----------------------------
    with st.spinner("🔍 Analyzing job posting metadata..."):
        time.sleep(1.5)

        prob_fake = model_a.predict_proba(X_input)[0, 1]
        pred_label = int(prob_fake >= 0.5)

        entropy = -(prob_fake * np.log(prob_fake + 1e-8) +
                    (1 - prob_fake) * np.log(1 - prob_fake + 1e-8))

    st.markdown("### 🔹 Model A Decision")

    if pred_label == 1:
        st.error(
            f"**Potentially Fake Job**  \n"
            f"Fake probability: **{prob_fake:.2f}**"
        )
    else:
        st.success(
            f"**Likely Real Job**  \n"
            f"Real probability: **{1 - prob_fake:.2f}**"
        )

    st.markdown("---")

    # ----------------------------
    # MODEL B — Trust Evaluation
    # ----------------------------
    with st.spinner("🧠 Evaluating prediction reliability..."):
        time.sleep(1.2)

        X_b = pd.DataFrame([{
            "predicted_probability": prob_fake,
            "confidence_margin": abs(prob_fake - 0.5),
            "entropy": entropy,
            "predicted_label": pred_label
        }])

        X_b = X_b.reindex(columns=features_b, fill_value=0)

        risk_prob = model_b.predict_proba(X_b)[0, 1]
        needs_human = int(risk_prob >= 0.5)

    st.markdown("### 🔹 Decision Layer Verdict")

    if needs_human:
        st.warning(
            f"⚠️ **Human review recommended**  \n"
            f"Escalation probability: **{risk_prob:.2f}**"
        )
    else:
        st.info(
            f"✅ **Auto-decision acceptable**  \n"
            f"Escalation probability: **{risk_prob:.2f}**"
        )

    st.caption("Decision threshold: 0.50")
