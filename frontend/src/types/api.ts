export type JobPosting = {
  has_company_logo: boolean
  has_questions: boolean
  has_salary_range: boolean
  experience_missing: boolean
  experience_not_applicable: boolean
  education_missing: boolean
  education_unspecified: boolean
  department: string
  employment_type: string
  required_experience: string
  required_education: string
}

export type PredictionResponse = {
  model_a: {
    predicted_label: 0 | 1
    classification: 'likely_real_job' | 'potentially_fake_job'
    fake_probability: number
    real_probability: number
    threshold: number
    entropy: number
    confidence_margin: number
  }
  model_b: {
    needs_human_review: 0 | 1
    risk_output: 'auto_decision_acceptable' | 'human_review_recommended'
    escalation_probability: number
    threshold: number
  }
  final_decision: 'auto_decision_acceptable' | 'human_review_recommended'
}

export type ModelInfoResponse = {
  model_a: Record<string, unknown>
  model_b: Record<string, unknown>
  input_fields: string[]
  ui_category_options: Record<string, string[]>
}

export type EvaluationResponse = {
  model: 'model_b_risk_classifier'
  metrics: Record<string, string | number | boolean | null>
}

export type ServiceState = 'checking' | 'ready' | 'unavailable'
