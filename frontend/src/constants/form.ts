import type { JobPosting } from '../types/api'

export const SAMPLE_POSTING: JobPosting = {
  has_company_logo: true,
  has_questions: false,
  has_salary_range: true,
  experience_missing: false,
  experience_not_applicable: false,
  education_missing: false,
  education_unspecified: false,
  department: 'engineering',
  employment_type: 'Full-time',
  required_experience: 'senior',
  required_education: 'bachelors'
}

export const EMPTY_POSTING: JobPosting = {
  ...SAMPLE_POSTING,
  has_company_logo: false,
  has_salary_range: false,
  department: '',
  employment_type: '',
  required_experience: '',
  required_education: ''
}

export const FALLBACK_OPTIONS: Record<string, string[]> = {
  department: ['sales', 'engineering', 'marketing', 'operations', 'development', 'it', 'product', 'other'],
  employment_type: ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship', 'unknown', 'Other'],
  required_experience: ['internship', 'junior', 'senior', 'unspecified'],
  required_education: ['high School or equivalent', 'bachelors', 'masters', 'unspecified', 'other']
}

export const FIELD_LABELS: Record<keyof JobPosting, string> = {
  has_company_logo: 'Company logo present',
  has_questions: 'Screening questions present',
  has_salary_range: 'Salary range provided',
  experience_missing: 'Experience not specified',
  experience_not_applicable: 'Experience not applicable',
  education_missing: 'Education missing',
  education_unspecified: 'Education unspecified',
  department: 'Department',
  employment_type: 'Employment type',
  required_experience: 'Required experience',
  required_education: 'Required education'
}
