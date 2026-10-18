import { type FormEvent, useState } from 'react'
import { EMPTY_POSTING, FIELD_LABELS, SAMPLE_POSTING } from '../constants/form'
import type { JobPosting } from '../types/api'

type Props = {
  categoryOptions: Record<string, string[]>
  disabled: boolean
  onSubmit: (posting: JobPosting) => void
}

type Errors = Partial<Record<keyof JobPosting, string>>

export function validatePosting(posting: JobPosting): Errors {
  const errors: Errors = {}
  ;(['department', 'employment_type', 'required_experience', 'required_education'] as const).forEach((field) => {
    if (!posting[field].trim()) errors[field] = `${FIELD_LABELS[field]} is required.`
  })
  return errors
}

const checkboxFields = [
  'has_company_logo', 'has_questions', 'has_salary_range',
  'experience_missing', 'experience_not_applicable', 'education_missing', 'education_unspecified'
] as const

const selects = [
  ['department', 'Department'], ['employment_type', 'Employment type'],
  ['required_experience', 'Required experience'], ['required_education', 'Required education']
] as const

export function JobPostingForm({ categoryOptions, disabled, onSubmit }: Props) {
  const [posting, setPosting] = useState<JobPosting>(EMPTY_POSTING)
  const [errors, setErrors] = useState<Errors>({})

  const update = <K extends keyof JobPosting>(field: K, value: JobPosting[K]) => {
    setPosting((current) => ({ ...current, [field]: value }))
    setErrors((current) => ({ ...current, [field]: undefined }))
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()
    const nextErrors = validatePosting(posting)
    setErrors(nextErrors)
    if (Object.keys(nextErrors).length === 0) onSubmit(posting)
  }

  const loadSample = () => {
    setPosting(SAMPLE_POSTING)
    setErrors({})
  }

  const reset = () => {
    setPosting(EMPTY_POSTING)
    setErrors({})
  }

  return <form className="analysis-form" onSubmit={submit} noValidate>
    <div className="form-heading">
      <div><p className="eyebrow">Posting metadata</p><h2>Describe the listing</h2></div>
      <div className="form-actions">
        <button className="text-button" type="button" onClick={loadSample} disabled={disabled}>Use sample</button>
        <button className="text-button" type="button" onClick={reset} disabled={disabled}>Reset</button>
      </div>
    </div>
    <p className="form-note">All fields are sent to the analysis service. This version analyzes structured metadata, not job-description text.</p>

    <fieldset>
      <legend>Listing signals</legend>
      <div className="checkbox-grid">
        {checkboxFields.slice(0, 3).map((field) => <label className="check-card" key={field}>
          <input type="checkbox" checked={posting[field]} onChange={(event) => update(field, event.target.checked)} disabled={disabled} />
          <span>{FIELD_LABELS[field]}</span>
        </label>)}
      </div>
    </fieldset>

    <fieldset>
      <legend>Requirements signals</legend>
      <div className="checkbox-grid">
        {checkboxFields.slice(3).map((field) => <label className="check-card" key={field}>
          <input type="checkbox" checked={posting[field]} onChange={(event) => update(field, event.target.checked)} disabled={disabled} />
          <span>{FIELD_LABELS[field]}</span>
        </label>)}
      </div>
    </fieldset>

    <fieldset>
      <legend>Role categories</legend>
      <div className="select-grid">
        {selects.map(([field, label]) => {
          const fieldName = field as keyof Pick<JobPosting, 'department' | 'employment_type' | 'required_experience' | 'required_education'>
          const error = errors[fieldName]
          return <div className="field" key={field}>
            <label htmlFor={field}>{label}</label>
            <select id={field} value={posting[fieldName]} aria-invalid={Boolean(error)} aria-describedby={error ? `${field}-error` : undefined}
              onChange={(event) => update(fieldName, event.target.value)} disabled={disabled}>
              <option value="">Select {label.toLowerCase()}</option>
              {(categoryOptions[field] ?? []).map((option) => <option value={option} key={option}>{option}</option>)}
            </select>
            {error && <span className="field-error" id={`${field}-error`}>{error}</span>}
          </div>
        })}
      </div>
    </fieldset>

    <button className="primary-button" type="submit" disabled={disabled}>
      {disabled ? 'Analyzing posting…' : 'Analyze job posting'} <span aria-hidden="true">→</span>
    </button>
  </form>
}
