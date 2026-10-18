import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { PredictionResults } from './PredictionResults'
import type { PredictionResponse } from '../types/api'

const result: PredictionResponse = {
  model_a: { predicted_label: 1, classification: 'potentially_fake_job', fake_probability: 0.72, real_probability: 0.28, threshold: 0.5, entropy: 0.59, confidence_margin: 0.22 },
  model_b: { needs_human_review: 1, risk_output: 'human_review_recommended', escalation_probability: 0.81, threshold: 0.5 },
  final_decision: 'human_review_recommended'
}

describe('PredictionResults', () => {
  it('renders returned model outputs with accurate score labels', () => {
    render(<PredictionResults result={result} onEdit={() => undefined} />)
    expect(screen.getByRole('heading', { name: /human review recommended/i })).toBeInTheDocument()
    expect(screen.getByText(/fake-job probability:.*72.0%/i)).toBeInTheDocument()
    expect(screen.getByText(/human-verification recommendation score:.*81.0%/i)).toBeInTheDocument()
    expect(screen.getByText(/not confidence ratings/i)).toBeInTheDocument()
  })
})
