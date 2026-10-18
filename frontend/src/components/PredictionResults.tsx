import type { PredictionResponse } from '../types/api'

const percentage = (value: number) => `${(value * 100).toFixed(1)}%`
const titleCase = (value: string) => value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())

export function PredictionResults({ result, onEdit }: { result: PredictionResponse; onEdit: () => void }) {
  const review = result.final_decision === 'human_review_recommended'
  return <section className="results" aria-live="polite" aria-labelledby="result-heading" tabIndex={-1}>
    <p className="eyebrow">Analysis complete</p>
    <div className={`decision-banner ${review ? 'review' : 'accepted'}`}>
      <span className="decision-icon" aria-hidden="true">{review ? '!' : '✓'}</span>
      <div><h2 id="result-heading">{titleCase(result.final_decision)}</h2>
        <p>{review ? 'Model B recommends additional human verification for this result. No human review has been performed or initiated automatically.' : 'Model B indicates that the automated result is acceptable under the existing decision-layer logic; this is not a human approval.'}</p>
      </div>
    </div>

    <div className="result-grid">
      <article className="metric-card"><p className="metric-label">Model A classification</p>
        <h3>{titleCase(result.model_a.classification)}</h3>
        <p>Fake-job probability: <strong>{percentage(result.model_a.fake_probability)}</strong></p>
      </article>
      <article className="metric-card"><p className="metric-label">Model B review recommendation</p>
        <h3>{titleCase(result.model_b.risk_output)}</h3>
        <p>Human-verification recommendation score: <strong>{percentage(result.model_b.escalation_probability)}</strong></p>
      </article>
    </div>
    <p className="result-note">These are model probability scores, not confidence ratings or explanations. A human-review recommendation is not an actual human intervention, and no listing is forwarded automatically.</p>
    <button className="secondary-button" type="button" onClick={onEdit}>Edit posting and analyze again</button>
  </section>
}
