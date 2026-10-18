import type { EvaluationResponse, ModelInfoResponse } from '../types/api'

const label = (value: string) => value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())

export function ModelDetails({ modelInfo, evaluation }: { modelInfo: ModelInfoResponse | null; evaluation: EvaluationResponse | null }) {
  if (!modelInfo && !evaluation) return null
  return <section className="details-section" aria-labelledby="details-heading">
    <div><p className="eyebrow">Service details</p><h2 id="details-heading">Model information</h2></div>
    {modelInfo && <div className="model-details-grid">
      <article><h3>Model A</h3><p>{String(modelInfo.model_a.purpose ?? 'Job posting classification')}</p><p className="detail-line">Threshold: {String(modelInfo.model_a.threshold ?? '—')}</p></article>
      <article><h3>Model B</h3><p>{String(modelInfo.model_b.purpose ?? 'Decision layer')}</p><p className="detail-line">Threshold: {String(modelInfo.model_b.threshold ?? '—')}</p></article>
    </div>}
    {evaluation && <div className="evaluation"><h3>Saved evaluation metrics</h3><p>These describe {label(evaluation.model)} evaluation data, not the current posting.</p>
      <dl>{Object.entries(evaluation.metrics).map(([key, value]) => <div key={key}><dt>{label(key)}</dt><dd>{typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : String(value)}</dd></div>)}</dl>
    </div>}
  </section>
}
