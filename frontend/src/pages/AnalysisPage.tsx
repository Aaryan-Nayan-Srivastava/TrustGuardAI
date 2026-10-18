import { useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError } from '../api/client'
import { DemonstrationDisclaimer } from '../components/DemonstrationDisclaimer'
import { JobPostingForm } from '../components/JobPostingForm'
import { ModelDetails } from '../components/ModelDetails'
import { PredictionResults } from '../components/PredictionResults'
import { FALLBACK_OPTIONS } from '../constants/form'
import type { EvaluationResponse, JobPosting, ModelInfoResponse, PredictionResponse, ServiceState } from '../types/api'

export function AnalysisPage({ serviceState }: { serviceState: ServiceState }) {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const formRef = useRef<HTMLElement>(null)
  useEffect(() => { if (serviceState !== 'ready') return; void Promise.allSettled([api.modelInfo(), api.evaluation()]).then(([info, savedEvaluation]) => { if (info.status === 'fulfilled') setModelInfo(info.value); if (savedEvaluation.status === 'fulfilled') setEvaluation(savedEvaluation.value) }) }, [serviceState])
  const options = useMemo(() => ({ ...FALLBACK_OPTIONS, ...(modelInfo?.ui_category_options ?? {}) }), [modelInfo])
  const submit = async (posting: JobPosting) => { setError(null); setResult(null); setIsSubmitting(true); try { const prediction = await api.predict(posting); setResult(prediction); setTimeout(() => document.querySelector<HTMLElement>('.results')?.focus(), 0) } catch (caught) { setError(caught instanceof ApiError ? caught.message : 'An unexpected error occurred. Please try again.') } finally { setIsSubmitting(false) } }
  const edit = () => { setResult(null); setTimeout(() => formRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 0) }
  const explain = () => { window.history.pushState({}, '', '/'); window.dispatchEvent(new PopStateEvent('popstate')); setTimeout(() => document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' }), 220) }

  return <main className="analysis-page"><section className="analysis-page-heading"><div><p className="eyebrow">TrustGuard AI demonstration</p><h1>Job Posting Fraud Analysis</h1><p>Enter the details of a job listing to evaluate its fraud risk and determine whether additional human verification is recommended.</p></div><button className="how-it-works" type="button" onClick={explain}>How It Works <span aria-hidden="true">←</span></button></section><section className="analysis-notice"><DemonstrationDisclaimer /></section><section className="analysis-layout" ref={formRef}><div className="form-panel">{error && <div className="request-error" role="alert"><strong>Unable to complete analysis.</strong><span>{error}</span></div>}<JobPostingForm categoryOptions={options} disabled={isSubmitting || serviceState !== 'ready'} onSubmit={(posting) => void submit(posting)} />{serviceState !== 'ready' && <p className="availability-note">Analysis is disabled until the service is ready. Start the backend, then use “Try again” in the navigation.</p>}</div><aside className="analysis-aside"><p className="eyebrow">Before you begin</p><h2>What to enter</h2><p>Use metadata visible on the job listing and select the closest category.</p><hr /><h3>What this does not do</h3><p>This interface does not interpret job-description text or explain why an individual prediction was produced.</p></aside></section>{result && <PredictionResults result={result} onEdit={edit} />}<ModelDetails modelInfo={modelInfo} evaluation={evaluation} /></main>
}
