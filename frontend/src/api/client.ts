import type { EvaluationResponse, JobPosting, ModelInfoResponse, PredictionResponse, ServiceState } from '../types/api'

const baseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(public readonly status: number | null, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${baseUrl}${path}`, {
      headers: { 'Content-Type': 'application/json', ...init?.headers },
      ...init
    })
  } catch {
    throw new ApiError(null, 'Unable to reach TrustGuardAI. Check that the backend is running.')
  }

  if (!response.ok) {
    if (response.status === 422) throw new ApiError(422, 'Please review the required fields and try again.')
    if (response.status === 503) throw new ApiError(503, 'The analysis service is not ready. Please retry shortly.')
    throw new ApiError(response.status, 'The service could not complete this request. Please try again.')
  }
  return response.json() as Promise<T>
}

export const api = {
  predict: (posting: JobPosting) => request<PredictionResponse>('/api/v1/predict', {
    method: 'POST', body: JSON.stringify(posting)
  }),
  modelInfo: () => request<ModelInfoResponse>('/api/v1/model-info'),
  evaluation: () => request<EvaluationResponse>('/api/v1/evaluation'),
  async serviceState(): Promise<ServiceState> {
    try {
      const health = await request<{ status: string }>('/health')
      if (health.status !== 'ok') return 'unavailable'
      const ready = await request<{ status: string; models_loaded: boolean }>('/ready')
      return ready.status === 'ready' && ready.models_loaded ? 'ready' : 'unavailable'
    } catch {
      return 'unavailable'
    }
  }
}
