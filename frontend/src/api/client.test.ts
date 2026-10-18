import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError } from './client'
import { SAMPLE_POSTING } from '../constants/form'

describe('API client', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('posts the exact job-posting payload to the backend prediction endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ final_decision: 'auto_decision_acceptable' }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await api.predict(SAMPLE_POSTING)
    expect(fetchMock).toHaveBeenCalledWith('http://127.0.0.1:8000/api/v1/predict', expect.objectContaining({
      method: 'POST', body: JSON.stringify(SAMPLE_POSTING)
    }))
  })

  it('turns a validation response into a readable error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 422 })))
    await expect(api.predict(SAMPLE_POSTING)).rejects.toMatchObject({ status: 422, message: 'Please review the required fields and try again.' })
  })

  it('turns connection failures into a readable error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')))
    await expect(api.predict(SAMPLE_POSTING)).rejects.toMatchObject({ status: null, message: 'Unable to reach TrustGuardAI. Check that the backend is running.' })
  })
})
