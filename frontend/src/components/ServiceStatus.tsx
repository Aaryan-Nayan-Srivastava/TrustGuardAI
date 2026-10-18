import type { ServiceState } from '../types/api'

type Props = { state: ServiceState; onRetry: () => void }

export function ServiceStatus({ state, onRetry }: Props) {
  if (state === 'ready') return <p className="service-status ready" role="status">● Analysis service ready</p>
  if (state === 'checking') return <p className="service-status" role="status">● Checking analysis service…</p>
  return <div className="service-status unavailable" role="alert">
    <span>! Analysis service is unavailable.</span>
    <button className="link-button" type="button" onClick={onRetry}>Try again</button>
  </div>
}
