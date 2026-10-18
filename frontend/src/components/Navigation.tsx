import { ServiceStatus } from './ServiceStatus'
import type { ServiceState } from '../types/api'
type Route = '/' | '/analyze'
type Props = { route: Route; serviceState: ServiceState; onRetry: () => void; onNavigate: (route: Route, sectionId?: string) => void }
export function Navigation({ route, serviceState, onRetry, onNavigate }: Props) {
  const link = (target: Route, label: string) => <a href={target} className={route === target ? 'nav-link active' : 'nav-link'} onClick={(event) => { event.preventDefault(); onNavigate(target) }}>{label}</a>
  return <header className="site-header"><a className="brand" href="/" aria-label="TrustGuard AI home" onClick={(event) => { event.preventDefault(); onNavigate('/') }}><span aria-hidden="true">◈</span> TrustGuard<span>AI</span></a><nav className="site-nav" aria-label="Primary navigation">{link('/', 'Home')}{link('/analyze', 'Analyze Job')}</nav><ServiceStatus state={serviceState} onRetry={onRetry} /></header>
}
