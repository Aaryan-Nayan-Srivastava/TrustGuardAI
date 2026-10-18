import { useEffect, useState } from 'react'
import { api } from './api/client'
import { Footer } from './components/Footer'
import { Navigation } from './components/Navigation'
import { AnalysisPage } from './pages/AnalysisPage'
import { HomePage } from './pages/HomePage'
import type { ServiceState } from './types/api'

type Route = '/' | '/analyze'
const currentRoute = (): Route => window.location.pathname === '/analyze' ? '/analyze' : '/'

export default function App() {
  const [route, setRoute] = useState<Route>(currentRoute)
  const [renderedRoute, setRenderedRoute] = useState<Route>(currentRoute)
  const [transitionPhase, setTransitionPhase] = useState<'enter' | 'leave'>('enter')
  const [serviceState, setServiceState] = useState<ServiceState>('checking')
  const refreshService = async () => { setServiceState('checking'); setServiceState(await api.serviceState()) }
  const navigate = (nextRoute: Route, sectionId?: string) => { window.history.pushState({}, '', nextRoute); setRoute(nextRoute); if (sectionId) setTimeout(() => document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' }), 220) }

  useEffect(() => { const handlePopState = () => setRoute(currentRoute()); window.addEventListener('popstate', handlePopState); void refreshService(); return () => window.removeEventListener('popstate', handlePopState) }, [])
  useEffect(() => {
    if (route === renderedRoute) return
    setTransitionPhase('leave')
    const timer = window.setTimeout(() => { setRenderedRoute(route); window.scrollTo({ top: 0, behavior: 'auto' }); setTransitionPhase('enter') }, 180)
    return () => window.clearTimeout(timer)
  }, [route, renderedRoute])

  return <div className="app-shell"><Navigation route={route} serviceState={serviceState} onRetry={() => void refreshService()} onNavigate={navigate} /><div className={`page-transition page-transition-${transitionPhase}`}>{renderedRoute === '/' ? <HomePage onNavigate={navigate} /> : <AnalysisPage serviceState={serviceState} />}</div><Footer /></div>
}
