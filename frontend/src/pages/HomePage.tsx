import { DemonstrationDisclaimer } from '../components/DemonstrationDisclaimer'
import { HumanInTheLoop } from '../components/HumanInTheLoop'
import { VerificationVisual } from '../components/VerificationVisual'

type Route = '/' | '/analyze'

export function HomePage({ onNavigate }: { onNavigate: (route: Route, sectionId?: string) => void }) {
  return <main id="top">
    <section className="hero home-hero">
      <div><p className="eyebrow">AI-assisted fraud detection</p><h1>TrustGuard AI</h1><p className="hero-tagline">AI-Powered Job Fraud Detection with Human-in-the-Loop Decision Making</p><p className="hero-copy">TrustGuard AI is an AI-assisted job fraud detection system designed to analyze job-posting metadata and identify potentially fraudulent listings. It combines machine learning with a Human-in-the-Loop decision framework to determine when automated predictions should be accepted and when additional human verification is recommended.</p><button className="primary-button hero-action" type="button" onClick={() => onNavigate('/analyze')}>Start Analysis <span aria-hidden="true">→</span></button></div>
      <aside className="hero-panel" aria-label="Job verification workflow"><VerificationVisual /><p>Decision workflow</p><ol><li><span>01</span> Analyze job metadata.</li><li><span>02</span> Classify and assess review need.</li><li><span>03</span> Present a clear recommendation.</li></ol></aside>
    </section>
    <section className="home-section two-column-section" aria-labelledby="about-heading"><div><p className="eyebrow">About the project</p><h2 id="about-heading">What is TrustGuard AI?</h2></div><div className="prose"><p>Fraudulent online job postings can make it difficult to assess whether an opportunity is trustworthy. TrustGuard AI uses structured details visible in a job listing—rather than job-description text—to identify patterns associated with suspicious postings.</p><p>Automated predictions are useful signals, but they are not always sufficient on their own. The system therefore also indicates when an additional human verification step may be appropriate.</p></div></section>
    <section className="home-section hitl-explainer" aria-labelledby="what-hitl-heading"><p className="eyebrow">Human judgment and AI</p><h2 id="what-hitl-heading">What is Human-in-the-Loop?</h2><p>Human-in-the-Loop (HITL) is an approach where human judgment is incorporated into an AI-assisted decision-making process. Instead of allowing AI to make every decision independently, certain cases are flagged for additional human verification.</p><p>In TrustGuard AI, the system uses a second machine-learning model to identify predictions that may require human review. This demonstrates how automated decision-making can be combined with human oversight.</p><p className="prototype-note"><strong>In this prototype:</strong> the system provides a recommendation for review. It does not connect to a reviewer or carry out human verification.</p></section>
    <div id="how-it-works"><HumanInTheLoop /></div>
    <section className="home-notice"><DemonstrationDisclaimer /></section>
    <section className="home-cta"><div><p className="eyebrow">Try the demonstration</p><h2>Ready to Analyze a Job Posting?</h2><p>Enter structured job-listing details to see the fraud classification and review recommendation.</p></div><button className="primary-button" type="button" onClick={() => onNavigate('/analyze')}>Try TrustGuard AI <span aria-hidden="true">→</span></button></section>
  </main>
}
