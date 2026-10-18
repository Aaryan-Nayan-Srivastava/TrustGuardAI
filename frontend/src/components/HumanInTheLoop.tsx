const steps = [
  {
    title: 'Job Metadata Input',
    model: 'Step 1',
    description: 'The user provides structured details about a job posting.'
  },
  {
    title: 'Model A: Fraud Classification',
    model: 'Step 2',
    description: 'The first machine-learning model predicts whether the posting is genuine or potentially fraudulent.'
  },
  {
    title: 'Model B: Review Assessment',
    model: 'Step 3',
    description: 'The second model assesses whether the prediction can be accepted automatically or flagged for human verification.'
  },
  {
    title: 'Decision Output',
    model: 'Step 4',
    description: 'The system displays the fraud prediction and whether human review is recommended.'
  }
]

export function HumanInTheLoop() {
  return <section className="hitl-section" aria-labelledby="hitl-heading">
    <div className="section-intro">
      <p className="eyebrow">Decision workflow</p>
      <h2 id="hitl-heading">How TrustGuard AI Works</h2>
      <p>Each stage presents a focused signal: a fraud classification, a review assessment, and a clear output for the user.</p>
    </div>
    <ol className="hitl-workflow">
      {steps.map((step, index) => <li key={step.title} className="hitl-step">
        <span className="step-number">0{index + 1}</span>
        <div className="step-icon" aria-hidden="true">{index === 0 ? 'AI' : index === 1 ? '?' : index === 2 ? 'H' : '✓'}</div>
        <div><p className="step-model">{step.model}</p><h3>{step.title}</h3><p>{step.description}</p></div>
      </li>)}
    </ol>
    <p className="hitl-note"><strong>Important:</strong> The human-review recommendation is produced by the existing Model B logic. It indicates that verification is recommended; it is not evidence that a human has reviewed the listing.</p>
  </section>
}
