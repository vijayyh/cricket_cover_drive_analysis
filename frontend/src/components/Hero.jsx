import { useCountUp } from '../hooks.js'
import { IconUpload } from './Icons.jsx'
import HeroVideo from './HeroVideo.jsx'

const STATUS = {
  ok: ['dot--ok', 'Model online'],
  degraded: ['dot--warn', 'Server up · model not trained yet'],
  offline: ['dot--off', 'Analysis server offline'],
  connecting: ['dot--warn', 'Connecting…'],
}

function Spec({ value, unit, label, decimals = 0 }) {
  const v = useCountUp(value, value != null)
  return (
    <div className="spec">
      <span className="spec__value">
        {value == null ? '–' : v.toFixed(decimals)}
        {unit && <span className="spec__unit">{unit}</span>}
      </span>
      <span className="spec__label">{label}</span>
    </div>
  )
}

export default function Hero({ health, onAnalyze, onModel }) {
  const acc = health?.holdout_accuracy != null ? health.holdout_accuracy * 100 : null
  const [dot, statusText] = STATUS[health?.status ?? 'connecting']

  return (
    <section className="hero container" id="top">
      <div className="hero__copy">
        <span className="overline">
          <span className={`dot ${dot}`} />
          {statusText}
        </span>
        <h1 className="hero__title">Every shot,<br />measured.</h1>
        <p className="hero__lead">
          Upload a clip or a photo of a batter. CricAI tracks 33 body landmarks, finds the moment of the stroke,
          names the shot and scores the technique, angle by angle.
        </p>
        <div className="hero__cta">
          <button className="btn btn--ink btn--lg" onClick={onAnalyze}>
            <IconUpload /> Analyze a shot
          </button>
          <button className="btn btn--clay btn--lg" onClick={onModel}>View accuracy</button>
        </div>
        <div className="specs">
          <Spec value={acc} unit="%" label="Accuracy on unseen clips" />
          <Spec value={5} label="Shot types" />
          <Spec value={33} label="Body landmarks" />
        </div>
      </div>

      <div className="hero__visual clay">
        <HeroVideo />
      </div>
    </section>
  )
}
