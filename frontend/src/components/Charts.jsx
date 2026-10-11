import { useEffect, useState } from 'react'
import { useCountUp } from '../hooks.js'
import { shotColor } from '../shots.js'

/** Animated ring gauge. value in [0, max]. */
export function RadialGauge({ value, max = 1, size = 140, stroke = 11, color = 'var(--ink)', label, format }) {
  const [shown, setShown] = useState(0)
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(value))
    return () => cancelAnimationFrame(id)
  }, [value])
  const num = useCountUp(value)
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const frac = Math.max(0, Math.min(1, shown / max))
  return (
    <div className="gauge" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} className="gauge__track" strokeWidth={stroke} />
        <circle
          cx={size / 2} cy={size / 2} r={r} className="gauge__value" strokeWidth={stroke}
          stroke={color} strokeDasharray={c} strokeDashoffset={c * (1 - frac)}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <div className="gauge__center">
        <span className="gauge__num">{format ? format(num) : num.toFixed(0)}</span>
        {label && <span className="gauge__label">{label}</span>}
      </div>
    </div>
  )
}

/** Horizontal probability bars, sorted high to low. */
export function ProbabilityBars({ probs, highlight }) {
  const [ready, setReady] = useState(false)
  useEffect(() => {
    const id = requestAnimationFrame(() => setReady(true))
    return () => cancelAnimationFrame(id)
  }, [])
  const rows = Object.entries(probs || {}).sort((a, b) => b[1] - a[1])
  return (
    <div className="pbars">
      {rows.map(([name, p], i) => (
        <div key={name} className={`pbars__row ${name === highlight ? 'is-top' : ''}`}>
          <span className="pbars__name">
            <span className="swatch" style={{ background: shotColor(name) }} />
            {name}
          </span>
          <span className="pbars__track">
            <span
              className="pbars__fill"
              style={{ width: ready ? `${Math.max(1, p * 100)}%` : '0%', background: shotColor(name),
                transitionDelay: `${i * 70}ms` }}
            />
          </span>
          <span className="pbars__val">{(p * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

/** Thin score bar 0..10 */
export function ScoreBar({ score }) {
  const [ready, setReady] = useState(false)
  useEffect(() => {
    const id = requestAnimationFrame(() => setReady(true))
    return () => cancelAnimationFrame(id)
  }, [])
  const tone = score >= 8 ? 'good' : score >= 5 ? 'ok' : 'bad'
  return (
    <span className="scorebar">
      <span className={`scorebar__fill scorebar__fill--${tone}`} style={{ width: ready ? `${score * 10}%` : '0%' }} />
    </span>
  )
}
