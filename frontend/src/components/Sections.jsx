import { useEffect, useState } from 'react'
import { getModelReport, getResults, mediaUrl } from '../api.js'
import { useReveal } from '../hooks.js'
import { pct, shotColor } from '../shots.js'
import { RadialGauge } from './Charts.jsx'
import { IconBrain, IconPose, IconTarget, IconWave } from './Icons.jsx'

const STEPS = [
  { icon: IconPose, title: 'Pose tracking', text: 'MediaPipe tracks 33 landmarks on every frame: shoulders, elbows, wrists, hips, knees and feet.' },
  { icon: IconWave, title: 'Swing detection', text: 'Hand speed is measured frame to frame to find the swing, so stance and walk-back frames don’t dilute the verdict.' },
  { icon: IconBrain, title: 'Two-model fusion', text: 'A fine-tuned ResNet-18 reads the batter’s shape while a pose-geometry model reads joint angles. Their votes are fused.' },
  { icon: IconTarget, title: 'Technique scoring', text: 'Knee flex, head position, arm extension, balance and follow-through are measured and compared with targets for that shot.' },
]

function StepCard({ s, i }) {
  const Icon = s.icon
  return (
    <div className="how__card clay" style={{ '--i': i }}>
      <span className="how__num">0{i + 1}</span>
      <span className="how__icon"><Icon width={22} height={22} /></span>
      <h3>{s.title}</h3>
      <p>{s.text}</p>
    </div>
  )
}

export function HowItWorks() {
  const reveal = useReveal()
  return (
    <section className="section container reveal" id="how" ref={reveal}>
      <div className="section__head">
        <span className="overline">How it works</span>
        <h2>From raw clip to coaching notes</h2>
      </div>
      <div className="how">
        {STEPS.map((s, i) => <StepCard key={s.title} s={s} i={i} />)}
      </div>
    </section>
  )
}

function Confusion({ matrix, classes }) {
  const max = Math.max(...matrix.flat(), 1)
  const short = (c) => c.replace(' Drive', ' Dr.').replace(' Shot', '').replace('Leg Glance', 'Glance')
  return (
    <div className="cm" style={{ '--n': classes.length }}>
      <span className="cm__corner">true ↓ / predicted →</span>
      {classes.map((c) => <span key={`h${c}`} className="cm__head">{short(c)}</span>)}
      {matrix.map((row, i) => (
        <div key={classes[i]} className="cm__row">
          <span className="cm__label">{short(classes[i])}</span>
          {row.map((v, j) => (
            <span key={j} className={`cm__cell ${i === j ? 'is-diag' : ''}`}
              title={`${classes[i]} predicted as ${classes[j]}: ${v}`}
              style={{ '--a': v / max, '--c': i === j ? shotColor(classes[i]) : 'var(--ink-3)' }}>
              {v}
            </span>
          ))}
        </div>
      ))}
    </div>
  )
}

export function ModelReport() {
  const reveal = useReveal()
  const [report, setReport] = useState(null)
  const [error, setError] = useState(false)
  useEffect(() => {
    getModelReport().then(setReport).catch(() => setError(true))
  }, [])

  const h = report?.holdout
  return (
    <section className="section container reveal" id="model" ref={reveal}>
      <div className="section__head">
        <span className="overline">Accuracy, measured honestly</span>
        <h2>Tested on clips the model never saw</h2>
        <p>
          Many dataset photos are neighbouring frames of the same video, and each comes with 6 augmented copies.
          Photos are grouped by source clip and a whole clip stays on one side of the train/test split, so the
          score below is never inflated by the model recognising a frame it has effectively already seen.
        </p>
      </div>

      {error && <div className="card clay muted">The model report isn’t available. Run <code>python train_shot_classifier.py</code>.</div>}
      {!report && !error && <div className="card clay"><span className="spinner" /> Loading report…</div>}

      {report && (
        <div className="model">
          <div className="model__scores">
            {[
              ['Fused model', h.fused.accuracy, 'var(--ink)', 'What the app uses'],
              ['Pose only', h.pose_only_on_detected.accuracy, 'var(--ink-3)', 'Joint geometry, when detected'],
              ['CNN only', h.cnn_only.accuracy, 'var(--ink-3)', 'ResNet-18 on the image'],
            ].map(([name, acc, color, note]) => (
              <div key={name} className="model__score clay">
                <RadialGauge value={acc} max={1} color={color} size={118} stroke={10}
                  format={(n) => `${Math.round(n * 100)}%`} />
                <div>
                  <strong>{name}</strong>
                  <span className="muted small">{note}</span>
                </div>
              </div>
            ))}
            <div className="model__score model__score--old clay">
              <span className="model__old-num">≈30%</span>
              <div>
                <strong>Previous model</strong>
                <span className="muted small">Trained on synthetic numbers; chance is 20%</span>
              </div>
            </div>
          </div>

          <div className="model__detail">
            <div className="card clay">
              <h3 className="card__title">Confusion matrix · held-out photos</h3>
              <Confusion matrix={h.fused.confusion_matrix} classes={report.dataset.classes} />
            </div>
            <div className="card clay">
              <h3 className="card__title">Per-shot F1 score</h3>
              <div className="pbars">
                {Object.entries(h.fused.per_class_f1).map(([c, f1]) => (
                  <div key={c} className="pbars__row">
                    <span className="pbars__name"><span className="swatch" style={{ background: shotColor(c) }} />{c}</span>
                    <span className="pbars__track"><span className="pbars__fill" style={{ width: `${f1 * 100}%`, background: shotColor(c) }} /></span>
                    <span className="pbars__val">{f1.toFixed(2)}</span>
                  </div>
                ))}
              </div>
              <dl className="model__facts">
                <div><dt>Photos · source clips</dt><dd>{report.dataset.original_photos} · {report.dataset.source_clips ?? '—'}</dd></div>
                <div><dt>Held-out test photos</dt><dd>{h.fused.n}</dd></div>
                <div><dt>Pose found in</dt><dd>{pct(report.dataset.pose_detected_fraction)}</dd></div>
                <div><dt>Pose weight in fusion</dt><dd>{report.fusion_pose_weight.toFixed(2)}</dd></div>
              </dl>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

export function History({ refreshKey, onOpen }) {
  const reveal = useReveal()
  const [items, setItems] = useState(null)
  useEffect(() => {
    getResults(9).then(setItems).catch(() => setItems([]))
  }, [refreshKey])

  return (
    <section className="section container reveal" id="history" ref={reveal}>
      <div className="section__head">
        <span className="overline">History</span>
        <h2>Recent analyses</h2>
      </div>
      {items && items.length === 0 && (
        <div className="card clay muted center">Nothing here yet. Your analyses will appear here.</div>
      )}
      <div className="history">
        {(items || []).map((r) => (
          <button key={r.id} className="history__item clay" onClick={() => onOpen(r.id)}>
            <span className="history__thumb">
              <img src={mediaUrl(r.thumbnail)} alt="" loading="lazy" />
              <span className="history__kind">{r.kind}</span>
            </span>
            <span className="history__body">
              <strong><span className="swatch" style={{ background: shotColor(r.shot_type) }} />{r.shot_type}</strong>
              <span className="muted small">{pct(r.confidence)} confidence{r.grade ? ` · ${r.grade}` : ''}</span>
              <span className="history__name" title={r.source_name}>{r.source_name}</span>
            </span>
          </button>
        ))}
      </div>
    </section>
  )
}
