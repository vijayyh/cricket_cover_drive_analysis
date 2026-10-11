import { useEffect, useRef, useState } from 'react'
import { getResult, mediaUrl } from '../api.js'
import { SHOTS, pct, shotColor } from '../shots.js'
import { ProbabilityBars, RadialGauge, ScoreBar } from './Charts.jsx'
import { IconAlert, IconArrowLeft, IconDownload } from './Icons.jsx'

function Timeline({ timeline, duration, impactTime, videoRef }) {
  const [now, setNow] = useState(0)
  const [hover, setHover] = useState(null) // { frame, x }
  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    const on = () => setNow(v.currentTime)
    v.addEventListener('timeupdate', on)
    return () => v.removeEventListener('timeupdate', on)
  }, [videoRef])

  if (!duration) return null
  const fraction = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width))
  }
  const seek = (e) => {
    if (videoRef.current) videoRef.current.currentTime = Math.min(duration - 0.01, fraction(e) * duration)
  }
  // Hover snaps to the nearest classified frame (hit area = the whole track, not the thin bar).
  const onMove = (e) => {
    const t = fraction(e) * duration
    let best = null
    for (const f of timeline) if (!best || Math.abs(f.time - t) < Math.abs(best.time - t)) best = f
    setHover(best ? { frame: best, x: (best.time / duration) * 100 } : null)
  }
  const present = [...new Set(timeline.map((f) => f.shot_type))]

  return (
    <div className="timeline">
      <div className="timeline__head">
        <span>Frame-by-frame classification</span>
        <span className="timeline__hint">Click to jump · bar height = confidence</span>
      </div>
      <div className="timeline__track" onClick={seek} onMouseMove={onMove} onMouseLeave={() => setHover(null)}
        role="slider" aria-label="Video position" aria-valuemin={0} aria-valuemax={duration} aria-valuenow={now}
        tabIndex={0}
        onKeyDown={(e) => {
          const v = videoRef.current
          if (!v) return
          if (e.key === 'ArrowRight') v.currentTime = Math.min(duration, v.currentTime + 0.1)
          if (e.key === 'ArrowLeft') v.currentTime = Math.max(0, v.currentTime - 0.1)
        }}>
        {timeline.map((f) => (
          <span key={f.frame} className={`timeline__bar ${hover?.frame === f ? 'is-hover' : ''}`}
            style={{ left: `${(f.time / duration) * 100}%`, height: `${18 + f.confidence * 72}%`,
              background: shotColor(f.shot_type) }} />
        ))}
        {impactTime != null && (
          <span className="timeline__impact" style={{ left: `${(impactTime / duration) * 100}%` }}>
            <span>Impact</span>
          </span>
        )}
        <span className="timeline__playhead" style={{ left: `${(now / duration) * 100}%` }} />
        {hover && (
          <span className="tooltip" style={{ left: `${Math.min(86, Math.max(14, hover.x))}%` }}>
            <span className="tooltip__time">{hover.frame.time.toFixed(2)} s</span>
            <span className="tooltip__row">
              <span className="swatch" style={{ background: shotColor(hover.frame.shot_type) }} />
              {hover.frame.shot_type}
              <b>{pct(hover.frame.confidence)}</b>
            </span>
          </span>
        )}
      </div>
      <div className="legend">
        {present.map((s) => (
          <span key={s} className="legend__item">
            <span className="swatch" style={{ background: shotColor(s) }} />{s}
          </span>
        ))}
      </div>
    </div>
  )
}

function TechniqueCard({ technique }) {
  if (!technique) {
    return (
      <div className="card clay">
        <h3 className="card__title">Technique</h3>
        <p className="muted">No reliable body pose was found, so technique could not be measured. Try a clearer, side-on clip.</p>
      </div>
    )
  }
  return (
    <div className="card clay technique">
      <div className="technique__summary">
        <RadialGauge value={technique.overall} max={10} color="var(--ink)" size={150} label="out of 10"
          format={(n) => n.toFixed(1)} />
        <div>
          <span className="overline">Technique</span>
          <h3 className="technique__grade">{technique.grade}</h3>
          <p className="muted small">{technique.method}</p>
        </div>
      </div>
      <div className="technique__grid">
        {technique.aspects.map((a, i) => (
          <div key={a.aspect} className="aspect clay" style={{ animationDelay: `${i * 80}ms` }}>
            <div className="aspect__top">
              <span className="aspect__name">{a.aspect}</span>
              <span className="aspect__score">{a.score.toFixed(1)}<small>/10</small></span>
            </div>
            <ScoreBar score={a.score} />
            <div className="aspect__meta">
              <span>Measured <b>{a.measured}</b></span>
              <span>Target <b>{a.target}</b></span>
            </div>
            <p className="aspect__tip">{a.feedback}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ResultsView({ id, initial, onBack }) {
  const [result, setResult] = useState(initial && initial.id === id ? initial : null)
  const [error, setError] = useState('')
  const videoRef = useRef(null)

  useEffect(() => {
    if (result?.id === id) return
    setError('')
    getResult(id).then(setResult).catch((e) => setError(e.message))
  }, [id]) // eslint-disable-line react-hooks/exhaustive-deps

  // Braces matter: newer browsers return a Promise from scrollTo, and React treats an
  // effect's return value as its cleanup function.
  useEffect(() => {
    window.scrollTo({ top: 0 })
  }, [id])

  if (error) {
    return (
      <section className="container results">
        <button className="btn btn--clay" onClick={onBack}><IconArrowLeft /> Back</button>
        <div className="alert" role="alert"><IconAlert /> {error}</div>
      </section>
    )
  }
  if (!result) {
    return (
      <section className="container results">
        <div className="results__loading clay"><span className="spinner" /> Loading analysis…</div>
      </section>
    )
  }

  const { shot, technique, media, stats, timeline } = result
  const color = shotColor(shot.shot_type)
  const impactTime = media.fps ? (technique?.impact_frame ?? shot.key_frame ?? 0) / media.fps : null

  return (
    <section className="container results">
      <div className="results__bar">
        <button className="btn btn--clay" onClick={onBack}><IconArrowLeft /> New analysis</button>
        <div className="results__file">
          <strong title={result.source_name}>{result.source_name}</strong>
          <span>{new Date(result.created).toLocaleString()}</span>
        </div>
        <a className="btn btn--clay" href={mediaUrl(media.annotated)} download>
          <IconDownload /> Download {result.kind === 'video' ? 'video' : 'image'}
        </a>
      </div>

      <div className="results__grid">
        <div className="card clay results__media">
          {result.kind === 'video' ? (
            <>
              <div className="player">
                <video ref={videoRef} src={mediaUrl(media.annotated)} controls autoPlay muted loop playsInline />
              </div>
              <Timeline timeline={timeline} duration={media.duration} impactTime={impactTime} videoRef={videoRef} />
            </>
          ) : (
            <div className="player player--image">
              <img src={mediaUrl(media.annotated)} alt={`Annotated ${shot.shot_type}`} />
            </div>
          )}
        </div>

        <div className="results__side">
          <div className="card clay verdict" style={{ '--shot': color }}>
            <span className="overline">Detected shot</span>
            <div className="verdict__main">
              <RadialGauge value={shot.confidence} max={1} color={color} size={132} label="confidence"
                format={(n) => `${Math.round(n * 100)}%`} />
              <div>
                <h2 className="verdict__name">
                  <span className="swatch swatch--lg" style={{ background: color }} />
                  {shot.shot_type}
                </h2>
                <p className="muted small">{SHOTS[shot.shot_type]?.blurb}</p>
                <p className="verdict__runner">Runner-up: <b>{shot.runner_up}</b> ({pct(shot.all_probabilities[shot.runner_up] || 0)})</p>
              </div>
            </div>
            {shot.low_confidence && (
              <div className="alert alert--warn">
                <IconAlert />
                <span>Low confidence: the camera angle or framing may differ from the training footage. Treat this as a best guess.</span>
              </div>
            )}
          </div>

          <div className="card clay">
            <h3 className="card__title">All shot probabilities</h3>
            <ProbabilityBars probs={shot.all_probabilities} highlight={shot.shot_type} />
          </div>

          <div className="card clay stats">
            <div><span>{media.frames}</span><small>{result.kind === 'video' ? 'Frames' : 'Image'}</small></div>
            <div><span>{shot.frames_used}</span><small>Classified</small></div>
            <div><span>{pct(stats.pose_detected_ratio)}</span><small>Pose coverage</small></div>
            <div><span>{stats.processing_seconds.toFixed(1)}<i>s</i></span><small>Processing</small></div>
          </div>
        </div>
      </div>

      <TechniqueCard technique={technique} />

      {result.kind === 'video' && (
        <div className="card clay keyframe">
          <div>
            <span className="overline">Key moment</span>
            <h3 className="card__title">Impact frame{impactTime != null ? ` · ${impactTime.toFixed(2)}s` : ''}</h3>
            <p className="muted small">The centre of the stroke, where the gripped hands sweep through the biggest, fastest arc. Technique angles are measured around this moment.</p>
            <button className="btn btn--clay btn--sm" onClick={() => {
              if (videoRef.current && impactTime != null) {
                videoRef.current.currentTime = impactTime
                videoRef.current.pause()
                videoRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
              }
            }}>Jump to impact in video</button>
          </div>
          <img src={mediaUrl(media.key_frame)} alt="Impact frame with pose overlay" loading="lazy" />
        </div>
      )}
    </section>
  )
}
