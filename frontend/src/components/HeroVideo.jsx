import { useEffect, useRef, useState } from 'react'
import { shotColor } from '../shots.js'

/*
 * Hero demo: a real batting clip with CricAI's live output drawn over it.
 * The clip, poster and per-frame data (pose joints + shot probabilities) are
 * produced by `python scripts/make_hero_clip.py --video <clip>` and live in /public/hero.
 */

const BASE = '/hero/'
const CLASS_ORDER = ['Cover Drive', 'Straight Drive', 'Pull Shot', 'Cut Shot', 'Leg Glance']
// Joint order written by make_hero_clip.py:
// 0 nose, 1/2 shoulders, 3/4 elbows, 5/6 wrists, 7/8 hips, 9/10 knees, 11/12 ankles
const BONES = [[1, 2], [1, 7], [2, 8], [7, 8], [1, 3], [3, 5], [2, 4], [4, 6], [7, 9], [9, 11], [8, 10], [10, 12]]
const MIN_VIS = 0.3

function Overlay({ meta, frame, showPose }) {
  const W = meta.width
  const H = meta.height
  const joints = meta.joints[frame]
  const locked = frame >= meta.center - 2
  const color = shotColor(meta.verdict.shot_type)
  const pt = (j) => [joints[j][0] * W, joints[j][1] * H]
  const vis = (j) => joints && joints[j][2] > MIN_VIS

  let box = null
  if (joints) {
    const xs = []
    const ys = []
    joints.forEach((j, i) => {
      if (vis(i)) {
        xs.push(j[0] * W)
        ys.push(j[1] * H)
      }
    })
    if (xs.length > 4) {
      const x0 = Math.max(10, Math.min(...xs) - W * 0.1)
      const x1 = Math.min(W - 10, Math.max(...xs) + W * 0.1)
      const y0 = Math.max(54, Math.min(...ys) - H * 0.09)
      const y1 = Math.min(H - 14, Math.max(...ys) + H * 0.05)
      box = { x0, y0, x1, y1 }
    }
  }

  const sinceImpact = frame - meta.center
  const pulse = sinceImpact >= 0 && sinceImpact < 14 ? 1 - sinceImpact / 14 : 0
  const hands = joints && vis(5) && vis(6) ? [(pt(5)[0] + pt(6)[0]) / 2, (pt(5)[1] + pt(6)[1]) / 2] : null
  const tagText = locked
    ? `${meta.verdict.shot_type.toUpperCase()}  ${Math.round(meta.verdict.confidence * 100)}%`
    : 'TRACKING'
  const tagW = locked ? 12 + tagText.length * 7.4 + 14 : 92

  return (
    <svg className="hv__overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
      {showPose && joints && (
        <g className="hv__pose">
          {BONES.map(([a, b]) => vis(a) && vis(b) && (
            <line key={`${a}-${b}`} x1={pt(a)[0]} y1={pt(a)[1]} x2={pt(b)[0]} y2={pt(b)[1]} />
          ))}
          {joints.map((_, i) => i > 0 && vis(i) && <circle key={i} cx={pt(i)[0]} cy={pt(i)[1]} r="3.6" />)}
        </g>
      )}
      {box && (
        <g className={`hv__box ${locked ? 'is-locked' : ''}`}>
          {[[box.x0, box.y0, 1, 1], [box.x1, box.y0, -1, 1], [box.x0, box.y1, 1, -1], [box.x1, box.y1, -1, -1]]
            .map(([x, y, sx, sy], i) => <path key={i} d={`M ${x} ${y + 18 * sy} L ${x} ${y} L ${x + 18 * sx} ${y}`} />)}
          <g transform={`translate(${Math.min(box.x0, W - tagW - 10)}, ${box.y0 - 30})`} className="hv__tag">
            <rect width={tagW} height="22" rx="11" />
            {locked && <circle cx="13" cy="11" r="4" style={{ fill: color }} />}
            <text x={locked ? 23 : 12} y="15">{tagText}</text>
          </g>
        </g>
      )}
      {pulse > 0 && hands && (
        <circle className="hv__impact" cx={hands[0]} cy={hands[1]} r={8 + 34 * (1 - pulse)} opacity={pulse} />
      )}
    </svg>
  )
}

function Probabilities({ meta, frame }) {
  const row = meta.probs[frame]
  const values = Object.fromEntries(meta.classes.map((c, i) => [c, row[i]]))
  const top = meta.classes[row.indexOf(Math.max(...row))]
  const locked = frame >= meta.center - 2
  return (
    <div className="hv__probs clay clay--sm">
      <div className="hv__probs-title">
        <span className={`dot ${locked ? 'dot--ok' : 'dot--warn'}`} />
        Frame-by-frame classifier
      </div>
      {CLASS_ORDER.map((c) => {
        const isTop = c === top
        return (
          <div key={c} className={`prob-row ${isTop ? 'is-top' : ''}`}>
            <span className="prob-row__label">{c}</span>
            <span className="prob-row__track">
              <span className="prob-row__fill"
                style={{ width: `${Math.max(2, values[c] * 100)}%`, background: isTop && locked ? shotColor(c) : undefined }} />
            </span>
            <span className="prob-row__val">{Math.round(values[c] * 100)}%</span>
          </div>
        )
      })}
    </div>
  )
}

export default function HeroVideo() {
  const [meta, setMeta] = useState(null)
  const [frame, setFrame] = useState(0)
  const [showPose, setShowPose] = useState(true)
  const [playing, setPlaying] = useState(false)
  const videoRef = useRef(null)
  const userPaused = useRef(false)

  useEffect(() => {
    fetch(`${BASE}hero.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setMeta)
      .catch(() => setMeta(null))
  }, [])

  // Keep the overlay in step with the video, frame by frame where the browser supports it.
  useEffect(() => {
    const v = videoRef.current
    if (!v || !meta) return
    let stopped = false
    let handle = 0
    let raf = 0
    const update = (t) => setFrame(Math.min(meta.frames - 1, Math.max(0, Math.floor(t * meta.fps + 0.001))))
    if ('requestVideoFrameCallback' in v) {
      const cb = (_now, m) => {
        if (stopped) return
        update(m.mediaTime)
        handle = v.requestVideoFrameCallback(cb)
      }
      handle = v.requestVideoFrameCallback(cb)
    } else {
      const loop = () => {
        if (stopped) return
        update(v.currentTime)
        raf = requestAnimationFrame(loop)
      }
      raf = requestAnimationFrame(loop)
    }
    return () => {
      stopped = true
      if (handle && v.cancelVideoFrameCallback) v.cancelVideoFrameCallback(handle)
      cancelAnimationFrame(raf)
    }
  }, [meta])

  // Play while on screen (pausing off-screen saves CPU), unless the viewer paused it.
  // The clip is the product demo, so it plays even with reduced motion; the pause
  // button below gives everyone control over it.
  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    const on = () => setPlaying(!v.paused)
    v.addEventListener('play', on)
    v.addEventListener('pause', on)
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !userPaused.current) v.play().catch(() => {})
      else if (!e.isIntersecting) v.pause()
    }, { threshold: 0.25 })
    io.observe(v)
    return () => {
      io.disconnect()
      v.removeEventListener('play', on)
      v.removeEventListener('pause', on)
    }
  }, [])

  const togglePlay = () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) {
      userPaused.current = false
      v.play().catch(() => {})
    } else {
      userPaused.current = true
      v.pause()
    }
  }

  const progress = meta ? (frame + 1) / meta.frames : 0
  return (
    <div className="hv-card">
      <div className="hv-card__head">
        <span className="overline overline--muted"><span className="dot dot--rec" /> Live demo · real clip</span>
        <button type="button" className={`switch ${showPose ? 'is-on' : ''}`} onClick={() => setShowPose((s) => !s)}
          aria-pressed={showPose}>
          <span className="switch__track"><span className="switch__thumb" /></span>
          Pose overlay
        </button>
      </div>
      <div className="hv" style={{ aspectRatio: meta ? `${meta.width} / ${meta.height}` : '4 / 5' }}>
        <video ref={videoRef} muted loop playsInline preload="auto" poster={`${BASE}hero.jpg`} autoPlay
          aria-label="Batter playing a shot, with CricAI tracking overlay">
          <source src={`${BASE}hero.webm`} type="video/webm" />
          <source src={`${BASE}hero.mp4`} type="video/mp4" />
        </video>
        {meta && <Overlay meta={meta} frame={frame} showPose={showPose} />}
        <button type="button" className="hv__play" onClick={togglePlay}
          aria-label={playing ? 'Pause demo video' : 'Play demo video'}>
          {playing ? (
            <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
          ) : (
            <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5Z" /></svg>
          )}
        </button>
        {meta && (
          <div className="hv__progress" aria-hidden="true">
            <span className="hv__progress-fill" style={{ width: `${progress * 100}%` }} />
            <span className="hv__progress-mark" style={{ left: `${((meta.center + 0.5) / meta.frames) * 100}%` }} />
          </div>
        )}
      </div>
      {meta && <Probabilities meta={meta} frame={frame} />}
    </div>
  )
}
