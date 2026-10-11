import { useEffect, useMemo, useRef, useState } from 'react'
import { prefersReducedMotion } from '../hooks.js'
import { shotColor } from '../shots.js'

/*
 * Animated "AI reading a shot" demo. A keyframed batsman skeleton plays a shot
 * while a detection HUD tracks him and the class probabilities settle.
 * Side-on view, bowler to the left. Bat angle = direction hands -> bat tip in
 * degrees (0 right, 90 down); angles are continuous so the swing direction is explicit.
 */

const J = ['head', 'neck', 'shF', 'shB', 'elF', 'elB', 'hands', 'hipF', 'hipB', 'knF', 'knB', 'anF', 'anB', 'toeF', 'toeB']

const STANCE = {
  head: [232, 92], neck: [236, 112], shF: [224, 126], shB: [246, 128], elF: [218, 168], elB: [252, 170],
  hands: [240, 200], hipF: [228, 205], hipB: [248, 207], knF: [220, 275], knB: [258, 277],
  anF: [214, 350], anB: [266, 350], toeF: [200, 356], toeB: [252, 356], bat: 80, batLen: 1,
}
const BACKLIFT = {
  ...STANCE, head: [230, 94], neck: [234, 114], shF: [222, 128], shB: [246, 130], elF: [234, 150],
  elB: [262, 158], hands: [262, 128], anF: [206, 344], toeF: [192, 350], bat: -62,
}

const SHOTS = [
  {
    name: 'Cover Drive',
    probs: { 'Cover Drive': 0.91, 'Straight Drive': 0.05, 'Leg Glance': 0.02, 'Cut Shot': 0.01, 'Pull Shot': 0.01 },
    keys: [
      [0, STANCE],
      [0.22, BACKLIFT],
      [0.45, { head: [196, 104], neck: [204, 124], shF: [192, 138], shB: [216, 138], elF: [210, 128], elB: [236, 140],
        hands: [236, 112], hipF: [214, 210], hipB: [234, 210], knF: [178, 272], knB: [252, 282], anF: [150, 350],
        anB: [272, 352], toeF: [136, 356], toeB: [258, 358], bat: -80, batLen: 1 }],
      [0.58, { head: [180, 116], neck: [190, 134], shF: [180, 148], shB: [204, 146], elF: [174, 180], elB: [204, 188],
        hands: [188, 214], hipF: [208, 214], hipB: [230, 214], knF: [172, 280], knB: [248, 286], anF: [150, 350],
        anB: [274, 352], toeF: [136, 356], toeB: [260, 358], bat: 20, batLen: 1 }],
      [0.68, { head: [170, 124], neck: [180, 142], shF: [172, 156], shB: [196, 152], elF: [162, 190], elB: [190, 206],
        hands: [168, 240], hipF: [206, 218], hipB: [228, 216], knF: [170, 282], knB: [246, 290], anF: [150, 350],
        anB: [276, 352], toeF: [136, 356], toeB: [262, 358], bat: 100, batLen: 1 }],
      [0.9, { head: [172, 116], neck: [182, 136], shF: [176, 150], shB: [198, 146], elF: [150, 118], elB: [172, 130],
        hands: [160, 96], hipF: [206, 214], hipB: [228, 214], knF: [170, 282], knB: [244, 292], anF: [150, 350],
        anB: [272, 350], toeF: [136, 356], toeB: [258, 356], bat: 235, batLen: 1 }],
      [1, { head: [174, 114], neck: [184, 134], shF: [178, 148], shB: [200, 144], elF: [152, 116], elB: [174, 128],
        hands: [162, 94], hipF: [208, 214], hipB: [230, 214], knF: [172, 282], knB: [246, 292], anF: [150, 350],
        anB: [272, 350], toeF: [136, 356], toeB: [258, 356], bat: 240, batLen: 1 }],
    ],
    // ball: [t, x, y, scale]
    ball: [[0.3, -20, 220, 1], [0.56, 92, 352, 1], [0.68, 150, 326, 1], [0.96, -60, 300, 2.2]],
  },
  {
    name: 'Pull Shot',
    probs: { 'Pull Shot': 0.88, 'Cut Shot': 0.06, 'Cover Drive': 0.03, 'Leg Glance': 0.02, 'Straight Drive': 0.01 },
    keys: [
      [0, STANCE],
      [0.22, { ...BACKLIFT, anB: [276, 350], toeB: [262, 356] }],
      [0.46, { head: [248, 100], neck: [250, 120], shF: [238, 134], shB: [262, 134], elF: [250, 150], elB: [278, 150],
        hands: [268, 116], hipF: [236, 210], hipB: [256, 210], knF: [222, 276], knB: [270, 280], anF: [206, 350],
        anB: [290, 350], toeF: [192, 356], toeB: [276, 356], bat: -95, batLen: 1 }],
      [0.58, { head: [246, 102], neck: [248, 122], shF: [236, 136], shB: [258, 136], elF: [228, 150], elB: [262, 154],
        hands: [240, 138], hipF: [236, 210], hipB: [256, 210], knF: [222, 276], knB: [270, 282], anF: [206, 350],
        anB: [290, 350], toeF: [192, 356], toeB: [276, 356], bat: -140, batLen: 1 }],
      [0.66, { head: [242, 102], neck: [246, 122], shF: [232, 136], shB: [252, 136], elF: [210, 152], elB: [236, 158],
        hands: [214, 150], hipF: [234, 210], hipB: [254, 210], knF: [222, 276], knB: [268, 282], anF: [206, 350],
        anB: [290, 350], toeF: [192, 356], toeB: [276, 356], bat: -180, batLen: 1 }],
      [0.77, { head: [242, 104], neck: [246, 124], shF: [236, 138], shB: [254, 138], elF: [226, 150], elB: [252, 156],
        hands: [236, 134], hipF: [236, 210], hipB: [256, 210], knF: [224, 276], knB: [268, 282], anF: [206, 350],
        anB: [290, 350], toeF: [192, 356], toeB: [276, 356], bat: -120, batLen: 0.4 }],
      [0.92, { head: [244, 104], neck: [248, 124], shF: [240, 138], shB: [258, 138], elF: [248, 120], elB: [270, 128],
        hands: [262, 104], hipF: [238, 210], hipB: [258, 210], knF: [224, 276], knB: [270, 282], anF: [206, 350],
        anB: [290, 350], toeF: [192, 356], toeB: [276, 356], bat: -35, batLen: 0.95 }],
      [1, { head: [244, 104], neck: [248, 124], shF: [240, 138], shB: [258, 138], elF: [250, 118], elB: [272, 126],
        hands: [264, 102], hipF: [238, 210], hipB: [258, 210], knF: [224, 276], knB: [270, 282], anF: [206, 350],
        anB: [290, 350], toeF: [192, 356], toeB: [276, 356], bat: -30, batLen: 1 }],
    ],
    ball: [[0.32, -20, 230, 1], [0.5, 70, 352, 1], [0.66, 142, 150, 1], [0.95, 520, 40, 0.55]],
  },
  {
    name: 'Straight Drive',
    probs: { 'Straight Drive': 0.86, 'Cover Drive': 0.09, 'Leg Glance': 0.03, 'Pull Shot': 0.01, 'Cut Shot': 0.01 },
    keys: [
      [0, STANCE],
      [0.22, BACKLIFT],
      [0.45, { head: [198, 102], neck: [206, 122], shF: [194, 136], shB: [216, 136], elF: [212, 126], elB: [236, 138],
        hands: [236, 108], hipF: [214, 208], hipB: [234, 208], knF: [180, 272], knB: [252, 282], anF: [156, 350],
        anB: [272, 352], toeF: [142, 356], toeB: [258, 358], bat: -84, batLen: 1 }],
      [0.58, { head: [184, 114], neck: [192, 132], shF: [182, 146], shB: [206, 144], elF: [180, 178], elB: [206, 186],
        hands: [194, 210], hipF: [208, 212], hipB: [230, 212], knF: [176, 280], knB: [248, 286], anF: [156, 350],
        anB: [274, 352], toeF: [142, 356], toeB: [260, 358], bat: 30, batLen: 1 }],
      [0.68, { head: [176, 120], neck: [184, 140], shF: [176, 154], shB: [198, 150], elF: [172, 186], elB: [194, 200],
        hands: [180, 236], hipF: [206, 216], hipB: [228, 216], knF: [174, 282], knB: [246, 290], anF: [156, 350],
        anB: [276, 352], toeF: [142, 356], toeB: [262, 358], bat: 92, batLen: 1 }],
      [0.9, { head: [178, 112], neck: [186, 132], shF: [180, 146], shB: [200, 142], elF: [170, 104], elB: [190, 118],
        hands: [178, 80], hipF: [206, 212], hipB: [228, 212], knF: [174, 282], knB: [244, 292], anF: [156, 350],
        anB: [272, 350], toeF: [142, 356], toeB: [258, 356], bat: 262, batLen: 1 }],
      [1, { head: [178, 110], neck: [186, 130], shF: [180, 144], shB: [200, 140], elF: [170, 102], elB: [190, 116],
        hands: [178, 78], hipF: [206, 212], hipB: [228, 212], knF: [174, 282], knB: [244, 292], anF: [156, 350],
        anB: [272, 350], toeF: [142, 356], toeB: [258, 356], bat: 266, batLen: 1 }],
    ],
    ball: [[0.3, -20, 222, 1], [0.56, 96, 352, 1], [0.68, 162, 330, 1], [0.97, -60, 344, 1]],
  },
]

const SHOT_SECONDS = 3.4
const HOLD_SECONDS = 1.4
const CYCLE = SHOT_SECONDS + HOLD_SECONDS
const IMPACT_T = 0.67
const BAT_LEN = 96
const CLASS_ORDER = ['Cover Drive', 'Straight Drive', 'Pull Shot', 'Cut Shot', 'Leg Glance']

const smooth = (x) => x * x * (3 - 2 * x)
const lerp = (a, b, t) => a + (b - a) * t

function poseAt(keys, t) {
  let i = 0
  while (i < keys.length - 2 && t > keys[i + 1][0]) i++
  const [t0, a] = keys[i]
  const [t1, b] = keys[i + 1]
  const u = smooth(Math.min(1, Math.max(0, (t - t0) / (t1 - t0 || 1))))
  const p = {}
  for (const k of J) p[k] = [lerp(a[k][0], b[k][0], u), lerp(a[k][1], b[k][1], u)]
  p.bat = lerp(a.bat, b.bat, u)
  p.batLen = lerp(a.batLen ?? 1, b.batLen ?? 1, u)
  return p
}

function ballAt(path, t) {
  if (t < path[0][0] || t > path[path.length - 1][0]) return null
  let i = 0
  while (i < path.length - 2 && t > path[i + 1][0]) i++
  const [t0, x0, y0, s0] = path[i]
  const [t1, x1, y1, s1] = path[i + 1]
  const u = (t - t0) / (t1 - t0)
  return { x: lerp(x0, x1, u), y: lerp(y0, y1, u), s: lerp(s0, s1, u) }
}

const bones = [
  ['head', 'neck'], ['neck', 'shF'], ['neck', 'shB'], ['shF', 'hipF'], ['shB', 'hipB'], ['hipF', 'hipB'],
  ['shF', 'elF'], ['elF', 'hands'], ['shB', 'elB'], ['elB', 'hands'],
  ['hipF', 'knF'], ['knF', 'anF'], ['anF', 'toeF'], ['hipB', 'knB'], ['knB', 'anB'], ['anB', 'toeB'],
]

export default function ShotAnimation() {
  const [clock, setClock] = useState(prefersReducedMotion() ? IMPACT_T * SHOT_SECONDS + 0.4 : 0)
  const wrapRef = useRef(null)

  useEffect(() => {
    if (prefersReducedMotion()) return
    let raf
    let visible = true
    let last = performance.now()
    const io = new IntersectionObserver(([e]) => (visible = e.isIntersecting))
    if (wrapRef.current) io.observe(wrapRef.current)
    const loop = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      if (visible && !document.hidden) setClock((c) => c + dt)
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => {
      cancelAnimationFrame(raf)
      io.disconnect()
    }
  }, [])

  const shotIdx = Math.floor(clock / CYCLE) % SHOTS.length
  const shot = SHOTS[shotIdx]
  const local = clock % CYCLE
  const t = Math.min(1, local / SHOT_SECONDS)
  const fade = local > CYCLE - 0.45 ? Math.max(0, (CYCLE - local) / 0.45) : local < 0.3 ? local / 0.3 : 1
  const p = poseAt(shot.keys, t)
  const ball = ballAt(shot.ball, t)
  const detected = t >= IMPACT_T
  const color = shotColor(shot.name)

  // Ball trail: the ball's own earlier positions (pure function of time)
  const trailPts = ball
    ? [7, 6, 5, 4, 3, 2, 1].map((k) => ballAt(shot.ball, t - k * 0.011)).filter(Boolean)
    : []

  const rad = (p.bat * Math.PI) / 180
  const tip = [p.hands[0] + Math.cos(rad) * BAT_LEN * p.batLen, p.hands[1] + Math.sin(rad) * BAT_LEN * p.batLen]
  const grip = [p.hands[0] + Math.cos(rad) * 26 * p.batLen, p.hands[1] + Math.sin(rad) * 26 * p.batLen]

  // Tracking box around body + bat
  const xs = [...J.map((k) => p[k][0]), tip[0]]
  const ys = [...J.map((k) => p[k][1]), tip[1]]
  // (the scene is drawn shifted 64 units left, so keep the box and its tag inside the frame)
  const box = { x0: Math.max(70, Math.min(...xs) - 22), y0: Math.min(...ys) - 30, x1: Math.max(...xs) + 22, y1: Math.max(...ys) + 8 }

  // Probability bars: noisy before impact, settle after
  const settle = detected ? smooth(Math.min(1, (t - IMPACT_T) / 0.2)) : 0
  const bars = useMemo(
    () => CLASS_ORDER.map((c, i) => ({ c, base: 0.12 + 0.1 * Math.abs(Math.sin(i * 1.7 + shotIdx)) })),
    [shotIdx],
  )

  const impactAge = (t - IMPACT_T) * SHOT_SECONDS
  const flash = impactAge > 0 && impactAge < 0.35 ? 1 - impactAge / 0.35 : 0
  const impactPt = shot.ball[2]

  return (
    <div className="shot-anim" ref={wrapRef} aria-label={`Animated demo: AI recognising a ${shot.name}`} role="img">
      <svg viewBox="0 0 480 400" className="shot-anim__svg">
        <defs>
          <radialGradient id="sa-ball" cx="35%" cy="35%" r="70%">
            <stop offset="0" stopColor="#ff9b8f" />
            <stop offset="0.6" stopColor="#e5384f" />
            <stop offset="1" stopColor="#8f1730" />
          </radialGradient>
        </defs>

        {/* Ground, crease and stumps */}
        <rect x="0" y="358" width="480" height="42" className="shot-anim__turf" />
        <line x1="0" y1="358" x2="480" y2="358" className="shot-anim__ground" />

        {/* Scene shifted left so the classifier panel never covers the batter or stumps */}
        <g transform="translate(-64 0)">
        <line x1="286" y1="358" x2="300" y2="372" className="shot-anim__crease" />
        <g className="shot-anim__stumps">
          <rect x="300" y="282" width="4" height="76" rx="2" />
          <rect x="309" y="282" width="4" height="76" rx="2" />
          <rect x="318" y="282" width="4" height="76" rx="2" />
          <rect x="298" y="279" width="26" height="3" rx="1.5" />
        </g>

        <g opacity={fade}>
          {/* Tracking box */}
          <g className={`shot-anim__box ${detected ? 'is-locked' : ''}`} style={{ '--shot': color }}>
            {[
              [box.x0, box.y0, 1, 1], [box.x1, box.y0, -1, 1], [box.x0, box.y1, 1, -1], [box.x1, box.y1, -1, -1],
            ].map(([x, y, sx, sy], i) => (
              <path key={i} d={`M ${x} ${y + 16 * sy} L ${x} ${y} L ${x + 16 * sx} ${y}`} />
            ))}
            {!detected && (
              <line x1={box.x0 + 4} x2={box.x1 - 4} y1={lerp(box.y0, box.y1, (local * 0.9) % 1)}
                y2={lerp(box.y0, box.y1, (local * 0.9) % 1)} className="shot-anim__scan" />
            )}
          </g>
          <g transform={`translate(${box.x0}, ${box.y0 - 30})`} className={`shot-anim__tag ${detected ? 'is-locked' : ''}`}>
            <rect width={detected ? 158 : 96} height="22" rx="11" />
            {detected && <circle cx="13" cy="11" r="4" style={{ fill: color }} />}
            <text x={detected ? 23 : 11} y="15">
              {detected ? `${shot.name.toUpperCase()}  ${Math.round(shot.probs[shot.name] * 100 * settle)}%` : 'ANALYSING'}
            </text>
          </g>

          {/* Body silhouette (soft) */}
          <g className="shot-anim__body">
            {bones.map(([a, b], i) => (
              <line key={i} x1={p[a][0]} y1={p[a][1]} x2={p[b][0]} y2={p[b][1]} />
            ))}
            <circle cx={p.head[0]} cy={p.head[1]} r="15" />
          </g>

          {/* Bat */}
          <g className="shot-anim__bat">
            <line x1={p.hands[0]} y1={p.hands[1]} x2={grip[0]} y2={grip[1]} className="shot-anim__handle" />
            <line x1={grip[0]} y1={grip[1]} x2={tip[0]} y2={tip[1]} className="shot-anim__blade" />
          </g>

          {/* Skeleton overlay */}
          <g>
            {bones.slice(1).map(([a, b], i) => (
              <line key={i} x1={p[a][0]} y1={p[a][1]} x2={p[b][0]} y2={p[b][1]} className="shot-anim__bone" />
            ))}
            {J.map((k) => (
              <circle key={k} cx={p[k][0]} cy={p[k][1]} r={k === 'head' ? 5 : 3.6} className="shot-anim__joint" />
            ))}
          </g>

          {/* Ball + trail */}
          {trailPts.map((b, i) => (
            <circle key={i} cx={b.x} cy={b.y} r={5.5 * b.s * ((i + 1) / trailPts.length)}
              className="shot-anim__trail" style={{ opacity: ((i + 1) / trailPts.length) * 0.35 }} />
          ))}
          {ball && <circle cx={ball.x} cy={ball.y} r={6.5 * ball.s} fill="url(#sa-ball)" />}
          {flash > 0 && (
            <circle cx={impactPt[1]} cy={impactPt[2]} r={10 + 30 * (1 - flash)} className="shot-anim__impact"
              opacity={flash} />
          )}
        </g>
        </g>
      </svg>

      {/* Live class probabilities */}
      <div className="shot-anim__probs clay clay--sm">
        <div className="shot-anim__probs-title">
          <span className={`dot ${detected ? 'dot--ok' : 'dot--warn'}`} /> Shot classifier
        </div>
        {bars.map(({ c, base }, i) => {
          const noisy = base + 0.08 * Math.sin(local * 7 + i * 2.1)
          const v = lerp(noisy, shot.probs[c], settle)
          const isTop = detected && c === shot.name
          return (
            <div key={c} className={`prob-row ${isTop ? 'is-top' : ''}`}>
              <span className="prob-row__label">{c}</span>
              <span className="prob-row__track">
                <span className="prob-row__fill"
                  style={{ width: `${Math.max(2, v * 100)}%`, background: isTop ? shotColor(c) : undefined }} />
              </span>
              <span className="prob-row__val">{Math.round(v * 100)}%</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
