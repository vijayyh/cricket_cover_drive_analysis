// Shot metadata shared across the UI. Colours reference tokens.css.
export const SHOTS = {
  'Cover Drive': {
    color: 'var(--shot-cover)',
    blurb: 'Front-foot drive through the off side, high front elbow, full face of the bat.',
  },
  'Straight Drive': {
    color: 'var(--shot-straight)',
    blurb: 'Front-foot drive back past the bowler with a vertical bat and long follow-through.',
  },
  'Pull Shot': {
    color: 'var(--shot-pull)',
    blurb: 'Back-foot, cross-bat shot to a short ball, rolling the wrists to the leg side.',
  },
  'Cut Shot': {
    color: 'var(--shot-cut)',
    blurb: 'Back-foot, horizontal-bat shot to a short, wide ball, slapped square on the off side.',
  },
  'Leg Glance': {
    color: 'var(--shot-glance)',
    blurb: 'Deft deflection off the pads, using the pace of the ball to guide it fine on the leg side.',
  },
}

export const shotColor = (name) => SHOTS[name]?.color ?? 'var(--ink-3)'

export const pct = (x, digits = 0) => `${(x * 100).toFixed(digits)}%`
