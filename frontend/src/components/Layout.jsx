import { useEffect, useState } from 'react'

export function Wordmark({ small = false }) {
  return (
    <span className={`wordmark ${small ? 'wordmark--sm' : ''}`}>
      CRIC<span className="wordmark__ai">AI</span>
    </span>
  )
}

const LINKS = [
  ['analyze', 'Analyze'],
  ['how', 'Method'],
  ['model', 'Accuracy'],
  ['history', 'History'],
]

export function Navbar({ onNavigate }) {
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 12)
    on()
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])

  return (
    <header className={`nav ${scrolled ? 'is-scrolled' : ''}`}>
      <nav className="nav__inner clay" aria-label="Main">
        <a className="nav__brand" href="#/" onClick={(e) => { e.preventDefault(); onNavigate('top') }}>
          <Wordmark />
        </a>
        <ul className="nav__links">
          {LINKS.map(([id, label]) => (
            <li key={id}>
              <a href={`#${id}`} onClick={(e) => { e.preventDefault(); onNavigate(id) }}>{label}</a>
            </li>
          ))}
        </ul>
        <button className="btn btn--ink btn--sm nav__cta" onClick={() => onNavigate('analyze')}>Analyze a shot</button>
      </nav>
    </header>
  )
}

export function Footer() {
  return (
    <footer className="footer container">
      <div className="footer__inner">
        <Wordmark small />
        <p>
          Shot recognition from a fine-tuned ResNet-18 fused with MediaPipe pose geometry, trained on real match
          photos. Technique scores are rule-based coaching heuristics.
        </p>
        <p className="footer__small">© {new Date().getFullYear()} CricAI</p>
      </div>
    </footer>
  )
}
