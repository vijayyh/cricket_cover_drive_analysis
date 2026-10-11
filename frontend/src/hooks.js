import { useEffect, useRef, useState } from 'react'

export const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

/** Adds .is-visible once the element scrolls into view (CSS handles the animation). */
export function useReveal(options = { threshold: 0.15 }) {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (!('IntersectionObserver' in window)) {
      el.classList.add('is-visible')
      return
    }
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        el.classList.add('is-visible')
        io.disconnect()
      }
    }, options)
    io.observe(el)
    return () => io.disconnect()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps
  return ref
}

/** Animates a number from 0 to `target` once `start` is true. */
export function useCountUp(target, start = true, duration = 1200) {
  const [value, setValue] = useState(0)
  useEffect(() => {
    if (!start || target == null) return
    if (prefersReducedMotion()) {
      setValue(target)
      return
    }
    let raf
    const t0 = performance.now()
    const tick = (now) => {
      const p = Math.min(1, (now - t0) / duration)
      setValue(target * (1 - Math.pow(1 - p, 3)))
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, start, duration])
  return value
}

/** Minimal hash router: '#/result/<id>' -> { view: 'result', id }. */
export function useHashRoute() {
  const parse = () => {
    const m = window.location.hash.match(/^#\/result\/([a-z0-9]+)/i)
    return m ? { view: 'result', id: m[1] } : { view: 'home', id: null }
  }
  const [route, setRoute] = useState(parse)
  useEffect(() => {
    const on = () => setRoute(parse())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return route
}
