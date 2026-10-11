import { useEffect, useState } from 'react'
import { getHealth } from './api.js'
import { useHashRoute } from './hooks.js'
import Hero from './components/Hero.jsx'
import { Footer, Navbar } from './components/Layout.jsx'
import ResultsView from './components/ResultsView.jsx'
import { History, HowItWorks, ModelReport } from './components/Sections.jsx'
import UploadPanel from './components/UploadPanel.jsx'

export default function App() {
  const route = useHashRoute()
  const [health, setHealth] = useState(null)
  const [lastResult, setLastResult] = useState(null)
  const [pendingScroll, setPendingScroll] = useState(null)

  // Poll until the backend reports a ready model (it may still be starting up).
  useEffect(() => {
    let timer
    let alive = true
    const check = async () => {
      let h
      try {
        h = await getHealth()
      } catch {
        h = { status: 'offline' }
      }
      if (!alive) return
      setHealth(h)
      if (h.status !== 'ok') timer = setTimeout(check, 4000)
    }
    check()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [])

  // After switching from the results view back home, scroll once the section exists.
  useEffect(() => {
    if (route.view !== 'home' || !pendingScroll) return
    const id = requestAnimationFrame(() => {
      if (pendingScroll === 'top') window.scrollTo({ top: 0, behavior: 'smooth' })
      else document.getElementById(pendingScroll)?.scrollIntoView({ behavior: 'smooth' })
      setPendingScroll(null)
    })
    return () => cancelAnimationFrame(id)
  }, [route.view, pendingScroll])

  const navigate = (section) => {
    if (route.view !== 'home') window.location.hash = '#/'
    setPendingScroll(section)
  }

  const openResult = (id) => {
    window.location.hash = `#/result/${id}`
  }

  return (
    <>
      <Navbar onNavigate={navigate} />
      <main>
        {route.view === 'result' ? (
          <ResultsView id={route.id} initial={lastResult} onBack={() => navigate('analyze')} />
        ) : (
          <>
            <Hero health={health} onAnalyze={() => navigate('analyze')} onModel={() => navigate('model')} />
            {health?.status === 'offline' && (
              <div className="container">
                <div className="alert" role="alert">
                  The analysis server isn’t reachable. Start it from the project folder with
                  <code>uvicorn backend.app:app --port 8000</code>
                </div>
              </div>
            )}
            <UploadPanel
              onResult={(r) => {
                setLastResult(r)
                openResult(r.id)
              }}
            />
            <HowItWorks />
            <ModelReport />
            <History refreshKey={lastResult?.id} onOpen={openResult} />
          </>
        )}
      </main>
      <Footer />
    </>
  )
}
