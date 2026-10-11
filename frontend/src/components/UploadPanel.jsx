import { useEffect, useRef, useState } from 'react'
import { analyzeFile } from '../api.js'
import { useReveal } from '../hooks.js'
import { IconAlert, IconCheck, IconFilm, IconImage, IconUpload, IconX } from './Icons.jsx'

const ACCEPT = 'video/*,image/jpeg,image/png,image/webp,.gif'
const STEPS = ['Uploading', 'Tracking body pose', 'Recognising the shot', 'Scoring technique', 'Rendering annotated video']

const isImage = (f) => f && f.type.startsWith('image/') && f.type !== 'image/gif'
const fmtSize = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1e3)} KB`)

export default function UploadPanel({ onResult }) {
  const reveal = useReveal()
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [duration, setDuration] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [phase, setPhase] = useState('idle') // idle | ready | uploading | processing | error
  const [upload, setUpload] = useState(0)
  const [step, setStep] = useState(0)
  const [error, setError] = useState('')

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview])

  // Advance the processing stepper on an estimate; the last step only completes when the server answers.
  useEffect(() => {
    if (phase !== 'processing') return
    const est = isImage(file) ? 3 : Math.max(6, (duration || 6) * 1.6 + 4)
    const marks = [0, 0.05, 0.55, 0.72, 0.82]
    const t0 = performance.now()
    const id = setInterval(() => {
      const f = (performance.now() - t0) / 1000 / est
      let s = 1
      for (let i = 1; i < marks.length; i++) if (f >= marks[i]) s = i
      setStep(isImage(file) ? Math.min(s, 3) : s)
    }, 250)
    return () => clearInterval(id)
  }, [phase, duration, file])

  const choose = (f) => {
    if (!f) return
    const ok = f.type.startsWith('video/') || f.type.startsWith('image/') || /\.(gif|mov|mkv|avi)$/i.test(f.name)
    if (!ok) {
      setError('Please choose a video (MP4, MOV, WEBM, GIF) or a photo (JPG, PNG).')
      setPhase('error')
      return
    }
    if (preview) URL.revokeObjectURL(preview)
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setDuration(null)
    setError('')
    setPhase('ready')
  }

  const reset = () => {
    setFile(null)
    setPreview(null)
    setPhase('idle')
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const start = async () => {
    if (!file) return
    setPhase('uploading')
    setUpload(0)
    setStep(0)
    try {
      const result = await analyzeFile(file, (f) => {
        setUpload(f)
        if (f >= 1) setPhase('processing')
      })
      setStep(STEPS.length)
      onResult(result)
      reset()
    } catch (e) {
      setError(e.message)
      setPhase('error')
    }
  }

  const busy = phase === 'uploading' || phase === 'processing'
  const steps = isImage(file) ? STEPS.slice(0, 4) : STEPS

  return (
    <section className="section container reveal" id="analyze" ref={reveal}>
      <div className="section__head">
        <span className="overline">Analyze</span>
        <h2>Drop in your shot</h2>
        <p>Side-on footage from square of the wicket works best. Clips of 2–10 seconds around the shot are ideal.</p>
      </div>

      <div className="upload clay">
        {!file && (
          <div
            className={`dropzone ${dragging ? 'is-dragging' : ''}`}
            role="button"
            tabIndex={0}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
            onDragEnter={(e) => { e.preventDefault(); setDragging(true) }}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false) }}
            onDrop={(e) => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files?.[0]) }}
          >
            <div className="dropzone__icon"><IconUpload width={30} height={30} /></div>
            <h3>{dragging ? 'Release to upload' : 'Drag a video or photo here'}</h3>
            <p>or <span className="link">browse your files</span> · MP4, MOV, WEBM, GIF, JPG, PNG · up to 200 MB</p>
            <div className="dropzone__formats">
              <span className="chip"><IconFilm width={14} height={14} /> Video clip</span>
              <span className="chip"><IconImage width={14} height={14} /> Single photo</span>
            </div>
          </div>
        )}
        <input ref={inputRef} type="file" accept={ACCEPT} hidden onChange={(e) => choose(e.target.files?.[0])} />

        {file && (
          <div className="upload__selected">
            <div className="upload__preview">
              {isImage(file) ? (
                <img src={preview} alt="Selected shot preview" />
              ) : file.type === 'image/gif' ? (
                <img src={preview} alt="Selected clip preview" />
              ) : (
                <video src={preview} muted playsInline loop autoPlay
                  onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)} />
              )}
              {busy && <div className="upload__scan" />}
            </div>

            <div className="upload__side">
              <div className="upload__file">
                <div>
                  <strong title={file.name}>{file.name}</strong>
                  <span>{fmtSize(file.size)}{duration ? ` · ${duration.toFixed(1)} s` : ''}</span>
                </div>
                {!busy && (
                  <button className="icon-btn" onClick={reset} aria-label="Remove file"><IconX /></button>
                )}
              </div>

              {phase === 'ready' && (
                <button className="btn btn--ink btn--lg btn--block" onClick={start}>Analyze this shot</button>
              )}

              {busy && (
                <ol className="steps" aria-live="polite">
                  {steps.map((s, i) => {
                    const state = i < step ? 'done' : i === step ? 'active' : 'todo'
                    return (
                      <li key={s} className={`steps__item is-${state}`}>
                        <span className="steps__mark">{state === 'done' ? <IconCheck width={14} height={14} /> : i + 1}</span>
                        <span className="steps__label">
                          {s}
                          {i === 0 && phase === 'uploading' && <em> {Math.round(upload * 100)}%</em>}
                        </span>
                      </li>
                    )
                  })}
                </ol>
              )}
            </div>
          </div>
        )}

        {phase === 'error' && (
          <div className="alert" role="alert">
            <IconAlert />
            <span>{error}</span>
            {file && <button className="btn btn--clay btn--sm" onClick={start}>Try again</button>}
          </div>
        )}
      </div>
    </section>
  )
}
