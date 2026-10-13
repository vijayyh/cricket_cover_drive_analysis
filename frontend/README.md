# CricAI frontend

React 19 + Vite. No UI framework: all styling lives in two files.

| File | Purpose |
|---|---|
| `src/styles/tokens.css` | Design tokens: colours, clay shadows, radii, motion, shot colours |
| `src/styles/app.css` | Component styles, built only from tokens |
| `src/components/HeroVideo.jsx` | Hero demo: real clip with CricAI's live tracking + classifier overlay |
| `src/components/ResultsView.jsx` | Analysis dashboard (video, timeline, verdict, technique) |
| `src/api.js` | Backend calls (relative `/api/...` URLs) |

```bash
npm install
npm run dev      # http://127.0.0.1:5173, proxies /api to http://127.0.0.1:8000
npm run build    # outputs dist/, which FastAPI serves at http://127.0.0.1:8000
```

With `prefers-reduced-motion`, sliding and bouncing are removed but fades and the hero clip (which has a
pause button) still play.

The hero clip is generated from any batting video by the real pipeline:

```bash
python scripts/make_hero_clip.py --video path/to/clip.mp4   # writes public/hero/*
```
