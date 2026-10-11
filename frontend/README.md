# CricAI frontend

React 19 + Vite. No UI framework: all styling lives in two files.

| File | Purpose |
|---|---|
| `src/styles/tokens.css` | Design tokens: colours, glass, radii, motion, shot colours (dark + light) |
| `src/styles/app.css` | Component styles, built only from tokens |
| `src/components/ShotAnimation.jsx` | Animated hero: keyframed batter skeleton + live classifier |
| `src/components/ResultsView.jsx` | Analysis dashboard (video, timeline, verdict, technique) |
| `src/api.js` | Backend calls (relative `/api/...` URLs) |

```bash
npm install
npm run dev      # http://127.0.0.1:5173, proxies /api to http://127.0.0.1:8000
npm run build    # outputs dist/, which FastAPI serves at http://127.0.0.1:8000
```

Animations respect `prefers-reduced-motion`, and the theme choice is saved in `localStorage`.
