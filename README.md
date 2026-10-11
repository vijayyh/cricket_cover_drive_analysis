# 🏏 CricAI: Cricket Shot Recognition & Technique Analysis

Upload a video or photo of a batter. CricAI tracks the body with MediaPipe, finds the moment of the
swing, **recognises the shot** (Cover Drive, Straight Drive, Pull, Cut, Leg Glance) and **scores the
technique** from measured joint angles. Results appear in a glassmorphism React dashboard with an
annotated video, a frame-by-frame classification timeline and coaching feedback.

---

## Accuracy, measured honestly

The model is trained on `data/shot_dataset`: **500 real photos** (100 per shot), each shipped with
6 augmented copies (brightness, ±30° rotation, shear, salt-and-pepper noise). Many of the photos
are neighbouring frames of the same video (e.g. Leg Glance 11–22), so the 500 photos come from
roughly **284 distinct source clips**.

> **Rule used for every number below:** photos are grouped by source clip, and a whole clip, with
> every augmented copy, stays on one side of the train/test split. Splitting copies or neighbouring
> frames across train and test lets the model "recognise" what it has already seen. The previous
> training script split images at random, which is why its numbers were meaningless.

| Model (held-out photos from clips the model never saw) | Accuracy |
|---|---|
| **Fused model (what the app uses)** | **84.0 %** (100 photos) |
| Pose only: joint geometry (when a pose is found) | 80.8 % |
| CNN only: ResNet-18 fine-tuned | 73.0 % |
| Fused, on the harder rotated/sheared/noisy copies | 78.2 % (600 images) |
| *Same model with the weaker photo-level split* | *90 % (inflated by near-duplicate frames)* |
| *Previous landmark model (trained on synthetic numbers)* | *≈ 30 %* |
| *Chance level* | *20 %* |

The stricter split exposed something useful. On its own, the CNN partly memorises clip appearance
(kit colour, background), while the pose model reads body geometry and generalises. The fusion
weight is tuned on held-out data, so the app now leans on pose (weight 0.60). Most remaining errors
are Cut → Pull (both back-foot cross-bat shots) and Straight Drive. The full report, with the
confusion matrix and per-class F1, is in `models/shot_classifier_report.json`, and the UI shows it
under **Accuracy**.

*Caveat:* the test set is 100 photos, so treat ±7 % as the uncertainty. Real videos filmed from
unusual angles (e.g. straight down the pitch) differ from the training photos. The app flags
low-confidence predictions rather than guessing silently.

---

## How it works

```
video ──► MediaPipe Pose (every frame) ──► hand-speed curve ──► key frames around the swing
                                                                      │
              ┌───────────────────────────────────────────────────────┤
              ▼                                                       ▼
  CNN branch: grayscale batter crop                    Pose branch: 140+ geometry features
  → ResNet-18 (fine-tuned)                             (normalised joints, angles, 3-D pose)
              │                                                       → SVM
              └──────────────► weighted fusion (pose weight 0.60) ◄───┘
                                       │
            per-frame probabilities, weighted by hand speed × confidence
                                       │
                         shot type  +  technique scoring  +  annotated video
```

* **Why two branches?** The CNN sees the bat and body shape; the pose model sees joint geometry
  and is less sensitive to background, lighting and jersey colour. Fusing them beats either one.
* **Why key frames?** Stance and walk-back frames look the same for every shot. Averaging over the
  whole clip (what the old pipeline did) washes out the signal. Instead, the stroke is found as the
  0.6 s window where the hands travel furthest through the biggest vertical arc, counted only while
  both hands grip the bat. That rules out runners and umpires, and also walking off, bat twirls and
  shadow practice. The search stays inside the camera shot that contains the stroke (scene cuts and
  tracker hops are detected), and frames are weighted by closeness to the stroke, grip and model
  confidence.
* **Left-handers:** every training sample is also mirrored, so left- and right-handers are handled.
* **Technique score** (`src/technique.py`) is **rule-based, not ML**: there is no labelled
  "quality" data, so instead of a model trained on invented numbers it measures knee flex, head
  position, arm extension, spine lean and follow-through around impact, and compares them with
  coaching targets *for the detected shot*. Every score shows the measured value and the target.

---

## Quick start (Windows, Python 3.11)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**1. Train** (once, about 1 h on a laptop CPU; skip if `models/shot_cnn.pt` already exists)
```bash
python train_shot_classifier.py
```

**2. Run the API** (from the project root)
```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

**3. Run the UI** (in a second terminal)
```bash
cd frontend
npm install
npm run dev
```
Open http://127.0.0.1:5173. The UI proxies `/api` to the backend.

*Single-server option:* `cd frontend && npm run build`, then just run uvicorn and open
http://127.0.0.1:8000. FastAPI serves the built UI.

**Command line:**
```bash
python main.py --video "test_folder/virat kohli.mp4"
python main.py --image path/to/photo.jpg
python main.py            # every clip in test_folder/
```

---

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | model status and held-out accuracy |
| GET | `/api/model` | full evaluation report |
| POST | `/api/analyze` | multipart `file` (video or photo) → analysis JSON |
| GET | `/api/results?limit=12` | recent analyses |
| GET | `/api/results/{id}` | one analysis |
| GET | `/api/media/{filename}` | annotated video/image (supports range requests for seeking) |

---

## Project layout

```
main.py                    analysis pipeline (video + photo) and CLI
train_shot_classifier.py   training + honest grouped evaluation + report
backend/app.py             FastAPI server
src/
  shot_recognizer.py       inference: key frames, crops, CNN + pose fusion
  shot_cnn.py              ResNet-18 model and transforms
  shot_pose.py             pose feature engineering (+ mirroring)
  technique.py             rule-based technique scoring
  pose_estimation.py       MediaPipe wrapper (video tracking)
  overlay_utils.py         skeleton / HUD drawing
  video_processing.py      capture + browser-friendly H.264 writer
  config.py                paths and settings
models/                    shot_cnn.pt, shot_pose.pkl, shot_recognizer.json, report
frontend/                  React 19 + Vite UI
  src/styles/tokens.css    ← the only place colours and glass are defined
```

### Changing the UI look

All colours, glass opacity, blur, radii and shot colours are CSS variables in
`frontend/src/styles/tokens.css`, defined once for the dark theme and once for the light theme.
Components never hard-code colours, so restyling means editing that one file.
