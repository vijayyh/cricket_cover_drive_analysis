"""
scripts/make_hero_clip.py — build the home-page hero demo from any batting clip.

    python scripts/make_hero_clip.py --video test_folder/1.1.mp4

Runs the real CricAI pipeline on the clip, finds the stroke, and writes into
frontend/public/hero/:

  hero.webm / hero.mp4   the stroke window, cropped around the batter (clean, no overlay)
  hero.jpg               poster frame at the moment of the stroke
  hero.json              per-frame pose joints + per-frame shot probabilities, which the
                         UI draws live over the playing video

Use footage you have the rights to if the site will be public.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import src.config as config  # noqa: E402
from main import _is_cut, _subject_jumped, _thumbnail  # noqa: E402
from src.pose_estimation import PoseEstimator  # noqa: E402
from src.shot_recognizer import (FramePose, ShotRecognizer, batsman_box, crop_box,  # noqa: E402
                                 hand_motion, segments, swing_center)

# Joints drawn in the UI (MediaPipe indices): nose, shoulders, elbows, wrists, hips, knees, ankles
JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def fit_aspect(box, aspect, W, H):
    """Grow a box to the target width/height aspect, then keep it inside the frame."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    if w > W:                        # too wide for the frame: shrink, keeping aspect
        w, h = W, W / aspect
    if h > H:
        h, w = H, H * aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = int(round(min(max(cx - w / 2, 0), W - w)))
    y0 = int(round(min(max(cy - h / 2, 0), H - h)))
    return x0, y0, x0 + int(w), y0 + int(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "frontend", "public", "hero"))
    ap.add_argument("--before", type=float, default=2.2, help="seconds kept before the stroke centre")
    ap.add_argument("--after", type=float, default=2.6, help="seconds kept after the stroke centre")
    ap.add_argument("--aspect", type=float, default=0.8, help="output width / height")
    ap.add_argument("--width", type=int, default=448)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, poses = [], []
    est, prev = PoseEstimator(), None
    while len(frames) < config.MAX_VIDEO_SECONDS * fps:
        ok, f = cap.read()
        if not ok:
            break
        thumb = _thumbnail(f)
        lm, wl = est.process(f)
        cut = (prev is not None and _is_cut(prev, thumb)) or (bool(poses) and _subject_jumped(poses[-1].img_lm, lm))
        poses.append(FramePose(len(poses), lm, wl, {"cut": cut}))
        frames.append(f)
        prev = thumb
    est.close()
    print(f"[hero] {len(frames)} frames @ {fps:.1f} fps, {W}x{H}")

    rec = ShotRecognizer(config.MODEL_DIR)
    verdict = rec.predict_video(args.video, poses, fps)
    center = swing_center(poses, hand_motion(poses), fps) or verdict.get("key_frame") or len(frames) // 2
    seg = segments(poses)
    shot_frames = np.where(seg == seg[center])[0]
    lo = max(int(shot_frames[0]), center - int(args.before * fps))
    hi = min(int(shot_frames[-1]), center + int(args.after * fps))
    print(f"[hero] verdict {verdict['shot_type']} {verdict['confidence']:.0%} | stroke centre {center} | window {lo}-{hi}")

    # One fixed crop for the whole window (no jitter): union of the batter boxes.
    boxes = [batsman_box(poses[i].img_lm, W, H) for i in range(lo, hi + 1) if poses[i].img_lm is not None]
    union = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    cx0, cy0, cx1, cy1 = fit_aspect(union, args.aspect, W, H)
    cw, ch = cx1 - cx0, cy1 - cy0
    out_w = args.width - args.width % 2
    out_h = int(round(out_w / args.aspect)) // 2 * 2

    # Per-frame probabilities exactly as the app computes them (CNN crop + pose, fused).
    idx = list(range(lo, hi + 1))
    crops = [crop_box(frames[i], batsman_box(poses[i].img_lm, W, H)) if poses[i].img_lm is not None
             else frames[i][cy0:cy1, cx0:cx1] for i in idx]
    p_cnn = rec._cnn_proba(crops)
    probs = np.array([rec._fuse(p_cnn[j], rec._pose_proba(poses[i].img_lm, poses[i].world_lm))
                      for j, i in enumerate(idx)])
    k = 5                                                    # light smoothing for display
    pad = np.pad(probs, ((k // 2, k // 2), (0, 0)), mode="edge")
    smooth = np.stack([pad[j:j + k].mean(0) for j in range(len(idx))])

    os.makedirs(args.out, exist_ok=True)
    writers = [cv2.VideoWriter(os.path.join(args.out, "hero.webm"), cv2.VideoWriter_fourcc(*"VP90"), fps, (out_w, out_h)),
               cv2.VideoWriter(os.path.join(args.out, "hero.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps, (out_w, out_h))]
    joints = []
    for i in idx:
        img = cv2.resize(frames[i][cy0:cy1, cx0:cx1], (out_w, out_h), interpolation=cv2.INTER_AREA)
        for w in writers:
            w.write(img)
        if i == center:
            cv2.imwrite(os.path.join(args.out, "hero.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 86])
        lm = poses[i].img_lm
        joints.append(None if lm is None else
                      [[round(float((lm[j, 0] - cx0) / cw), 4), round(float((lm[j, 1] - cy0) / ch), 4),
                        round(float(lm[j, 3]), 2)] for j in JOINTS])
    for w in writers:
        w.release()

    meta = {
        "fps": round(fps, 3),
        "frames": len(idx),
        "width": out_w,
        "height": out_h,
        "center": center - lo,
        "classes": rec.classes,
        "verdict": {"shot_type": verdict["shot_type"], "confidence": verdict["confidence"]},
        "joints": joints,
        "probs": [[round(float(v), 3) for v in row] for row in smooth],
        "source": os.path.basename(args.video),
    }
    with open(os.path.join(args.out, "hero.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"))
    sizes = {n: os.path.getsize(os.path.join(args.out, n)) / 1e6 for n in ("hero.webm", "hero.mp4", "hero.jpg", "hero.json")}
    print("[hero] wrote " + ", ".join(f"{n} {s:.2f} MB" for n, s in sizes.items()))


if __name__ == "__main__":
    main()
