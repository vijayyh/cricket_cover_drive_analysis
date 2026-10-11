"""
main.py — analyse a cricket shot video or photo.

    python main.py --video path/to/clip.mp4
    python main.py --image path/to/photo.jpg
    python main.py                       # every video/photo in test_folder/

Pipeline (video)
  1. MediaPipe pose tracking on every frame
  2. Shot type: CNN + pose fusion on the frames around the swing   (src/shot_recognizer.py)
  3. Technique: measured angles vs shot-specific coaching targets   (src/technique.py)
  4. Annotated video with skeleton, shot label and impact marker
Results are written to output/results/<id>.json and returned as a dict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.config as config
from src.overlay_utils import draw_hud, draw_progress, draw_skeleton
from src.pose_estimation import PoseEstimator
from src.shot_recognizer import FramePose, ShotRecognizer, detect_pose_static, segments
from src.technique import assess
from src.video_processing import setup_video_capture, setup_video_writer

_recognizer = None


def get_recognizer() -> ShotRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = ShotRecognizer(config.MODEL_DIR)
    return _recognizer


def analyze(path: str, analysis_id: str | None = None, source_name: str | None = None) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext in config.IMAGE_EXTS:
        return analyze_image(path, analysis_id, source_name)
    return analyze_video(path, analysis_id, source_name)


def _save(result: dict) -> dict:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, f"{result['id']}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def _thumbnail(frame):
    """Tiny grayscale frame + histogram for scene-cut detection."""
    g = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    h = cv2.calcHist([g], [0], None, [32], [0, 256])
    cv2.normalize(h, h)
    return g, h


def _is_cut(a, b) -> bool:
    """A camera cut changes both the brightness histogram and the pixels at once."""
    hist_d = cv2.compareHist(a[1], b[1], cv2.HISTCMP_BHATTACHARYYA)
    pix_d = float(np.mean(cv2.absdiff(a[0], b[0]))) / 255
    return hist_d > 0.5 or (hist_d > 0.2 and pix_d > 0.1)


def _subject_jumped(prev_lm, lm) -> bool:
    """The tracker hopped to a different person: hips move > 1.5 torso lengths in one frame."""
    if prev_lm is None or lm is None:
        return False
    hip_a = (prev_lm[23, :2] + prev_lm[24, :2]) / 2
    hip_b = (lm[23, :2] + lm[24, :2]) / 2
    torso = np.linalg.norm((prev_lm[11, :2] + prev_lm[12, :2]) / 2 - hip_a) + 1e-6
    return float(np.linalg.norm(hip_b - hip_a)) > 1.5 * torso


def analyze_video(path: str, analysis_id: str | None = None, source_name: str | None = None) -> dict:
    analysis_id = analysis_id or uuid.uuid4().hex[:12]
    t0 = time.time()
    cap, W, H, fps = setup_video_capture(path)
    if cap is None:
        raise ValueError("Could not open the video file")

    # ── Pass 1: pose tracking + scene-cut detection ──────────────────────────
    max_frames = int(config.MAX_VIDEO_SECONDS * fps)
    estimator = PoseEstimator()
    poses: list[FramePose] = []
    prev_thumb = None
    while len(poses) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        thumb = _thumbnail(frame)
        img_lm, world_lm = estimator.process(frame)
        cut = prev_thumb is not None and _is_cut(prev_thumb, thumb)
        cut = cut or (bool(poses) and _subject_jumped(poses[-1].img_lm, img_lm))
        poses.append(FramePose(len(poses), img_lm, world_lm, {"cut": cut}))
        prev_thumb = thumb
    cap.release()
    estimator.close()
    n = len(poses)
    if n == 0:
        raise ValueError("The video has no readable frames")

    # ── Shot type + technique ────────────────────────────────────────────────
    shot = get_recognizer().predict_video(path, poses, fps)
    timeline = shot.pop("timeline", [])
    key = shot.get("key_frame")
    seg = segments(poses)
    # Only measure technique inside the camera shot that contains the swing.
    tech_poses = [p.img_lm if key is None or seg[i] == seg[key] else None for i, p in enumerate(poses)]
    technique = assess(shot["shot_type"], tech_poses, key, fps)
    impact = technique["impact_frame"] if technique else key

    # ── Pass 2: annotated video ──────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    video_name, key_name = f"{analysis_id}_annotated.mp4", f"{analysis_id}_key.jpg"
    out = setup_video_writer(os.path.join(config.OUTPUT_DIR, video_name), W, H, fps)
    cap = cv2.VideoCapture(path)
    label = f"{shot['shot_type'].upper()}  {shot['confidence']:.0%}"
    near_impact = max(1, int(0.1 * fps))
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frame = draw_skeleton(frame, poses[i].img_lm)
        is_impact = impact is not None and abs(i - impact) <= near_impact
        frame = draw_hud(frame, label, "IMPACT" if is_impact else "CRICAI", is_impact)
        frame = draw_progress(frame, (i + 1) / n, impact / n if impact is not None else None)
        if i == impact or (impact is None and i == n // 2):
            cv2.imwrite(os.path.join(config.OUTPUT_DIR, key_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.write(frame)
    cap.release()
    out.release()

    elapsed = time.time() - t0
    detected = sum(p.img_lm is not None for p in poses)
    print(f"[main] {n} frames | pose on {detected} | {elapsed:.1f}s | "
          f"{shot['shot_type']} ({shot['confidence']:.0%})")

    return _save({
        "id": analysis_id,
        "kind": "video",
        "source_name": source_name or os.path.basename(path),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shot": shot,
        "timeline": timeline,
        "technique": technique,
        "media": {
            "annotated": video_name, "key_frame": key_name,
            "width": W, "height": H, "fps": round(fps, 2), "frames": n,
            "duration": round(n / fps, 2),
        },
        "stats": {
            "pose_detected_ratio": round(detected / n, 3),
            "processing_seconds": round(elapsed, 2),
        },
    })


def analyze_image(path: str, analysis_id: str | None = None, source_name: str | None = None) -> dict:
    analysis_id = analysis_id or uuid.uuid4().hex[:12]
    t0 = time.time()
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read the image file")
    if max(img.shape[:2]) > 1600:
        s = 1600 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]

    pose = detect_pose_static(img)
    shot = get_recognizer().predict_image(img, pose)
    lm = pose[0] if pose else None
    technique = assess(shot["shot_type"], [lm], 0, 1.0) if lm is not None else None

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    name = f"{analysis_id}_annotated.jpg"
    annotated = draw_skeleton(img, lm) if lm is not None else img.copy()
    annotated = draw_hud(annotated, f"{shot['shot_type'].upper()}  {shot['confidence']:.0%}", "CRICAI")
    cv2.imwrite(os.path.join(config.OUTPUT_DIR, name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    elapsed = time.time() - t0
    print(f"[main] image | pose {'yes' if lm is not None else 'no'} | {elapsed:.1f}s | "
          f"{shot['shot_type']} ({shot['confidence']:.0%})")
    return _save({
        "id": analysis_id,
        "kind": "image",
        "source_name": source_name or os.path.basename(path),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shot": shot,
        "timeline": [],
        "technique": technique,
        "media": {"annotated": name, "key_frame": name, "width": W, "height": H,
                  "fps": 0, "frames": 1, "duration": 0},
        "stats": {"pose_detected_ratio": 1.0 if lm is not None else 0.0,
                  "processing_seconds": round(elapsed, 2)},
    })


def _print_result(r: dict):
    s = r["shot"]
    probs = ", ".join(f"{k} {v:.0%}" for k, v in sorted(s["all_probabilities"].items(), key=lambda kv: -kv[1]))
    print(f"  Shot      : {s['shot_type']}  ({s['confidence']:.1%}{', LOW CONFIDENCE' if s['low_confidence'] else ''})")
    print(f"  All       : {probs}")
    t = r.get("technique")
    if t:
        print(f"  Technique : {t['overall']}/10 ({t['grade']})")
        for a in t["aspects"]:
            print(f"     {a['aspect']:<15} {a['score']:>4}/10  {a['measured']}")
    print(f"  Output    : {os.path.join(config.OUTPUT_DIR, r['media']['annotated'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cricket shot analysis")
    ap.add_argument("--video", type=str)
    ap.add_argument("--image", type=str)
    args = ap.parse_args()

    if args.video or args.image:
        targets = [args.video or args.image]
    else:
        folder = os.path.join(config.ROOT, "test_folder")
        exts = config.VIDEO_EXTS | config.IMAGE_EXTS
        targets = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                   if os.path.splitext(f)[1].lower() in exts]

    for t in targets:
        print(f"\n[main] Analysing {os.path.basename(t)}")
        _print_result(analyze(t))
