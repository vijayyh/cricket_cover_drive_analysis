"""
src/shot_recognizer.py
======================
Shot-type recognition for photos and videos.

    CNN branch   ResNet-18 fine-tuned on grayscale batsman crops   (src/shot_cnn.py)
    Pose branch  MediaPipe pose geometry -> SVM / boosted trees    (src/shot_pose.py)
    Fusion       weighted geometric mean, weight chosen on held-out photos

For a video, every frame is not equally informative: the stance and backlift
look alike for every shot. We therefore locate the stroke (the 0.6 s window
where the gripped hands travel furthest through the biggest vertical arc),
stay inside the camera shot that contains it (broadcast clips cut or pan to
umpires and fielders), classify frames around it, and weight each frame by
closeness to the stroke, bat grip and model confidence.

Train the models with:  python train_shot_classifier.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.shot_pose import (L_HIP, L_SH, L_WR, R_HIP, R_SH, R_WR,
                           pose_features_with_mirror, pose_is_usable)

MODEL_DIR = "models"
CNN_FILE = "shot_cnn.pt"
POSE_FILE = "shot_pose.pkl"
META_FILE = "shot_recognizer.json"

LOW_CONFIDENCE = 0.50
MAX_KEY_FRAMES = 16

_BODY = [0, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28, 31, 32]


# ─────────────────────────────────────────────────────────────────────────────
# Pose helpers
# ─────────────────────────────────────────────────────────────────────────────

_static_pose = None


def detect_pose_static(img_bgr: np.ndarray):
    """
    Single-image pose detection. Tight crops (like the training photos) often
    defeat MediaPipe's person detector, so we retry with upscaling + padding.
    Returns (img_lm_pixels (33,4), world_lm (33,4)) or None.
    """
    global _static_pose
    import mediapipe as mp
    if _static_pose is None:
        _static_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1,
                                              min_detection_confidence=0.3)
    for scale, pad in ((2, 0.25), (1, 0.4), (1, 0.0)):
        im = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) \
            if scale != 1 else img_bgr
        p = int(max(im.shape[:2]) * pad)
        if p:
            im = cv2.copyMakeBorder(im, p, p, p, p, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        H, W = im.shape[:2]
        r = _static_pose.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if r.pose_landmarks:
            a = landmarks_to_array(r.pose_landmarks.landmark, W, H)
            a[:, 0] = (a[:, 0] - p) / scale
            a[:, 1] = (a[:, 1] - p) / scale
            a[:, 2] = a[:, 2] / scale
            return a, world_to_array(r.pose_world_landmarks.landmark)
    return None


def landmarks_to_array(landmarks, width: int, height: int) -> np.ndarray:
    """MediaPipe normalised landmarks -> (33,4) array in pixel units."""
    return np.array([[l.x * width, l.y * height, l.z * width, l.visibility] for l in landmarks],
                    np.float32)


def world_to_array(landmarks) -> np.ndarray:
    return np.array([[l.x, l.y, l.z, l.visibility] for l in landmarks], np.float32)


def batsman_box(img_lm: np.ndarray, frame_w: int, frame_h: int):
    """Crop box around the batsman resembling the framing of the training photos."""
    pts = img_lm[_BODY]
    pts = pts[pts[:, 3] > 0.3] if np.sum(pts[:, 3] > 0.3) >= 6 else pts
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    bh = max(y1 - y0, 1.0)
    h = bh * 1.25                                   # head room + bat below the hands
    w = max(h * 1.35, (x1 - x0) * 1.35)             # photos are landscape, bat sticks out
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - bh * 0.02
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def crop_box(frame: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    H, W = frame.shape[:2]
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return frame[max(y0, 0):y1, max(x0, 0):x1]


# ─────────────────────────────────────────────────────────────────────────────
# Recogniser
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FramePose:
    """Per-frame pose record collected by the video pipeline."""
    index: int
    img_lm: np.ndarray | None = None     # (33,4) pixels
    world_lm: np.ndarray | None = None   # (33,4) metres
    extra: dict = field(default_factory=dict)


class ShotRecognizer:
    def __init__(self, model_dir: str = MODEL_DIR):
        import joblib
        import torch
        from src.shot_cnn import build_model

        meta_path = os.path.join(model_dir, META_FILE)
        cnn_path = os.path.join(model_dir, CNN_FILE)
        if not (os.path.exists(meta_path) and os.path.exists(cnn_path)):
            raise FileNotFoundError("Shot models missing — run:  python train_shot_classifier.py")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.classes: list[str] = meta["classes"]
        self.pose_weight: float = float(meta.get("pose_weight", 0.3))
        self.holdout_accuracy = meta.get("holdout_accuracy")

        ckpt = torch.load(cnn_path, map_location="cpu", weights_only=False)
        self.img_size = int(ckpt.get("img_size", 192))
        self.cnn = build_model(len(self.classes), pretrained=False)
        self.cnn.load_state_dict(ckpt["state_dict"])
        self.cnn.eval()

        pose_path = os.path.join(model_dir, POSE_FILE)
        self.pose_model = joblib.load(pose_path)["model"] if os.path.exists(pose_path) else None
        print(f"[ShotRecognizer] classes={self.classes} pose_weight={self.pose_weight} "
              f"pose_branch={'on' if self.pose_model is not None else 'off'}")

    # ── branch probabilities ────────────────────────────────────────────────

    def _cnn_proba(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        from PIL import Image
        from src.shot_cnn import predict_proba
        imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)) for c in crops_bgr]
        return predict_proba(self.cnn, imgs, self.img_size)

    def _pose_proba(self, img_lm, world_lm) -> np.ndarray | None:
        if self.pose_model is None or img_lm is None or not pose_is_usable(img_lm):
            return None
        X = pose_features_with_mirror(img_lm, world_lm)
        return self.pose_model.predict_proba(X).mean(0)

    def _fuse(self, p_cnn: np.ndarray, p_pose: np.ndarray | None) -> np.ndarray:
        if p_pose is None:
            return p_cnn
        w = self.pose_weight
        log = (1 - w) * np.log(np.clip(p_cnn, 1e-6, 1)) + w * np.log(np.clip(p_pose, 1e-6, 1))
        e = np.exp(log - log.max())
        return e / e.sum()

    def _result(self, probs: np.ndarray, extra: dict) -> dict:
        order = np.argsort(probs)[::-1]
        conf = float(probs[order[0]])
        return {
            "shot_type": self.classes[order[0]],
            "confidence": round(conf, 4),
            "runner_up": self.classes[order[1]],
            "all_probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
            "low_confidence": conf < LOW_CONFIDENCE,
            "model": "ResNet-18 + pose-geometry fusion",
            "model_holdout_accuracy": self.holdout_accuracy,
            **extra,
        }

    # ── public API ──────────────────────────────────────────────────────────

    def predict_image(self, img_bgr: np.ndarray, pose=None) -> dict:
        """Classify a single photo. `pose` = (img_lm, world_lm) if already known."""
        pose = pose if pose is not None else detect_pose_static(img_bgr)
        if pose is not None and pose_is_usable(pose[0]):
            h, w = img_bgr.shape[:2]
            crop = crop_box(img_bgr, batsman_box(pose[0], w, h))
            # A photo that is already a tight crop should be used as-is.
            if crop.shape[0] * crop.shape[1] > 0.8 * h * w:
                crop = img_bgr
        else:
            crop = img_bgr
        p_cnn = self._cnn_proba([crop])[0]
        p_pose = self._pose_proba(*pose) if pose is not None else None
        return self._result(self._fuse(p_cnn, p_pose), {"pose_used": p_pose is not None,
                                                        "frames_used": 1})

    def predict_video(self, video_path: str, poses: list[FramePose], fps: float) -> dict:
        """
        Classify a video given per-frame poses from the tracking pass.
        Re-reads only the selected key frames from disk.
        """
        n = len(poses)
        if n == 0:
            return self._result(np.full(len(self.classes), 1 / len(self.classes)),
                                {"frames_used": 0, "pose_used": False, "timeline": []})

        motion = hand_motion(poses)
        keys, center = select_key_frames(motion, poses, fps)
        sigma = 0.4 * (fps if fps and fps > 0 else 25.0)

        # Carry the last known batsman box over frames where pose dropped out.
        cap = cv2.VideoCapture(video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        boxes = _boxes_with_fallback(poses, W, H)

        # Sequential read: seeking is unreliable for GIFs and some codecs.
        crops, valid = [], []
        wanted = set(keys)
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                crops.append(crop_box(frame, boxes[i]) if boxes[i] else frame)
                valid.append(i)
        cap.release()
        if not crops:
            return self._result(np.full(len(self.classes), 1 / len(self.classes)),
                                {"frames_used": 0, "pose_used": False, "timeline": []})

        p_cnn = self._cnn_proba(crops)
        per_frame, weights, timeline, pose_used = [], [], [], 0
        m_max = max(float(motion.max()), 1e-6)
        for j, i in enumerate(valid):
            fp = poses[i]
            p_pose = self._pose_proba(fp.img_lm, fp.world_lm)
            pose_used += p_pose is not None
            p = self._fuse(p_cnn[j], p_pose)
            conf = float(p.max())
            # Frames near the stroke, a confident model and a bat grip mark the shot itself.
            near = np.exp(-0.5 * ((i - center) / sigma) ** 2) if center is not None else 0.5
            w = (0.1 + near) * conf ** 2 * (0.25 + 0.75 * grip_score(fp.img_lm))
            per_frame.append(p)
            weights.append(w)
            timeline.append({"frame": int(i), "time": round(i / fps, 3) if fps else 0.0,
                             "shot_type": self.classes[int(p.argmax())],
                             "confidence": round(conf, 3),
                             "motion": round(float(motion[i] / m_max), 3)})

        weights = np.asarray(weights)
        probs = (np.asarray(per_frame) * weights[:, None]).sum(0) / weights.sum()
        key = valid[int(np.argmax(weights))]
        return self._result(probs, {
            "frames_used": len(valid),
            "pose_used": pose_used > 0,
            "key_frame": int(key),
            "key_time": round(key / fps, 3) if fps else 0.0,
            "timeline": timeline,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Key-frame selection
# ─────────────────────────────────────────────────────────────────────────────

def _torso(lm: np.ndarray) -> float:
    sh = (lm[L_SH, :2] + lm[R_SH, :2]) / 2
    hp = (lm[L_HIP, :2] + lm[R_HIP, :2]) / 2
    return float(np.linalg.norm(sh - hp)) + 1e-6


def grip_score(lm: np.ndarray | None) -> float:
    """
    1.0 when both hands are together (on a bat handle), 0.0 when they are far
    apart. Separates the batter from runners, umpires and bowlers that the
    pose tracker may latch onto after a camera cut.
    """
    if lm is None:
        return 0.0
    gap = float(np.linalg.norm(lm[L_WR, :2] - lm[R_WR, :2])) / _torso(lm)
    return float(np.clip(1.0 - (gap - 0.35) / 0.6, 0.0, 1.0))


def segments(poses: list[FramePose]) -> np.ndarray:
    """Camera-shot index per frame, split wherever the pipeline flagged a scene cut."""
    seg = np.zeros(len(poses), int)
    s = 0
    for i, fp in enumerate(poses):
        if i and fp.extra.get("cut"):
            s += 1
        seg[i] = s
    return seg


def hand_motion(poses: list[FramePose]) -> np.ndarray:
    """
    Per-frame hand speed (torso-lengths per frame) counted only while the hands
    grip the bat, reset at scene cuts, lightly smoothed.
    """
    n = len(poses)
    speed = np.zeros(n, np.float32)
    prev = None
    for i, fp in enumerate(poses):
        if fp.img_lm is None or fp.extra.get("cut"):
            prev = None
        if fp.img_lm is None:
            continue
        hands = (fp.img_lm[L_WR, :2] + fp.img_lm[R_WR, :2]) / 2
        grip = grip_score(fp.img_lm)
        if prev is not None:
            speed[i] = np.linalg.norm(hands - prev[0]) / prev[1] * min(grip, prev[2])
        prev = (hands, _torso(fp.img_lm), grip)
    if n > 10:
        speed = np.clip(speed, 0, np.percentile(speed, 99))
    k = 5
    return np.convolve(speed, np.ones(k) / k, mode="same") if n >= k else speed


def swing_center(poses: list[FramePose], motion: np.ndarray, fps: float) -> int | None:
    """
    Centre of the batting stroke. A stroke moves the gripped hands fast AND
    through a big vertical arc (backlift -> through the ball -> follow-through);
    walking off, bat twirls and shadow practice keep the hands near the waist.
    Each 0.6 s window inside one camera shot is scored by hand travel x
    vertical hand range, and the best window's centre is returned.
    """
    n = len(poses)
    seg = segments(poses)
    height = np.full(n, np.nan)
    for i, fp in enumerate(poses):
        lm = fp.img_lm
        if lm is not None and grip_score(lm) > 0.3:
            shoulders = (lm[L_SH, 1] + lm[R_SH, 1]) / 2
            hands = (lm[L_WR, 1] + lm[R_WR, 1]) / 2
            height[i] = (shoulders - hands) / _torso(lm)
    w = min(n, max(3, int(0.6 * fps)))
    best_score, best = 0.0, None
    for s in range(0, n - w + 1):
        if seg[s] != seg[s + w - 1]:
            continue
        h = height[s:s + w]
        h = h[~np.isnan(h)]
        if len(h) < 0.5 * w:
            continue
        arc = float(np.clip(np.percentile(h, 95) - np.percentile(h, 5), 0, 2.0))
        score = float(motion[s:s + w].sum()) * arc
        if score > best_score:
            best_score, best = score, s + w // 2
    return best


def select_key_frames(motion: np.ndarray, poses: list[FramePose], fps: float) -> tuple[list[int], int | None]:
    """
    Frames spanning the swing (just before the stroke to the follow-through),
    restricted to the camera shot that contains it. Returns (frames, centre).
    """
    n = len(motion)
    fps = fps if fps and fps > 0 else 25.0
    seg = segments(poses)
    has_pose = np.array([p.img_lm is not None for p in poses])

    peak = swing_center(poses, motion, fps) if has_pose.sum() >= 3 else None
    if peak is None and motion.max() > 0:
        peak = int(np.argmax(motion))
    if peak is None:
        # No usable swing: sample the camera shot with the most pose detections.
        best = np.bincount(seg, weights=has_pose.astype(float)).argmax()
        frames = np.where(seg == best)[0]
        return np.unique(np.linspace(frames[0], frames[-1], min(len(frames), MAX_KEY_FRAMES))
                         .astype(int)).tolist(), None

    shot = np.where(seg == seg[peak])[0]
    lo = max(int(shot[0]), peak - int(0.45 * fps))
    hi = min(int(shot[-1]), peak + int(0.75 * fps))
    window = np.arange(lo, hi + 1)
    # Also take a sparse look at the rest of this camera shot, in case the
    # biggest hand movement was not the shot itself (e.g. a practice swing).
    rest = np.setdiff1d(np.linspace(shot[0], shot[-1], 6).astype(int), window)
    picks = np.unique(np.linspace(lo, hi, min(len(window), MAX_KEY_FRAMES - len(rest))).astype(int))
    return sorted(set(picks.tolist()) | set(rest.tolist())), peak


def _boxes_with_fallback(poses: list[FramePose], W: int, H: int):
    boxes = [batsman_box(p.img_lm, W, H) if p.img_lm is not None and pose_is_usable(p.img_lm)
             else None for p in poses]
    last = None
    for i in range(len(boxes)):                    # forward fill
        boxes[i] = boxes[i] or last
        last = boxes[i]
    nxt = None
    for i in range(len(boxes) - 1, -1, -1):        # back fill the start
        boxes[i] = boxes[i] or nxt
        nxt = boxes[i]
    return boxes
