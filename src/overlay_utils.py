"""Drawing helpers for the annotated output video / image (matches the UI's monochrome style)."""
import cv2
import numpy as np

# MediaPipe pose topology (body only — face points are noise at video scale)
_BONES = [(11, 12), (11, 23), (12, 24), (23, 24),
          (11, 13), (13, 15), (15, 19), (23, 25), (25, 27), (27, 31),
          (12, 14), (14, 16), (16, 20), (24, 26), (26, 28), (28, 32)]
_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# BGR
C_INK = (17, 15, 14)
C_WHITE = (246, 248, 248)
C_MUTED = (200, 200, 196)
C_BRAND = (28, 0, 213)       # brand red, used only to mark the impact


def draw_skeleton(img: np.ndarray, lm: np.ndarray, min_vis: float = 0.3) -> np.ndarray:
    """lm: (33,4) pixel landmarks. White bones with a dark outline read on any footage."""
    if lm is None:
        return img
    scale = max(img.shape[:2]) / 720
    th = max(2, int(2.5 * scale))
    r = max(3, int(3.5 * scale))
    out = img.copy()
    visible = lambda j: lm[j, 3] > min_vis  # noqa: E731
    pt = lambda j: tuple(map(int, lm[j, :2]))  # noqa: E731
    for a, b in _BONES:
        if visible(a) and visible(b):
            cv2.line(out, pt(a), pt(b), C_INK, th + 3, cv2.LINE_AA)
    for a, b in _BONES:
        if visible(a) and visible(b):
            cv2.line(out, pt(a), pt(b), C_WHITE, th, cv2.LINE_AA)
    for j in _JOINTS:
        if visible(j):
            cv2.circle(out, pt(j), r + 2, C_INK, -1, cv2.LINE_AA)
            cv2.circle(out, pt(j), r, C_WHITE, -1, cv2.LINE_AA)
    return cv2.addWeighted(out, 0.9, img, 0.1, 0)


def draw_hud(img: np.ndarray, title: str, subtitle: str = "", highlight: bool = False) -> np.ndarray:
    """Dark rounded label in the top-left corner; a red rule marks the impact moment."""
    scale = max(img.shape[:2]) / 900
    fs = max(0.5, 0.82 * scale)
    th = max(1, int(1.6 * scale))
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, tht), _ = cv2.getTextSize(title, font, fs, th)
    (sw, sh), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.55, 1)
    pad = int(14 * scale) + 4
    w = max(tw, sw) + 2 * pad
    h = tht + (sh + pad // 2 if subtitle else 0) + 2 * pad
    x0, y0 = pad, pad
    box = img.copy()
    rad = min(h // 2, int(14 * scale) + 4)
    _rounded_rect(box, (x0, y0), (x0 + w, y0 + h), C_INK, rad)
    img = cv2.addWeighted(box, 0.78, img, 0.22, 0)
    if highlight:
        cv2.rectangle(img, (x0 + rad, y0 + h - max(3, th + 1)), (x0 + w - rad, y0 + h), C_BRAND, -1)
    cv2.putText(img, title, (x0 + pad, y0 + pad + tht), font, fs, C_WHITE, th, cv2.LINE_AA)
    if subtitle:
        cv2.putText(img, subtitle, (x0 + pad, y0 + pad + tht + sh + pad // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 0.55, C_MUTED, 1, cv2.LINE_AA)
    return img


def draw_progress(img: np.ndarray, t: float, impact_t: float | None) -> np.ndarray:
    """Thin timeline bar along the bottom with a red impact marker."""
    h, w = img.shape[:2]
    bh = max(4, h // 170)
    cv2.rectangle(img, (0, h - bh), (w, h), C_INK, -1)
    cv2.rectangle(img, (0, h - bh), (int(w * t), h), C_WHITE, -1)
    if impact_t is not None:
        x = int(w * impact_t)
        cv2.rectangle(img, (x - 2, h - 3 * bh), (x + 2, h), C_BRAND, -1)
    return img


def _rounded_rect(img, p0, p1, color, r):
    (x0, y0), (x1, y1) = p0, p1
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
