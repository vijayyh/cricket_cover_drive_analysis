"""
src/shot_pose.py
================
Pose branch of the shot-type recogniser.

Turns MediaPipe Pose landmarks into a scale/position-invariant feature vector
(normalised joint coordinates, joint angles, limb orientations, hand and foot
placement, plus the 3-D world landmarks). Because a left-hander's cover drive
is the mirror image of a right-hander's, every sample can be mirrored without
changing its label; we use that both as training augmentation and as
test-time augmentation.

Landmark arrays are (33, 4) float arrays of (x, y, z, visibility) where x/y
are in PIXELS of the source image (not MediaPipe's 0-1 normalised units, which
distort angles on non-square frames).
"""

from __future__ import annotations

import numpy as np

# MediaPipe indices
NOSE = 0
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_IDX, R_IDX = 19, 20
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
L_FT, R_FT = 31, 32

KEY_JOINTS = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_IDX, R_IDX,
              L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN, L_FT, R_FT]

# Left/right swap permutation for mirroring (MediaPipe's 33-point topology)
_SWAP = list(range(33))
for a, b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
             (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]:
    _SWAP[a], _SWAP[b] = b, a
SWAP = np.array(_SWAP)

MIN_VISIBILITY = 0.35


def mirror(img_lm: np.ndarray, world_lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a pose horizontally (x -> -x) and swap left/right joints."""
    a = img_lm[SWAP].copy()
    a[:, 0] = -a[:, 0]
    w = world_lm[SWAP].copy()
    w[:, 0] = -w[:, 0]
    return a, w


def _angle(a, b, c) -> float:
    """Angle at b (degrees) between b->a and b->c, in 2-D or 3-D."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _orient(a, b) -> tuple[float, float]:
    """Unit direction a->b as (cos, sin) — wrap-free orientation encoding."""
    d = b - a
    n = np.linalg.norm(d) + 1e-9
    return float(d[0] / n), float(d[1] / n)


def pose_is_usable(img_lm: np.ndarray) -> bool:
    if img_lm is None or not np.any(img_lm):
        return False
    upper = img_lm[[L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP], 3]
    return float(np.mean(upper)) >= MIN_VISIBILITY


def pose_features(img_lm: np.ndarray, world_lm: np.ndarray) -> np.ndarray:
    """Fixed-length feature vector from one pose (see module docstring)."""
    p = img_lm[:, :2].astype(np.float64)
    vis = img_lm[:, 3]
    sh_mid = (p[L_SH] + p[R_SH]) / 2
    hip_mid = (p[L_HIP] + p[R_HIP]) / 2
    torso = np.linalg.norm(sh_mid - hip_mid) + 1e-6
    q = (p - hip_mid) / torso                       # origin at hips, unit = torso length

    f: list[float] = []
    f += q[KEY_JOINTS].ravel().tolist()             # 34 normalised coordinates

    # Joint angles
    for a, b, c in [(L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR),      # elbows
                    (L_HIP, L_SH, L_EL), (R_HIP, R_SH, R_EL),    # shoulders
                    (L_SH, L_HIP, L_KN), (R_SH, R_HIP, R_KN),    # hips
                    (L_HIP, L_KN, L_AN), (R_HIP, R_KN, R_AN)]:   # knees
        f.append(_angle(q[a], q[b], q[c]) / 180.0)

    # Segment orientations
    for a, b in [(hip_mid, sh_mid), (q[L_SH], q[R_SH]), (q[L_HIP], q[R_HIP]),
                 (q[L_EL], q[L_WR]), (q[R_EL], q[R_WR]),
                 (q[L_SH], q[L_EL]), (q[R_SH], q[R_EL]),
                 (q[L_WR], q[L_IDX]), (q[R_WR], q[R_IDX]),
                 (q[L_KN], q[L_AN]), (q[R_KN], q[R_AN])]:
        if a is hip_mid:
            a, b = np.zeros(2), (sh_mid - hip_mid) / torso
        f += _orient(a, b)

    # Hands (the bat lives between them)
    hands = (q[L_WR] + q[R_WR]) / 2
    head = q[NOSE]
    shm = (q[L_SH] + q[R_SH]) / 2
    f += [hands[0] - head[0], hands[1] - head[1], hands[0] - shm[0], hands[1] - shm[1],
          float(np.linalg.norm(q[L_WR] - q[R_WR]))]

    # Feet / stance
    f += [q[L_AN][0] - q[R_AN][0], q[L_AN][1] - q[R_AN][1],
          abs(q[L_AN][0] - q[R_AN][0]), head[0] - (q[L_AN][0] + q[R_AN][0]) / 2]

    # 3-D world landmarks (metres, hip-centred), torso-normalised
    w = world_lm[:, :3].astype(np.float64)
    w_sh = (w[L_SH] + w[R_SH]) / 2
    w_hip = (w[L_HIP] + w[R_HIP]) / 2
    wt = np.linalg.norm(w_sh - w_hip) + 1e-6
    wq = (w - w_hip) / wt
    f += wq[KEY_JOINTS].ravel().tolist()            # 51
    for a, b, c in [(L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR), (L_HIP, L_KN, L_AN), (R_HIP, R_KN, R_AN)]:
        f.append(_angle(wq[a], wq[b], wq[c]) / 180.0)
    sh_line, hip_line = wq[R_SH] - wq[L_SH], wq[R_HIP] - wq[L_HIP]
    # Shoulder vs hip rotation around the vertical axis (x-z plane)
    f.append(float(np.arctan2(sh_line[2], sh_line[0]) - np.arctan2(hip_line[2], hip_line[0])) / np.pi)

    # Reliability
    f += [float(np.mean(vis[[L_SH, R_SH, L_EL, R_EL, L_WR, R_WR]])),
          float(np.mean(vis[[L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]]))]
    return np.nan_to_num(np.asarray(f, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def pose_features_with_mirror(img_lm: np.ndarray, world_lm: np.ndarray) -> np.ndarray:
    """(2, D) array: the pose and its mirror image."""
    m_img, m_world = mirror(img_lm, world_lm)
    return np.stack([pose_features(img_lm, world_lm), pose_features(m_img, m_world)])
