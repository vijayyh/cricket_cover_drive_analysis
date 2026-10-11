"""
src/technique.py
================
Rule-based technique assessment from measured pose geometry.

There is no labelled "shot quality" data in this project, so instead of a model
trained on made-up numbers this module measures real angles and positions from
the batsman's pose around the moment of impact and scores them against
shot-specific coaching targets. Every score comes with the measured value and
the target range, so the result is transparent and explainable.

2-D pose measurements depend on the camera angle; side-on footage works best.
"""

from __future__ import annotations

import numpy as np

from src.shot_pose import (L_AN, L_EL, L_HIP, L_KN, L_SH, L_WR, NOSE, R_AN, R_EL,
                           R_HIP, R_KN, R_SH, R_WR, _angle, pose_is_usable)

FRONT_FOOT = {"Cover Drive", "Straight Drive"}
BACK_FOOT = {"Cut Shot", "Pull Shot"}

# (ideal_low, ideal_high, zero_low, zero_high) per shot — trapezoid scoring
KNEE_TARGET = {
    "Cover Drive": (110, 155, 80, 178), "Straight Drive": (115, 160, 85, 178),
    "Leg Glance": (120, 165, 90, 180), "Cut Shot": (130, 175, 95, 181), "Pull Shot": (125, 175, 90, 181),
}
LEAN_TARGET = {
    "Cover Drive": (8, 32, -5, 55), "Straight Drive": (5, 28, -8, 50),
    "Leg Glance": (3, 28, -10, 50), "Cut Shot": (0, 30, -12, 55), "Pull Shot": (-5, 25, -20, 50),
}
ELBOW_TARGET = {
    "Cover Drive": (110, 170, 60, 181), "Straight Drive": (115, 172, 65, 181),
    "Leg Glance": (90, 160, 50, 181), "Cut Shot": (120, 176, 70, 181), "Pull Shot": (100, 170, 55, 181),
}
STANCE_TARGET = {   # 2-D ankle distance in torso lengths (robust to the stride pointing at the camera)
    "Cover Drive": (1.1, 2.2, 0.4, 3.2), "Straight Drive": (1.0, 2.1, 0.4, 3.0),
    "Leg Glance": (0.8, 1.9, 0.3, 2.8), "Cut Shot": (0.7, 1.8, 0.25, 2.8), "Pull Shot": (0.7, 1.9, 0.25, 2.9),
}
FOLLOW_TARGET = {   # hands height above shoulders after impact, torso lengths
    "Cover Drive": (0.25, 1.2, -0.6, 2.0), "Straight Drive": (0.35, 1.3, -0.5, 2.0),
    "Pull Shot": (0.1, 1.2, -0.7, 2.0), "Cut Shot": (-0.3, 0.9, -1.1, 1.8), "Leg Glance": (-0.5, 0.6, -1.3, 1.5),
}


def _trap(x: float, lo: float, hi: float, z_lo: float, z_hi: float) -> float:
    """Trapezoid membership: 1 inside [lo, hi], linear to 0 at z_lo / z_hi."""
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return float(np.clip((x - z_lo) / (lo - z_lo + 1e-9), 0, 1))
    return float(np.clip((z_hi - x) / (z_hi - hi + 1e-9), 0, 1))


def frame_measurements(lm: np.ndarray) -> dict:
    """Geometry of a single pose (pixel landmarks, y grows downward)."""
    p = lm[:, :2].astype(np.float64)
    sh = (p[L_SH] + p[R_SH]) / 2
    hip = (p[L_HIP] + p[R_HIP]) / 2
    torso = float(np.linalg.norm(sh - hip)) + 1e-6
    knees = [_angle(p[L_HIP], p[L_KN], p[L_AN]), _angle(p[R_HIP], p[R_KN], p[R_AN])]
    elbows = [_angle(p[L_SH], p[L_EL], p[L_WR]), _angle(p[R_SH], p[R_EL], p[R_WR])]
    feet_mid_x = (p[L_AN, 0] + p[R_AN, 0]) / 2
    hands = (p[L_WR] + p[R_WR]) / 2
    # forward lean = angle of hip->shoulder from vertical, sign toward the head side
    d = sh - hip
    lean = float(np.degrees(np.arctan2(abs(d[0]), -d[1] + 1e-9)))
    return {
        "front_knee": min(knees),
        "lead_elbow": max(elbows),
        "spine_lean": lean,
        "stance_ratio": float(np.linalg.norm(p[L_AN] - p[R_AN])) / torso,
        "head_offset": abs(p[NOSE, 0] - feet_mid_x) / torso,
        "hands_height": (sh[1] - hands[1]) / torso,
        "nose": p[NOSE] / torso,
    }


def _aspect(name, score, measured, unit, target, tip_good, tip_low, tip_high, value):
    lo, hi = target
    if score >= 0.8:
        tip = tip_good
    else:
        tip = tip_low if value < lo else tip_high
    return {
        "aspect": name,
        "score": round(10 * score, 1),
        "measured": measured,
        "target": f"{lo:g}–{hi:g}{unit}",
        "feedback": tip,
    }


def assess(shot: str, poses: list, impact_index: int | None, fps: float = 25.0) -> dict | None:
    """
    poses        list of (33,4) pixel-landmark arrays or None, one per frame
    impact_index frame index of the key moment (from the shot recogniser)
    """
    valid = [i for i, lm in enumerate(poses) if lm is not None and pose_is_usable(lm)]
    if not valid:
        return None
    shot = shot if shot in KNEE_TARGET else "Cover Drive"
    if impact_index not in valid:
        target = impact_index if impact_index is not None else valid[len(valid) // 2]
        impact_index = min(valid, key=lambda i: abs(i - target))

    fps = fps if fps and fps > 0 else 25.0
    win = [i for i in valid if abs(i - impact_index) <= int(0.25 * fps)] or [impact_index]
    after = [i for i in valid if impact_index <= i <= impact_index + int(0.9 * fps)] or [impact_index]
    m_win = [frame_measurements(poses[i]) for i in win]
    imp = {k: float(np.median([m[k] for m in m_win])) for k in
           ("front_knee", "lead_elbow", "spine_lean", "stance_ratio", "head_offset")}
    follow = max(frame_measurements(poses[i])["hands_height"] for i in after)
    nose_path = np.array([m["nose"] for m in m_win])
    stillness = float(np.linalg.norm(nose_path.std(0))) if len(nose_path) > 1 else 0.0
    single_frame = len(valid) == 1

    aspects = []

    # Footwork: stride length + knee flex at impact (feedback names the weaker of the two)
    st = STANCE_TARGET[shot]
    kn = KNEE_TARGET[shot]
    s_st = _trap(imp["stance_ratio"], *st)
    s_kn = _trap(imp["front_knee"], *kn)
    foot_word = "back" if shot in BACK_FOOT else "front"
    score = 0.5 * s_st + 0.5 * s_kn
    if score >= 0.8:
        tip = f"Good base: {foot_word}-foot movement and knee flex suit a {shot.lower()}."
    elif s_kn <= s_st:
        tip = (f"Knee is very bent ({imp['front_knee']:.0f}°). Stay taller through the shot to keep the bat path clean."
               if imp["front_knee"] < kn[0] else
               f"Leg is nearly straight ({imp['front_knee']:.0f}°). Bend the {foot_word} knee and get closer to the pitch of the ball.")
    else:
        tip = (f"Stride is short ({imp['stance_ratio']:.1f} torso-lengths). Move your {foot_word} foot further towards the line of the ball."
               if imp["stance_ratio"] < st[0] else
               f"Stride is very long ({imp['stance_ratio']:.1f} torso-lengths). A shorter step keeps you balanced and in control.")
    aspects.append({
        "aspect": "Footwork", "score": round(10 * score, 1),
        "measured": f"knee {imp['front_knee']:.0f}°, stride {imp['stance_ratio']:.1f} torso-lengths",
        "target": f"knee {kn[0]:g}–{kn[1]:g}°, stride {st[0]:g}–{st[1]:g}",
        "feedback": tip,
    })

    # Head position: over the base and still (feedback names the weaker of the two)
    s_head = _trap(imp["head_offset"], 0, 0.35, -1, 1.1)
    s_still = 1.0 if single_frame else _trap(stillness, 0, 0.06, -1, 0.3)
    score = 0.6 * s_head + 0.4 * s_still
    if score >= 0.8:
        tip = "Head is over your base and steady at contact, which is great for judging line and length."
    elif s_head <= s_still:
        tip = "Head is falling away from your base. Lead with the head and keep it over the front knee at contact."
    else:
        tip = "Head is moving through the shot. Keep it still from the bowler's release to contact."
    aspects.append({
        "aspect": "Head Position", "score": round(10 * score, 1),
        "measured": f"{imp['head_offset']:.2f} torso-lengths from base"
                    + ("" if single_frame else f", sway {stillness:.2f}"),
        "target": "offset 0–0.35" + ("" if single_frame else ", sway 0–0.06"),
        "feedback": tip,
    })

    # Bat swing / arm extension at impact
    el = ELBOW_TARGET[shot]
    s_el = _trap(imp["lead_elbow"], *el)
    aspects.append(_aspect(
        "Swing Control", s_el, f"lead elbow {imp['lead_elbow']:.0f}°", "°", el[:2],
        "Arms extend well through the ball: a full, controlled swing.",
        f"Arms are cramped ({imp['lead_elbow']:.0f}°). Create room and extend through the line of the ball.",
        "Arms are locked straight. Keep a slight bend to control the bat face.",
        imp["lead_elbow"]))

    # Balance: spine lean
    ln = LEAN_TARGET[shot]
    s_ln = _trap(imp["spine_lean"], *ln)
    aspects.append(_aspect(
        "Balance", s_ln, f"spine lean {imp['spine_lean']:.0f}°", "°", ln[:2],
        "Upper body is balanced over the hips.",
        "Very upright. Lean slightly into the shot to transfer your weight.",
        f"Leaning too far ({imp['spine_lean']:.0f}°). Stay more upright to stay balanced after contact.",
        imp["spine_lean"]))

    # Follow-through (needs frames after impact)
    if not single_frame:
        fo = FOLLOW_TARGET[shot]
        s_fo = _trap(follow, *fo)
        aspects.append(_aspect(
            "Follow-through", s_fo, f"hands {follow:+.2f} torso-lengths vs shoulders", "", fo[:2],
            "Follow-through finishes in a strong, complete position.",
            "Follow-through is cut short. Let the hands carry on past the shoulders.",
            "Hands finish very high. Make sure that is control, not a slog.",
            follow))

    overall = float(np.mean([a["score"] for a in aspects]))
    grade = ("Excellent" if overall >= 8 else "Good" if overall >= 6.5
             else "Average" if overall >= 5 else "Needs Work")
    return {
        "overall": round(overall, 1),
        "grade": grade,
        "aspects": aspects,
        "impact_frame": int(impact_index),
        "method": "Rule-based biomechanics (measured pose angles vs coaching targets for this shot)",
    }
