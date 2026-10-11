"""MediaPipe Pose wrapper for video (tracking mode)."""
import cv2

import src.config as config
from src.shot_recognizer import landmarks_to_array, world_to_array

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:          # pragma: no cover - depends on the install
    mp_pose = None


class PoseEstimator:
    def __init__(self):
        if mp_pose is None:
            raise ImportError("MediaPipe is not installed correctly. "
                              "Run: pip install mediapipe==0.10.11")
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.POSE_MODEL_COMPLEXITY,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def process(self, frame_bgr):
        """Returns (img_lm (33,4) pixels, world_lm (33,4)) or (None, None)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None, None
        return (landmarks_to_array(res.pose_landmarks.landmark, w, h),
                world_to_array(res.pose_world_landmarks.landmark))

    def close(self):
        self.pose.close()
