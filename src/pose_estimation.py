import mediapipe as mp
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE, min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE)

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def get_landmarks(self, results):
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None