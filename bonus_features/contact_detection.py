import numpy as np
import mediapipe as mp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def detect_contact_moment(current_landmarks, previous_landmarks, mp_pose, threshold=config.CONTACT_WRIST_VELOCITY_THRESHOLD):
    if not previous_landmarks:
        return False

    # Calculate wrist velocity
    current_wrist_x = current_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
    current_wrist_y = current_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    
    prev_wrist_x = previous_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
    prev_wrist_y = previous_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    
    wrist_velocity = np.sqrt((current_wrist_x - prev_wrist_x)**2 + (current_wrist_y - prev_wrist_y)**2)

    # A simple heuristic: contact moment is when wrist velocity peaks (or is above a certain threshold)
    # This is a very basic approximation and would need more sophisticated logic for accuracy.
    if wrist_velocity > threshold:
        return True
    return False
