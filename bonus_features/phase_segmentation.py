import numpy as np
import mediapipe as mp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_joint_velocity(current_landmarks, previous_landmarks, joint_name, mp_pose):
    if not previous_landmarks:
        return 0
    
    current_x = current_landmarks[getattr(mp_pose.PoseLandmark, joint_name.upper()).value].x
    current_y = current_landmarks[getattr(mp_pose.PoseLandmark, joint_name.upper()).value].y
    
    prev_x = previous_landmarks[getattr(mp_pose.PoseLandmark, joint_name.upper()).value].x
    prev_y = previous_landmarks[getattr(mp_pose.PoseLandmark, joint_name.upper()).value].y
    
    return np.sqrt((current_x - prev_x)**2 + (current_y - prev_y)**2)

def detect_phase(current_landmarks, previous_landmarks, mp_pose):
    phase = "Unknown"
    
    if not previous_landmarks:
        return "Stance"

    # Simple heuristics for phase detection
    # Stride: Significant hip movement
    hip_velocity = get_joint_velocity(current_landmarks, previous_landmarks, "left_hip", mp_pose)
    if hip_velocity > config.HIP_VELOCITY_THRESHOLD: # Threshold for hip movement
        phase = "Stride"

    # Downswing: Significant elbow/wrist movement
    elbow_velocity = get_joint_velocity(current_landmarks, previous_landmarks, "left_elbow", mp_pose)
    wrist_velocity = get_joint_velocity(current_landmarks, previous_landmarks, "left_wrist", mp_pose)
    if elbow_velocity > config.ARM_VELOCITY_THRESHOLD or wrist_velocity > config.ARM_VELOCITY_THRESHOLD: # Threshold for arm movement
        phase = "Downswing"

    # Impact (very basic, could be improved with bat/ball detection)
    # If downswing is detected and wrist velocity peaks
    if phase == "Downswing" and wrist_velocity > config.IMPACT_WRIST_VELOCITY_THRESHOLD: # Higher threshold for impact
        phase = "Impact"

    # Follow-through: Continued arm movement after impact
    if phase == "Impact" and (elbow_velocity > 0.01 or wrist_velocity > 0.01): # Continued movement
        phase = "Follow-through"

    # Recovery: Reduced movement after follow-through
    if phase == "Follow-through" and hip_velocity < 0.01 and elbow_velocity < 0.01 and wrist_velocity < 0.01:
        phase = "Recovery"

    return phase
