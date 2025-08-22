import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def compare_to_reference(metrics):
    # Define ideal ranges for key metrics (these are example values and should be refined)
    ideal_ranges = {
        "elbow_angle": config.ELBOW_ANGLE_GOOD_THRESHOLD,  # Now a range
        "spine_lean": config.SPINE_LEAN_GOOD_RANGE,   
        "head_over_knee": config.HEAD_OVER_KNEE_GOOD_THRESHOLD, 
        "foot_direction": config.FOOT_DIRECTION_GOOD_RANGE 
    }

    deviation_score = 0
    feedback = []

    if metrics:
        # Elbow Angle
        if not (ideal_ranges["elbow_angle"][0] <= metrics["elbow_angle"] <= ideal_ranges["elbow_angle"][1]):
            deviation_score += 1
            feedback.append(f"Elbow angle ({int(metrics['elbow_angle'])}°) is outside the ideal range ({ideal_ranges['elbow_angle'][0]}-{ideal_ranges['elbow_angle'][1]}°).")

        # Spine Lean
        if not (ideal_ranges["spine_lean"][0] <= metrics["spine_lean"] <= ideal_ranges["spine_lean"][1]):
            deviation_score += 1
            feedback.append(f"Spine lean ({int(metrics['spine_lean'])}°) is outside the ideal range ({ideal_ranges['spine_lean'][0]}-{ideal_ranges['spine_lean'][1]}°).")

        # Head Over Knee
        if not (0 <= metrics["head_over_knee"] <= ideal_ranges["head_over_knee"]): # Smaller is better
            deviation_score += 1
            feedback.append(f"Head over knee alignment ({metrics['head_over_knee']:.2f}) is not ideal (should be < {ideal_ranges['head_over_knee']}).")

        # Foot Direction
        if not (ideal_ranges["foot_direction"][0] <= metrics["foot_direction"] <= ideal_ranges["foot_direction"][1]):
            deviation_score += 1
            feedback.append(f"Foot direction ({int(metrics['foot_direction'])}°) is outside the ideal range ({ideal_ranges['foot_direction'][0]}-{ideal_ranges['foot_direction'][1]}°).")

    return deviation_score, feedback