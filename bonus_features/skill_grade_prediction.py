import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def predict_skill_grade(evaluation_data):
    footwork_score = evaluation_data["Footwork"]["score"]
    head_position_score = evaluation_data["Head Position"]["score"]
    swing_control_score = evaluation_data["Swing Control"]["score"]
    balance_score = evaluation_data["Balance"]["score"]
    follow_through_score = evaluation_data["Follow-through"]["score"]

    avg_score = (footwork_score + head_position_score + swing_control_score + balance_score + follow_through_score) / 5

    if avg_score >= config.ADVANCED_SCORE:
        return "Advanced"
    elif avg_score >= config.INTERMEDIATE_SCORE:
        return "Intermediate"
    else:
        return "Beginner"
