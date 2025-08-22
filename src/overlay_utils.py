import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 

def draw_pose_landmarks(image, results):
    if results.pose_landmarks:
        # Use bright, distinct colors for better visibility
        landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=4)  # bright green, larger
        connection_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=2)  # bright red

        
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_spec,
            connection_drawing_spec=connection_spec
        )

def display_metrics_on_frame(image, metrics):
    cv2.putText(image, f"Elbow Angle: {int(metrics['elbow_angle'])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Spine Lean: {int(metrics['spine_lean'])}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Head Over Knee: {'Good' if metrics['head_over_knee'] < 0.1 else 'Needs Improvement'}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Foot Direction: {int(metrics['foot_direction'])}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
