import os
import sys
import cv2
import time
import json
import mediapipe as mp

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_processing import download_video, setup_video_capture, setup_video_writer
from src.pose_estimation import PoseEstimator
from src.biomechanical_metrics import calculate_metrics
from src.overlay_utils import draw_pose_landmarks, display_metrics_on_frame
from src.evaluation import save_evaluation
from bonus_features.phase_segmentation import detect_phase
from bonus_features.contact_detection import detect_contact_moment
from bonus_features.temporal_smoothness import TemporalSmoothness
from bonus_features.bat_tracking import detect_and_draw_bat
from bonus_features.reference_comparison import compare_to_reference
from bonus_features.new_report_export import generate_new_html_report

import config

def analyze_video(video_path, progress_bar=None, status_text=None):
    pose_estimator = PoseEstimator()
    mp_pose = mp.solutions.pose

    cap, width, height, fps = setup_video_capture(video_path)
    if cap is None:
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = setup_video_writer(os.path.join(config.OUTPUT_DIR, 'annotated_video.mp4'), width, height, fps)

    footwork_scores = []
    head_position_scores = []
    swing_control_scores = []
    balance_scores = []
    follow_through_scores = []
    all_metrics = []

    previous_landmarks = None
    temporal_smoothness_analyzer = TemporalSmoothness()

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if progress_bar and total_frames > 0:
            progress = frame_count / total_frames
            progress_bar.progress(min(int(progress * 100), 100))
        if status_text:
            status_text.text(f"Processing frame {frame_count} of {total_frames}...")

        image, results = pose_estimator.process_frame(frame)

        try:
            landmarks = pose_estimator.get_landmarks(results)
            if landmarks:
                metrics = calculate_metrics(landmarks, mp_pose)
                if metrics:
                    all_metrics.append(metrics)
                    display_metrics_on_frame(image, metrics)
                    temporal_smoothness_analyzer.record_elbow_angle(metrics['elbow_angle'])

                    # Phase Segmentation
                    current_phase = detect_phase(landmarks, previous_landmarks, mp_pose)
                    cv2.putText(image, f"Phase: {current_phase}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Contact Moment Detection
                    if detect_contact_moment(landmarks, previous_landmarks, mp_pose):
                        cv2.putText(image, "CONTACT!", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    # Reference Comparison - Removed as per user request
                    # deviation_score, ref_feedback = compare_to_reference(metrics)
                    # if ref_feedback:
                    #     all_reference_feedback.extend(ref_feedback)
                    #     for i, fb in enumerate(ref_feedback):
                    #         cv2.putText(image, fb, (10, 270 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                    previous_landmarks = landmarks

                    # Collect scores for final evaluation
                    if config.ELBOW_ANGLE_GOOD_THRESHOLD[0] < metrics['elbow_angle'] < config.ELBOW_ANGLE_GOOD_THRESHOLD[1]:
                        swing_control_scores.append(1)
                    else:
                        swing_control_scores.append(0)

                    if metrics['head_over_knee'] < config.HEAD_OVER_KNEE_GOOD_THRESHOLD:
                        head_position_scores.append(1)
                    else:
                        head_position_scores.append(0)
                    
                    if config.FOOT_DIRECTION_GOOD_RANGE[0] < metrics['foot_direction'] < config.FOOT_DIRECTION_GOOD_RANGE[1]:
                        footwork_scores.append(1)
                    else:
                        footwork_scores.append(0)
                    
                    if config.SPINE_LEAN_GOOD_RANGE[0] < metrics['spine_lean'] < config.SPINE_LEAN_GOOD_RANGE[1]:
                        balance_scores.append(1)
                    else:
                        balance_scores.append(0)
                    
                    if metrics['wrist'][0] > metrics['shoulder'][0]:
                        follow_through_scores.append(1)
                    else:
                        follow_through_scores.append(0)

                    # Always try to draw landmarks and skeleton if detected
                    if results.pose_landmarks:
                        draw_pose_landmarks(image, results)

                        detect_and_draw_bat(image)

        except Exception as e:
            print(f"Error processing frame: {e}")
            pass
               
        out.write(image)

    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    if status_text:
        status_text.text(f"Analysis complete! Processed {frame_count} frames at {avg_fps:.2f} FPS.")

    print(f"Average FPS: {avg_fps:.2f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save evaluation and get the data back for report generation
    save_evaluation(footwork_scores, head_position_scores, swing_control_scores, balance_scores, follow_through_scores)
    with open(os.path.join(config.OUTPUT_DIR, 'evaluation.json'), 'r') as f:
        evaluation_data = json.load(f)
    
    # Temporal Smoothness Analysis
    smoothness_variance, smoothness_plot_path = temporal_smoothness_analyzer.analyze_and_plot(config.OUTPUT_DIR)
    if smoothness_variance is not None:
        print(f"Elbow Angle Smoothness Variance: {smoothness_variance:.2f}")
        print(f"Elbow Angle Plot saved to: {smoothness_plot_path}")

    # Generate HTML Report
    html_report_path = generate_new_html_report(evaluation_data, all_metrics, smoothness_plot_path, config.OUTPUT_DIR)
    print(f"HTML Report generated at: {html_report_path}")

    return html_report_path, smoothness_plot_path

if __name__ == '__main__':
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        
    # For direct execution, you can specify a video URL or path here
    video_url = config.VIDEO_URL
    video_path = os.path.join(config.OUTPUT_DIR, "input_video.mp4")
    download_video(video_url, video_path)
    analyze_video(video_path)
    
    # Example of how to run if you have a local video file
    # analyze_video("path/to/your/video.mp4")