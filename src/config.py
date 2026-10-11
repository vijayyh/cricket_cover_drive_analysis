# config.py — paths are resolved from the project root so every entry point
# (main.py, backend/app.py, training scripts) works regardless of the CWD.
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
UPLOAD_DIR = os.path.join(ROOT, "uploads")
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_PATH = os.path.join(MODEL_DIR, "shot_classifier_report.json")

# MediaPipe Pose (video tracking mode)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
POSE_MODEL_COMPLEXITY = 1

# Longest clip we analyse; longer uploads are trimmed to this many seconds.
MAX_VIDEO_SECONDS = 30

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".gif"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
