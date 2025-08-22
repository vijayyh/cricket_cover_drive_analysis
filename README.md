# AthleteRise - AI-Powered Cricket Analytics

This project builds a Python-based system for real-time cover drive analysis from full cricket videos, performing frame-by-frame pose estimation and outputting an annotated video with live overlays and a final shot evaluation.

## Objective

Build a Python-based system that processes the entire cricket video in real time (no screenshots/keyframe exports), performs pose estimation frame-by-frame, and outputs an annotated video with live overlays and a final shot evaluation.

## Video to Analyze

YouTube Short: https://youtube.com/shorts/vSX3IRxGnNY

## Scope & Requirements (Base)

1.  **Full Video Processing (Real-Time Flow)**
    *   Download input video and process all frames sequentially (OpenCV).
    *   Normalize FPS/resolution if needed but preserve real-time or near-real-time flow.
    *   Output: a single annotated .mp4 (or .avi) saved to `/output/`.
2.  **Pose Estimation (Per Frame)**
    *   Use MediaPipe, OpenPose, or similar.
    *   Extract keypoints each frame for: head, shoulders, elbows, wrists, hips, knees, ankles.
    *   Gracefully handle missing joints/occlusions.
3.  **Biomechanical Metrics (Per Frame or Rolling)**
    Compute and log:
    *   Front elbow angle (shoulder–elbow–wrist)
    *   Spine lean (hip–shoulder line vs. vertical)
    *   Head-over-knee vertical alignment (projected distance)
    *   Front foot direction (toe/foot angle vs. crease or video x-axis surrogate)
    (Bat tracking is not required in base scope.)
4.  **Live Overlays in the Output Video**
    *   Draw pose skeleton on each frame.
    *   Display real-time metric readouts (e.g., “Elbow: 115°”).
    *   Short feedback cues when thresholds are breached:
        *   ✅ “Good elbow elevation”
        *   ❌ “Head not over front knee”
5.  **Final Shot Evaluation (End of Video)**
    *   Compute and save a summary score (1–10) for:
        *   Footwork
        *   Head Position
        *   Swing Control
        *   Balance
        *   Follow-through
    *   Include 1–2 lines of actionable feedback per category.
    *   Save summary to `evaluation.json` or `evaluation.txt`.

## Deliverables

*   `cover_drive_analysis_realtime.py` (main script - likely `src/main.py`)
*   `/output/`
    *   `annotated_video.mp4` (with overlays, full-length)
    *   `evaluation.json` (or `.txt`) with category scores & comments
    *   `elbow_angle_plot.png` (example matplotlib graph)
*   `requirements.txt` (or `environment.yml`)
*   `README.md` (this file)
    *   Setup & run instructions
    *   Notes on assumptions/limitations

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AthleteRise.git
    cd AthleteRise
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
There are two ways to run the analysis:

### Option 1: Run via the Command Line (main.py)

1.  **Prepare the script:** The `src/main.py` script is configured to run directly from the command line. It will download the video from the URL specified in `config.py` and perform the analysis. You can edit the `if __name__ == '__main__':` block in `src/main.py` to analyze a local file instead.

2.  **Run the analysis:** From the project's root directory, execute the following command:
    ```bash
    python src/main.py
    ```
    The script will download the video, process it, and save the output files to the `/output/` directory.


### Option 2: Run the Streamlit Web Application
The web app provides an interactive interface to upload your own videos.
    ```bash
    streamlit run bonus_features/streamlit_app.py
    ```

## Output

*   An annotated video (`annotated_video.mp4` or `.avi`) will be saved to the `/output/` directory. This video will include pose overlays and real-time metric readouts.
*   A summary evaluation file (`evaluation.json` or `evaluation.txt`) with category scores and comments will be saved to the `/output/` directory.
*   A graph visualizing biomechanical metrics (e.g., `elbow_angle_plot.png` for elbow angle over time) generated using Matplotlib will be saved in the `/output/` directory.

## Project Structure

```
AthleteRise/
├── src/
│   ├── main.py
│   ├── video_processing.py
│   ├── pose_estimation.py
│   ├── biomechanical_metrics.py
│   ├── overlay_utils.py
│   └── evaluation.py
├── bonus_features/
│   ├── phase_segmentation.py
│   ├── contact_detection.py
│   ├── temporal_smoothness.py
│   ├── performance_target.py
│   ├── reference_comparison.py
│   ├── bat_tracking.py
│   ├── skill_grade_prediction.py
│   ├── streamlit_app.py
│   ├── robustness_ux.py
│   └── report_export.py
├── output/
├── GEMINI.md
├── README.md
└── requirements.txt
```

## Notes on Assumptions/Limitations

*   The system assumes the input video is a cricket video, specifically for cover drive analysis.
*   Bat tracking is not required in the base scope.
*   The system handles missing joint detections/occlusions gracefully.
*   The system aims for real-time or near-real-time processing.
*   The analysis is based on a single video and may not be representative of a player's overall skill.
*   The evaluation metrics and scoring are based on general cricketing principles and can be customized.
