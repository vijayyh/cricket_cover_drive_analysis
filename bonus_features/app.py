import streamlit as st
import os
import sys
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import analyze_video, download_video
from bonus_features.skill_grade_prediction import predict_skill_grade
import config

st.title("AthleteRise - AI-Powered Cricket Analysis")

st.write("Upload a cricket video to analyze the cover drive.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

if uploaded_file is not None:
    # Create the output directory if it doesn't exist
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    # Save the uploaded video to a temporary path
    video_path = os.path.join(config.OUTPUT_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Video uploaded successfully to {video_path}")

    st.subheader("Analysis Progress")

    progress_bar = st.progress(0) 
    status_text = st.empty()
    
    # Call the analysis function and pass the progress bar and status text elements
    html_report_path, smoothness_plot_path = analyze_video(video_path, progress_bar, status_text)
    progress_bar.progress(100) # Ensure it completes
    st.success("Video analysis complete!")

    # --- Results Display Section ---

    annotated_video_path = os.path.join(config.OUTPUT_DIR, "annotated_video.mp4")
    if os.path.exists(annotated_video_path):
        st.video(annotated_video_path)
        with open(annotated_video_path, "rb") as file:
            st.download_button(
                label="Download Annotated Video",
                data=file.read(),
                file_name="annotated_video.mp4",
                mime="video/mp4"
            )
    else:
        st.warning("Annotated video not found. Please ensure analysis completed successfully.")

    # Display the Matplotlib graph
    if os.path.exists(smoothness_plot_path):
        st.subheader("Elbow Angle Over Time")
        st.image(smoothness_plot_path, caption="Elbow Angle Over Time")

    # html_report_path = os.path.join(config.OUTPUT_DIR, "new_analysis_report.html") # This is now returned by analyze_video

    # Check if all output files exist before trying to display them
    if os.path.exists(html_report_path):
        with open(html_report_path, "r", encoding='latin-1') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=1200, scrolling=True)

        with open(html_report_path, "rb") as f:
            st.download_button(
                label="Download Analysis Report (HTML)",
                data=f.read(),
                file_name="new_analysis_report.html",
                mime="text/html"
            )
    else:
        st.error("Analysis failed. Could not find the necessary output files. Please check the console for errors.")