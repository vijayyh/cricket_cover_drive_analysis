import os
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from bonus_features.skill_grade_prediction import predict_skill_grade
import matplotlib.pyplot as plt
import io
import base64

def generate_elbow_angle_plot(all_metrics):
    if not all_metrics:
        return None

    elbow_angles = [m['elbow_angle'] for m in all_metrics if 'elbow_angle' in m]
    if not elbow_angles:
        return None

    frames = range(len(elbow_angles))

    plt.figure(figsize=(10, 6))
    plt.plot(frames, elbow_angles, label='Elbow Angle')
    plt.xlabel('Frame')
    plt.ylabel('Elbow Angle (degrees)')
    plt.title('Elbow Angle Over Time')
    plt.grid(True)
    plt.legend()

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def generate_new_html_report(evaluation_data, all_metrics, smoothness_plot_path, output_dir):
    # Create a pandas DataFrame from the metrics data
    df = pd.DataFrame(all_metrics)

    # Calculate summary statistics
    metrics_summary_html = "<p>No metrics data available to generate summary.</p>"
    if not df.empty:
        # Define the columns we want to summarize
        cols_to_summarize = ["elbow_angle", "spine_lean", "head_over_knee", "foot_direction"]
        # Filter to only columns that actually exist in the DataFrame
        existing_cols = [col for col in cols_to_summarize if col in df.columns]
        
        if existing_cols:
            summary_df = df[existing_cols].describe().round(2)
            # Rename columns for better readability in the report
            summary_df.rename(columns={
                "elbow_angle": "Elbow Angle",
                "spine_lean": "Spine Lean",
                "head_over_knee": "Head Over Knee",
                "foot_direction": "Foot Direction"
            }, inplace=True)
            # Convert the summary DataFrame to an HTML table
            metrics_summary_html = summary_df.to_html(classes='table', border=0)

    # Predict skill grade
    skill_grade = predict_skill_grade(evaluation_data)

    # Generate elbow angle plot as a base64 string to embed in HTML
    elbow_angle_plot_base64 = generate_elbow_angle_plot(all_metrics)

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__))))
    template = env.get_template("new_report_template.html")

    # Render the HTML template with data
    html_content = template.render(
        annotated_video_path="annotated_video.mp4",  # Relative path for portability
        metrics_summary=summary_df.to_dict(orient='index') if not df.empty and existing_cols else {},
        elbow_angle_plot_base64=elbow_angle_plot_base64,
        skill_grade=skill_grade,
        evaluation_data=evaluation_data
    )

    # Save the HTML report
    report_path = os.path.join(output_dir, "new_analysis_report.html")
    with open(report_path, "w") as f:
        f.write(html_content)

    return report_path