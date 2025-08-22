from jinja2 import Environment, FileSystemLoader
import os

def generate_html_report(evaluation_data, smoothness_plot_path, output_dir="output"):
    env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = env.get_template("report_template.html")

    # Prepare data for the template
    report_data = {
        "evaluation": evaluation_data,
        "smoothness_plot": os.path.basename(smoothness_plot_path) if smoothness_plot_path else None
    }

    html_content = template.render(report_data)

    report_path = os.path.join(output_dir, "analysis_report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    return report_path