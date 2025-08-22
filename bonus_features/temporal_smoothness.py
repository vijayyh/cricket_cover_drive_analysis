import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class TemporalSmoothness:
    def __init__(self):
        self.elbow_angles = []

    def record_elbow_angle(self, angle):
        self.elbow_angles.append(angle)

    def analyze_and_plot(self, output_dir=config.OUTPUT_DIR):
        if not self.elbow_angles:
            return None, None

        # Calculate variance as a smoothness metric
        variance = np.var(self.elbow_angles)

        # Plotting the elbow angle over time
        plt.figure(figsize=(10, 6))
        plt.plot(self.elbow_angles)
        plt.title("Elbow Angle Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Elbow Angle (degrees)")
        plt.grid(True)

        plot_path = os.path.join(output_dir, "elbow_angle_plot.png")
        plt.savefig(plot_path)
        plt.close()

        return variance, plot_path
