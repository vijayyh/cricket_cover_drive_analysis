import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def detect_and_draw_bat(image):
    # Convert image to HSV for color-based segmentation (example: dark bat)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for dark colors (adjust as needed)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 50]) # Adjust V for darkness

    mask = cv2.inRange(hsv, lower_dark, upper_dark)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find bat-like shapes (elongated, certain area)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > config.BAT_MIN_AREA: # Minimum area for a bat (adjust as needed)
            rect = cv2.minAreaRect(cnt)
            width, height = rect[1]
            if min(width, height) > 0: # Avoid division by zero
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > config.BAT_ASPECT_RATIO_THRESHOLD: # Elongated shape (adjust as needed)
                    # Draw the rotated rectangle (approximating the bat)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(image, [box], 0, (0, 255, 0), 2) # Green color for bat
                    return True # Bat detected
    return False # No bat detected
