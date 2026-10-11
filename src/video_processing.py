import cv2


def setup_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None, None, None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 240:     # GIFs / broken headers
        fps = 25.0
    return cap, width, height, fps


def setup_video_writer(output_path, width, height, fps):
    """H.264 so browsers can play the result; falls back to MPEG-4 Part 2."""
    for code in ("avc1", "H264", "mp4v"):
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*code), fps, (width, height))
        if out.isOpened():
            return out
    raise RuntimeError("No usable video codec found for the annotated output")
