# src/Base/Video.py

from pathlib import Path
import cv2
from .File import File

class Video: 

    def __init__(self, video_path: Path):

        self.file = File(video_path)

        # open video to get properties
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.date = self.file.date
        self.name = self.file.name
        self.path = self.file.path
        
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()