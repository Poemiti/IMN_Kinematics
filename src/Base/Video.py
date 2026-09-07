# src/Base/Video.py

from pathlib import Path
import cv2
from .File import File

class Video: 

    def __init__(self, video_path: Path):

        self.file = File(video_path)
        self.is_openable = True

        # open video to get properties
        cap = cv2.VideoCapture(str(video_path))

        # verify state of the video
        if not cap.isOpened():
            self.is_openable = False
            print(f"[ERROR] {self.path}: cannot be opened")

        self.date = self.file.date
        self.name = self.file.name
        self.path = self.file.path
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()