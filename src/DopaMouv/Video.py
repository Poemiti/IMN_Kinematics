# src/DopaMouv/Video.py

from pathlib import Path
import cv2
from .File import File
from src.Base.Video import Video as BaseVideo


class Video(BaseVideo): 

    file_cls = File

    def __init__(self, video_path):
        super().__init__(video_path)

        self.clean_name = self.file.clean_name
