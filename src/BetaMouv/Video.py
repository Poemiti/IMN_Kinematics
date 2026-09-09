# src/BetaMouv/Video.py


from src.Base.Video import Video as BaseVideo
from .File import File

class Video(BaseVideo): 

    def __init__(self, video_path):
        super().__init__(video_path)

        self.file = File(self.path)