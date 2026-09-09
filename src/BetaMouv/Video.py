# src/BetaMouv/Video.py


from src.Base.Video import Video as BaseVideo
from .File import File

class Video(BaseVideo): 

    file_cls = File 