# src/BetaMouv/Video.py


from src.Base.Video import Video as BaseVideo
from .File import File
import cv2

class Video(BaseVideo): 

    file_cls = File 


    def extract_1_frame(self, nb:int, output_path):

        cap = cv2.VideoCapture(self.path)

        for i in range(1, nb+1): 
            _, frame = cap.read()

            if i == nb: 
                cv2.imwrite(output_path, frame)
        cap.release()


