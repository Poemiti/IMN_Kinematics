# src/BetaMouv/Prediction.py

from src.Base.Prediction import Prediction as BasePrediction
from src.Base.Video import Video

from pathlib import Path

class Prediction(BasePrediction): 

    def __init__(self, 
                 video: Video, 
                 paths: dict[Path], 
                 clip_duration: int,):

        super().__init__(video)
        self.paths = paths

        output_clips_dir = self.paths["raw_clips"] / self.video.name
        output_csv_dir = self.paths["dlc"] / self.video.name
        output_csv_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------- video splitting --------------------------------------------------
        
        print(f"\nSplitting video : {self.video.name}")
        print(f"clip duration: {clip_duration}")

        if not output_clips_dir.exists() : 
            self.split_video(input_path= self.video.path, 
                            output_dir= output_clips_dir, 
                            CLIP_DURATION= clip_duration)
        else : 
            print(f"Has already been splitted !")

        # ----------------------------------------------- prediction of each clip --------------------------------------------------

        for clip_path in output_clips_dir.iterdir():

            clip = Video(clip_path)

            if not clip.is_openable : 
                continue

            clip_number = clip_path.stem[-7:]
            csv_path = output_csv_dir / f"pred_results_{self.video.name}_{clip_number}.csv"

            print(f"\nPrediction of clip : {clip_path.stem}\n")

            self.dlc_predict(model_path=self.paths["model"],
                            video_path=clip_path,
                            output_csv_path=csv_path)
        