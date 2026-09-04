# src/BetaMouv/Prediction.py

from src.Base.Prediction import Prediction as BasePrediction

from pathlib import Path

class Prediction(BasePrediction): 

    def __init__(self, 
                 video, 
                 output_dir: Path, 
                 clip_duration: int, 
                 fps, 
                 model_path):

        super().__init__(video)

        output_clips_dir = self.paths["raw_clips"] / video.name
        output_csv_dir = self.paths["dlc"] / video.name
        output_csv_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------- video splitting --------------------------------------------------
        
        print(f"\nSplitting video : {self.video.name}")
        print(f"clip duration: {clip_duration}")

        if not output_clips_dir.exists() : 
            self.prediction_class.split_video(input_path= self.video.path, 
                                            output_path= output_clips_dir, 
                                            CLIP_DURATION= clip_duration)
        else : 
            print(f"Has already been splitted !")
        

        # ----------------------------------------------- prediction --------------------------------------------------

        for j, clip_path in enumerate(output_clips_dir.iterdir() ):

            clip_number = clip_path.stem[-7:]
            csv_path = output_csv_dir / f"pred_results_{video.name}_{clip_number}.csv"

            print(f"\n[{j+1}/{i+1}/{len(DATABASE)}]")
            print(f"\nPrediction of clip : {clip_path.stem}\n")

            if not verify_video(clip_path): ############ TODO
                continue

            self.prediction_class.dlc_predict(
                                model_path=self.paths["model"],
                                video_path=clip_path,
                                output_csv_path=csv_path,
                            )
        