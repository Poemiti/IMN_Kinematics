# src/DopaMouv/Project.py

from src.Base.Project import Project as BaseProject
from src.Base.PathConfig import PathConfig
import src.Base.gui.database_filter as Data_filter
import src.Base.gui.preprocess_validator as Validator

from .File import File
from .Video import Video
from .Trial import Trial
from .TrialGroup import TrialGroup
from .Leds import Leds
from .Trajectory import TrajectoryDLC, DLCTracking
import src.BetaMouv.Behavior as Behavior

from pathlib import Path
import yaml, joblib, sys, shutil, traceback
from src.utils import process_time
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import numpy as np

import src.utils as u

import matplotlib.pyplot as plt

# nohup python3 -u main.py > main.out &
# tail -f main.out 

class Project(BaseProject):

    def __init__(self, name: str):
        super().__init__(name)

        self.config_dir: Path = Path(f"./config/{self.name}/")

        self.conditions: dict = u.load_config(self.config_dir / "conditions.yaml")
        self.subject_info: dict = u.load_config(self.config_dir / "subject_info.yaml")
        self.project_info: dict = u.load_config(self.config_dir / "project_info.yaml")

        # setup rules
        # self.annotation_rules: dict = u.load_config(self.config_dir / "rules/annotation_rules.yaml")
        # self.clip_duration_rules: dict = u.load_config(self.config_dir / "rules/clip_duration_rules.yaml")
        # self.exclusion_rules: dict = u.load_config(self.config_dir / "rules/exclusion_rules.yaml")
        # self.camera_shift_rules: dict = u.load_config(self.config_dir / "rules/camera_shift_rules.yaml")

        # setup path
        self.paths: PathConfig = PathConfig(project_name=self.name, 
                                            model=Path(self.project_info["model_path"]), 
                                            raw_videos=Path(self.project_info["raw_videos_path"]))

        # # trial group of the project
        # joblib_filenames = list(self.paths.trials_metadata.glob("*.joblib"))
        # if joblib_filenames == []: 
        #     print("No Joblib metadata found")
        #     self.trialgroup = None
        # else: 
        #     try:
        #         self.trialgroup = (TrialGroup(joblib_filenames, self.conditions) if joblib_filenames else None)
        #     except EOFError as e:
        #         print(f"\nWARNING: Could not load joblib files: {e}")
        #         self.trialgroup = None


    @process_time
    def split_trials(self): 

        print(f"""
        =============== Video Splitting ==============
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        records = {}
        all_records=[]
        dataset = list(Data_filter.load_database(File, self.paths.raw_videos, self.paths.database, "*.avi")["filename"])

        print(len(dataset))

        for i, video_path in enumerate(dataset):             

            raw_video = Video(video_path=video_path)

            output_dir = u.make_dir(self.paths.raw / f"{raw_video.file.subject}"/ raw_video.full_name)
            subject = raw_video.file.subject            

            clip_duration = 12.5    # sec

            if subject not in records : 
                records[subject] = []

            rec = {"subject": subject, "raw_path": video_path, "clean_name": raw_video.full_name, "clip_duration": clip_duration}
            records[subject].append(rec)
            all_records.append(rec)

            print(f"\nSplitting video : {raw_video.full_name}")
            print(f"clip duration: {clip_duration}")

            if (list(output_dir.glob("*.mp4"))) == [] : 
                print("Splitting..")
                raw_video.split_video(input_path= raw_video.path, 
                                output_dir= output_dir, 
                                CLIP_DURATION= clip_duration, 
                                CRF=0)
            else : 
                print(f"Has already been splitted !")

        # save metadata
        for subject, data in records.items():

            save_as = self.paths.raw / subject / "metadata.csv"
            df_new = pd.DataFrame(data)

            if save_as.exists():
                df_existing = pd.read_csv(save_as, index_col=0)
                df = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df = df_new
            df.to_csv(save_as, index=False)

    
        # check duplicates
        df = pd.DataFrame(all_records)
        duplicates_cleannames = df[df.duplicated(subset=["clean_name"], keep=False)].sort_values("clean_name")
        duplicates_paths = df[df.duplicated(subset=["raw_path"], keep=False)]

        print("\n========== DUPLICATE CHECK ==========")

        print(f"Unique clean_names: {df['clean_name'].nunique()}")
        print(f"Total videos: {len(df)}")

        print(f"\n Duplicate clean names: {len(duplicates_cleannames)}")
        print(f"Duplicate raw paths: {len(duplicates_paths)}")

        if not duplicates_cleannames.empty:
            print("\nGLOBAL DUPLICATES:")
            print(duplicates_cleannames[["subject", "raw_path", "clean_name"]].to_string(index=False))

        if not duplicates_paths.empty:
            print("\nDUPLICATE RAW PATHS:")
            print(duplicates_paths.to_string(index=False))


