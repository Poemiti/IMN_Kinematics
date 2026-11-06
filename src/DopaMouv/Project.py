# src/DopaMouv/Project.py

from src.Base.Project import Project as BaseProject
from src.Base.PathConfig import PathConfig
import src.Base.gui.database_filter as Data_filter

from .File import File
from .Video import Video
from .Trial import Trial
from .TrialGroup import TrialGroup
from .Leds import Leds
# from .Trajectory import TrajectoryDLC, DLCTracking
# import src.BetaMouv.Behavior as Behavior

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
        self.annotation_rules: dict = u.load_config(self.config_dir / "rules/annotation_rules.yaml")
        # self.clip_duration_rules: dict = u.load_config(self.config_dir / "rules/clip_duration_rules.yaml")
        # self.exclusion_rules: dict = u.load_config(self.config_dir / "rules/exclusion_rules.yaml")

        # setup path
        self.paths: PathConfig = PathConfig(project_name=self.name, 
                                            model=Path(self.project_info["model_path"]), 
                                            raw_videos=Path(self.project_info["raw_videos_path"]))

        # trial group of the project
        joblib_filenames = list(self.paths.trials_metadata.glob("*.joblib"))
        if joblib_filenames == []: 
            print("No Joblib metadata found")
            self.trialgroup = None
        else: 
            try:
                self.trialgroup = (TrialGroup(joblib_filenames, self.conditions) if joblib_filenames else None)
            except EOFError as e:
                print(f"\nWARNING: Could not load joblib files: {e}")
                self.trialgroup = None


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



    def build_metadata(self):
        print(f"""
        =============== Build Metadata ===============
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        joblib_files = list(self.paths.trials_metadata.glob("*.joblib"))
        print(f"{len(joblib_files)} joblib metadata files ({self.paths.trials_metadata})")
        res = input("Overwrite (o), update (u) or quit (q) ? : ").strip().lower()

        if res in ("o", ""):
            return self._run_metadata(update=False)
        if res == "u":
            return self._run_metadata(update=True)
        if res == "q":
            print("\nQuit !\n")
            sys.exit()
        raise ValueError(f"'{res}' is not valid, must be 'y', 'n' or 'q'")


    @process_time
    def _run_metadata(self, update: bool):

        if update:
            clip_paths = list(self.paths.raw.rglob("*.mp4"))
        else:
            clip_paths = list(Data_filter.load_database(File, self.paths.raw, self.paths.database, "*.mp4")["filename"])

        trials_by_group, errors = {}, {}

        for clip_path in tqdm(clip_paths, desc="Metadata update" if update else "Metadata init"):
            try:
                trial = self._process_clip(clip_path, update)
                trial.to_yaml_dict()                   # strict check, nothing written yet
            except Exception as e:
                errors[str(clip_path)] = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                continue
            trials_by_group.setdefault(trial.group, []).append(trial)

        #  report and decide BEFORE touching the disk 
        u._save_error_report(output_dir=u.make_dir(self.paths.data_root / "logs"), 
                             errors=errors, update=update)

        n_ok = sum(len(t) for t in trials_by_group.values())
        print(f"\n{n_ok} trials OK, {len(errors)} errors")

        for p, e in list(errors.items())[:10]:
            print(f"  {Path(p).name}: {e}")
        if n_ok == 0 or len(errors) > 0.1 * len(clip_paths):
            raise RuntimeError("Too many failures, nothing was written.")
        if errors and input("Errors found. Save anyway? (y/n) : ").lower() != "y":
            print("Aborted, nothing written.")
            return

        #  backup, then write 

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = self.paths.data_root / "trials_metadata_backup" / f"metabuilding_{stamp}"
        shutil.copytree(self.paths.trials_metadata, backup)
        print(f"Backup: {backup}")

        # save yaml

        print("\nSaving YAML")
        for group, trials in trials_by_group.items():
            for t in tqdm(trials, desc=f"Saving {group}"):
                t.save_yaml()      

        # save joblib

        for group, trials in tqdm(trials_by_group.items(), desc="Saving joblib"):
            u.save(u.make_path(self.paths.trials_metadata, f"{group}.joblib"),
                   lambda p, t=trials: joblib.dump(t, p),)

        if self.trialgroup is None: 
            joblib_filenames = list(self.paths.trials_metadata.glob("*.joblib"))
            self.trialgroup = TrialGroup(joblib_filenames, self.conditions)
        else: 
            self.trialgroup.trials = [trial
                    for trial_list in trials_by_group.values()
                    for trial in trial_list
                ]

    def _process_clip(self, clip_path, update: bool) -> Trial:
        trial = Trial(clip_path=clip_path)
        previous = None

        # get the raw video path it comes from
        raw_video_df = pd.read_csv(trial.clip_path.parent.parent / "metadata.csv")
        raw_path = raw_video_df.loc[raw_video_df["clean_name"] == trial.name[:-8], "raw_path"]
        trial.update(raw_video_path=raw_path)

        # 1. is the clip usable?
        if update:
            previous = Trial(clip_path=clip_path)
            previous.load_yaml()
            readable = previous.trial_outcomes["clip_openable"].success
        else:
            clip = Video(clip_path)
            readable = clip.is_openable and clip.is_readable

        if not readable:
            for stage in trial.STAGES:
                trial.set_success(stage=stage, success=False, reason="Clip not readable")
            trial.set_group(laser_state="UNKNOWN")
            return trial

        trial.set_success(stage="clip_openable", success=True, reason="Clip openable")
        
        # 2. the ONLY step that differs: LED info
        if update:
            trial.update(
                task=previous.task,
                laser_state=previous.laser_state,
                time_pad_off=previous.time_pad_off,
                time_laser_on=previous.time_laser_on,
                time_reward=previous.time_reward,
            )
        else:
            rule_key = {
                "view": trial.camera_view,
                "month": trial.file.date.month,
            }
            annotation = u.match_rule(rule_key, self.annotation_rules)
            trial.set_led_info(Leds(video_path=clip_path, label_studio_annotation=annotation, 
                                    fig_output_path=trial.clip_path.parent / f'{trial.name}.html', 
                                    csv_output_path=trial.clip_path.parent / f'{trial.name}.csv'))

        trial.set_task_success(self.project_info["laser_on_duration"])
        trial.set_group()

        pred_path = trial.file.path.parent / f"pred_results_{trial.name}.csv"
        if pred_path.exists():
            trial.update(pred_path=str(pred_path))
        
        return trial

