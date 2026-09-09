# src/BetaMouv/Project.py

from src.Base.Project import Project as BaseProject
from .Video import Video

from .PathConfig import PathConfig
from .Trial import Trial
from .TrialGroup import TrialGroup
from .Leds import Leds

from pathlib import Path
import yaml

import src.utils as u
import src.BetaMouv.gui.database_filter as data_filter

class Project(BaseProject):

    def __init__(self, name: str):
        super().__init__(name)

        self.config_dir: Path = Path(f"./config/{self.name}/")

        self.conditions: dict = self._load_config(self.config_dir / "conditions.yaml")
        self.subject_info: dict = self._load_config(self.config_dir / "subject_info.yaml")

        # setup rules
        self.annotation_rules: dict = self._load_config(self.config_dir / "rules/annotation_rules.yaml")
        self.clip_duration_rules: dict = self._load_config(self.config_dir / "rules/clip_duration_rules.yaml")
        self.ethology_rules: dict = self._load_config(self.config_dir / "rules/ethology_rules.yaml")
        self.exclusion_rules: dict = self._load_config(self.config_dir / "rules/exclusion_rules.yaml")

        # setup path
        self.paths: PathConfig = PathConfig(project_name=self.name)

    @staticmethod
    def _load_config(filename: Path):
        with open(filename, "r") as cfg_file:
            cfg = yaml.safe_load(cfg_file)
        return cfg



    def split_trials(self): 

        print(f"""
        =============== Video Splitting ==============
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        dataset = data_filter.load_database(self.paths.raw_videos, self.paths.database, "video")

        for i, video_path in enumerate(dataset["filename"].iloc[:]): 

            raw_video = Video(video_path=video_path)

            output_dir = self.paths.raw / raw_video.file.subject / raw_video.name
            output_dir.mkdir(parents=True, exist_ok=True)

            meta = {"month": raw_video.date.month}
            clip_duration = u.match_rule(meta, self.clip_duration_rules)

            print(f"\nSplitting video : {self.video.name}")
            print(f"clip duration: {clip_duration}")

            if not output_dir.exists() : 
                self.split_video(input_path= self.video.path, 
                                output_dir= output_dir, 
                                CLIP_DURATION= clip_duration)
            else : 
                print(f"Has already been splitted !")

        print("Done !")



    def build_metadata(self): 
        import joblib

        print(f"""
        =============== Build Metadata ===============
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        dataset = data_filter.load_database(self.paths.raw, self.paths.database, "video")

        output_dir = self.paths.trials_metadata
        output_dir.mkdir(parents=True, exist_ok=True)

        trials_by_group = {}

        for i, clip_path in enumerate(dataset["filename"].iloc[:]):

            trial = Trial(clip_path=clip_path)

            print(f"Building metadata of: {trial.file.name}")

            # get annotation number
            annotation_meta = {
                "laser_type": trial.file.laser_type, 
                "view": trial.file.camera_view,
                "month": trial.file.date.month,
            }
            label_studio_annotation = u.match_rule(annotation_meta, self.annotation_rules)

            print(f"label studio annotation: {label_studio_annotation}")
            print(f"annotation meta: {annotation_meta}")

            # get Leds info to tell the Laser state (LaserOn, LaserOff)
            leds = Leds(video_path=clip_path, 
                        label_studio_annotation=label_studio_annotation,
                        )

            if (trial.file.camera_view == "left" and leds.cue_type == "CueL2") or \
               (trial.file.camera_view == "right" and leds.cue_type == "CueL1") : 
                print(f"Camera: {trial.file.camera_view} | cue: {leds.cue_type} ! Not compatible\n")
                continue

            trial.set_led_info(leds)
            trial.set_mvt_type(self.subject_info[trial.file.subject]["hemi"])
            trial.set_group()
            trial.save_trial()

            # Add trial to its group
            trials_by_group.setdefault(trial.group, []).append(trial)

        # Save one big joblib per group
        for group, trials in trials_by_group.items():

            joblib.dump(trials, output_dir / f"{group}.joblib")
            print(f"  {group}: {len(trials)}")


        print("\nVisualisation of the proportion of each experimental condition\n")

        # TODO
        # refaire la fonction de metadata report pour afficher le nombre
        # d'essai par groupe
        # u.metadata_report(...)

        print("Done!")




    def run_prediction(self): 

        print(f"""
        ================= Prediction =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        joblib_filenames = self.paths.trials_metadata.glob("*.joblib")
        
        trialgroup = TrialGroup(joblib_filenames, self.conditions)

        for i, trial in enumerate(trialgroup.trials):

            if "FIBER_BROKEN" in trial.file.name:        # for the rat 521
                print("FIBER_BROKEN, skipped")
                continue

            print(f"\n[{i+1}/{len(trialgroup.trials)}]")
            print(f"Prediction of video: {trial.file.name}\n")

            # prediction 
            trial.dlc_predict(model_path=self.paths.model,
                              output_csv_path=trial.file.path.parent / f"{trial.file.name}.csv")

            break

        print("Done!")
            
            
            
    def run_preprocessing(self): 

        joblib_filenames = self.paths.trials_metadata.glob("*.joblib")

        trialgroup = TrialGroup(joblib_filenames, self.conditions)

        for t in trialgroup.trials: 
            print(t.file.name)


    def run_analysis(self): 

        joblib_filenames = self.paths.trials_metadata.glob("*.joblib")

        trialgroup = TrialGroup(joblib_filenames, self.conditions)

        analysis_output_path = self.paths.analysis(trialgroup.keep_val)
        analysis_output_path.mkdir(parents=True, exist_ok=True)
        print(analysis_output_path)