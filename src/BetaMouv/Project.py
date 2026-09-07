# src/BetaMouv/Project.py

from src.Base.Project import Project as BaseProject
from src.Base.Video import Video

from .Trial import Trial
from .Prediction import Prediction
from .Leds import Leds

from pathlib import Path
import yaml

import src.utils as u
import src.BetaMouv.gui.database_filter as data_filter

class Project(BaseProject):

    def __init__(self, name, config_dir):
        super().__init__(name, config_dir)

        self.config_dir = config_dir
        self.name = name

        # setup config path
        self.paths = self._load_config(self.config_dir / "paths.yaml")
        for pname, path in self.paths.items(): 
            self.paths[pname] = Path(path)

        self.conditions = self._load_config(self.config_dir / "conditions.yaml")

        # setup rules
        self.annotation_rules = self._load_config(self.config_dir / "rules/annotation_rules.yaml")
        self.clip_duration_rules = self._load_config(self.config_dir / "rules/clip_duration_rules.yaml")
        self.ethology_rules = self._load_config(self.config_dir / "rules/ethology_rules.yaml")
        self.exclusion_rules = self._load_config(self.config_dir / "rules/exclusion_rules.yaml")


    @staticmethod
    def _load_config(filename: Path):
        with open(filename, "r") as cfg_file:
            cfg = yaml.safe_load(cfg_file)
        return cfg


    def run_prediction(self): 

        print(f"""
        ================= Prediction =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        dataset = data_filter.load_database(self.paths["raw_videos"], self.paths["database"], "video")

        for i, video_path in enumerate(dataset["filename"].iloc[:]):

            video = Video(video_path)

            if "FIBER_BROKEN" in video.name:        # for the rat 521
                print("FIBER_BROKEN, skipped")
                continue

            print(f"\n[{i+1}/{len(dataset)}]")
            print(f"Prediction of video: {video_path}\n")

            # get clips lenght by looking at the month 
            
            meta = {"month": video.date.month}
            clip_duration = u.match_rule(meta, self.clip_duration_rules)

            # prediction 

            Prediction(video_obj=video,
                            paths=self.paths,
                            clip_duration=clip_duration)

        print("Done!")


    def build_metadata(self): 
        import joblib

        dataset = data_filter.load_database(self.paths["raw_clip"], self.paths["database"], "video")

        output_dir = self.paths["metrics"]
        output_dir.mkdir(parents=True, exist_ok=True)

        trials_by_group = {}

        for i, clip_path in enumerate(dataset["filename"].iloc[:]):

            trial = Trial(clip_path=clip_path)

            # get annotation number
            with open("./annotation_rules.yaml") as f:
                annotation_rules = yaml.safe_load(f)

            annotation_meta = {
                "condition": trial.file.condition, 
                "view": trial.file.camera_view,
                "month": trial.file.date.month,
            }
            label_studio_annotation = u.match_rule(annotation_meta, annotation_rules)

            # get Leds info to tell the Laser state (LaserOn, LaserOff)
            leds = Leds(video_path=clip_path, 
                        label_studio_annotation=label_studio_annotation)

            trial.set_led_info(leds)
            trial.save_trial()

            # Add trial to its group
            trials_by_group.setdefault(trial.group, []).append(trial)

        # Save one big joblib per group
        for group, trials in trials_by_group.items():

            joblib.dump(trials, output_dir / f"{group}.joblib")
            print(f"  {group}: {len(trials)}")

        print("Done!")
            
            
            

            