# src/BetaMouv/Project.py

from src.Base.Project import Project as BaseProject
from .Video import Video

from .PathConfig import PathConfig
from .Trial import Trial
from .TrialGroup import TrialGroup
from .Leds import Leds
from .Trajectory import Trajectory



from pathlib import Path
import yaml, time
from src.utils import process_time

import src.utils as u
import src.BetaMouv.gui.database_filter as Data_filter
import src.BetaMouv.gui.preprocess_validator as Validator


class Project(BaseProject):

    def __init__(self, name: str):
        super().__init__(name)

        self.config_dir: Path = Path(f"./config/{self.name}/")

        self.conditions: dict = self._load_config(self.config_dir / "conditions.yaml")
        self.subject_info: dict = self._load_config(self.config_dir / "subject_info.yaml")
        self.project_info: dict = self._load_config(self.config_dir / "project_info.yaml")

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

    @process_time
    def split_trials(self): 

        print(f"""
        =============== Video Splitting ==============
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        dataset = Data_filter.load_database(self.paths.raw_videos, self.paths.database, "video")

        for i, video_path in enumerate(dataset["filename"].iloc[:]): 

            raw_video = Video(video_path=video_path)

            output_dir = self.paths.raw / raw_video.file.subject / raw_video.name
            output_dir.mkdir(parents=True, exist_ok=True)

            meta = {"month": raw_video.date.month}
            clip_duration = u.match_rule(meta, self.clip_duration_rules)

            print(f"\nSplitting video : {raw_video.name}")
            print(f"clip duration: {clip_duration}")

            if not output_dir.exists() : 
                self.split_video(input_path= raw_video.path, 
                                output_dir= output_dir, 
                                CLIP_DURATION= clip_duration)
            else : 
                print(f"Has already been splitted !")

        


    @process_time
    def build_metadata(self): 
        import joblib

        print(f"""
        =============== Build Metadata ===============
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        dataset = Data_filter.load_database(self.paths.raw, self.paths.database, "video")

        trials_by_group = {}

        for i, clip_path in enumerate(dataset["filename"].iloc[:]):

            trial = Trial(clip_path=clip_path)

            print(f"Building metadata of: {trial.name}")

            # get annotation number
            annotation_meta = {
                "laser_type": trial.laser_type, 
                "view": trial.camera_view,
                "month": trial.file.date.month,
            }
            label_studio_annotation = u.match_rule(annotation_meta, self.annotation_rules)

            print(f"label studio annotation: {label_studio_annotation}")
            print(f"annotation meta: {annotation_meta}")

            # get Leds info to tell the Laser state (LaserOn, LaserOff)
            leds = Leds(video_path=clip_path, 
                        label_studio_annotation=label_studio_annotation,
                        )

            if (trial.camera_view == "left" and leds.cue_type == "CueL2") or \
               (trial.camera_view == "right" and leds.cue_type == "CueL1") : 
                print(f"Camera: {trial.camera_view} | cue: {leds.cue_type} ! Not compatible\n")
                continue

            # get lever_position
            etho_meta = {
                "rat": int(trial.subject[1:]),
                "day": trial.file.date.day,
                "condition": trial.condition,
                "view": trial.camera_view,
                "month": trial.file.date.month,
            }

            anchors_position = u.match_rule(etho_meta, self.ethology_rules)
            trial.update(lever_position=anchors_position["lever"])

            trial.set_led_info(leds)
            trial.set_task_success(self.project_info["laser_on_duration"])
            trial.set_mvt_type(self.subject_info[trial.subject]["hemi"])
            trial.set_group()

            pred_path = trial.file.path.parent / f"pred_results_{trial.name}.csv"
            if pred_path.exists(): 
                trial.update(pred_path=str(pred_path))

            trial.save_yaml()

            # Add trial to its group
            trials_by_group.setdefault(trial.group, []).append(trial.to_dict())

        # Save one big joblib per group
        print("\nTrials saved as joblibs: ")
        for group, trials in trials_by_group.items():

            joblib.dump(trials, u.make_path(self.paths.trials_metadata, f"{group}.joblib"))
            print(f"  {group}: {len(trials)}")


        print("\nVisualisation of the proportion of each experimental condition\n")

        # TODO
        # refaire la fonction de metadata report pour afficher le nombre
        # d'essai par groupe
        # u.metadata_report(...)

        



    @process_time
    def run_prediction(self): 

        print(f"""
        ================= Prediction =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        joblib_filenames = self.paths.trials_metadata.glob("*.joblib")
        
        trialgroup = TrialGroup(joblib_filenames, self.conditions)

        for i, trial in enumerate(trialgroup.trials):

            if "FIBER_BROKEN" in trial.name:        # for the rat 521
                print("FIBER_BROKEN, skipped")
                continue

            print(f"\n[{i+1}/{len(trialgroup.trials)}]")
            print(f"Prediction of video: {trial.name}\n")

            # prediction 
            trial.dlc_predict(model_path=self.paths.model,
                              output_csv_path=trial.clip_path.parent / f"{trial.name}.csv")

            break

        
            
            
    @process_time  
    def run_preprocessing(self): 
        print(f"""
        ============= Run Preprocessing ==============
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        joblib_filenames = self.paths.trials_metadata.glob("*.joblib")

        trialgroup = TrialGroup(joblib_filenames, self.conditions)

        preprocess_dir = self.paths.results_root / "preprocess"

        if  any(preprocess_dir.iterdir()): 

            for i, trial in enumerate(trialgroup.trials):

                print("\n", trial.name)

                # construct a traj object
                traj = Trajectory(coords_path=trial.pred_path, 
                                view=trial.camera_view,
                                bodypart="finger_3",
                                cm_per_pixel=trial.cm_per_pixel, 
                                lever_position=trial.lever_position)

                # do some filtration + launch validation gui
                likelihood_thresh = 0.7

                trial.set_task_success(self.project_info["laser_on_duration"])
                print(trial.task_success_reason, trial.time_pad_off)
                if not trial.task_success: 
                    print("TASK NOT SUCCESSFUL:", trial.task_success_reason)
                    continue

                outlier_filtered_coords, params = traj.filter_outliers(traj.raw_coords, stat_method='eucli')
                likelihood_filtered_coords, likelihood_threshold = traj.filter_likelihood(outlier_filtered_coords, likelihood_thresh)
                traj.interpolated_coords = traj.interpolate_data(likelihood_filtered_coords, method="spline", max_gap=5)

                # make figures that will be used for the validation
                interpolation_path = u.make_path(preprocess_dir, f"interpolation_{trial.name}png")
                traj.make_interpolation_figures(interpolated_coords=traj.interpolated_coords, 
                                                likelihood_filtered_coords=likelihood_filtered_coords,
                                                outlier_filtered_coords=outlier_filtered_coords,
                                                raw_coords=traj.coords,
                                                time_pad_off=trial.time_pad_off,
                                                title=f"{trial.group} - {trial.camera_view} view", 
                                                save_as=interpolation_path)


                # save traj info (before validation)
                trial.update( 
                        model_success = None,
                        traj = traj
                )

        # launch validation

        trial_states: list[str] = Validator.load_preprocess_validator(preprocess_dir)

        for trial, state in zip(trialgroup.trials, trial_states): 

            if not trial.task_success or not state:  # pass when the validation has been stop in the middle
                continue

            elif state == "rejected" : 
                trial.update(model_success = False) 

            elif state == "raw" : 
                trial.update(model_success = True,
                             coords = trial.traj.coords)

            else : 
                trial.update(model_success = True, 
                             coords = trial.traj.interpolated_coords)

        trialgroup.save(self.paths.trials_metadata)

        

## TODO
# review this code: 
    # def run_preprocessing_compute(self):
    #     preprocess_dir = self.paths.results_root / "preprocess"
    #     cache_dir = preprocess_dir / "cache"
    #     cache_dir.mkdir(parents=True, exist_ok=True)

    #     joblib_filenames = self.paths.trials_metadata.glob("*.joblib")
    #     trialgroup = TrialGroup(joblib_filenames, self.conditions)

    #     for trial in trialgroup.trials:
    #         cache_path = cache_dir / f"{trial.name}.joblib"
    #         if cache_path.exists():
    #             continue   # already computed, skip — makes this step resumable

    #         trial.set_task_success(self.project_info["laser_on_duration"])
    #         if not trial.task_success:
    #             continue

    #         traj = Trajectory(
    #             coords_path=trial.prediction_csv_path,
    #             view=trial.camera_view,
    #             cm_per_pixel=self.camera_calibration[trial.camera_view],  # project config, not per-trial
    #             lever_position=self.lever_positions[trial.camera_view],
    #         )

    #         outlier_filtered, _ = traj.filter_outliers(traj.coords, stat_method="eucli")
    #         likelihood_filtered, _ = traj.filter_likelihood(outlier_filtered, 0.7)
    #         interpolated = traj.interpolate_data(likelihood_filtered, method="spline", max_gap=5)

    #         traj.make_interpolation_figures(
    #             interpolated_coords=interpolated,
    #             likelihood_filtered_coords=likelihood_filtered,
    #             outlier_filtered_coords=outlier_filtered,
    #             raw_coords=traj.raw_coords,
    #             time_pad_off=trial.time_pad_off,
    #             title=f"{trial.group} - {trial.camera_view} view",
    #             save_as=preprocess_dir / f"interpolation_{trial.name}.png",
    #         )

    #         joblib.dump({
    #             "raw": traj.raw_coords,
    #             "outlier_filtered": outlier_filtered,
    #             "likelihood_filtered": likelihood_filtered,
    #             "interpolated": interpolated,
    #         }, cache_path)


    # def run_preprocessing_finalize(self):
    # preprocess_dir = self.paths.results_root / "preprocess"
    # cache_dir = preprocess_dir / "cache"

    # joblib_filenames = self.paths.trials_metadata.glob("*.joblib")
    # trialgroup = TrialGroup(joblib_filenames, self.conditions)

    # trial_states = Validator.load_preprocess_validator(preprocess_dir)

    # for trial in trialgroup.trials:
    #     state = trial_states.get(trial.name)
    #     cache_path = cache_dir / f"{trial.name}.joblib"

    #     if not trial.task_success or not state or not cache_path.exists():
    #         continue

    #     if state == "rejected":
    #         trial.update(model_success=False)
    #         continue

    #     candidates = joblib.load(cache_path)
    #     coords = candidates["raw"] if state == "raw" else candidates["interpolated"]

    #     metrics = compute_metrics(coords, fps=125)  # velocity, tortuosity, etc. — plain function or Trajectory.compute_metrics

    #     trial.update(model_success=True, coords=coords, **metrics)

    # trialgroup.save(self.paths.trials_metadata)



            



    def run_analysis(self): 

        joblib_filenames = self.paths.trials_metadata.glob("*.joblib")

        trialgroup = TrialGroup(joblib_filenames, self.conditions)

        analysis_output_path = self.paths.analysis(trialgroup.keep_val)
        analysis_output_path.mkdir(parents=True, exist_ok=True)
        print(analysis_output_path)