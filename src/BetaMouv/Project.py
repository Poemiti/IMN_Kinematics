# src/BetaMouv/Project.py

from src.Base.Project import Project as BaseProject
from .Video import Video

from .PathConfig import PathConfig
from .Trial import Trial
from .TrialGroup import TrialGroup
from .Leds import Leds
from .Trajectory import Trajectory
from .BehaviorBox import BehaviorBox



from pathlib import Path
import yaml, time, joblib, sys, shutil
from src.utils import process_time
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.camera_shift_rules: dict = self._load_config(self.config_dir / "rules/camera_shift_rules.yaml")

        # setup path
        self.paths: PathConfig = PathConfig(project_name=self.name)

        # trial group of the project
        joblib_filenames = list(self.paths.trials_metadata.glob("*.joblib"))
        self.trialgroup = TrialGroup(joblib_filenames, self.conditions) if joblib_filenames else None
        print("\n-->",len(joblib_filenames), "FILES LOADED")


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

            output_dir = u.make_dir(self.paths.raw / raw_video.file.subject / raw_video.name)

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

    def build_metadata(self): 
        print(f"""
        =============== Build Metadata ===============
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        joblib_filenames = list(self.paths.trials_metadata.glob("*.joblib"))
        print(f"{len(joblib_filenames)} joblib metadata files ({self.paths.trials_metadata})")
        res = input("Overwrite those metadata ? (y/n/q) : ")

        if res == "y" or res == "": 
            print("\nOVEWRITING METADATA\n")
            return self.init_metadata()

        elif res == "n" : 
            print("\nUPDATING METADATA\n")
            return self.update_metadata()

        elif res == "q" : 
                    print("\nQuit !\n")
                    sys.exit()

        else : 
            raise ValueError(f"'{res}' is not valid, must be 'y' or 'n'")
         

    def define_camera_shift(self):  
        # TODO MAKE IT BETTER LOL 

        print(f"""
        =============== Define Camera shift ===============
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")


        output_dir = u.make_dir(self.paths.data_root / "camera_shift")
        raw_frames_dir = u.make_dir(output_dir / "raw_frames")
        superimpose_dir = u.make_dir(output_dir / "superimposed")

        frames_paths = set()
        frames_paths.update(raw_frames_dir.glob("*.png")) 

        ref_path = Path("/home/poemiti/IMN_Kinematics/data/BetaMouv/camera_shift/raw_frames/#517_CHR_CONTRA_Beta_RightHemi_leftView_high_LaserOff_2024-07-05_5.png")

        shift_config: dict = {"rules":[]}

        for trial in self.trialgroup.trials: 

            trial_comb = trial.group + "_" + trial.date.isoformat()
            output_path = u.make_path(raw_frames_dir, f"{trial_comb}")

            # 1. extract frame (if not already done)
            if not output_path in frames_paths :
                frames_paths.add(output_path)
                print("extracting:", trial_comb)

                video = Video(trial.clip_path)
                video.extract_frames(frame_range=[5], output_base=output_path)

                # 2. compute camera shift
                dx, dy = video.compute_camera_shift(ref_path=ref_path, img_path=f"{output_path}_5.png",
                                            save_as=superimpose_dir / f"{trial_comb}.png")
                shift_config["rules"].append({
                                "when": {
                                    "date": trial.date.isoformat(),
                                    "group": trial.group
                                },
                                "value": {
                                    "flip": trial.camera_view == "right",
                                    "dx": dx.round(3).item(),
                                    "dy": dy.round(3).item()
                                }
                                
                            })

        with open(self.config_dir / "rules/camera_shift_rules.yaml", "w") as f: 
            yaml.safe_dump(shift_config, f) 
                    


    @process_time
    def init_metadata(self): 

        dataset = Data_filter.load_database(self.paths.raw, self.paths.database, "video")

        trials_by_group = {}

        for clip_path in dataset["filename"].iloc[:]:

            clip = Video(clip_path)
            trial = Trial(clip_path=clip_path)

            print("\nBuilding metadata:", clip.path.stem)

            if not clip.is_openable or not clip.is_readable: 
                trial.set_success(stage="clip_openable", success=False, reason="Clip not readable")
                trial.set_success(stage="view_match_task", success=False, reason="Clip not readable")
                trial.set_success(stage="task", success=False, reason="Clip not readable")

                trial.set_group(laser_state="UNKNOWN", mvt_type="UNKNOWN")
                trials_by_group.setdefault(trial.group, []).append(trial)
                trial.save_yaml()
                continue

            trial.set_success(stage="clip_openable", success=True, reason="Clip openable")

            # get annotation number
            annotation_meta = {
                "laser_type": trial.laser_type, 
                "view": trial.camera_view,
                "month": trial.file.date.month,
            }
            label_studio_annotation = u.match_rule(annotation_meta, self.annotation_rules)

            # get Leds info to tell the Laser state (LaserOn, LaserOff)
            leds = Leds(video_path=clip_path, 
                        label_studio_annotation=label_studio_annotation,)

            if (trial.camera_view == "left" and leds.cue_type == "CueL2") or \
               (trial.camera_view == "right" and leds.cue_type == "CueL1") : 
                trial.set_success(stage="view_match_task", success=False, reason=f"'{trial.camera_view}' not compatible with '{leds.cue_type}'")
            else:
                trial.set_success(stage="view_match_task", success=True, reason=f"Compatible view with task")

            trial.set_led_info(leds)
            trial.set_task_success(self.project_info["laser_on_duration"])
            trial.set_mvt_type(self.subject_info[trial.subject]["hemi"])
            trial.set_group()

            # get camera shift info
            shift_meta = {
                "date": trial.date.isoformat(),
                "group": trial.group,
            }
            print(shift_meta)
            
            shift = u.match_rule(shift_meta, self.camera_shift_rules)
            trial.update(camera_shift=shift)

            pred_path = trial.file.path.parent / f"pred_results_{trial.name}.csv"
            if pred_path.exists(): 
                trial.update(pred_path=str(pred_path))

            trial.save_yaml()

            # Add trial to its group
            trials_by_group.setdefault(trial.group, []).append(trial)

        n_trial=0
        # Save one big joblib per group
        print("\nTrials saved as joblibs: ")
        for group, trials in trials_by_group.items():

            joblib.dump(trials, u.make_path(self.paths.trials_metadata, f"{group}.joblib"))
            print(f"  {group}: {len(trials)}")
            n_trial+=len(trials)

        print(f"\nTotal trials processed: {n_trial}")


    @process_time   
    def update_metadata(self): 
        """Update every metadata of each trial EXEPT the LEDs info"""

        updated_trials = []

        for previous_trial in tqdm(self.trialgroup.trials, desc="Metadata update"): 

            updated_trial = Trial(previous_trial.clip_path)
            clip = Video(previous_trial.clip_path)

            # if not clip.is_openable or not clip.is_readable:
            if not previous_trial.trial_outcomes["clip_openable"].success:  
                updated_trial.set_success(stage="clip_openable", success=False, reason="Clip not readable")
                updated_trial.set_success(stage="view_match_task", success=False, reason="Clip not readable")
                updated_trial.set_success(stage="task", success=False, reason="Clip not readable")

                updated_trial.set_group(laser_state="UNKNOWN", mvt_type="UNKNOWN")
                updated_trials.append(updated_trial)
                updated_trial.save_yaml()
                continue
            
            updated_trial.set_success(stage="clip_openable", success=True, reason="Clip openable")

            # get camera shift info
            shift_meta = {
                "date": previous_trial.date.isoformat(),
                "group": previous_trial.group,
            }

            shift = u.match_rule(shift_meta, self.camera_shift_rules)
            updated_trial.update(camera_shift=shift)

            # get leds info
            updated_trial.update(
                            laser_state=previous_trial.laser_state,
                            cue_type=previous_trial.cue_type,
                            time_pad_off=previous_trial.time_pad_off,
                            time_laser_on=previous_trial.time_laser_on,
                            time_reward=previous_trial.time_reward,
                        )

            if (previous_trial.camera_view == "left" and previous_trial.cue_type == "CueL2") or \
                (previous_trial.camera_view == "right" and previous_trial.cue_type == "CueL1") : 
                updated_trial.set_success(stage="view_match_task", success=False, reason=f"'{updated_trial.camera_view}' X '{updated_trial.cue_type}'")
            else: 
                updated_trial.set_success(stage="view_match_task", success=True, reason=f"Compatible view X task")


            updated_trial.set_task_success(self.project_info["laser_on_duration"])
            updated_trial.set_mvt_type(self.subject_info[previous_trial.subject]["hemi"])
            updated_trial.set_group()

            pred_path = previous_trial.file.path.parent / f"pred_results_{previous_trial.name}.csv"
            if pred_path.exists(): 
                updated_trial.update(pred_path=str(pred_path))

            updated_trial.save_yaml()
            updated_trials.append(updated_trial)

        self.trialgroup.save(self.paths.trials_metadata, trial_list=updated_trials)
        self.trialgroup.trials = updated_trials


    @process_time
    def run_prediction(self): 

        print(f"""
        ================= Prediction =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        for trial in tqdm(self.trialgroup.trials, desc="Prediction"):
            if not trial.is_valid(): 
                continue

            if "FIBER_BROKEN" in trial.name:        # for the rat 521
                print("FIBER_BROKEN, skipped")
                continue


            # prediction 
            trial.dlc_predict(model_path=self.paths.model,
                              output_csv_path=trial.clip_path.parent / f"pred_results_{trial.stem}.csv")

            break

        
            
            
    @process_time  
    def run_preprocessing(self):
        print(f"""
        =============== Preprocessing =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")
        preprocess_dir = u.make_dir(self.paths.results_root / "preprocess")
        preprocess_data_path = preprocess_dir / "preprocess_data.csv"
        outlier_fig_path = preprocess_dir / "outlier_distri.png"

        interpolation_dir = preprocess_dir / "interpolation"
        shutil.rmtree(interpolation_dir, ignore_errors=True)   # clear stale figures first
        u.make_dir(interpolation_dir)                          # then (re)create

        MAX_OUTLIER = self.project_info["max_outlier"]
        DIST_THRESH = self.project_info["distance_thresh"]
        GAP = self.project_info["gap_scale"]
        BODYPARTS = self.project_info["bodyparts"]
        LEVER_POS = self.project_info["lever_position"]

        preprocess_records = []

        for trial in tqdm(self.trialgroup.trials, desc="Preprocessing"):

            if not trial.is_valid():
                continue

            for bodypart in BODYPARTS: 

                traj = Trajectory(
                    coords_path=trial.pred_path,
                    view=trial.camera_view,
                    bodypart="finger_3",
                    cm_per_pixel=trial.cm_per_pixel,
                    lever_position=LEVER_POS,
                )
                traj.apply_shift(trial.camera_shift)

                raw_coords = traj.compute_instant_metrics(traj.coords)
                raw_outlier_mask = traj.outlier_euclidian_dist_anchored(coords=raw_coords, threshold=DIST_THRESH, gap_scale=GAP)
                n_raw_outliers = int(raw_outlier_mask.sum())

                outlier_filtered_coords = traj.filter_outliers(coords=raw_coords, method="eucli_anchored", gap_scale=GAP)

                traj.interpolated_coords = traj.interpolate_data(coords=outlier_filtered_coords, method="spline", max_gap=5)
                traj.interpolated_coords = traj.compute_instant_metrics(traj.interpolated_coords)

                interpolated_outlier_mask = traj.outlier_euclidian_dist_anchored(traj.interpolated_coords, threshold=DIST_THRESH, gap_scale=GAP)
                n_interpolated_outliers = int(interpolated_outlier_mask.sum())

                preprocess_records.append({"trial": trial.name, "bodypart": bodypart, "n_outlier": n_raw_outliers, "type": "raw"})
                preprocess_records.append({"trial": trial.name, "bodypart": bodypart, "n_outlier": n_interpolated_outliers, "type": "interpolated"})

                # set outlier stage outcome 

                if n_interpolated_outliers == 0:
                    traj.set_success("outlier", success=True, reason=f"0 outliers found")
                    traj.clean_coords = traj.coords

                elif n_interpolated_outliers >= MAX_OUTLIER:
                    traj.set_success("outlier", success=False, reason=f"outliers >= {MAX_OUTLIER}")
                    traj.set_success("validation", success=False, reason=f"outliers >= {MAX_OUTLIER}")
                
                else:
                    traj.set_success("outlier", success=False, reason=f"outliers >= {MAX_OUTLIER} need validation")
                    traj.plot_preprocess(
                                        interpolated_coords=traj.interpolated_coords,
                                        outlier_filtered_coords=outlier_filtered_coords,
                                        raw_coords=raw_coords,
                                        time_pad_off=trial.time_pad_off,
                                        title=trial.name,
                                        save_as=interpolation_dir / f"interpolation_{bodypart}_{trial.name}.png",
                                    )
                    
                trial.trajectories[bodypart] = traj

        outlier_df = pd.DataFrame(preprocess_records)
        outlier_df.to_csv(preprocess_data_path)

        self.trialgroup.save(self.paths.trials_metadata)
        # self.trialgroup.distri_outlier(data=outlier_df, save_as=outlier_fig_path)
        # self.trialgroup.lineplot_all_traj(save_as=preprocess_dir / f"{bodypart}_traj.png")


    @process_time
    def run_validation(self):
        print(f"""  
        ================= Validation =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        BODYPARTS = self.project_info["bodyparts"]
        
        preprocess_dir = self.paths.results_root / "preprocess" / "interpolation"
        print("\n-->", len(list(preprocess_dir.iterdir())), "FILES TO VALIDATE")
        trial_states = Validator.load_preprocess_validator(preprocess_dir)

        for trial in tqdm(self.trialgroup.trials, desc="Saving validation"):
            if not trial.is_valid(): 
                continue

            for bodypart in BODYPARTS: 

                state = trial_states.get(f"{bodypart}_{trial.name}")

                if not state:
                    continue

                traj: Trajectory = trial.trajectories.get(bodypart)

                if state == "rejected":
                    traj.set_success(stage="validation", success=False, reason="manually rejected")
 
                elif state == "raw":
                    traj.set_success(stage="validation", success=True, reason="manually accepted, raw")
                    traj.clean_coords = traj.coords
                else:
                    traj.set_success(stage="validation", success=True, reason="manually accepted, interpolated")
                    traj.clean_coords = traj.interpolated_coords

        self.trialgroup.save(self.paths.trials_metadata)



    @process_time 
    def compute_metrics(self): 
        print(f"""
        ============== Compute Metrics ===============
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        BODYPARTS = self.project_info["bodyparts"]        
        preprocess_dir = self.paths.results_root / "preprocess"

        for trial in tqdm(self.trialgroup.trials, desc="Computing metrics"):
            if not trial.is_valid():    
                continue

            for bodypart in BODYPARTS: 

                traj = trial.trajectories.get(bodypart)
                
                if not traj.is_valid(): 
                    continue

                Boxes = BehaviorBox(coords=trial.coords,
                                    time_pad_off=trial.time_pad_off,
                                    shift=trial.camera_shift,)
                
                coords = trial.traj.compute_instant_metrics(trial.coords)
                coords = trial.behaviorBox.classify_trajectory(coords)
                ## TODO
                # compute scalar metrics that will then be added to SCALAR_FIELD in Trial
                
                trial.update(behaviorBox=Boxes)

            self.trialgroup.save(self.paths.trials_metadata)


    @process_time 
    def run_analysis(self): 
        print(f"""
        =============== Analysis =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        # self.define_camera_shift()
        
        analysis_dir = self.paths.analysis(self.trialgroup.keep_val)

        # trial = self.trialgroup.trials[0]
        # vid = Video(trial.clip_path)
        # vid.annotate_video(output_path=self.paths.data_root / "annotated_vid" / f"annotated_{trial.name}.mp4",)
        # vid.extract_frames(frame_range=range(0, 150), 
        #                    video_path=self.paths.data_root / "annotated_vid" / f"annotated_{trial.name}.mp4",
        #                    output_base=u.make_path(self.paths.data_root / "annotated_vid" , "frame"))

        # self.trialgroup.crop_coords(True)
        # self.trialgroup.buils_timeseries_df(init=False, save_as=analysis_dir / "timeseries_df.csv")
        # print(self.trialgroup._timeseries_df)

        # for val in ["instant_velocity", "instant_acc", "lever_distance"]:
        #     self.trialgroup.plot_tendency(
        #         value=val,
        #         save_as=u.make_path(analysis_dir / "tendency", f"{val}.svg")
        #     )
        
        # self.trialgroup.lineplot_all_traj(save_as=analysis_dir / "all_traj.svg")

        self.trialgroup.trial_success_rate(u.make_path(analysis_dir, "trial_success_rate.png"))
        
        