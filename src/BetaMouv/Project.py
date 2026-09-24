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

        ref_path = Path("/home/ninjayu/IMN_Kinematics/data/BetaMouv/camera_shift/raw_frames/#517_CHR_CONTRA_Beta_RightHemi_leftView_high_LaserOff_2024-07-05.png")

        shift_config: dict = {"rules":[]}

        for trial in self.trialgroup.trials: 

            trial_comb = trial.group + "_" + trial.date.isoformat()
            output_path = u.make_path(raw_frames_dir, f"{trial_comb}.png")

            # 1. extract frame (if not already done)
            if not output_path in frames_paths :
                frames_paths.add(output_path)
                print("extracting:", trial_comb)

                video = Video(trial.clip_path)
                video.extractframes(frame_range=[5], output_path=output_path)
            else : 
                print("loading:", trial_comb)

            # 2. compute camera shift
            dx, dy = video.compute_camera_shift(ref_path=ref_path, 
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
            print("\nBuilding metadata:", clip.path.stem)

            if not clip.is_openable or not clip.is_readable: 
                continue

            if (clip.path.parent / f"{clip.path.stem}.yaml").exists(): 
                print("Metadata already builded !")
                continue

            trial = Trial(clip_path=clip_path)

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
                # print(f"Camera: {trial.camera_view} | cue: {leds.cue_type} ! Not compatible\n")
                continue

            # get camera shift info
            shift_meta = {
                "date": trial.date,
                "group": trial.group,
            }

            shift = u.match_rule(shift_meta, self.camera_shift_rules)
            trial.update(camera_shift=shift)

            trial.set_led_info(leds)
            trial.set_task_success(self.project_info["laser_on_duration"])
            trial.set_mvt_type(self.subject_info[trial.subject]["hemi"])
            trial.set_group()

            pred_path = trial.file.path.parent / f"pred_results_{trial.name}.csv"
            if pred_path.exists(): 
                trial.update(pred_path=str(pred_path))

            trial.save_yaml()

            # Add trial to its group
            trials_by_group.setdefault(trial.group, []).append(trial)

        # Save one big joblib per group
        print("\nTrials saved as joblibs: ")
        for group, trials in trials_by_group.items():

            joblib.dump(trials, u.make_path(self.paths.trials_metadata, f"{group}.joblib"))
            print(f"  {group}: {len(trials)}")


        # print("\nVisualisation of the proportion of each experimental condition\n")

        # TODO
        # refaire la fonction de metadata report pour afficher le nombre
        # d'essai par groupe
        # u.metadata_report(...)


    @process_time   
    def update_metadata(self): 
        """Update every metadata of each trial EXEPT the LEDs info"""

        updated_trials = []

        for previous_trial in tqdm(self.trialgroup.trials, desc="Metadata update"): 
        
            updated_trial = Trial(previous_trial.clip_path)

            # get camera shift info
            shift_meta = {
                "date": previous_trial.date,
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
            if not trial._is_successful(): 
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

        interpolation_dir = u.make_dir(preprocess_dir / "interpolation")
        shutil.rmtree(interpolation_dir, ignore_errors=False)
        
        preprocess_data = {
            "trial": [],
            "n_outlier": [],
            "type": [],
            "accepted": []
        }

        MAX_OUTLIER = self.project_info["max_outlier"]

        for trial in tqdm(self.trialgroup.trials, desc="Preprocessing"):

            if not trial.task_success:
                trial.update(coords_success_reason="Task failed, skipped")
                continue

            traj = Trajectory(
                coords_path=trial.pred_path,
                view=trial.camera_view,
                cm_per_pixel=trial.cm_per_pixel, 
                lever_position=trial.lever_position,
            )

            raw_coords = traj.coords
            raw_coords = traj.compute_instant_metrics(raw_coords)
            raw_outlier_mask = traj.outlier_euclidian_dist_anchored(coords=raw_coords, 
                                                                    threshold=self.project_info["distance_thresh"], gap_scale=self.project_info["gap"])

            outlier_coords = traj.filter_outliers(coords=raw_coords, method="eucli_anchored", 
                                                  gap_scale=self.project_info["gap"])

            traj.interpolated_coords = traj.interpolate_data(coords=outlier_coords, method="spline", max_gap=5)
            traj.interpolated_coords = traj.compute_instant_metrics(traj.interpolated_coords)
            interpolated_outlier_mask = traj.outlier_euclidian_dist_anchored(traj.interpolated_coords, 
                                                                             threshold=self.project_info["distance_thresh"], gap_scale=self.project_info["gap"])

            n_raw_outliers = len([o for o in raw_outlier_mask if o==True])
            n_interpolated_outliers = len([o for o in interpolated_outlier_mask if o==True])
            traj_accepted = n_raw_outliers <= 3 

            preprocess_data["trial"].append(trial.name)
            preprocess_data["n_outlier"].append(n_raw_outliers)
            preprocess_data["type"].append("raw")
            preprocess_data["accepted"].append(traj_accepted)

            preprocess_data["trial"].append(trial.name)
            preprocess_data["n_outlier"].append(n_interpolated_outliers)
            preprocess_data["type"].append("interpolated")
            preprocess_data["accepted"].append(traj_accepted)

            if not traj_accepted and n_raw_outliers < MAX_OUTLIER: # if more than MAX_OUTLIER outler, no need for verification 
                traj.plot_preprocess(
                    interpolated_coords=traj.interpolated_coords,
                    outlier_filtered_coords=outlier_coords,
                    raw_coords=raw_coords,
                    time_pad_off=trial.time_pad_off,
                    title=f"{trial.name}",
                    save_as=interpolation_dir / f"interpolation_{trial.name}.png",
                )

                trial.update(traj=traj, 
                            coords=traj.interpolated_coords,
                            coords_success=traj_accepted, 
                            coords_success_reason=f"Less than {MAX_OUTLIER} outliers (need validation)",)
                continue

            if not traj_accepted and n_raw_outliers > MAX_OUTLIER: 
                trial.update(traj=traj, 
                            coords=traj.interpolated_coords,
                            coords_success=traj_accepted, 
                            coords_success_reason=f"Rejected, more than {MAX_OUTLIER} outliers (no validation)",
                        validation_success=False, 
                        validation_success_reason=f"Rejected, more than {MAX_OUTLIER} outliers (no validation)")
                continue

            trial.update(traj=traj, 
                        coords=traj.interpolated_coords,
                        coords_success=traj_accepted, 
                        coords_success_reason="Interpolated coordinates (no validation)",
                        validation_success=True, 
                        validation_success_reason="Interpolated coordinates (no validation)")


        outlier_df = pd.DataFrame(preprocess_data)
        outlier_df.to_csv(preprocess_data_path)

        self.trialgroup.save(self.paths.trials_metadata)

        self.trialgroup.distri_outlier(data=outlier_df, save_as=outlier_fig_path)
        self.trialgroup.lineplot_all_traj(save_as=preprocess_dir / "all_traj(laser period only).png")



    @process_time
    def run_validation(self):
        print(f"""  
        ================= Validation =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")
        
        preprocess_dir = self.paths.results_root / "preprocess" / "interpolation"
        print("\n-->", len(list(preprocess_dir.iterdir())), "FILES TO VALIDATE")
        trial_states = Validator.load_preprocess_validator(preprocess_dir)

        for trial in tqdm(self.trialgroup.trials, desc="Saving validation"):
            state = trial_states.get(trial.name)

            if not trial.task_success:
                trial.update(validation_success=False, 
                            validation_success_reason="Task failed, skipped")
                continue

            if not state:
                trial.update(validation_success=True, 
                            validation_success_reason="Successful, no validation needed")
                continue

            if state == "rejected":
                trial.update(validation_success=False, 
                             validation_success_reason="Manually rejected")
            elif state == "raw":
                trial.update(validation_success=True, 
                             validation_success_reason="Raw coordinates", 
                             coords=trial.traj.coords)
            else:
                trial.update(validation_success=True, 
                             validation_success_reason="Interpolated coordinates",
                            coords=trial.traj.interpolated_coords)

        self.trialgroup.save(self.paths.trials_metadata)



    @process_time 
    def compute_metrics(self): 
        print(f"""
        ============== Compute Metrics ===============
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")
        
        preprocess_dir = self.paths.results_root / "preprocess"

        for trial in tqdm(self.trialgroup.trials, desc="Computing metrics"):
            if not trial._is_successful():    
                continue

            Boxes = BehaviorBox(coords=trial.coords,
                                time_pad_off=trial.time_pad_off,
                                shift=trial.camera_shift,)
            

            coords = trial.traj.compute_instant_metrics(trial.coords)
            coords = trial.behaviorBox.classify_trajectory(coords)
            ## TODO
            # compute scalar metrics that will then be added to SCALAR_FIELD in Trial
             
            trial.update(coords=coords,
                         behaviorBox=Boxes)

        self.trialgroup.save(self.paths.trials_metadata)



            



    def run_analysis(self): 
        print(f"""
        =============== Analysis =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        # self.define_camera_shift()
        
        analysis_dir = self.paths.analysis(self.trialgroup.keep_val)

        trial = self.trialgroup.trials[0]
        vid = Video(trial.clip_path)
        vid.annotate_video(output_path=self.paths.data_root / "annotated_vid" / f"annotated_{trial.name}.mp4",)
        vid.extract_frames(frame_range=range(0, 150), 
                           video_path=self.paths.data_root / "annotated_vid" / f"annotated_{trial.name}.mp4",
                           output_base=u.make_path(self.paths.data_root / "annotated_vid" , "frame"))

        # self.trialgroup.crop_coords(True)
        # self.trialgroup.buils_timeseries_df(init=False, save_as=analysis_dir / "timeseries_df.csv")
        # print(self.trialgroup._timeseries_df)

        # for val in ["instant_velocity", "instant_acc", "lever_distance"]:
        #     self.trialgroup.plot_tendency(
        #         value=val,
        #         save_as=u.make_path(analysis_dir / "tendency", f"{val}.svg")
        #     )
        
        # self.trialgroup.lineplot_all_traj(save_as=analysis_dir / "all_traj.svg")

        # self.trialgroup.success_rate_report(u.make_path(analysis_dir, "success_rate.txt"))
        
        