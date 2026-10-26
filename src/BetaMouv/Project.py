# src/BetaMouv/Project.py

from src.Base.Project import Project as BaseProject
from src.Base.PathConfig import PathConfig
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
import src.BetaMouv.gui.database_filter as Data_filter
import src.BetaMouv.gui.preprocess_validator as Validator
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
        self.clip_duration_rules: dict = u.load_config(self.config_dir / "rules/clip_duration_rules.yaml")
        self.exclusion_rules: dict = u.load_config(self.config_dir / "rules/exclusion_rules.yaml")
        self.camera_shift_rules: dict = u.load_config(self.config_dir / "rules/camera_shift_rules.yaml")

        # setup path
        self.paths: PathConfig = PathConfig(project_name=self.name, model=self.project_info["model_path"], raw_videos=self.project_info["raw_videos_path"])

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

        dataset = list(Data_filter.load_database(
                self.paths.raw_videos, self.paths.database, "video")["filename"])

        for i, video_path in enumerate(dataset): 

            if "FIBER_BROKEN" in video_path or "NO_TRUST" in video_path: 
                print("\nskip")
                continue

            raw_video = Video(video_path=video_path)


            if raw_video.file.condition == "Unknown" or \
            raw_video.file.stim_location == "Unknown" or \
            raw_video.file.laser_intensity == "Unknown": 
                video_name = raw_video.path.parent.stem
            else: 
                video_name = raw_video.name

            output_dir = u.make_dir(self.paths.raw / f"subject_{raw_video.file.subject}"/ video_name)

            meta = {"month": raw_video.date.month}
            clip_duration = u.match_rule(meta, self.clip_duration_rules)

            print(f"\nSplitting video : {video_name}")
            print(f"clip duration: {clip_duration}")


            if (list(output_dir.glob("*.mp4"))) == [] : 
                raw_video.split_video(input_path= raw_video.path, 
                                output_dir= output_dir, 
                                CLIP_DURATION= clip_duration, 
                                CRF=13)
            else : 
                print(f"Has already been splitted !")


    def build_metadata(self):
        self.trialgroup.load_trials()
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


    def _process_clip(self, clip_path, update: bool, shift_ctx: dict) -> Trial:
        trial = Trial(clip_path=clip_path)
        previous = None

        # 1. is the clip usable?
        if update:
            previous = Trial(clip_path=clip_path)
            previous.load_yaml()
            readable = previous.trial_outcomes["clip_openable"].success
        else:
            clip = Video(clip_path)
            readable = clip.is_openable and clip.is_readable

        if not readable:
            for stage in ("clip_openable", "view_match_task", "task"):
                trial.set_success(stage=stage, success=False, reason="Clip not readable")
            trial.set_group(laser_state="UNKNOWN", mvt_type="UNKNOWN")
            return trial

        trial.set_success(stage="clip_openable", success=True, reason="Clip openable")

        if trial.date.isoformat() in self.exclusion_rules[trial.subject] :
            for stage in ("clip_openable", "view_match_task", "task"):
                trial.set_success(stage=stage, success=False, reason="Excluded by date")
            trial.set_group(laser_state="UNKNOWN", mvt_type="UNKNOWN")
            return trial
        
        
        # 2. the ONLY step that differs: LED info
        if update:
            trial.update(
                laser_state=previous.laser_state,
                cue_type=previous.cue_type,
                time_pad_off=previous.time_pad_off,
                time_laser_on=previous.time_laser_on,
                time_reward=previous.time_reward,
            )
        else:
            rule_key = {
                "laser_type": trial.laser_type,
                "view": trial.camera_view,
                "month": trial.file.date.month,
            }
            annotation = u.match_rule(rule_key, self.annotation_rules)
            trial.set_led_info(Leds(video_path=clip_path, label_studio_annotation=annotation))

        # 3. shared logic
        incompatible = (
            (trial.camera_view == "left" and trial.cue_type == "CueL2") or
            (trial.camera_view == "right" and trial.cue_type == "CueL1")
        )
        trial.set_success(
            stage="view_match_task",
            success=not incompatible,
            reason=(f"'{trial.camera_view}' not compatible with '{trial.cue_type}'"
                    if incompatible else "Compatible view with task"),
        )

        trial.set_task_success(self.project_info["laser_on_duration"])
        trial.set_mvt_type(self.subject_info[trial.subject]["hemi"])
        trial.set_group()

        self._add_camera_shift(trial, update, shift_ctx, save_superimposed=True)

        pred_path = trial.file.path.parent / f"pred_results_{trial.name}.csv"
        if pred_path.exists():
            trial.update(pred_path=str(pred_path))

        return trial


    def _add_camera_shift(self, trial, update, ctx, save_superimposed: bool = False):

        def read_shift_dict(shift_dict): 
            shift_meta = {"date": trial.date.isoformat(), "subject": trial.subject, "view": trial.camera_view} 
            shift = u.match_rule(shift_meta, shift_dict)
            dx, dy = shift["dx"], shift["dy"]
            return dx, dy
              
        if update: 
            dx, dy = read_shift_dict(self.camera_shift_rules)

        else: 
            tag = f"{trial.subject}_{trial.camera_view}_{trial.date.isoformat()}"
            base = u.make_path(ctx["raw_frames_dir"], tag)
            frame_num = 5
            frame_path = f"{base}_{frame_num}.png"

            if frame_path in ctx["frame_paths_list"] :
                dx, dy = read_shift_dict(ctx)

            else :
                clip = Video(trial.clip_path)
                clip.extract_frames(frame_range=[frame_num], output_base=base)
                dx, dy = clip.compute_camera_shift(view=trial.camera_view,
                    ref_path=ctx["ref_path"], img_path=frame_path,
                    save_as=ctx["superimpose_dir"] / f"{tag}.png" if save_superimposed else None,
                )
                dx, dy = dx.round(3).item(), dy.round(3).item()

                ctx["rules"].append({
                            "when": {"date": trial.date.isoformat(), "subject": trial.subject, "view": trial.camera_view},
                            "value": {"dx": dx, "dy": dy},
                        })

                ctx["frame_paths_list"].append(frame_path)

        trial.update(camera_shift=(dx, dy))



    @process_time
    def _run_metadata(self, update: bool):
        shift_dir = u.make_dir(self.paths.data_root / "camera_shift")
        ctx = {
            "raw_frames_dir": u.make_dir(shift_dir / "raw_frames"),
            "superimpose_dir": u.make_dir(shift_dir / "superimposed"),
            "ref_path": self.project_info["shift_ref"],
            "rules": [],
            "frame_paths_list": [],
        }

        if update:
            clip_paths = list(self.paths.raw.rglob("*.mp4"))
        else:
            clip_paths = list(Data_filter.load_database(
                self.paths.raw, self.paths.database, "video")["filename"])

        trials_by_group, errors = {}, {}

        for clip_path in tqdm(clip_paths, desc="Metadata update" if update else "Metadata init"):
            try:
                trial = self._process_clip(clip_path, update, ctx)
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

        # save camera shift rules when it s not an update
        if not update:
            u.save(self.config_dir / "rules/camera_shift_rules.yaml",
                       lambda p: p.write_text(yaml.safe_dump({"rules": ctx["rules"]})),)

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

    @process_time
    def run_prediction(self): 
        self.trialgroup.load_trials()
        print(f"""
        ================= Prediction =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        for trial in tqdm(self.trialgroup.trials, desc="Prediction"):
            if not trial.is_valid(): 
                continue

            output_path = trial.clip_path.parent / f"pred_results_{trial.name}.csv"
            trial.update(pred_path=output_path)

            trial.save_yaml()

            if output_path.exists(): 
                print(f"{output_path} already exist, prediction skipped!")
                continue 

            # prediction 
            trial.dlc_predict(model_path=self.paths.model,
                              output_csv_path=output_path)

        #  backup, save joblib
        self.trialgroup.save_backup(dir_to_copy=self.paths.trials_metadata, output_dir=self.paths.data_root / "trials_metadata_backup", 
                                    step="prediction")
        self.trialgroup.save(self.paths.trials_metadata)


            
    @process_time  
    def run_preprocessing(self):
        self.trialgroup.load_trials()
        print(f"""
        =============== Preprocessing =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")
        preprocess_dir = u.make_dir(self.paths.analysis(self.trialgroup.keep_val) / "preprocess")
        preprocess_data_path = preprocess_dir / "preprocess_data.csv"

        interpolation_dir = preprocess_dir / "interpolation"
        u.make_dir(interpolation_dir)                          # then (re)create

        MAX_OUTLIER = self.project_info["max_outlier"]
        DIST_THRESH = self.project_info["distance_thresh"]
        GAP = self.project_info["gap_scale"]
        BODYPARTS = self.project_info["bodyparts"]

        preprocess_data = []

        for trial in tqdm(self.trialgroup.trials, desc="Preprocessing"):
            if not trial.is_valid():
                continue
            trial.update(pred_path = Path(trial.pred_path))
            trial.update(clean_pred_path = trial.pred_path.parent / f"clean_{trial.name}.csv")

            tracking = DLCTracking(path=trial.pred_path,
                                        view=trial.camera_view, 
                                        bodyparts=BODYPARTS,
                                        time_pad_off=trial.time_pad_off,
                                        cm_per_pixel=trial.cm_per_pixel,
                                        shift=trial.camera_shift)

            raw_coords_df, outlier_results, interpolation_results = tracking.preprocess(DIST_THRESH, GAP, "spline", 5)

            # set outlier stage outcome 
            for bp in BODYPARTS: 
                raw_coords, outlier_traj, inter_traj = raw_coords_df[bp], outlier_results[bp], interpolation_results[bp]
                raw_outlier_mask = outlier_traj["mask"]
                n_raw_outliers = int(raw_outlier_mask.sum())
                n_interpolated_outliers = int(inter_traj["mask"].sum())
                longest_consecutive_outlier = tracking.longest_consecutive_true(raw_outlier_mask)

                if n_raw_outliers == 0 :
                    tracking.update_status(bp, success=True, stage="outlier", reason="0 out")
                    tracking.update_status(bp, success=True, stage="validation", reason="0 out",
                                                new_coords=raw_coords,  state="raw",)

                elif n_raw_outliers <= MAX_OUTLIER//2 or n_interpolated_outliers == 0:
                    tracking.update_status(bp, success=True, stage="outlier", reason="few out")
                    tracking.update_status(bp, success=True, stage="validation", reason="few out",
                                                new_coords=inter_traj[["x", "y", "t"]], state="interpolated")
                    
                elif n_raw_outliers > MAX_OUTLIER and longest_consecutive_outlier > MAX_OUTLIER//2:
                    tracking.update_status(bp, success=False, stage="outlier", reason=f"out cons>{MAX_OUTLIER//2}")
                    tracking.update_status(bp, success=False, stage="validation", reason=f"out cons>{MAX_OUTLIER//2}",
                                                new_coords=raw_coords, state="raw")

                elif n_raw_outliers >= 2*MAX_OUTLIER :
                    tracking.update_status(bp, success=False, stage="outlier", reason=f"out>={MAX_OUTLIER*2}")
                    tracking.update_status(bp, success=False, stage="validation", reason=f"out>={MAX_OUTLIER*2}",
                                                new_coords=raw_coords, state="raw")

                elif n_raw_outliers >= MAX_OUTLIER and n_interpolated_outliers >= MAX_OUTLIER:
                    tracking.update_status(bp, success=False, stage="outlier", reason=f"out>={MAX_OUTLIER}")
                    tracking.update_status(bp, success=False, stage="validation", reason=f"out>={MAX_OUTLIER}",
                                                new_coords=raw_coords, state="raw")

                else:
                    tracking.update_status(bp, success=False, stage="outlier", reason=f"out<={MAX_OUTLIER}", 
                                           new_coords=raw_coords, state="need_validation")
                    save_as = interpolation_dir / f"interpolation_{bp}_{trial.name}.png"

                    if not save_as.exists():  # plot only if not exist
                        tracking.plot_preprocess(
                                        interpolated_coords=inter_traj[["x", "y", "t"]], interpolated_dist=inter_traj["local_dists"],
                                        outlier_filtered_coords=outlier_traj[["x", "y", "t"]],
                                        raw_coords=raw_coords, raw_dist=outlier_traj["local_dists"],
                                        local_thresholds=outlier_traj["local_thresh"],
                                        time_pad_off=trial.time_pad_off,
                                        title=f"{trial.group}\nn_out: {n_raw_outliers}|{n_interpolated_outliers} - shift: {trial.camera_shift}",
                                        save_as=save_as,
                                    )

            # saving for each trial
            tracking.save_csv(trial.clean_pred_path)        
            trial.update(trajectories = tracking.status_to_dict())
            trial.save_yaml()
                        
        #  backup, save joblib
        self.trialgroup.save_backup(dir_to_copy=self.paths.trials_metadata, output_dir=self.paths.data_root / "trials_metadata_backup", 
                                    step="preprocessing")
        self.trialgroup.save(self.paths.trials_metadata)



    @process_time
    def run_validation(self):
        self.trialgroup.load_trials()
        print(f"""  
        ================= Validation =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        BODYPARTS = self.project_info["bodyparts"]
        DIST_THRESH = self.project_info["distance_thresh"]
        GAP = self.project_info["gap_scale"]
        
        preprocess_dir = self.paths.analysis(self.trialgroup.keep_val) / "preprocess"
        interpolation_dir = preprocess_dir / "interpolation"

        # Find existing validation states
        validation_files = sorted(preprocess_dir.glob("validation_state_*.csv"))

        if validation_files:
            print("Validation has already been computed. Select option:")
            print("q\t: Quit")
            print("c\t: Compute again")
            for i, filename in enumerate(validation_files):
                print(f"{i}\t: load {filename.name}")

            res = input("Option selected: ").strip()

            if res == "q":
                print("\nQuit !\n")
                sys.exit()

            elif res == "c": 
                print("Re-computing validation")
                print("\n-->", len(list(interpolation_dir.iterdir())), "FILES TO VALIDATE")
                trial_states: dict = Validator.load_preprocess_validator(interpolation_dir)

                # save validation state
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pd.Series(trial_states).to_csv(preprocess_dir / f"validation_state_{stamp}.csv")

            elif res.isdigit():
                index = int(res) 

                if index >= len(validation_files):
                    raise ValueError(f"Invalid file index: {index}. Choose between 0 and {len(validation_files) - 1}.")

                selected_file = validation_files[index]
                print(f"\nLoading validation state: {selected_file.name}")
                trial_states = pd.read_csv(selected_file, index_col=0).iloc[:, 0].to_dict() 

            else: 
                raise ValueError(f"'{res}' is not valid. Choose a file number, 'c', or 'q'")

        else :
            print("No validation file found")
            print("\n-->", len(list(interpolation_dir.iterdir())), "FILES TO VALIDATE")
            trial_states: dict = Validator.load_preprocess_validator(interpolation_dir)
            
            # save validation state
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pd.Series(trial_states).to_csv(preprocess_dir / f"validation_state_{stamp}.csv")

        # save validation state
        for trial in tqdm(self.trialgroup.trials, desc="Saving validation"):
            if not trial.is_valid(): 
                continue

            if not any(stat["state"] == "need_validation" for bp, stat in trial.trajectories.items()) :
                continue

            tracking = DLCTracking(path=trial.pred_path,
                                        view=trial.camera_view, 
                                        bodyparts=BODYPARTS,
                                        time_pad_off=trial.time_pad_off,
                                        cm_per_pixel=trial.cm_per_pixel,
                                        shift=trial.camera_shift,
                                        status=trial.trajectories)

            for bp in BODYPARTS: 
                
                state = trial_states.get(f"{bp}_{trial.name}")

                if not state:
                    continue

                if state == "rejected" :
                    tracking.update_status(bp, stage="validation", success=False, reason="man reject", state="raw")
                    continue

                elif state == "raw":
                    tracking.update_status(bp, stage="validation", success=True, reason="man acc, raw", state="raw")
                else:
                    tracking.update_status(bp, stage="validation", success=True, reason="man acc, inter", state="interpolated")
                    tracking.load()
                    interpolated_coords = tracking.interpolate_data(tracking.filter_outliers(tracking.coords_set[bp], DIST_THRESH, GAP))
                    tracking.coords_set[bp] = interpolated_coords
                    tracking.save_csv(trial.clean_pred_path)

            trial.update(trajectories = tracking.status_to_dict())
            trial.save_yaml()
            
        #  backup, then save
        self.trialgroup.save_backup(dir_to_copy=self.paths.trials_metadata, output_dir=self.paths.data_root / "trials_metadata_backup", 
                                    step="validation")
        self.trialgroup.save(self.paths.trials_metadata)



    @process_time 
    def compute_metrics(self): 
        self.trialgroup.load_trials()
        print(f"""
        ============== Compute Metrics ===============
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        analysis_dir = self.paths.analysis(self.trialgroup.keep_val)
        output_dir = u.make_dir(analysis_dir / "trials_df")

        dfs = [
            "trial_success",
            "bodypart_success",
            "behavior",
            "kinematics",
            "scalar",
        ]

        for df in dfs: 
            self.trialgroup.build_df(output_dir, df)