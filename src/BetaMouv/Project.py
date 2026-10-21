# src/BetaMouv/Project.py

from src.Base.Project import Project as BaseProject
from .Video import Video

from .PathConfig import PathConfig
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
        print(f"""
        =============== Preprocessing =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")
        preprocess_dir = u.make_dir(self.paths.analysis(self.trialgroup.keep_val) / "preprocess")
        preprocess_data_path = preprocess_dir / "preprocess_data.csv"

        interpolation_dir = preprocess_dir / "interpolation"
        # shutil.rmtree(interpolation_dir, ignore_errors=True)   # clear figures first
        u.make_dir(interpolation_dir)                          # then (re)create

        MAX_OUTLIER = self.project_info["max_outlier"]
        DIST_THRESH = self.project_info["distance_thresh"]
        GAP = self.project_info["gap_scale"]
        BODYPARTS = self.project_info["bodyparts"]

        preprocess_data = []

        for trial in tqdm(self.trialgroup.trials, desc="Preprocessing"):

            if not trial.is_valid():
                continue

            tracking = DLCTracking(raw_path=trial.pred_path,
                                        view=trial.camera_view, 
                                        bodyparts=BODYPARTS,
                                        time_pad_off=trial.time_pad_off,
                                        cm_per_pixel=trial.cm_per_pixel,
                                        shift=trial.camera_shift)

            outlier_results, interpolation_results = tracking.preprocess(DIST_THRESH, GAP, "sline", 5)

            # set outlier stage outcome 
            for bp in BODYPARTS: 
                outlier_traj, inter_traj = outlier_results[bp], interpolation_results[bp]
                raw_outlier_mask = outlier_traj["raw_mask"]
                n_raw_outliers = int(raw_outlier_mask.sum())
                n_interpolated_outliers = int(outlier_traj["inter_mask"].sum())

                if n_raw_outliers == 0 :
                    tracking.update_status(bp, success=True, stage="outlier", reason="0 out")
                    tracking.update_status(bp, success=True, stage="validation", reason="0 out",
                                                new_coords=tracking.coords_set[bp],  state="raw",)

                elif n_raw_outliers <= MAX_OUTLIER//2 or n_interpolated_outliers == 0:
                    tracking.update_status(bp, success=True, stage="outlier", reason="few out")
                    tracking.update_status(bp, success=True, stage="validation", reason="few out",
                                                new_coords=inter_traj[["x", "y"]], state="interpolated")
                                           
                elif n_raw_outliers > MAX_OUTLIER and tracking.longest_consecutive_true(raw_outlier_mask) > MAX_OUTLIER//2:
                    tracking.update_status(bp, success=False, stage="outlier", reason=f"out cons>{MAX_OUTLIER//2}")
                    tracking.update_status(bp, success=False, stage="validation", reason=f"out cons>{MAX_OUTLIER//2}",
                                                new_coords=tracking.coords_set[bp], state="raw")
                                           
                elif n_raw_outliers >= 2*MAX_OUTLIER :
                    tracking.update_status(bp, success=False, stage="outlier", reason=f"out>={MAX_OUTLIER*2}")
                    tracking.update_status(bp, success=False, stage="validation", reason=f"out>={MAX_OUTLIER*2}",
                                                new_coords=tracking.coords_set[bp], state="raw")

                elif n_raw_outliers >= MAX_OUTLIER and n_interpolated_outliers >= MAX_OUTLIER:
                    tracking.update_status(bp, success=False, stage="outlier", reason=f"out>={MAX_OUTLIER}")
                    tracking.update_status(bp, success=False, stage="validation", reason=f"out>={MAX_OUTLIER}",
                                                new_coords=tracking.coords_set[bp], state="raw")

                else:
                    tracking.update_status(bp, success=False, stage="outlier", reason=f"out<={MAX_OUTLIER}", 
                                           new_coords=tracking.coords_set[bp], state="raw")
                    save_as = interpolation_dir / f"interpolation_{bp}_{trial.name}.png"

                    if not save_as.exists():  # plot only if not exist
                        tracking.plot_preprocess(
                                        interpolated_coords=inter_traj, interpolated_dist=inter_traj["local_dist"],
                                        outlier_filtered_coords=outlier_traj,
                                        raw_coords=tracking.coords_set[bp], raw_dist=outlier_traj["local_dist"],
                                        local_thresholds=outlier_traj["local_thresh"],
                                        time_pad_off=trial.time_pad_off,
                                        title=f"{trial.group}\nn_out: {n_raw_outliers}|{n_interpolated_outliers} - shift: {trial.camera_shift}",
                                        save_as=save_as,
                                    )

            # saving for each trial
            tracking.save_clean_csv()        
            trial.trajectories = tracking.status
            trial.save_yaml()
        
        #  backup, save joblib
        self.trialgroup.save_backup(dir_to_copy=self.paths.trials_metadata, output_dir=self.paths.data_root / "trials_metadata_backup", 
                                    step="preprocessing")
        self.trialgroup.save(self.paths.trials_metadata)



    @process_time
    def run_validation(self):
        print(f"""  
        ================= Validation =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        BODYPARTS = self.project_info["bodyparts"]
        
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


            tracking = DLCTracking(aw_path=trial.pred_path,
                                        view=trial.camera_view, 
                                        bodyparts=BODYPARTS,
                                        time_pad_off=trial.time_pad_off,
                                        cm_per_pixel=trial.cm_per_pixel,
                                        shift=trial.camera_shift)

            for bodypart in BODYPARTS: 
                
                state = trial_states.get(f"{bodypart}_{trial.name}")

                if not state:
                    continue

                trajectory_status = trial.trajectories.get(bodypart)

                if state == "rejected" :
                    tracking.update_status(stage="validation", success=False, reason="man reject")
                    tracking.update_clean_coords()
                    trajectory_status.set_success(stage="validation", success=False, reason="man reject")
                    continue

                elif state == "raw":
                    traj.set_success(stage="validation", success=True, reason="man acc, raw")
                    traj.clean_coords = traj.coords
                else:
                    traj.set_success(stage="validation", success=True, reason="man acc, inter")
                    traj.clean_coords = traj.interpolated_coords

        #  backup, then save
        self.trialgroup.save_backup(dir_to_copy=self.paths.trials_metadata, output_dir=self.paths.data_root / "trials_metadata_backup", 
                                    step="validation")
        self.trialgroup.save(self.paths.trials_metadata)



    @process_time 
    def compute_metrics(self): 
        print(f"""
        ============== Compute Metrics ===============
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")

        BODYPARTS = self.project_info["bodyparts"]        

        for trial in tqdm(self.trialgroup.trials, desc="Computing metrics"):
            if not trial.is_valid():    
                continue

            # video = Video(trial.clip_path)
            # video.annotate_video(output_path=self.paths.data_root / "annotated_vid" / f"{trial.name}.mp4")
            # video.extract_frames(frame_range=range(video.frame_count), output_base=self.paths.data_root / "annotated_vid" / "frame", video_path=self.paths.data_root / "annotated_vid" / f"{trial.name}.mp4")

            # compute behavior state

            if all(traj.is_valid() for traj in trial.trajectories.values()): 
                print(trial.name)
                # for body, traj in trial.trajectories.items(): 
                #     print(body, traj.is_valid(), traj.clean_coords is None)

                behavior = Behavior.Behavior(coords_set=trial.trajectories,
                                    time_pad_off=trial.time_pad_off, 
                                    cm_per_pixel=trial.cm_per_pixel)

                # fig, ax = plt.subplots(figsize=(6, 6))
                # behavior.draw_boxes(ax)
                # ax.plot(behavior.centroid.coords["x"], behavior.centroid.coords["y"], color = "k")
                # plt.show()

                trial.update(behavior=behavior)

            # compute metrics for each trajectories

            for bodypart in BODYPARTS: 

                traj: TrajectoryDLC = trial.trajectories.get(bodypart)
                
                if not traj.is_valid(): 
                    continue
                
                traj.clean_coords = traj.compute_instant_metrics(traj.clean_coords)
                traj.scalar_metrics = traj.compute_scalar_metrics(traj.clean_coords)

        self.trialgroup.save_backup(dir_to_copy=self.paths.trials_metadata, output_dir=self.paths.data_root / "trials_metadata_backup", 
                                            step="metrics")
        self.trialgroup.save(self.paths.trials_metadata)


    @process_time 
    def run_analysis(self): 
        print(f"""
        =============== Analysis =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ==============================================\n""")
        import matplotlib.pyplot as plt
        import seaborn as sns
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme("talk", style="ticks", rc=custom_params, palette="pastel")
        
        analysis_dir = self.paths.analysis(self.trialgroup.keep_val)
        u.make_dir(analysis_dir)
        self.trialgroup.crop_laser_period = True
        self.trialgroup.remove_NOstim = True

        traj_data = []

        for trial in self.trialgroup.trials:
            if not trial.is_valid():
                continue

            trial_traj_data = []

            for bp, traj in trial.trajectories.items():

                coords = traj.clean_coords if traj.is_valid() else traj.coords

                # Store trajectory-level information
                traj_info = {
                    "trial": trial.name,
                    "bodypart": bp,
                    "n_frames": len(coords),
                    "is_valid": traj.is_valid(),
                    "valid_reason": traj.failure(),
                }

                traj_data.append(traj_info)

                # Store frame-level coordinates for plotting
                coords_df = coords[["x", "y", "t"]].copy()
                coords_df["trial"] = trial.name
                coords_df["bodypart"] = bp
                coords_df["is_valid"] = traj.is_valid()

                trial_traj_data.append(coords_df)

            # Combine all bodypart trajectories for this trial
            if trial_traj_data:
                print(trial.name)
                plt.close()

                trial_traj_data_df = pd.concat(
                    trial_traj_data,
                    ignore_index=True
                )

                # Plot trajectories
                plt.figure(figsize=(8, 6))

                sns.lineplot(
                    data=trial_traj_data_df,
                    x="x",
                    y="y",
                    hue="bodypart", palette="bright",
                    sort=False,
                    estimator=None
                )

                plt.title(f"{trial.name}")
                plt.axis("equal")
                plt.show()


        traj_data_df = pd.DataFrame(traj_data)
        traj_data_df.to_csv("traj_data_df.csv")

        # display the proportion of trajectories that don't have the same number of frames
        # display the proportion of trajectories where not all bodyparts are valid
        # Group trajectories by trial
        trial_stats = traj_data_df.groupby("trial").agg(
            n_bodyparts=("bodypart", "nunique"),
            n_frames_unique=("n_frames", "nunique"),
            all_bodyparts_valid=("is_valid", "all"),
            all_bodyparts_have_same_frames=("n_frames", lambda x: x.nunique() == 1)
        )
        trial_stats.to_csv("trial_stat.csv")

        # Proportion of trials where bodyparts have different numbers of frames
        prop_different_frames = (
            ~trial_stats["all_bodyparts_have_same_frames"]
        ).mean()

        # Proportion of trials where at least one bodypart is invalid
        prop_not_all_valid = (
            ~trial_stats["all_bodyparts_valid"]
        ).mean()

        print(f"Number of trials analyzed: {len(trial_stats)}")

        print(
            f"Trials where bodyparts have different numbers of frames: "
            f"{prop_different_frames:.1%}"
        )

        print(
            f"Trials where not all bodyparts are valid: "
            f"{prop_not_all_valid:.1%}"
        )
        # Trials with different numbers of frames
        different_frames = trial_stats[
            ~trial_stats["all_bodyparts_have_same_frames"]
        ]

        # Trials with at least one invalid bodypart
        invalid_trials = trial_stats[
            ~trial_stats["all_bodyparts_valid"]
        ]

        print("\nTrials with different frame counts:")
        print(different_frames)

        print("\nTrials with invalid bodyparts:")
        print(invalid_trials)
        invalid_trials.to_csv("invalid_trials.csv")

        # --- Proportion of trials where trajectories don't all have the same number of frames ---
        frames_per_trial = traj_data_df.groupby("trial")["n_frames"].nunique()
        mismatch_frames = frames_per_trial > 1
        prop_frame_mismatch = mismatch_frames.mean()

        print(
            f"Trials with mismatched frame counts across bodyparts: "
            f"{mismatch_frames.sum()} / {len(mismatch_frames)} "
            f"({prop_frame_mismatch:.1%})"
        )

        # --- Proportion of trials where not all bodyparts are valid ---
        all_valid_per_trial = traj_data_df.groupby("trial")["is_valid"].all()
        not_all_valid = ~all_valid_per_trial
        prop_not_all_valid = not_all_valid.mean()

        print(
            f"Trials where not all bodyparts are valid: "
            f"{not_all_valid.sum()} / {len(not_all_valid)} "
            f"({prop_not_all_valid:.1%})"
        )

        # Optional: breakdown of failure reasons among invalid trajectories
        print("\nFailure reasons (invalid trajectories only):")
        print(
            traj_data_df.loc[~traj_data_df["is_valid"], "valid_reason"]
            .value_counts()
        )
        
        # ___________________________________________________________________________________________
        ################################# SUCCESS RATE / METADATA REPORT #############################



        # self.trialgroup.trial_success_rate(u.make_path(analysis_dir, "trial_success_rate.png"))
        # self.trialgroup.trajectories_success_rate(u.make_path(analysis_dir, "trajectories_success_rate.png"))

        # self.trialgroup.sunburst_metadata(output_dir=u.make_dir(analysis_dir / "metadata_sunburst"),
        #                                   subfig_group="subject", 
        #                                   groups=["laser_type"],
        #                                   title=self.trialgroup.group_name)

        # self.trialgroup.sunburst_metadata(output_dir=u.make_dir(analysis_dir / "metadata_sunburst"), 
        #                                   subfig_group="condition", 
        #                                   groups=["subject", "laser_type", "laser_intensity"],
        #                                   title=self.trialgroup.group_name)

        # self.trialgroup.sunburst_metadata(output_dir=u.make_dir(analysis_dir / "metadata_sunburst"), 
        #                                   subfig_group=None, 
        #                                   groups=["subject", "is_valid", "failing_step"],
        #                                   title=self.trialgroup.group_name)


        # _________________________________________________________________________________
        ################################# BEHAVIOR ANALYSIS #############################

              
        features = [
                    "relative_t",
                    "x", 
                    "y",
                    "area", 
                    "centroid_direction", 
                    "centroid_velocity", 
                    "centroid_speed", 
                    "centroid_acc", 
                    "f3_sp_distance", 
                    "f3_sp_angle", 
                    "centroid_lever_dist"
        ]

        # for feat in features: 
        #     self.trialgroup.plot_tendency_beha_features(
        #                         save_as=u.make_path(analysis_dir / "behavior" / "with_behavior_span", f"{feat}.png"),
        #                         x="relative_t", y=feat, show_beha="spatial_behavior")
            
        #     self.trialgroup.plot_tendency_beha_features(
        #                         save_as=u.make_path(analysis_dir / "behavior", f"{feat}.png"),
        #                         x="relative_t", y=feat, show_beha=None)
        
        # self.trialgroup.lineplot_traj_centroid(save_as=u.make_path(analysis_dir / "behavior", f"centroid_traj.png"),) 


        # for l_type in ["Beta","Conti"] : 
        #     for l_intensity in ["low", "high"]: 
        #         for l_state in ["LaserOn", "LaserOff"] : 

        #             self.trialgroup.distri_feat_per_behavior(laser_type=l_type, laser_intensity=l_intensity, laser_state=l_state,
        #                                                     save_as=u.make_path(analysis_dir / "behavior" / "distri_feat_per_behavior", f"spatial_behavior_{l_type}_{l_intensity}_{l_state}.svg"),
        #                                                     features_list=features)

        #             self.trialgroup.distri_feat_per_behavior(data=df, behavior_varname="pred_state", laser_type=l_type, laser_intensity=l_intensity, laser_state=l_state,
        #                                                     save_as=u.make_path(analysis_dir / "behavior" / "distri_feat_per_behavior", f"HMM_pred_state_{l_type}_{l_intensity}_{l_state}.svg"),
        #                                                     features_list=features)



        # self.trialgroup.distri_feat_per_behavior(
        #                                         save_as=u.make_path(analysis_dir / "behavior" / "distri_feat_per_behavior", "spatial_behavior_state_all.svg"),
        #                                         features_list=features)

        # self.trialgroup.distri_feat_per_behavior(data=df, behavior_varname="pred_state", 
        #                                         save_as=u.make_path(analysis_dir / "behavior" / "distri_feat_per_behavior", "HMM_pred_state_all.svg"),
        #                                         features_list=features)

        # ___________________________________________________________________________________________
        ################################# BEHAVIOR EXPLORATION #############################


        # HMM

        # df = self.trialgroup.behavior_features_df(add_identity=True)
        # Behavior.HMM_exploration(df, analysis_dir, init=False)


        # # Keypoint MoSeq

        # kpms_dir = u.make_dir(self.paths.data_root / "Keypoint-MoSeq")
        # Behavior.Keypoint_MoSeq_exploration(kpms_dir, self.paths.model, self.paths.raw_videos)


        # _________________________________________________________________________________
        ################################# TRAJECTORY ANALYSIS #############################

        ################################# ALL TRAJ
        ################################# Timeseries


        bodypart = "finger_3"
        self.trialgroup.crop_laser_period = True
        self.trialgroup.remove_NOstim = True

        # self.trialgroup.buils_timeseries_df(init=False, save_as=analysis_dir / f"{bodypart}_timeseries_df.csv")

        # trial = self.trialgroup.trials[1]

        # behavior_obj = Behavior(trial.trajectories, trial.time_pad_off, trial.cm_per_pixel)
        # self.trialgroup.lineplot_traj(save_as=analysis_dir / f"{bodypart}_all_traj_with_beha_zones.svg", Beha=behavior_obj)
        # self.trialgroup.lineplot_traj(save_as=u.make_path(analysis_dir, f"{bodypart}_all_traj.svg"))
        # self.trialgroup.lineplot_traj_per_indentity(save_as=analysis_dir / f"{bodypart}_traj_per_indentity.svg")

        # timeseries_metric = [
        #                     "x", 
        #                     "y", 
        #                     "instant_velocity", 
        #                     "instant_speed", 
        #                     "acc", 
        #                     "signed_acc", 
        #                     "lever_distance",
        #                     "angle"
        #                     ] 
        # for val in timeseries_metric:
        #     self.trialgroup.plot_tendency_metrics(
        #         value=val,
        #         save_as=u.make_path(analysis_dir / "tendency" / bodypart, f"{val}.svg"),
        #         # show_units=True, 
        #     )

        


        ################################# ALL TRAJ
        ################################# Scalar series

        # comparisons = [
        #     # Conti vs Beta
        #     ("Conti_LaserOff.low",  "Beta_LaserOff.low"),
        #     ("Conti_LaserOff.high", "Beta_LaserOff.high"),
        #     ("Conti_LaserOn.low",   "Beta_LaserOn.low"),
        #     ("Conti_LaserOn.high",  "Beta_LaserOn.high"),

        #     # Off vs On
        #     ("Conti_LaserOff.low",  "Conti_LaserOn.low"),
        #     ("Conti_LaserOff.high", "Conti_LaserOn.high"),
        #     ("Beta_LaserOff.low",   "Beta_LaserOn.low"),
        #     ("Beta_LaserOff.high",  "Beta_LaserOn.high"),

        #     # low vs high
        #     ("Beta_LaserOff.low",   "Beta_LaserOff.high"),
        #     ("Beta_LaserOn.low",    "Beta_LaserOn.high"),
        #     ("Conti_LaserOff.low",  "Conti_LaserOff.high"),
        #     ("Conti_LaserOn.low",   "Conti_LaserOn.high"),
        # ]

        # comparisons_merge_Off = [
        #     # conti vs beta
        #     ("Conti.low",  "Beta.low"),
        #     ("Conti.high", "Beta.high"),

        #     # low vs high
        #     ("LaserOff.high", "Conti.high"),
        #     ("LaserOff.high",  "Beta.high"),
        #     ("LaserOff.low",    "Beta.low"),
        #     ("LaserOff.low",   "Conti.low"),

        #     # condition high vs condition low
        #     ("Conti.low", "Conti.high"),
        #     ("Beta.low",  "Beta.high"),

        #     # Beta high ON vs Conti low ON 
        #     # (because conti low is suppose to be as damagefull as the beta high)
        #     ("Beta.high", "Conti.low"),
        #     ("Beta.low", "Conti.high")
        # ]

        # metrics = [
        #     "avg_velocity_laser_period", 
        #     "tortuosity_laser_period", 
        #     "tortuosity_pad_to_lever", 
        # ]

        # for metric in metrics: 
        #     self.trialgroup.violin_statistic(comparisons, metric=metric, 
        #                                      save_as=u.make_path(analysis_dir / "distribution" / bodypart, f"{metric}.svg"))
        #     self.trialgroup.violin_statistic(comparisons_merge_Off, metric=metric,  merge_Off=True,
        #                                      save_as=u.make_path(analysis_dir / "distribution" / bodypart, f"{metric}_merged_off.svg"),
        #                                      order=["LaserOff", "Conti", "Beta"])


        ################################# SINGLE TRAJ 
        ################################# Timeseries

        # i = 0
        # for trial in tqdm(self.trialgroup.trials, desc="plotting timeseries"): 
        #     if not trial.is_valid(): 
        #         continue
            
        #     traj: TrajectoryDLC = trial.trajectories.get(bodypart)

        #     if not traj.is_valid(): 
        #         continue

        #     fig, ax = plt.subplots()

        #     ax = traj.plot_traj(traj.coords, ax, fig, color = "orange", label="raw")
        #     ax = traj.plot_traj(traj.interpolated_coords, ax, fig, color = "green", label="interpolated")
        #     ax = traj.plot_traj(traj.clean_coords, ax, fig, color = "blue", label="clean")


        #     name = trial.name
        #     title = name[:len(name)//2] + "\n" + name[len(name)//2:]

        #     ax.set(
        #         title=title,
        #         xlabel="X position (cm)",
        #         ylabel="Y position (cm)",
        #     )
        #     fig.savefig(u.make_path(analysis_dir / "traj_per_trial" / bodypart, f"{trial.name}.svg"))
        #     plt.close()

        #     i += 1
        #     if i > 300 : 
        #         break
            # for val in timeseries_metric: 

            #     traj.plot_tendency(value=val, time_pad_off=trial.time_pad_off,
            #                        laser_state=trial.laser_state,
            #                        group=trial.group,
            #                        show_angle= val == "angle",
            #                        save_as=u.make_path(analysis_dir / "tendency_per_trial" / bodypart / trial.name , f"{val}.svg"))

    
