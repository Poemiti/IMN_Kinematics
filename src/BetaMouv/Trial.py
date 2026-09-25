# src/BetaMouv/Trial.py

from src.Base.Trial import Trial as BaseTrial
from .Leds import Leds
from .File import File
from .Trajectory import Trajectory
from.BehaviorBox import BehaviorBox
from src.Base.Outcome import Outcome

import pandas as pd

class Trial(BaseTrial):
        
    STAGES = ("clip_openable", "view_match_task", "task") # order is important, all done in "build_metadata"

    IDENTITY_FIELDS = (
        "name", "clip_path", "date", "camera_view", "clip_number",
        "laser_intensity", "handedness", "laser_type", "subject",
        "condition", "session", "stim_location", "laser_state", 
        "movement_type", "cue_type",
        "time_pad_off", "time_laser_on", "time_reward", "group",
        "task_success", "task_success_reason",
    )

    SCALAR_METRICS = (
        "avg_velocity_laser_period", 
        "tortuosity_laser_period", "tortuosity_pad_to_lever", 
    )

    FIELDS = BaseTrial.FIELDS + IDENTITY_FIELDS + SCALAR_METRICS + (
        # from filename
        "frame_width_cm", "cm_per_pixel", "frame_width_px",

        # computed after prediction and metadata_building
        "pred_path", "lever_position", "behaviorBox",
        "validation_success", "validation_success_reason", "traj", 
        "coords", "coords_success", "coords_success_reason",
        "camera_shift", "stage_outcomes"
    )


    YAML_FIELDS = tuple(f for f in FIELDS if f not in ["traj", "behaviorBox"])


    def __init__(self, clip_path: str, yaml_path: str = None):
        super().__init__(clip_path, yaml_path)

        self.file = File(clip_path)

        # identity fields, filled immediately from the parsed filename
        self.name = self.file.name
        self.clip_path = self.file.path
        self.date = self.file.date
        self.camera_view = self.file.camera_view
        self.clip_number = self.file.clip_number
        self.laser_intensity = self.file.laser_intensity
        self.handedness = self.file.handedness
        self.laser_type = self.file.laser_type
        self.subject = self.file.subject
        self.condition = self.file.condition
        self.session = self.file.session
        self.stim_location = self.file.stim_location
        self.frame_width_px = self.file.frame_width_px
        self.frame_width_cm = self.file.frame_width_cm
        self.cm_per_pixel = self.file.cm_per_pixel

        # pipeline-computed fields, unknown at construction time
        # set after build_metadata
        self.lever_position: tuple[int] = None
        self.movement_type: str = None
        self.laser_state: str = None
        self.cue_type: str = None
        self.time_pad_off: float = None
        self.time_laser_on: float = None
        self.time_reward: float = None
        self.group: str = None

        # set after prediction
        self.pred_path: str = None
        self.task_success: bool = False
        self.task_success_reason: str = "unknown"

        # set after preprocessing + validation
        self.coords: pd.Dataframe = None
        self.coords_success: bool = False
        self.coords_success_reason: str = "unknown"
        self.validation_success: bool = False
        self.validation_success_reason: str = "unknown"

        self.traj: Trajectory = None
        self.behaviorBox: BehaviorBox = None
        self.camera_shift: tuple[float] = None

        self.trajectories: dict[str, Trajectory] = {}


    def bodypart_valid(self, bodypart: str) -> bool:
        traj = self.trajectories.get(bodypart)
        return self.is_valid() and traj is not None and traj.is_valid()
        

    def identity(self) -> dict:
        return {f: getattr(self, f, None) for f in self.IDENTITY_FIELDS}


    def set_led_info(self, led_obj: Leds):
        self.update(
            laser_state=led_obj.laser_state,
            cue_type=led_obj.cue_type,
            time_pad_off=led_obj.time_pad_off,
            time_laser_on=led_obj.time_laser_on,
            time_reward=led_obj.time_reward,
        )

    def set_mvt_type(self, hemi_info: str):
        contra_rule = {"CueL1": "RightHemi", 
                       "CueL2": "LeftHemi"}
        if self.cue_type == "NoCue": 
            self.update(movement_type="UNKNOWN")
            return

        is_contra = (
            hemi_info[self.condition] == self.stim_location
            and contra_rule[self.cue_type] == self.stim_location
        )
        self.update(movement_type="CONTRA" if is_contra else "IPSI")

    def set_group(self):
        self.update(group=(
            f"{self.subject}_{self.condition}_{self.movement_type}_"
            f"{self.laser_type}_{self.stim_location}_"
            f"{self.camera_view}View_{self.laser_intensity}_{self.laser_state}"
        ))


    def set_task_success(self, laser_duration: int) : 
        """Set if the rat did the correct task
        A correct task is when the rat lift the paw associated with the task.
        Unsuccessful is when the paw we're looking at has not been lift in time"""
        if self.cue_type == "noCue": 
            self.set_success(stage="task", success=False, reason = "Rejected, no cue detected")

        elif self.time_pad_off == 0 : 
            self.set_success(stage="task", success=False, reason = "Rejected, bad split video")

        elif self.time_pad_off is None:
            self.set_success(stage="task", success=False, reason = "Rejected, no pad off")

        elif self.time_laser_on is not None and self.time_laser_on + laser_duration > 3:
            self.set_success(stage="task", success=False, reason = "Rejected, late pad off")

        else : 
            self.set_success(stage="task", success=True, reason = "Successful, paw lifted")


    # saving methods 

    def trajectory_outcomes_dict(self) -> dict:
        """Just the pass/fail per bodypart, yaml-safe — no coordinate data."""
        return {bp: entry["outcome"].to_dict() 
                for bp, entry in self.trajectories.items()}

    def to_yaml_dict(self) -> dict:
        data = super().to_yaml_dict()  # everything in YAML_FIELDS, normally serialized
        data["trajectory_outcomes"] = self.trajectory_outcomes_dict()
        return data