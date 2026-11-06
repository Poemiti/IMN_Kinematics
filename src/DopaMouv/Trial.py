# src/DopaMouv/Trial.py

from src.Base.Trial import Trial as BaseTrial
from .Leds import Leds
from .File import File

import pandas as pd
from pathlib import Path
from datetime import datetime


class Trial(BaseTrial):
    STAGES = ("clip_openable", "task") # order is important, all done in "build_metadata"
    
    IDENTITY_FIELDS = (
        "name", "clip_path", "raw_video_path", "date", "camera_view", "clip_number", 
        "handedness", "subject",
        "condition", "cue_condition", "cue_declared", "laser_test", 
        "protocol", "pulse_type", "pulse_num", "pulse_freq",  
        "session", "subsession", "laser_state",
        "time_pad_off", "time_laser_on", "time_reward", 
        "group", "trial_outcomes", "task",
    )

    SCALAR_METRICS = (
        "avg_velocity_laser_period", 
        "tortuosity_laser_period", "tortuosity_pad_to_lever", 
    )

    FIELDS = BaseTrial.FIELDS + IDENTITY_FIELDS + SCALAR_METRICS + (
        # from filename
        "frame_width_cm", "cm_per_pixel", "frame_width_px",

        # computed after prediction and metadata_building
        "pred_path", "clean_pred_path", "trajectories"
    )

    YAML_FIELDS = tuple(f for f in FIELDS)
        
    # methods to save or updates Trial attributs
    def __init__(self, clip_path: str, yaml_path: str = None):
        super().__init__(clip_path, yaml_path)

        self.file = File(clip_path)

        # identity fields, filled immediately from the parsed filename
        self.name: str = self.file.name
        self.clip_path: Path = self.file.path
        self.date: datetime = self.file.date
        self.camera_view: str = self.file.camera_view
        self.clip_number: int = self.file.clip_number
        self.handedness: str = self.file.handedness
        self.subject: str = self.file.subject
        self.condition: str = self.file.condition
        self.cue_condition: str = self.file.cue_condition
        self.cue_declared: str = self.file.cue_type
        self.laser_test: str = self.file.laser_test
        self.protocol: str = self.file.protocol
        self.pulse_type: str = self.file.pulse_type 
        self.pulse_num: str = self.file.pulse_num    
        self.pulse_frec: str = self.file.pulse_frec    
        self.session: str = self.file.session
        self.subsession: str = self.file.subsession

        self.frame_width_px: str = self.file.frame_width_px
        self.frame_width_cm: str = self.file.frame_width_cm
        self.cm_per_pixel: str = self.file.cm_per_pixel

        # pipeline-computed fields, unknown at construction time
        # set after build_metadata
        self.raw_video_path: Path = None
        self.task: str = None
        self.laser_state: str = None
        self.time_pad_off: float = None
        self.time_laser_on: float = None
        self.time_reward: float = None
        self.group: str = None

        # set after prediction
        self.pred_path: Path = None
        self.clean_pred_path: Path = self.pred_path.parent / f"clean_{self.pred_path.stem}.csv" if self.pred_path else None

        # set after preprocessing + validation
        self.trajectories: dict[str, dict] = {}
        
    def bodypart_valid(self, bodypart: str) -> bool:
            traj = self.trajectories.get(bodypart)
            return self.is_valid() and traj is not None and traj.is_valid()
            
    
    def identity(self) -> dict:
        return {f: getattr(self, f, None) for f in self.IDENTITY_FIELDS}


    def set_led_info(self, led_obj: Leds):
        self.update(
            task=led_obj.task,
            laser_state=led_obj.laser_state,
            time_pad_off=led_obj.time_pad_off,
            time_laser_on=led_obj.time_laser_on,
            time_reward=led_obj.time_reward,
        )

    def set_group(self, laser_state: str = None):
        if laser_state is None: 
            laser_state= self.laser_state


        self.update(group=(
            f"{self.subject}_{self.condition}_{self.cue_condition}_{self.laser_test}_"
            f"{self.protocol}_{self.pulse_num}"
            f"{self.camera_view}View_{self.laser_intensity}_{laser_state}"
        ))


    def set_task_success(self, laser_duration: int) : 
        """Set if the rat did the correct task
        A correct task is when the rat lift the paw associated with the task.
        Unsuccessful is when the paw we're looking at has not been lift in time"""
        if self.task == "noCue": 
            self.set_success(stage="task", success=False, reason = "Rejected, no cue detected")

        elif self.task is None: 
            self.set_success(stage="task", success=False, reason = "Rejected, clip not readable")
        
        elif self.time_pad_off == 0 : 
            self.set_success(stage="task", success=False, reason = "Rejected, bad split video")

        elif self.time_pad_off is None:
            self.set_success(stage="task", success=False, reason = "Rejected, no pad off")

        elif self.time_laser_on is not None and self.time_laser_on + laser_duration > 3:
            self.set_success(stage="task", success=False, reason = "Rejected, late pad off")

        else : 
            self.set_success(stage="task", success=True, reason = "Successful, paw lifted")

