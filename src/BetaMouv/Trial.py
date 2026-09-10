# src/BetaMouv/Trial.py

from src.Base.Trial import Trial as BaseTrial
from .Leds import Leds
from .File import File


class Trial(BaseTrial):

    FIELDS = BaseTrial.FIELDS + (
        # from filename
        "name", "clip_path", "date", "camera_view", "clip_number",
        "laser_intensity", "handedness", "laser_type", "subject",
        "condition", "session", "stim_location", "frame_width_cm", "cm_per_pixel",
        "frame_width_px",

        # computed during build_metadata
        "movement_type", "laser_state", "cue_type", "lever_position",
        "time_pad_off", "time_laser_on", "time_reward", "group",
        "task_success", "task_success_reason", 

        # computed after prediction
        "pred_path",

        # computed during preprocessing
        "model_success", "traj", "coords",
    )

    def __init__(self, clip_path: str, yaml_path: str = None):
        super().__init__(clip_path, yaml_path)

        self.file = File(clip_path)

        # identity fields, filled immediately from the parsed filename
        self.name = self.file.name
        self.clip_path = str(self.file.path)
        self.date = self.file.date.isoformat()
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
        self.time_pad_off = None
        self.time_laser_on = None
        self.time_reward = None
        self.group = None

        # set after prediction
        self.pred_path: str = None
        self.task_success: bool = None
        self.task_success_reason: str = None


    def set_led_info(self, led_obj: Leds):
        self.update(
            laser_state=led_obj.laser_state,
            cue_type=led_obj.cue_type,
            time_pad_off=led_obj.time_pad_off,
            time_laser_on=led_obj.time_laser_on,
            time_reward=led_obj.time_reward,
        )

    def set_mvt_type(self, hemi_info: str):
        contra_rule = {"CueL1": "RightHemi", "CueL2": "LeftHemi"}
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
        if self.cue_type is None: 
            self.update(task_success = False,
                        task_success_reason = "no_cue_detected")

        elif self.time_pad_off is None:
            # print("  ! Pad off time is None")
            self.update(task_success = False,
                        task_success_reason = "no_pad_off")

        elif self.time_laser_on is not None and self.time_laser_on + laser_duration > 3:
            # print(f"  ! Laser window out of bounds (laser_on={time_laser_on})")
            self.update(task_success = False,
                        task_success_reason = "late_pad_off")

        else : 
            self.update(task_success = True,
                        task_success_reason = "paw_lifted")
