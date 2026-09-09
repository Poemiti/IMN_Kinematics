# src/BetaMouv/Trial.py

from src.Base.Trial import Trial as BaseTrial
from .Leds import Leds
from .File import File


class Trial(BaseTrial):

    FIELDS = BaseTrial.FIELDS + (
        # from filename
        "name", "clip_path", "date", "camera_view", "clip_number",
        "laser_intensity", "handedness", "laser_type", "subject",
        "condition", "session", "stim_location",
        # computed during build_metadata
        "movement_type", "laser_state", "cue_type",
        "time_pad_off", "time_laser_on", "time_reward", "group",
        # computed during preprocessing
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

        # pipeline-computed fields, unknown at construction time
        self.movement_type = None
        self.laser_state = None
        self.cue_type = None
        self.time_pad_off = None
        self.time_laser_on = None
        self.time_reward = None
        self.group = None

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