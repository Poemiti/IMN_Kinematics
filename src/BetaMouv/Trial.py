# src/BetaMouv/Trial.py


from src.Base.Trial import Trial as BaseTrial
from .Leds import Led
import yaml

from .File import File

class Trial(BaseTrial): 

    def __init__(self, clip_path: str):
        
        super().__init__(clip_path)

        self.file = File(clip_path)
        self.laser_state: str = "Not_defined"
        self.cue_type: str = "Not_defined"
        self.time_pad_off: float = "Not_defined"
        self.time_laser_on: float = "Not_defined"
        self.time_reward: float = "Not_defined"

    def to_dict(self) -> dict:
        return {
            "name": self.file.name,
            "clip_path": str(self.file.path),
            "date": self.file.date.isoformat(),
            "camera_view": self.file.camera_view,
            "clip_number": self.file.clip_number,
            "laser_intensity": self.file.laser_intensity,
            "handedness": self.file.handedness,
            "condition": self.file.condition,
            "rat_name": self.file.rat_name,
            "rat_type": self.file.rat_type,
            "session": self.file.session,
            "stim_location": self.file.stim_location,

            "laser_state": self.laser_state,
            "cue_type": self.cue_type,
            "time_pad_off": self.time_pad_off,
            "time_laser_on": self.time_laser_on,
            "time_reward": self.time_reward,

            "group": self.file.set_group(),
            }


    def set_led_info(self, led_obj: Led): 
        
        self.laser_state = led_obj.laser_state
        self.cue_type = led_obj.cue_type
        self.time_pad_off = led_obj.time_pad_off
        self.time_laser_on = led_obj.time_laser_on
        self.time_reward = led_obj.time_reward
