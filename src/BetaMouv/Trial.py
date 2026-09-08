# src/BetaMouv/Trial.py


from src.Base.Trial import Trial as BaseTrial
from .Leds import Leds
import yaml

from .File import File

class Trial(BaseTrial): 

    def __init__(self, clip_path: str, yaml_path: str = None):
        
        super().__init__(clip_path)

        self.file = File(clip_path)
        self.movement_type: str = "Not_defined"
        self.laser_state: str = "Not_defined"
        self.cue_type: str = "Not_defined"
        self.time_pad_off: float = "Not_defined"
        self.time_laser_on: float = "Not_defined"
        self.time_reward: float = "Not_defined"

    def to_dict(self) -> dict:
        return {
            # metadata extracted from the filename
            "name": self.file.name,
            "clip_path": str(self.file.path),
            "date": self.file.date.isoformat(),
            "camera_view": self.file.camera_view,
            "clip_number": self.file.clip_number,
            "laser_intensity": self.file.laser_intensity,
            "handedness": self.file.handedness,
            "laser_type": self.file.laser_type,
            "subject": self.file.subject,
            "condition": self.file.condition,
            "session": self.file.session,
            "stim_location": self.file.stim_location,
            "movement_type": self.movement_type,

            # metadata extracted from the video itself (LED observation)
            "laser_state": self.laser_state,
            "cue_type": self.cue_type,
            "time_pad_off": self.time_pad_off,
            "time_laser_on": self.time_laser_on,
            "time_reward": self.time_reward,

            "group": self.group,
            }

    def load_yaml_trial(self): 
            with open(self.yaml_path, "r") as f:
                data = yaml.safe_load(f)

            # TODO
            # finir l'initialisation des attribut (respect des object comme date)
    
            self.laser_state = data.get("laser_state", "Not_defined")
            self.cue_type = data.get("cue_type", "Not_defined")
            self.time_pad_off = data.get("time_pad_off", "Not_defined")
            self.time_laser_on = data.get("time_laser_on", "Not_defined")
            self.time_reward = data.get("time_reward", "Not_defined")
            self.set_group()


    def set_mvt_type(self, hemi_info: str):
        contra_rule = {
             "CueL1": "RightHemi",  # if stimulation in right hemi - task must be left L1
             "CueL2": "LeftHemi"    # if stimulation in left hemi - task must be right L2
        }
        print(hemi_info)

        if hemi_info[self.file.condition]  == self.file.stim_location and \
           contra_rule[self.cue_type] == self.file.stim_location : 

            self.movement_type = "CONTRA"
        else : 
            self.movement_type = "IPSI"        



    def set_led_info(self, led_obj: Leds): 
        
        self.laser_state = led_obj.laser_state
        self.cue_type = led_obj.cue_type
        self.time_pad_off = led_obj.time_pad_off
        self.time_laser_on = led_obj.time_laser_on
        self.time_reward = led_obj.time_reward

    def set_group(self):
        self.group = (
            f"{self.file.subject}_"
            f"{self.file.condition}_"
            f"{self.movement_type}_"
            f"{self.file.laser_type}_"
            f"{self.file.stim_location}_"
            f"{self.file.camera_view}View_"
            f"{self.file.laser_intensity}_"
            f"{self.laser_state}"
        )