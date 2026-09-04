# src/BetaMouv/Trial.py


from src.Base.Trial import Trial as BaseTrial
import yaml

from .File import File

class Trial(BaseTrial): 

    def __init__(self, clip_path: str):
        
        super().__init__(clip_path)

        self.file = File(clip_path)


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
        }