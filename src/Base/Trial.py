# src/Base/Trial.py

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import yaml

class Trial:

    def __init__(self, clip_path: Path):

        self.clip_path = clip_path
        self.id = clip_path.stem

        # build metadate from the trial id
        trial_metadata = self._parse_filename(self.id)

        self.date = date = datetime.strptime(trial_metadata["date"], "%Y%m%d").date()
        self.camera_view = trial_metadata["camera_view"]
        self.clip_number = trial_metadata["clip"]

    @staticmethod
    def _parse_filename(filename: str) -> dict:
        import re
        
        PATTERNS = {
            "camera_view": r"H\d+",
            "date": r"(\d{4}20\d{2}|20\d{6})",
            "clip": r"clip_(\d+)",
        }

        result = {key: "Unknown" for key in PATTERNS.keys()}
        result["task"] = "Unknown"

        # First pass: regex extraction
        for key, regex in PATTERNS.items():

            if result[key] == "Unknown"  :
                match = re.search(regex, filename)
                if match:
                    if key == "clip" : 
                        result[key] = match.group(1)
                        continue
                    result[key] = match.group(0)

        return result


    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "clip_path": str(self.clip_path),
            "date": self.date.isoformat(),
            "camera_view": self.camera_view,
            "clip_number": self.clip_number,
        }


    def save_trial(self):
        yaml_path: Path =  self.clip_path.parent / f"{self.id}.yaml"

        with open((yaml_path) , "w") as f : 
            yaml.safe_dump(self.to_dict(), f)





if __name__ == "__main__": 

    trial1 = Trial(clip_path=Path("/home/poemiti/IMN_Kinematics/data/BetaMouv/Rat_#531LeftHanded_20240526_ContiMT300_RightHemiCHR_L1-60_L2-40_C001H001S0003_clip_00.mp4"),)

    print(trial1)
    trial1.save_trial()
    