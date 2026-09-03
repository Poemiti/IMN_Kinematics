# src/BetaMouv/Trial.py


from src.Base.Trial import Trial as BaseTrial
from pathlib import Path


class Trial(BaseTrial): 
    def __init__(self, clip_path: Path):
        
        super().__init__(clip_path)

        self.subject = subject
        self.condition = condition
        self.laser_intensity = laser_intensity
        self.task = task

    @staticmethod
    def parse_filename(name: str) -> dict:
        import re
        
        PATTERNS = {
                "rat_name": r"#\d{3}",
                "rat_type": r"(CTRL|CHR)",
                "condition": r"(Conti|NOstim|Beta)",
                "stim_location": r"(LeftHemi|RightHemi|Ipsi|ipsi|Bilateral|Contra|contra)",
                "handedness": r"(Ambidexter|LeftHanded|RightHanded)",
                "session": r"S\d+",
                "view": r"H\d+",
                "laser_intensity": r"\d,\d*mW|\d+mW",
                "date": r"(\d{4}20\d{2}|20\d{6})",
                "clip": r"clip_(\d+)",
                }
        
        TASKS = ["onlyL1LeftHand", "onlyL2", "onlyL1", "onlyL2RightHand", "CueL2RightHand", 
                "L1", "L2", "L1L2", "L1L26040", "L1L25050", "L1-60", "L2-40",
                "NoCue", "CueL1", "CueL2"]

        result = {key: "Unknown" for key in PATTERNS.keys()}
        result["task"] = "Unknown"

        # First pass: regex extraction
        for key, regex in PATTERNS.items():

            if result[key] == "Unknown"  :
                match = re.search(regex, name)
                if match:
                    if key == "clip" : 
                        result[key] = match.group(1)
                        continue
                    result[key] = match.group(0)

        # Task handling (not regex)
        for t in TASKS:
            if t in name.split("_"):
                result["task"] = t
                break

        # Second pass: derived defaults 
        if result["laser_intensity"] == "Unknown" :

            if result["condition"] == "Beta":
                result["laser_intensity"] = "1mW"
            elif result["condition"] == "Conti":
                result["laser_intensity"] = "0,5mW"
            elif result["condition"] == "NOstim":
                result["laser_intensity"] = "NOstim"

        return result



if __name__ == "__main__": 

    trial1 = Trial(clip_path=Path("/home/poemiti/IMN_Kinematics/data/BetaMouv/Rat_#531LeftHanded_20240526_ContiMT300_RightHemiCHR_L1-60_L2-40_C001H001S0003_clip_00.mp4"),)
