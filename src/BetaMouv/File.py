# src/BetaMouv/File.py


from src.Base.File import File as BaseFile
from datetime import datetime


class File(BaseFile): 

    def __init__(self, path: str):
        super().__init__(path)

        # get metadata out of the filename
        trial_metadata = self.parse_filename()

        self.date = datetime.strptime(trial_metadata["date"], "%Y%m%d").date()
        self.camera_view = "left" if trial_metadata["camera_view"] == "H001" else "right"
        self.clip_number = trial_metadata["clip"]
        self.subject = trial_metadata["subject"]
        self.condition = trial_metadata["condition"]
        self.laser_type = trial_metadata["laser_type"]
        self.stim_location = trial_metadata["stim_location"]
        self.handedness = trial_metadata["handedness"]
        self.session = trial_metadata["session"]
        self.laser_intensity = trial_metadata["laser_intensity"]


    def parse_filename(self) -> dict:
        import re
        
        PATTERNS = {
                "subject": r"#\d{3}",
                "condition": r"(CTRL|CHR)",
                "laser_type": r"(Conti|NOstim|Beta)",
                "stim_location": r"(LeftHemi|RightHemi|Ipsi|ipsi|Bilateral|Contra|contra)",
                "handedness": r"(Ambidexter|LeftHanded|RightHanded)",
                "session": r"S\d+",
                "camera_view": r"H\d+",
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
                match = re.search(regex, self.name)
                if match:
                    if key == "clip" : 
                        result[key] = match.group(1)
                        continue
                    result[key] = match.group(0)

        # Task handling (not regex)
        for t in TASKS:
            if t in self.name.split("_"):
                result["task"] = t
                break

        # Second pass: derived defaults 
        if result["laser_intensity"] == "Unknown" :

            if result["laser_type"] == "Beta":
                result["laser_intensity"] = "1mW"
            elif result["laser_type"] == "Conti":
                result["laser_intensity"] = "0,5mW"
            elif result["laser_type"] == "NOstim":
                result["laser_intensity"] = "NOstim"

        return result




    def classify_file(self, file_list: list) -> None:
        """
        Parse a video file_path and extract experimental metadata.

        The function decomposes the file_path into tokens that are then
        used to classify the file in certain categories.
        The extracted metadata is appended as a dictionary to `file_list`.

        Parameters
        ----------
        file_path : pathlib.Path
            Full path or name of the video file.
        videos : list
            List to which the extracted metadata dictionary is appended.

        Returns
        -------
        None"""
        
        metadata = self.parse_filename()

        if metadata["condition"] == "Unknown" : 
            metadata = self.parse_filename(self.path.parent.name)

        metadata.pop("clip", None)
        metadata.pop("date", None)

        file_list.append({
            "filename": str(self.path),
            **metadata
        })


