# src/DopaMouv/File.py


from src.Base.File import File as BaseFile
from datetime import datetime


class File(BaseFile): 

    def __init__(self, path: str):
        super().__init__(path)

        # get metadata out of the filename
        trial_metadata = self.parse_filename()

        self.camera_view = trial_metadata["camera_view"]
        self.date = datetime.strptime(trial_metadata["date"], "%Y%m%d").date()
        self.clip_number = int(trial_metadata["clip"]) if trial_metadata["clip"] != "Unknown" else None
        self.subject = trial_metadata["subject"]
        self.condition = trial_metadata["condition"]
        self.cue_condition = trial_metadata["cue_condition"]
        self.cue_type = trial_metadata["cue_type"]
        self.laser_test = trial_metadata["laser_test"]
        self.protocol = trial_metadata["protocol"]
        self.handedness = trial_metadata["handedness"]
        self.pulse_type = trial_metadata["pulse_type"]
        self.pulse_num = int(trial_metadata["pulse_num"]) if trial_metadata["pulse_num"] != "Unknown" else "Unknown"
        self.pulse_frec = trial_metadata["pulse_frec"]
        self.session = trial_metadata["session"]
        self.subsession = trial_metadata["subsession"]

        self.frame_width_px = 512
        self.frame_width_cm = 8.7 if self.camera_view == "left" else 8.3
        self.cm_per_pixel = self.frame_width_cm / self.frame_width_px

        self.full_name = self.build_full_name()

    def build_full_name(self): 
        meta = [
            self.subject, self.handedness, self.date.isoformat(), self.condition,
            self.cue_condition, self.cue_type,  self.laser_test,self.protocol, 
            f"{self.pulse_type}", f"{str(self.pulse_num)}p", self.pulse_frec,
            self.camera_view, self.session, self.subsession
        ]
        return "_".join(meta)


    def parse_filename(self, path: str = None) -> dict:
        import re

        if path is None: 
            path = str(self.path)
        
        PATTERNS = {
                "subject": r"Rat\d{3}",
                "condition": r"(Chrimson|Control|Jaw)",
                "cue_condition": r"(NoNoGo|NoGo)",
                "cue_type": r"S\d",
                "laser_test": r"(Cuedriven|NoStimLaser|opto_LaserTest|Selfpaced_stim)",
                "protocol": r"opto_[^/_]+_\d+%",
                "handedness": r"(Left|Righthand|Ambi)",
                "pulse_type": r"Stop|FULL",
                "pulse_num": r"\d+(?=p)",
                "pulse_frec": r"\d+Hz",
                "camera_view": r"(RightCam|LeftCam)",
                "date": r"(?<!\d)20\d{6}(?!\d)",
                "clip": r"clip_(\d+)",
                "session": r"#\d+",
                "subsession": r"s\d"
                }

        result = {key: "Unknown" for key in PATTERNS.keys()}

        # First pass: regex extraction
        for key, regex in PATTERNS.items():

            if result[key] == "Unknown"  :
                match = re.search(regex, path)
                if match:
                    if key == "clip" : 
                        result[key] = match.group(1)
                        continue
                    result[key] = match.group(0)


        # set pulse based on laser_test
        if result["laser_test"] == "NoStimLaser": 
            result["pulse_type"] = "NoPulse"
            result["pulse_num"] = "0"
            result["pulse_frec"] = "0Hz"

        if result["subsession"] == "Unknown": 
            result["subsession"] = "s1"

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

        metadata.pop("clip", None)
        metadata.pop("date", None)

        file_list.append({
            "filename": str(self.path),
            **metadata
        })