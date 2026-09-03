# src/Base/Project.py

from pathlib import Path


class Project:

    def __init__(self, name: str, config_dir: Path):
        self.name = name
        self.config_dir = config_dir

        self.trial_class = None
        self.prediction_class = None
        self.validation_class = None

    def __str__(self):
        desc = f"""
        ================= Project info =================
        Project name: {self.name}
        Config directory: {self.config_dir}
        ================================================\n"""
        return desc
