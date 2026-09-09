# src/Base/Config.py

from datetime import datetime
from functools import wraps
from importlib import import_module
from pathlib import Path
import shutil

import yaml


class Config:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.load_cfg()

    @staticmethod
    def log_cfg(loader):
        @wraps(loader)
        def wrapper(self, *args, **kwargs):
            cfg_filename = self.path
            log_dir = Path("./logs")
            log_dir.mkdir(parents=True, exist_ok=True)

            # Look for previous logs of this config
            previous_logs = sorted(
                log_dir.glob(
                    f"{cfg_filename.stem}_*{cfg_filename.suffix}"
                )
            )

            # Read current config
            current_content = cfg_filename.read_bytes()

            # Check whether it is identical to the most recent log
            if previous_logs:
                last_log = previous_logs[-1]

                if current_content == last_log.read_bytes():
                    print(f"log: config unchanged ({last_log.name})")
                    return loader(self, *args, **kwargs)

            # Config has changed ==> create a new log
            cfg_date = datetime.now()
            cfg_new_filename = (
                f"{cfg_filename.stem}_"
                f"{cfg_date.strftime('%Y%m%d_%H%M%S')}"
                f"{cfg_filename.suffix}"
            )

            shutil.copy2(
                cfg_filename,
                log_dir / cfg_new_filename,
            )

            print(f"log: {cfg_new_filename}")

            return loader(self, *args, **kwargs)

        return wrapper

    @log_cfg
    def load_cfg(self):
        with self.path.open("r", encoding="utf-8") as cfg_file:
            cfg = yaml.safe_load(cfg_file)

        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid configuration file: {self.path}")

        self.project_name = cfg["project_name"]

        steps = cfg["steps"]
        self.split_trials = steps["split_trials"]
        self.run_prediction = steps["run_prediction"]
        self.build_metadata = steps["build_metadata"]
        self.preprocessing = steps["preprocessing"]
        self.analysis = steps["analysis"]

    def load_project(self) -> object:
        module = import_module(f"src.{self.project_name}.Project")

        return module.Project(name=self.project_name)
