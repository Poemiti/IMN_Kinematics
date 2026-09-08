# src/Base/Config.py

from pathlib import Path
from datetime import datetime
import shutil, yaml
from functools import wraps
from importlib import import_module


class Config: 

    def __init__(self, path: Path):

        self.path = path
        self.load_cfg()


    def log_cfg(loader):
        @wraps(loader)
        def wrapper(*args, **kwargs):
            cfg_filename: Path = args[0]
            log_dir = Path("./logs")
            log_dir.mkdir(exist_ok=True)

            # Look for previous logs of this config
            previous_logs = sorted(
                log_dir.glob(f"{cfg_filename.stem}_*{cfg_filename.suffix}")
            )

            # Read current config
            current_content = cfg_filename.read_bytes()


            # Check whether it is identical to the most recent log
            if previous_logs:
                last_log = previous_logs[-1]

                if current_content == last_log.read_bytes():
                    print(f"log: config unchanged ({last_log.name})")
                    return loader(*args, **kwargs)


            # Config has changed ==> create a new log
            cfg_date = datetime.datetime.now()
            cfg_new_filename = (
                cfg_filename.stem
                + "_"
                + cfg_date.strftime("%Y%m%d_%H%M%S")
                + cfg_filename.suffix
            )

            shutil.copy(cfg_filename, log_dir / cfg_new_filename)

            print(f"log: {cfg_new_filename}")

            return loader(*args, **kwargs)

        return wrapper


    @log_cfg
    def load_cfg(self) :
        with open(self.path, "r") as cfg_file:
            cfg = yaml.safe_load(cfg_file)

        self.project_name = cfg["project_name"]
        self.prediction = cfg["steps"]["prediction"]
        self.build_metadata = cfg["steps"]["build_metadata"]
        self.preprocessing = cfg["steps"]["preprocessing"]


    def load_project(self) -> object : 
        module = import_module(f"src.{self.project_name}.Project")

        return module.Project(name=self.project_name)
