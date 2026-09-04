import datetime
import shutil
import yaml
from functools import wraps
from pathlib import Path
from importlib import import_module


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
def load_cfg(filename: Path):
    with open(filename, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    return cfg


def load_project(project_name) -> object : 
    module = import_module(f"src.{project_name}.Project")
    config_dir = Path(f"./config/{project_name}")

    return module.Project(name=project_name, 
                          config_dir=config_dir)


# --------------------- load metaconfig ----------------------

cfg = load_cfg(Path("./config.yaml"))

project_name = cfg["project_name"]
project = load_project(project_name)


if cfg["steps"]["prediction"]:
    project.run_prediction()

# if cfg["steps"]["preprocessing"]:
#     project.Validation.run()

# if cfg["steps"]["compute_metrics"]:
#     project.compute_metrics()

# if cfg["steps"]["make_figures"]["single_subject"]:
#     project.make_single_subject_figures()