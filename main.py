import datetime
import shutil
import yaml
from pathlib import Path
from importlib import import_module

from src.Base.Config import Config


cfg = Config(Path("./config.yaml"))
project = cfg.load_project()

if cfg.prediction:
    project.run_prediction()

if cfg.build_metadata:
    project.build_metadata()

if cfg.preprocessing:
    project.run_preprocessing()
