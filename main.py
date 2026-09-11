import datetime
import shutil
import yaml
from pathlib import Path
from importlib import import_module

from src.Base.Config import Config


cfg = Config(Path("./config.yaml"))
project = cfg.load_project()

if cfg.split_trials: 
    project.split_trials()

if cfg.build_metadata:
    project.build_metadata()

if cfg.run_prediction:
    project.run_prediction()

if cfg.run_preprocessing:
    project.run_preprocessing()

if cfg.run_validation:
    project.run_validation()

if cfg.compute_metrics:
    project.compute_metrics()

if cfg.analysis:
    project.run_analysis()


print(f"\n{project.name} run finished !")