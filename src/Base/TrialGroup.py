# src/Base/TrialGroup.py

from .Trial import Trial as BaseTrial
import joblib, os, shutil
from pathlib import Path
import pandas as pd
from datetime import datetime


class TrialGroup:
    """Manager to filter trials to use by their group name"""

    trial_cls = BaseTrial  

    def __init__(self, filenames: list[Path], condition: dict):
        self.filenames = filenames
        self.condition = condition

        self.keep_joblib = []
        self.trials: list[BaseTrial] = self._filter_joblib()

        self.keep_val = []
        for condition, criteria in self.condition.items():
            self.keep_val.extend(value for value, keep in criteria.items() if keep)
        self.group_name = "_".join(self.keep_val)

        # display output

        print(f"\n--> {len(self.keep_joblib)}/{len(self.filenames)} FILES LOADED")
        print(self.group_name)


    def _filter_joblib(self) -> list[dict]:
        trials = []

        for filename in self.filenames:
            if self._keep_file(filename):

                self.keep_joblib.append(filename)
                records: list = joblib.load(filename)   # list of Trial objects
                trials.extend(records)
            # else:
            #     print(f"Not Keep: {filename.name}")

        return trials

    def _keep_file(self, filename: Path) -> bool:
        name = filename.name

        for condition, criteria in self.condition.items():
            val_to_keep = [value for value, keep in criteria.items() if keep]

            if val_to_keep and not any(value in name for value in val_to_keep):
                return False

            not_keep_val = [value for value, keep in criteria.items() if not keep]
            if any(value in name for value in not_keep_val):
                return False

        return True

    def save(self, output_dir: Path, trial_list: list = None):
        by_group = {}

        if trial_list is None : 
            print("Saving trials from trialgroup")
            trial_list = self.trials

        for trial in trial_list:
            by_group.setdefault(trial.group, []).append(trial)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for group, records in by_group.items():
            path = output_dir / f"{group}.joblib"
            tmp = path.with_name(path.name + ".tmp")
            try:
                joblib.dump(records, tmp)
                if tmp.stat().st_size == 0:
                    raise IOError(f"Refusing to write empty file: {path}")
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)


    def save_backup(self, dir_to_copy: Path, output_dir: Path, step: str) -> None:
        """Save trial metadata backup""" 
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output_dir / f"{step}_{stamp}"
        shutil.copytree(dir_to_copy, backup)
        print(f"Backup: {backup}")
