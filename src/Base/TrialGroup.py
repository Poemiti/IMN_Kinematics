# src/Base/TrialGroup.py

from .Trial import Trial as BaseTrial
import joblib
from pathlib import Path
import pandas as pd


class TrialGroup:
    """Manager to filter trials to use by their group name"""

    trial_cls = BaseTrial  

    def __init__(self, filenames: list[Path], condition: dict):
        self.filenames = filenames
        self.condition = condition

        self.keep_val = None

        self.trials: list[BaseTrial] = self._filter_joblib()



    def _filter_joblib(self) -> list[dict]:
        trials = []

        for filename in self.filenames:
            if self._keep_file(filename):
                print(f"Keep: {filename.name}")

                records: list = joblib.load(filename)   # list of Trial objects
                trials.extend(records)
            else:
                print(f"Not Keep: {filename.name}")

        return trials

    def _keep_file(self, filename: Path) -> bool:
        name = filename.name

        for condition, criteria in self.condition.items():
            self.keep_val = [value for value, keep in criteria.items() if keep]

            if self.keep_val and not any(value in name for value in self.keep_val):
                return False

            not_keep_val = [value for value, keep in criteria.items() if not keep]

            if any(value in name for value in not_keep_val):
                return False

        return True

    def save(self, output_dir: Path):
        by_group = {}
        for trial in self.trials:
            by_group.setdefault(trial.group, []).append(trial)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for group, records in by_group.items():
            joblib.dump(records, output_dir / f"{group}.joblib")



