# src/Base/TrialGroup.py

from .Trial import Trial
import joblib
from pathlib import Path

class TrialGroup: 
    """Manager to filter trials to use by their group name"""

    def __init__(self, filenames: list[Path], condition: dict):

        self.filenames = filenames
        self.condition = condition

        self.trials = self._filter_trials()


    def _filter_trials(self) -> list[Trial]:
        trials = []

        for filename in self.filenames:
            if self._keep_file(filename):
                print(f"{filename.name}: keep")

                file_trials = joblib.load(filename)
                trials.extend(file_trials)
            else : 
                print(f"{filename.name}: not keep")

        return trials


    def _keep_file(self, filename: Path) -> bool:

        name = filename.name

        self.keep_val = [value
            for value, keep in self.condition.items()
            if keep]

        # If this category has active filters,
        # filename must match one of them.
        if self.keep_val and not all(value in name for value in self.keep_val):
            return False

        # Explicit exclusions
        self.not_keep_val = [value
            for value, keep in self.condition.items()
            if not keep]

        if any(value in name for value in self.not_keep_val):
            return False

        return True