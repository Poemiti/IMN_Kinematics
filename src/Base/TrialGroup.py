# src/Base/TrialGroup.py

from .Trial import Trial as BaseTrial
import joblib, os
from pathlib import Path
import pandas as pd


class TrialGroup:
    """Manager to filter trials to use by their group name"""

    trial_cls = BaseTrial  

    def __init__(self, filenames: list[Path], condition: dict):
        self.filenames = filenames
        self.condition = condition

        self.trials: list[BaseTrial] = self._filter_joblib()

        self.keep_val = []
        for condition, criteria in self.condition.items():
            self.keep_val.extend(value for value, keep in criteria.items() if keep)
        self.group_name = "_".join(self.keep_val)

    def _filter_joblib(self) -> list[dict]:
        trials = []

        for filename in self.filenames:
            if self._keep_file(filename):

                records: list = joblib.load(filename)   # list of Trial objects
                trials.extend(records)
            else:
                print(f"Not Keep: {filename.name}")

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
            print("Loading trials from trialgroup")
            trial_list = self.trials

        for trial in trial_list:
            by_group.setdefault(trial.group, []).append(trial)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for group, records in by_group.items():
            joblib.dump(records, output_dir / f"{group}.joblib")



def save(self, output_dir: Path, trial_list: list = None):
    # 1. get the trials (accept either a list or a {group: [trials]} dict)
    if trial_list is None:
        print("Loading trials from trialgroup")
        trial_list = self.trials
    if isinstance(trial_list, dict):
        trial_list = [t for trials in trial_list.values() for t in trials]

    if not trial_list:
        raise ValueError("TrialGroup.save: no trials to save, nothing written")

    # 2. validate before touching the disk
    by_group, problems = {}, []
    for trial in trial_list:
        group = getattr(trial, "group", None)
        if not group:
            problems.append(f"{getattr(trial, 'name', trial)}: no group")
            continue
        try:
            trial.to_yaml_dict()          # same strict check as for the YAML
        except Exception as e:
            problems.append(f"{trial.name}: {e!r}")
            continue
        by_group.setdefault(group, []).append(trial)

    if problems:
        shown = "\n  ".join(problems[:10])
        raise ValueError(f"{len(problems)} invalid trials, nothing written:\n  {shown}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. phase 1: write and verify every temp file (real files untouched)
    tmp_files = {}
    try:
        for group, records in by_group.items():
            final = output_dir / f"{group}.joblib"
            tmp = final.with_name(final.name + ".tmp")
            joblib.dump(records, tmp)

            if tmp.stat().st_size == 0:
                raise IOError(f"Empty joblib for group '{group}'")
            reloaded = joblib.load(tmp)           # read back to check it
            if len(reloaded) != len(records):
                raise IOError(f"Group '{group}': saved {len(records)}, "
                              f"reloaded {len(reloaded)}")
            tmp_files[tmp] = final

        # 4. phase 2: everything is valid, swap the files in
        for tmp, final in tmp_files.items():
            os.replace(tmp, final)
        tmp_files.clear()
    finally:
        for tmp in tmp_files:                     # only leftovers if something failed
            tmp.unlink(missing_ok=True)

    for group, records in by_group.items():
        print(f"  {group}: {len(records)}")
    print(f"Total saved: {sum(len(r) for r in by_group.values())}")