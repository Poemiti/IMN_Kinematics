# src/BetaMouv/TrialGroup.py


from src.Base.TrialGroup import TrialGroup as BaseTrialGroup
from .Trial import Trial
import pandas as pd

class TrialGroup(BaseTrialGroup): 

    trial_cls = Trial

    def __init__(self, filenames, condition):
        super().__init__(filenames, condition)

        self.success_df = None
        self.scalar_df = None
        self.timeseries_df = None


    def success_df(self) -> pd.DataFrame:
        """One row per trial, including failures — for QC / yield figures
        (e.g. success rate by condition, why trials were excluded)."""

        if self.success_df is None:
            records = []

            for trial in self.trials:
                records.append(trial.identity() | {
                    "task_success": getattr(trial, "task_success", None),
                    "task_success_reason": getattr(trial, "task_success_reason", None),
                    "model_success": getattr(trial, "model_success", None),
                    "model_success_reason": getattr(trial, "model_success_reason", None),
                })

            self.success_df = pd.DataFrame(records)

        return self.success_df


    def scalar_df(self, metrics: list[str] = None) -> pd.DataFrame:
        """One row per trial — for distributions/scatter (avg velocity, tortuosity, ...)."""

        if self.scalar_df is None:
            records = []

            for trial in self.trials:
                if not trial._is_successful() :
                    continue

                records.append(trial.identity() | {
                    f: getattr(trial, f, None) for f in (metrics or trial.SCALAR_METRIC_FIELDS)
                })

            self.scalar_df = pd.DataFrame(records)

        return self.scalar_df



    def timeseries_df(self, value_cols: list[str] = ("x", "y", "velocity")) -> pd.DataFrame:
        """Many rows per trial (one per frame/timestamp) — for plots over time."""

        if self.timeseries_df is None:
            frames = []

            for trial in self.trials:
                if not trial._is_successful() :
                    continue

                df = trial.coords[["t", *value_cols]].copy()
                for col, val in trial.identity().items():
                    df[col] = val
                frames.append(df)

            self.timeseries_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        return self.timeseries_df
    
    

    
    ####################### plotting methods ###########################


    def plot_instant_velocity(self): 

        for t in self.trials: 

            print(t.traj)