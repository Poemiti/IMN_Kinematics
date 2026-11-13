# src/DopaMouv/TrialGroup.py


from src.Base.TrialGroup import TrialGroup as BaseTrialGroup
from src.BetaMouv.Statistics import Statistics

from .Trial import Trial
from .Trajectory import TrajectoryDLC, DLCTracking
from .Behavior import Behavior

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
from pathlib import Path
from datetime import datetime


class TrialGroup(BaseTrialGroup): 

    trial_cls = Trial

    def __init__(self, filenames, condition):
        super().__init__(filenames, condition)

        self.crop_laser_period = False

    def build_df(self, output_dir: Path, kind: str) -> pd.DataFrame:
        """
        Build a dataframe of all valid trials, or load it if already saved.
            - trial_success
            - bodypart_success
            - behavior
            - kinematics
            - scalar
        """
        # Map each kind to the method (NO parentheses -> nothing is executed here)
        builders = {
            "trial_success": self.trial_success_df,
            "bodypart_success": self.bodypart_success_df,
            "behavior": self.behavior_df,
            "kinematics": self.kinematics_df,
            "scalar": self.scalar_df,
        }

        if kind not in builders:
            raise ValueError(f"Unknown kind '{kind}'. Choose from {list(builders)}")

        output_dir = Path(output_dir)
        output_path = output_dir / f"{kind}.csv"

        # If it exists, load it
        if output_path.exists():
            print(f"\n'{kind}' already exist: '{output_path}'")
            print("Loading...")

            if kind in ["kinematics", "scalar"]: 
                df = pd.read_csv(output_path, header=[0, 1])
            else : 
                df = pd.read_csv(output_path)

            setattr(self, f"_{kind}_df", df)
            return df

        # or compute it
        print(f"\nComputing '{kind}'")
        self.load_trials()
        df = builders[kind]()
        df.to_csv(output_path, index=False)
        setattr(self, f"_{kind}_df", df)

        # Backup
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = output_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        backup = backup_dir / f"{kind}_{stamp}.csv"
        df.to_csv(backup, index=False)

        print(f"\nSaved '{kind}': {output_path}\nbackup: {backup}")
        return df


    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame: 
        """Check 'remove_NOstim' flag and 'crop_laser_period' flag
        remove_NOstim : remove trials with the NOstim condition
        crop_laser_period : crop trial timeseries around the laser period 
        (0.1 sec before pad off --> 0.1 after end of the opto stim)"""

        has_2_levels = df.columns.nlevels == 2

        if self.crop_laser_period :
            if has_2_levels :
                if ("trial", "t") in df.columns :
                    t = df["trial"]["t"]
                    pad_off = df["trial"]["time_pad_off"]
                else:
                    return df
                
            else: 
                t = df["t"]
                pad_off = df["time_pad_off"]

            df = df[
                (t >= pad_off - 0.1) &
                (t <= pad_off + 0.4)
            ]
            
        return df


    def behavior_df(): 
        pass

    def kinematics_df(): 
        pass

    def scalar_df(): 
        pass



    #################### plotting ##########################



    def trial_success_rate(self, save_as, show: bool = False): 
        return self._countplot_success_rate(self.trial_success_df(), None, save_as, show)

    def trajectories_success_rate(self, save_as, show: bool = False): 
            return self._countplot_success_rate(self.bodypart_success_df(), "bodypart", save_as, show)
