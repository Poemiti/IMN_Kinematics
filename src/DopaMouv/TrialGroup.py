# src/DopaMouv/TrialGroup.py


from src.Base.TrialGroup import TrialGroup as BaseTrialGroup
from src.BetaMouv.Statistics import Statistics

from .Trial import Trial
from .Trajectory import TrajectoryDLC, DLCTracking
from .Behavior import Behavior
from statannotations.Annotator import Annotator

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
from pathlib import Path
from datetime import datetime


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("talk", style="ticks", rc=custom_params, palette="pastel")

BOOL_PALETTE = {True: "yellowgreen", False: "tomato"}


LASER_STATE_PALETTE = {
    "NOstim": "slategray",
    "LaserOff" :"slategray",
    "LaserOn" : "tomato"
}

LASER_STATE_DASH = {
    "LaserOff" : (4,2),
    "LaserOn" : ""
}

LASER_STATE_HATCH = {          
    "LaserOff": "oo",
    "LaserOn": "",
}

LASER_PERIOD_COLOR = "royalblue"

BEHAVIOR_PALETTE = {
    "reach": "yellow", 
    "open": "orange", 
    "grasp": "blue", 
    "press": "green",
    "none": "gray"
}

LASER_INTENSITY_PALETTE = {"low" : "lightblue",
                            "high" : "salmon",
                            "NOstim" : "gray"}

LASER_INTENSITY_PALETTE_DARK = {"low" : "steelblue",
                            "high" : "tomato",
                            "NOstim" : "black"}




class TrialGroup(BaseTrialGroup): 

    trial_cls = Trial

    def __init__(self, filenames, condition):
        super().__init__(filenames, condition)

        self._trial_success_df = None
        self._scalar_df = None
        self._kinematics_df = None
        self._bodypart_success_df = None
        self._behavior_df = None

        self.crop_laser_period = False
