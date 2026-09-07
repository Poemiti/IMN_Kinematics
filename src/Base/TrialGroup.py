# src/Base/TrialGroup.py

from .Trial import Trial

class TrialGroup: 
    """Manager to filter trials to use by their group name"""

    def __init__(self, condition):

        trials: list[Trial] = ...
        