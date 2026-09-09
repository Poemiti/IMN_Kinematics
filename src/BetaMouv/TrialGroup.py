# src/BetaMouv/TrialGroup.py


from src.Base.TrialGroup import TrialGroup as BaseTrialGroup
from .Trial import Trial

class TrialGroup(BaseTrialGroup): 

    def __init__(self, filenames, condition):
        super().__init__(filenames, condition)
