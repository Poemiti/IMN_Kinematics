# src/BetaMouv/TrialGroup.py


from src.Base.TrialGroup import TrialGroup as BaseTrialGroup
from .Leds import Leds
import yaml

from .File import File

class TrialGroup(BaseTrialGroup): 

    def __init__(self, filenames, condition):
        super().__init__(filenames, condition)
