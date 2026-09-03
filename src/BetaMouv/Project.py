# src/BetaMouv/Project.py

from src.Base.Project import Project as BaseProject

from .Trial import Trial
from .Prediction import Prediction
from .Validation import Validation


class Project(BaseProject):

    def __init__(self, name, config_dir):
        super().__init__(name, config_dir)

        self.trial_class = Trial
        self.prediction_class = Prediction
        self.validation_class = Validation