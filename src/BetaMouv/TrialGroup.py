# src/BetaMouv/TrialGroup.py


from src.Base.TrialGroup import TrialGroup as BaseTrialGroup
from .Trial import Trial

class TrialGroup(BaseTrialGroup): 

    trial_cls = Trial

    def __init__(self, filenames, condition):
        super().__init__(filenames, condition)



    def filter_successful(self): 

        # TODO
        # mettre a jour la list des essais selon sil sont successful ou non

        raise NotImplementedError("filter_successful")
        