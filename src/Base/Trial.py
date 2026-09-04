# src/Base/Trial.py


from .File import File
from pathlib import Path
import yaml

class Trial:

    def __init__(self, clip_path: Path):

        self.file = File(clip_path)


    def to_dict(self) -> dict:
        return {"name": self.file.name,
                "clip_path": str(self.file.path),}


    def save_trial(self):
        yaml_path =  self.file.path.parent / f"{self.file.name}.yaml"

        with open((yaml_path) , "w") as f : 
            yaml.safe_dump(self.to_dict(), f)

        # TODO
        # verifier si le fichier existe deja, 
        # si cest deja le cas, demander veification ????
        # handle existance



if __name__ == "__main__": 

    trial1 = Trial(clip_path=Path("/home/poemiti/IMN_Kinematics/data/BetaMouv/Rat_#531LeftHanded_20240526_ContiMT300_RightHemiCHR_L1-60_L2-40_C001H001S0003_clip_00.mp4"),)

    print(trial1)
    trial1.save_trial()
    