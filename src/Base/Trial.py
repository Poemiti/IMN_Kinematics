# src/Base/Trial.py


from .File import File
from pathlib import Path
import yaml
import pandas as pd

class Trial:

    def __init__(self, clip_path: Path):

        self.file = File(clip_path)
        self.group = "Not_defined"


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


    def dlc_predict(self, model_path: Path, 
                    output_csv_path : Path = None) -> pd.DataFrame:
        import tempfile
        import deeplabcut
        from deeplabcut.pose_estimation_pytorch import set_load_weights_only

        set_load_weights_only(False)

        with tempfile.TemporaryDirectory() as dlc_dest:
            # print(dlc_dest)
            deeplabcut.analyze_videos(
                f'{model_path}/config.yaml',
                [str(self.file.path)],
                save_as_csv=False,
                # gputouse=0,
                destfolder=dlc_dest
            )

            h5_file = next(Path(dlc_dest).glob("*.h5"), None)
            df = pd.read_hdf(h5_file)

        df.index.name="frame_num"

        if output_csv_path : 
            df.to_csv(output_csv_path)

        return df



if __name__ == "__main__": 

    trial1 = Trial(clip_path=Path("/home/poemiti/IMN_Kinematics/data/BetaMouv/Rat_#531LeftHanded_20240526_ContiMT300_RightHemiCHR_L1-60_L2-40_C001H001S0003_clip_00.mp4"),)

    print(trial1)
    trial1.save_trial()
    