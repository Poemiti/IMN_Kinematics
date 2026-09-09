# src/Base/Trial.py


from .File import File
from pathlib import Path
import yaml
import pandas as pd

class Trial:
    # Subclasses extend this list with their own fields.
    FIELDS: tuple = ()

    def __init__(self, clip_path: str, yaml_path: str = None):
        self.clip_path = clip_path
        self.yaml_path = Path(yaml_path) if yaml_path else Path(clip_path).with_suffix(".yaml")


    def update(self, **kwargs):
        """Generic setter for any declared field """

        for key, value in kwargs.items():
            if key not in self.FIELDS:
                raise AttributeError(
                    f"'{key}' is not a declared field of {type(self).__name__} "
                    f"(check FIELDS or a typo)"
                )
            setattr(self, key, value)
        return self  # allows chaining: trial.update(...).update(...)


    def to_dict(self) -> dict:
        return {field: getattr(self, field, None) for field in self.FIELDS}


    def from_dict(self, data: dict):
        for field in self.FIELDS:
            if field in data:
                setattr(self, field, data[field])
        return self


    def save_yaml(self, path: str = None):
        path = path or self.yaml_path
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


    def load_yaml(self, path: str = None):
        path = path or self.yaml_path
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return self.from_dict(data)


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
                [str(self.clip_path)],
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
    