# src/Base/Trial.py


from pathlib import Path
import yaml
import pandas as pd
from datetime import date, datetime


class Trial:
    FIELDS: tuple = ()          # everything — goes into joblib as-is
    YAML_FIELDS: tuple = ()     # subset that gets written to the per-trial yaml

    def __init__(self, clip_path: str, yaml_path: str = None):
        self.clip_path = clip_path
        self.yaml_path = Path(yaml_path) if yaml_path else Path(clip_path).with_suffix(".yaml")

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.FIELDS:
                raise AttributeError(f"'{key}' is not a declared field of {type(self).__name__}")
            setattr(self, key, value)
        return self

    def to_dict(self) -> dict:
        """Native objects, untouched — used for joblib (object identity preserved)."""
        return {f: getattr(self, f, None) for f in self.FIELDS}

    def from_dict(self, data: dict):
        for f in self.FIELDS:
            if f in data:
                setattr(self, f, data[f])
        return self

    @staticmethod
    def _serialize_value(value):
        """Convert one value into something yaml.safe_dump can handle."""
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "to_dict"):          # e.g. a Trajectory-like object
            return value.to_dict()
        return value

    def to_yaml_dict(self) -> dict:
        fields = self.YAML_FIELDS or self.FIELDS
        return {f: self._serialize_value(getattr(self, f, None)) for f in fields}

    def save_yaml(self, path=None):
        path = path or self.yaml_path
        with open(path, "w") as f:
            yaml.safe_dump(self.to_yaml_dict(), f, sort_keys=False)

    def load_yaml(self, path=None):
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
    