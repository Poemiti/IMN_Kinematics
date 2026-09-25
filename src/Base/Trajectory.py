# src/Base/Trajectory.py

from .File import File as BaseFile
import pandas as pd
from pathlib import Path
import numpy as np
from .Outcome import Outcome

import matplotlib.pyplot as plt



class Trajectory: 
    """A trajectory is defined in a classical cartesian plane, in cm"""

    file_cls = BaseFile
    STAGES = ()

    def __init__(self,
                coords_path: Path,
                view: str,
                bodypart: str = "finger_3",
                fps: int = 125,
                frame_width: int = 512,     # px
                frame_height: int = 512,    #px
                cm_per_pixel: float | None = None,
                lever_position: float | None = None): 

        self.file = self.file_cls(coords_path)

        self.coords_path = coords_path
        self.view = view
        self.bodypart = bodypart
        self.fps = fps
        self.dt = 1 / fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cm_per_pixel = cm_per_pixel
        self.stage_outcomes = {
            name: Outcome(stage=name, order=i) for i, name in enumerate(self.STAGES)
        }

        # raw pixel coordinates, untouched — kept for debugging / overlaying on the source video
        self.raw_coords = self._open_DLC_results()
        self.raw_coords = self.raw_coords[self.bodypart].copy()
        self.raw_coords = self.raw_coords.assign(t=np.arange(len(self.raw_coords)) / self.fps)  # add time column

        # setup coordinates into cartesian plane (bottom-left origin) + cm units
        self.coords = self._array_to_scaled_cartesian(self.raw_coords)
        self.lever_position = self._point_to_scaled_cartesian(*lever_position) if lever_position else None

    ############## Success function #############

    def set_success(self, stage: str, success: bool, reason: str):
        outcome = self.stage_outcomes.get(stage)
        if outcome is None:
            raise ValueError(f"Unknown stage '{stage}' for bodypart '{self.bodypart}'")
        outcome.success = success
        outcome.reason = reason

    def is_valid(self, upto: str = None) -> bool:
        limit = self.stage_outcomes[upto].order if upto else float("inf")
        return all(o.success for o in self.stage_outcomes.values() if o.order <= limit)

    def failure(self) -> Outcome | None:
        for outcome in sorted(self.stage_outcomes.values()):
            if not outcome.success:
                return outcome
        return None


    ################ saving function ###############

    def to_dict(self) -> dict:
        return {
            "bodypart": self.bodypart,
            "coords": self.coords,
            "raw_coords": self.raw_coords,
            "stage_outcomes": {k: o.to_dict() for k, o in self.stage_outcomes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        obj = cls.__new__(cls)   # bypass __init__ (no coords_path to re-read from disk)
        obj.bodypart = data["bodypart"]
        obj.coords = data.get("coords")
        obj.raw_coords = data.get("raw_coords")
        obj.stage_outcomes = {k: Outcome.from_dict(v) for k, v in data["stage_outcomes"].items()}
        return obj



    ################ preprocess function ################


    def _to_cartesian(self, x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Flip pixel axes (top-left origin) to a bottom-left-origin cartesian plane."""
        y = self.frame_height - y
        if self.view != "left":   # non-left views are also mirrored horizontally
            x = self.frame_width - x
        return x, y


    def _scale(self, x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Convert pixel distances to cm, if a calibration factor is available."""
        if self.cm_per_pixel is None:
            return x, y
        return x * self.cm_per_pixel, y * self.cm_per_pixel


    def _array_to_scaled_cartesian(self, coords: pd.DataFrame) -> pd.DataFrame:
        """Full transformation : pixel/top-left -> cartesian/cm, bottom-left origin."""
        coords = coords.copy()
        x, y = self._to_cartesian(coords["x"], coords["y"])
        x, y = self._scale(x, y)
        coords["x"], coords["y"] = x, y
        
        return coords


    def _point_to_scaled_cartesian(self, x: float, y: float) -> tuple[float, float]:
        """Apply the same flip+scale transform to a single (x, y) point, e.g. lever_position."""
        x_series = pd.Series([x])
        y_series = pd.Series([y])
        x_t, y_t = self._to_cartesian(x_series, y_series)
        x_t, y_t = self._scale(x_t, y_t)

        return float(x_t.iloc[0]), float(y_t.iloc[0])


    def _open_DLC_results(self) -> pd.DataFrame : 
        """
        Load and clean a DeepLabCut CSV file.
        """

        # DLC CSV has 3 header rows (scorer, bodyparts, coords)
        df = pd.read_csv(self.coords_path, header=[0, 1, 2])
        df.columns = df.columns.droplevel(0)  # remove scorer row
        clean_df = df.iloc[1:].reset_index(drop=True)

        return clean_df


    def show_traj(self, coords: pd.DataFrame):
        if coords is None:
            coords = self.coords

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(coords["x"], coords["y"], color="lightblue")
        ax.scatter(coords["x"], coords["y"], marker="|", )

        # sns.scatterplot(
        #     data=coords,
        #     x="x", y="y", 
        #     ax=ax,
        #     markers="|",
        #     linewidth=0,
        #     hue="instant_velocity"
        # )
        name = self.file.name.replace("pred_results_", "")
        title = name[:len(name)//2] + "\n" + name[len(name)//2:]

        ax.set(
            # xlim=(0, self.frame_height * self.cm_per_pixel),
            # ylim=(0, self.frame_height * self.cm_per_pixel),
            title=title,
            xlabel="X position (cm)",
            ylabel="Y position (cm)",
        )

        plt.show()
        plt.close()




    ############## Compute some metrics ###################

    def crop_xy(self, coords: pd.DataFrame = None, start: float = 0, end: float = 0.4) -> pd.DataFrame :  
        """Crop coordinates from [start : end]"""
        if coords is None:
            coords = self.coords

        return coords.loc[
            (coords["t"] >= start) &
            (coords["t"] <= end)
        ].reset_index(drop=True)

    def compute_instant_metrics(self, coords: pd.DataFrame | None = None) -> pd.DataFrame:
        if coords is None:
            coords = self.coords

        coords["instant_velocity"] = self.instant_velocity(coords)
        coords["instant_acc"] = self.acceleration(coords)
        coords["lever_distance"] = self.lever_bodypart_distance(coords)
        coords["distances"] = self.distances(coords)

        return coords


    def distances(self, coords: pd.DataFrame | None = None) -> pd.DataFrame:
        """Instantaneous distances between each points"""
        if coords is None:
            coords = self.coords

        v = coords[["x", "y"]].diff()

        return np.sqrt(v["x"]**2 + v["y"]**2)


    def velocity_vector(self, coords: pd.DataFrame | None = None) -> pd.DataFrame:
        """Instantaneous velocity components."""
        if coords is None:
            coords = self.coords

        v = coords[["x", "y"]].diff() / self.dt

        return pd.DataFrame({
            "t": coords["t"],
            "vx": v["x"],
            "vy": v["y"]
        })

    def instant_velocity(self, coords: pd.DataFrame | None = None) -> np.array  :
        """Instantaneous velocity"""
        if coords is None:
            coords = self.coords

        v = self.velocity_vector(coords)

        return np.sqrt(v["vx"]**2 + v["vy"]**2)


    def acceleration(self, coords: pd.DataFrame | None = None) -> np.array :
        """ Instantaneous acceleration magnitude """
        if coords is None:
            coords = self.coords

        v = self.velocity_vector(coords)[["vx", "vy"]]
        a = v.diff() / self.dt
        
        return (np.sqrt(a["vx"]**2 + a["vy"]**2))


    def lever_bodypart_distance(self, 
                                coords: pd.DataFrame | None = None) -> np.array : 
        """Compute the straight/net distance between the lever and the bodypart choosen"""
        if coords is None:
            coords = self.coords

        xy = coords[["x", "y"]]
        disp = xy - self.lever_position

        return np.linalg.norm(disp, axis=1)