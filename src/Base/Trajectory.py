# src/Base/Trajectory.py

from .File import File as BaseFile
import pandas as pd
from pathlib import Path
import numpy as np
from .Outcome import Outcome

import matplotlib.pyplot as plt


class Trajectory: 
    """Method only class to compute stuff on coordinates (pd.Dataframes) containing: x, y, t"""

    def __init__(self, coords: pd.DataFrame | None = None):
        self.coords = coords   
        self.dt = coords["t"].diff()


    # ------------- plotting function ------------    

    def plot_traj(self, coords: pd.DataFrame, ax: plt.axes, fig, 
                  color: str = "k", show_time: bool= False) -> plt.axes: 
        
        ax.plot(coords["x"], coords["y"], color=color)
        sc = ax.scatter(coords["x"], coords["y"], 
                        c=coords["t"] if show_time else color, 
                        cmap="viridis" if show_time else None)
        
        if show_time: 
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("t")

        return ax


    def show_traj(self, coords: pd.DataFrame, show: bool=False):
        if coords is None:
            coords = self.coords.copy()

        fig, ax = plt.subplots(figsize=(8, 6))

        ax = self.plot_traj(coords, ax, fig)

        name = self.file.name.replace("pred_results_", "")
        title = name[:len(name)//2] + "\n" + name[len(name)//2:]

        ax.set(
            # xlim=(0, self.frame_height * self.cm_per_pixel),
            # ylim=(0, self.frame_height * self.cm_per_pixel),
            title=title,
            xlabel="X position (cm)",
            ylabel="Y position (cm)",
        )

        if show: 
            plt.show()
        plt.close()

        return fig


    ############## Compute some metrics ###################

    def crop_xy(self, coords: pd.DataFrame = None, start: float = 0, end: float = 0.4) -> pd.DataFrame :  
        """Crop coordinates from [start : end]"""
        if coords is None:
            coords = self.coords

        return coords.loc[
            (coords["t"] >= start) &
            (coords["t"] <= end)
        ].reset_index(drop=True)
    

    def distances(self, coords: pd.DataFrame | None = None) -> pd.DataFrame:
        """Instantaneous distances between each points"""
        if coords is None:
            coords = self.coords

        v = coords[["x", "y"]].diff()

        return np.hypot(v["x"], v["y"])


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

    def instant_speed(self, coords=None):
        """Scalar, >= 0, no direction."""
        v = self.velocity_vector(coords)

        return np.hypot(v["vx"], v["vy"])


    def instant_velocity(self, coords=None, direction=(-1, 1)) -> pd.Series:
        """Velocity projected on `direction`.
        > 0 : moving toward `direction` (default: upper-left)
        < 0 : moving away from it
        """

        u = np.asarray(direction, float)
        u /= np.linalg.norm(u)
        v = self.velocity_vector(coords)

        cond = (v["vx"] < 0) & (v["vy"] > 0)          # upper-left, as in your docstring
        sign = np.where(cond, 1, -1)
        return sign * np.hypot(v["vx"], v["vy"])
        
        # return v["vx"] * u[0] + v["vy"] * u[1]


    def radial_velocity(self, coords=None): 
        
        xl, yl = self._point_to_scaled_cartesian(self.XL, self.YL)

        disp = (xl, yl) - coords[["x", "y"]]
        u = disp.div(np.linalg.norm(disp, axis=1), axis=0)      # unit vector lever -> point
        v = self.velocity_vector(coords)

        return v["vx"] * u["x"] + v["vy"] * u["y"]        



    def acceleration(self, coords: pd.DataFrame | None = None) -> np.array :
        v = self.velocity_vector(coords)[["vx", "vy"]]
        a = v.diff() / self.dt
        return (np.sqrt(a["vx"]**2 + a["vy"]**2))


    def signed_acceleration(self, coords=None, direction=(-1, 1)):
        return self.instant_velocity(coords, direction).diff() / self.dt


    def obj_bodypart_distance(self, bodypart_coords: pd.DataFrame | None = None, obj_coord: list = [0,0]) -> np.array : 
        """Compute the straight/net distance between an object (obj_coords) and the bodypart choosen"""
        if bodypart_coords is None:
            bodypart_coords = self.coords

        return np.linalg.norm(bodypart_coords[["x", "y"]] - obj_coord, axis=1)


    def angle(self, coords=None):
        """Compute angle over time"""
        if coords is None: 
            coords = self.coords

        dx = coords["x"].diff()
        dy = coords["y"].diff()

        return np.degrees(np.arctan2(dy, -dx))





class TrajectoryDLC(Trajectory):

    """Allow to open the DLC output and have more information
    related to the trial"""

    file_cls = BaseFile
    STAGES = ()

    def __init__(self,
                coords_path: Path,
                view: str,
                bodypart: str = "finger_3",
                cm_per_pixel: float | None = None,
                time_pad_off: float = None,
                shift: tuple[float] | None = (0,0), 
                fps: float = 125, 
                frame_height_px: int = 512, 
                frame_width_px: int = 512): 

        self.file = self.file_cls(coords_path)
        self.time_pad_off = time_pad_off
        self.coords_path = coords_path
        self.view = view
        self.bodypart = bodypart
        self.cm_per_pixel = cm_per_pixel
        self.shift = shift

        # constants
        self.fps = fps
        self.frame_height_px = frame_height_px
        self.frame_width_px = frame_width_px

        # traj validity
        self.stage_outcomes = {name: Outcome(stage=name, order=i) for i, name in enumerate(self.STAGES)}

        # raw pixel coordinates, untouched — kept for debugging / overlaying on the source video
        self.raw_coords = self._open_DLC_results()
        self.raw_coords = self.raw_coords[self.bodypart].copy()
        self.raw_coords = self.raw_coords.assign(t=np.arange(len(self.raw_coords)) / self.fps)  # add time column

        # apply camera shift BEFORE SCALING (because shift is in pixel)
        self.raw_coords[["x", "y"]] = self.raw_coords[["x", "y"]] + self.shift

        # setup coordinates into cartesian plane (bottom-left origin) + cm units
        self.raw_coords = self._array_to_scaled_cartesian(self.raw_coords)

        super().__init__(self.raw_coords)


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
        """Flip pixel axes (top-left origin) to a bottom-left-origin cartesian plane
        Coordinates must be in pixels"""
        y = self.frame_height_px - y
        if self.view != "left":   # non-left views are also mirrored horizontally
            x = self.frame_width_px - x
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