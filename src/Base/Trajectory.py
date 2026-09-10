# src/Base/Trajectory.py

from .File import File as BaseFile
import pandas as pd
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("talk", style="ticks", rc=custom_params, palette="pastel")


class Trajectory: 

    file_cls = BaseFile

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

        # raw pixel coordinates, untouched — kept for debugging / overlaying on the source video
        self.raw_coords = self._open_DLC_results()
        self.raw_coords = self.raw_coords[self.bodypart].copy()
        self.raw_coords = self.raw_coords.assign(t=np.arange(len(self.raw_coords)) / self.fps)  # add time column

        # setup coordinates into cartesian plane (bottom-left origin) + cm units
        self.coords = self._array_to_scaled_cartesian(self.raw_coords)
        self.lever_position = self._point_to_scaled_cartesian(*lever_position) if lever_position else None



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

        sns.lineplot(
            data=coords,
            x="x", y="y",
            ax=ax,
            linewidth=0.5, color="gray",
            sort=False
        )

        # sns.scatterplot(
        #     data=coords,
        #     x="x", y="y", 
        #     ax=ax,
        #     markers="|",
        #     linewidth=0,
        #     hue="instant_velocity"
        # )

        ax.set(
            xlim=(0, self.frame_height * self.cm_per_pixel),
            ylim=(0, self.frame_height * self.cm_per_pixel),
            title=self.file.name,
            xlabel="X position (cm)",
            ylabel="Y position (cm)",
        )

        plt.show()
        plt.close()




    ############## Compute some metrics ###################



    def compute_instant_metrics(self, coords: pd.DataFrame | None = None) -> pd.DataFrame:
        if coords is None:
            coords = self.coords

        coords["instant_velocity"] = self.instant_velocity(coords)
        coords["instant_acc"] = self.acceleration(coords)
        coords["lever_distance"] = self.lever_bodypart_distance(coords)

        return coords



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