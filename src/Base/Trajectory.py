# src/Base/Trajectory.py

import pandas as pd
from pathlib import Path
import numpy as np

class Trajectory: 

    def __init__(self,
                coords_path: Path,
                view: str,
                bodypart: str = "finger_3",
                fps: int = 125,
                frame_width: float = 512,
                cm_per_pixel: float | None = None,
                lever_position: float | None = (55, 230)): 

        self.coords_path = coords_path
        self.view = view
        self.bodypart = bodypart
        self.fps = fps
        self.dt = 1 / fps
        self.cm_per_pixel = cm_per_pixel
        self.lever_position = lever_position
        self.frame_width = frame_width  # in pixel

        # setup coordinates
        self.raw_coords = self._open_DLC_results(self.coords_path)
        self.raw_coords = self.raw_coords[self.bodypart].copy()
        self.raw_coords = self.raw_coords.assign(t=np.arange(len(self.raw_coords)) / self.fps)
        self.raw_coords = self._cartesian_xy()

    def _scale(self, values):
        if self.cm_per_pixel is None:
            return values
        return values * self.cm_per_pixel


    def _cartesian_xy(self) : 
        """change to a correct cartesian axis (bottom left corner)"""
        if self.view == "left" :  # invert y axis
            self.coords["y"] = self.frame_width - self.coords["y"]
            self.coords["x"]  = self.coords["x"]
        else :              # invert both x and y axis
            self.coords["y"]  = self.frame_width - self.coords["y"]
            self.coords["x"]  = self.frame_width - self.coords["x"]

        return self.coords


    def _open_DLC_results(self) -> pd.DataFrame : 
        """
        Load and clean a DeepLabCut CSV file.
        """

        # DLC CSV has 3 header rows (scorer, bodyparts, coords)
        df = pd.read_csv(self.coords_path, header=[0, 1, 2])

        # clean dataframe
        df.columns = df.columns.droplevel(0)  # remove scorer row
        clean_df = df.iloc[1:].reset_index(drop=True)

        return clean_df