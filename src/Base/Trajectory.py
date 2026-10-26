# src/Base/Trajectory.py

from .File import File as BaseFile
import pandas as pd
from pathlib import Path
import numpy as np
from .Outcome import Outcome
from dataclasses import dataclass, field

import matplotlib.pyplot as plt


class Trajectory: 
    """Method only class to compute stuff on coordinates (pd.Dataframes) containing: x, y, t"""

    def __init__(self, coords: pd.DataFrame | None = None):
        self.coords = coords 
        self.dt = coords["t"].iloc[1] 
        self.duration_sec = len(coords) * self.dt
        self.duration_frame = len(coords)


    # ------------- plotting function ------------    

    def plot_traj(self, coords: pd.DataFrame, ax: plt.axes, fig, 
                  color: str = "k", show_time: bool= False, label: str = None) -> plt.axes: 
        
        ax.plot(coords["x"], coords["y"], color=color)
        sc = ax.scatter(coords["x"], coords["y"], 
                        c=coords["t"] if show_time else color, 
                        cmap="viridis" if show_time else None, 
                        label=label)
         
        if show_time: 
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("t")

        return ax


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


    def instant_velocity(self, coords=None,) -> pd.Series:
        """Velocity is a vector composed of a direction (sign) and a norm (distance)
        Return signed speed
        """
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


    def signed_acceleration(self, coords=None):
        return self.instant_velocity(coords).diff() / self.dt


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


    ############## Compute averaging metrics ###################


    def net_displacement(self, coords: pd.DataFrame | None = None) -> float:
        """Straight-line distance from start to end."""
        if coords is None:
            coords = self.coords

        start = coords[["x", "y"]].iloc[0]
        end = coords[["x", "y"]].iloc[-1]

        return np.linalg.norm(end - start)


    def path_length(self, coords: pd.DataFrame | None = None) -> float:
        """Total traveled distance (path length). Sum of distance between each points"""
        if coords is None:
            coords = self.coords

        dist = self.distances(coords)
        return dist.sum()

    

    def _get_lever_reaching_time(self, coords: pd.DataFrame, lever_pos: tuple[int, int]) -> float: 
        """Return the time at which the coordinates where as close 
        a possible to the lever position"""

        lever_x, lever_y = lever_pos

        # measure distance between coords and lever
        distances = np.sqrt(
            (coords["x"] - lever_x) ** 2 +
            (coords["y"] - lever_y) ** 2
        )

        # return the time at which the distance was the smallest 
        return coords.loc[distances.idxmin(), "t"]
    

    def tortuosity_from_obj(self, coords, obj_coords, time_pad_off):
        """Path length to get to the lever / straight-line distance from pad off to lever.""" 
        # crop at pad off to skip beginning
        pad_off_coords = coords.loc[coords["t"] >= time_pad_off].reset_index(drop=True)

        # crop coordinates from the pad off to the lever reach
        time_reach_lever =  self._get_lever_reaching_time(pad_off_coords, obj_coords)
        cropped_coords = pad_off_coords.loc[pad_off_coords["t"] <= time_reach_lever].reset_index(drop=True)
        
        direct = self.net_displacement(cropped_coords)
        path = self.path_length(cropped_coords)

        # Avoid division by zero
        if direct == 0:
            return 0

        return path / direct
    

    def tortuosity_fixed_period(self, coords: pd.DataFrame | None = None, 
                   lever_pos: tuple[int, int] = [0, 0],
                   start = None, end = None) -> float:
        """Torutosity for the fixed time period. 
        Path lenght during the time period / straight line distance from pad off to lever pos"""
        if coords is None:
            coords = self.coords

        cropped_coords = self.crop_xy(coords, start, end)

        # calculated straigth distance between pad off and lever position
        dist = lever_pos - cropped_coords[["x", "y"]].iloc[0]
        direct = np.linalg.norm(dist)

        return self.path_length(cropped_coords) / direct


    def duration(self, coords: pd.DataFrame | None = None) -> float:
        """Return duration in sec of the dataframe coords"""
        if coords is None:
            coords = self.coords
        return coords["t"].iloc[-1] - coords["t"].iloc[0]


    def avg_velocity(self, coords, start, end): 
        """Path length to get to the lever / straight-line distance from pad off to lever."""
        if coords is None:
            coords = self.coords

        cropped_coords = self.crop_xy(coords, start, end)

        return self.path_length(cropped_coords) / self.duration(cropped_coords)


        






class TrajectoryDLC(Trajectory):

    """Allow to open the DLC output and have more information
    related to the trial"""

    file_cls = BaseFile
    STAGES = ()

    def __init__(self, coords: pd.DataFrame,
                view: str,
                bodypart: str,
                cm_per_pixel: float,
                time_pad_off: float,
                shift: tuple[float], 
                fps: float = 125, 
                frame_height_px: int = 512, 
                frame_width_px: int = 512): 

        self.time_pad_off = time_pad_off
        self.view = view
        self.bodypart = bodypart
        self.cm_per_pixel = cm_per_pixel
        self.shift = shift

        # constants
        self.fps = fps
        self.frame_height_px = frame_height_px
        self.frame_width_px = frame_width_px

        super().__init__(coords)

    def show_traj(self, coords: pd.DataFrame, show: bool=False, crop_coords: bool=True):
        if coords is None:
            coords = self.coords.copy()

        if crop_coords: 
            coords = self.crop_xy(coords, self.time_pad_off - 0.1, self.time_pad_off + 0.4)

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


    def _array_to_scaled_cartesian(self, coords: pd.DataFrame = None) -> pd.DataFrame:
        """Full transformation : pixel/top-left -> cartesian/cm, bottom-left origin."""
        if coords is None: 
            coords = self.coords

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


        
     
     # ------------------------ preprocessing methods --------------------
   


@dataclass
class BodypartStatus:
    STAGES = ("outlier", "validation")
    bodypart: str
    state: str
    duration_frame: int | None = None
    duration_sec: float | None = None
    stage_outcomes: dict = field(default_factory=lambda: {
                                name: Outcome(stage=name, order=i)
                                for i, name in enumerate(BodypartStatus.STAGES)})

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
        return "successful"


    def to_dict(self) -> dict:
        return {
            "bodypart": self.bodypart,
            "stage_outcomes": {k: o.to_dict() for k, o in self.stage_outcomes.items()},
            "state": self.state,
            "duration_frame": self.duration_frame,
            "duration_sec": self.duration_sec,
            "is_valid": self.is_valid(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BodypartStatus":
        return cls(bodypart=data["bodypart"],
                    state=data["state"],
                    duration_frame=data.get("duration_frame"),
                    duration_sec=data.get("duration_sec"),
                    stage_outcomes={k: Outcome.from_dict(v)
                                    for k, v in data.get("stage_outcomes", {}).items()},)




class DLCTracking:
    """Defines a whole tracking and methods to save or load DLC output"""

    file_cls = BaseFile
    trajectory_cls = TrajectoryDLC

    def __init__(self, 
                 path: Path,
                 view: str, 
                 bodyparts: list[str], 
                 cm_per_pixel: float, 
                 time_pad_off: float, 
                 shift: tuple[float, float], 
                 status: dict[str, dict] | None = None,
                 fps=125, frame_h=512, frame_w=512):
        
        self.file = self.file_cls(path)
        self.time_pad_off = time_pad_off
        self.path = path
        self.view = view
        self.cm_per_pixel = cm_per_pixel
        self.shift = shift

        # constants
        self.bodyparts = bodyparts
        self.fps = fps
        self.frame_height_px = frame_h
        self.frame_width_px = frame_w

        if status is None: 
            self.status = {bp: BodypartStatus(bp, state="raw") for bp in bodyparts}
        else : 
            self.status = {bp: BodypartStatus.from_dict(stat) for bp, stat in status.items()}
            
        self.coords_set: dict[str, TrajectoryDLC] = None
        self.scorer: str = None

    # --- I/O ---

    def save_csv(self, save_as: Path):
        data = {bp: traj.coords.set_index("t")
                for bp, traj in self.coords_set.items()}

        df = pd.concat({self.scorer: pd.concat(data, axis=1)}, axis=1)
        df.to_csv(save_as)


    def load(self, only_valid=True) -> dict[str, Trajectory]:
        df = pd.read_csv(self.path, header=[0, 1, 2], index_col=0)
        self.scorer = df.columns[0][0]
        df = df.droplevel(0, axis=1)
        df = df.iloc[1:].reset_index(drop=True)
        df.index = np.arange(len(df)) / self.fps
        df.index.name = "t"

        if only_valid: 
            bps = [bp for bp in self.bodyparts if self.status[bp].is_valid()]
            # print([f"{bp}: {stat}" for bp, stat in self.status.items()]) 
        else:
            bps = self.bodyparts

        if bps == []: 
            return None

        self.coords_set = {bp: self.trajectory_cls(df[bp].reset_index(), self.view,
                            bp, self.cm_per_pixel, self.time_pad_off,
                            self.shift, self.fps, self.frame_height_px, self.frame_width_px) 
                            for bp in bps}

        for bp, stat in self.status.items():
            if only_valid and not stat.is_valid() : 
                continue
            stat.duration_frame = self.coords_set[bp].duration_frame
            stat.duration_sec = self.coords_set[bp].duration_sec

        return self.coords_set

    # --- preprocess ---

    def preprocess(self, thresh, gap_scale, interp_method, max_gap) -> pd.DataFrame:
        """Open raw DLC output
        1. detect outlier of raw trajectories
        2. remove thus outliers
        3. interpolated missing coordinates
        4. detect outliers of interpolated coordinates
        Returns a big dataframe containing 2 levels: 
            1st level = bodypart name
            2nd level = raw coordinates, raw outlier mask, interpolated coordinates, inter outlier mask"""
        
        raw = self.load(only_valid=False)
        outlier_results = {}
        interpolated_results = {}
        raw_coords = {}

        for bp in self.bodyparts:

            traj: TrajectoryDLC = raw[bp]
            c = traj._array_to_scaled_cartesian()

            # 1. outlier removal of raw
            outlier: pd.DataFrame = self.filter_outliers(c, thresh, gap_scale)
            
            # 2. outlier removal of interpolated
            interpolated = self.interpolate_data(outlier, interp_method, max_gap)
            interpolated_outlier = self.filter_outliers(interpolated, thresh, gap_scale)

            interpolated["mask"] = interpolated_outlier["mask"]
            interpolated["local_dists"] = interpolated_outlier["local_dists"]
            interpolated["local_thresh"] = interpolated_outlier["local_thresh"]

            raw_coords[bp] = c
            outlier_results[bp] = outlier
            interpolated_results[bp] = interpolated

        raw_coords_df: pd.DataFrame = pd.concat(raw_coords, axis=1)    # columns: (bodypart, coords)
        outlier_df: pd.DataFrame = pd.concat(outlier_results, axis=1)          # columns: (bodypart, coords)
        inter_df: pd.DataFrame = pd.concat(interpolated_results, axis=1)          # columns: (bodypart, coords)

        outliers_masks = {bp: data["mask"] for bp, data in outlier_results.items()}
        crop_idx, crop_necessary = self.crop_begining_frames(outliers_masks)

        if crop_necessary: 
            raw_coords_df = raw_coords_df.iloc[crop_idx:]
            outlier_df = outlier_df.iloc[crop_idx:]
            inter_df = inter_df.iloc[crop_idx:]

        return raw_coords_df, outlier_df, inter_df



    def crop_begining_frames(self, outliers_masks, lim=10):
        """Return the crop index and whether cropping is necessary.

        Crop from the earliest first-outlier index if it occurs within
        the first `lim` frames for any body part.
        """
        first_outlier_idxs = [
            idx
            for mask in outliers_masks.values()
            if (idx := np.flatnonzero(mask)[0:1]).size > 0
        ]

        if not first_outlier_idxs:
            return None, False

        first_outlier_idxs = [idx[0] for idx in first_outlier_idxs]
        crop_idx = min(first_outlier_idxs)

        crop_necessary = crop_idx < lim

        return (crop_idx if crop_necessary else None), crop_necessary

    def longest_consecutive_true(self, mask):
        mask = np.asarray(mask, dtype=bool)
        padded = np.concatenate(([False], mask, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        
        return int((edges[1::2] - edges[::2]).max(initial=0))



    def outlier_euclidian_dist_anchored(self, coords: pd.DataFrame, threshold: float, gap_scale: float = 0.0):
        """
        Sequential outlier detection anchored on the last CONFIRMED-good point.
        Correctly catches 'stuck cluster' outliers (consecutive bad points that
        are close to each other but far from the real trajectory), which a
        naive previous-point diff cannot see.

        :param threshold: max plausible distance for a single-frame step
        :param gap_scale: extra allowed distance per frame since last good point,
                        to avoid falsely flagging legitimate fast movement
                        after a gap (e.g. during peak reach velocity)
        """
        xy = coords[["x", "y"]].to_numpy()
        n = len(xy)
        is_outlier = np.zeros(n, dtype=bool)
        local_thresholds = np.zeros(n, dtype=float)
        local_thresholds[0] = threshold
        local_dists = np.zeros(n, dtype=float)
        local_dists[0] = 0

        last_good_idx = 0
        for i in range(1, n):
            if np.isnan(xy[i]).any():
                continue
            gap = i - last_good_idx
            local_threshold = threshold + gap_scale * (gap-1)
            local_thresholds[i] = local_threshold
            dist = np.linalg.norm(xy[i] - xy[last_good_idx])
            local_dists[i] = dist

            if dist < local_threshold:
                last_good_idx = i      # accept -> becomes new anchor
            else:
                is_outlier[i] = True   # stays flagged; anchor NOT updated
        return is_outlier, local_thresholds, local_dists


    def filter_outliers(self, coords: pd.DataFrame, thresh: float=0.6, gap_scale: float= 0.0) -> pd.DataFrame : 
        """
        Detect outliers in coordinates and put them to NaN

        :param coords: coordinates (must contain x, y columns)
        :param stat_method:
        - 'regression': dists distance to computed polynomial regression < threshold
        - 'eucli': euclidian distance between 2 consecutive points > threshold
        """
        if coords is None:
            coords = self.coords
        
        filtered_coords: pd.DataFrame = coords.copy()

        outlier_mask, local_thresholds, local_dists = self.outlier_euclidian_dist_anchored(coords, threshold=thresh, gap_scale=gap_scale)

        filtered_coords.loc[outlier_mask, ["x", "y"]] = np.nan
        filtered_coords["mask"] = outlier_mask
        filtered_coords["local_thresh"] = local_thresholds
        filtered_coords["local_dists"] = local_dists

        return filtered_coords



    def interpolate_data(self, coords: pd.DataFrame, method: str, max_gap: int) -> pd.DataFrame:
        """
        Interpolates missing values (NaN) in coordinates dataframe.
        Only for a number of consecutive missing values under max_gap

        :param method: 
            - 'zero'
            - 'linear'
            - 'splinear'
            - 'cubic'
            - 'spline'
        """
        if coords is None:
            coords = self.coords

        coords_interpolated = coords.copy() 

        # Minimum valid points required per method
        min_points = {
            'zero': 2,
            'linear': 2,
            'slinear': 2,
            'cubic': 4,
            'spline': 4
        }

        for col in ["x", "y"]:
            series = coords[col]
            before_nans = series.isna().sum()
            valid = series.dropna()

            # Determine if fallback to linear is needed
            use_method = method
            if len(valid) < min_points.get(method, 2):
                print(f"Column {col} has only {len(valid)} valid points; falling back to linear interpolation.")
                use_method = 'linear'

            # Perform interpolation for interior gaps
            if use_method == 'spline':
                # Use a cubic spline of order 3
                interp_series = series.interpolate(
                    method='spline',
                    order=3,
                    s=0,
                    limit=max_gap,
                    limit_direction='both'
                )
            else:
                interp_series = series.interpolate(
                    method=use_method,
                    limit=max_gap,
                    limit_direction='both'
                )
            # Fill leading/trailing small gaps via backward/forward fill
            interp_series = interp_series.bfill(limit=max_gap)
            interp_series = interp_series.ffill(limit=max_gap)

            after_nans = interp_series.isna().sum()
            # print(f"Number of NaNs before interpolation : {before_nans}, after : {after_nans}")

            coords_interpolated[col] = interp_series

        return coords_interpolated


    # --- validation ---

    def update_status(self, bp, new_coords: pd.DataFrame = None, state=None,
                    success=None, stage=None, reason=None):
        if state:
            self.status[bp].state = state

            if new_coords is not None:
                self.coords_set[bp].coords = new_coords
                self.status[bp].duration_frame = len(new_coords)
                self.status[bp].duration_sec = len(new_coords) / self.fps

        if success is not None and stage is not None and reason is not None:
            self.status[bp].set_success(stage, success, reason)

    def status_to_dict(self): 
        return {bp: stat.to_dict() for bp, stat in self.status.items()}