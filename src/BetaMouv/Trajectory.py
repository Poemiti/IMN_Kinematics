# src/BetaMouv/Trajectory.py


from src.Base.Trajectory import Trajectory as BaseTrajectory
from .File import File
from .BehaviorBox import BehaviorBox

import pandas as pd
import numpy as np
from scipy.interpolate import make_splrep, splev

import matplotlib.pyplot as plt
import seaborn as sns



custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("talk", style="ticks", rc=custom_params, palette="pastel")


class Trajectory(BaseTrajectory): 

    file_cls = File 


    ################## Trajectory filtration method ###############
    # specific to this project


    @staticmethod
    def _define_likelihood_threshold(coords: pd.DataFrame, thresh: float, percentile: float = None) -> float : 
        """
        Define the threshold of low likelihood coordinates
        Coords must be a dataframe containing the following columns ['x', 'y', 'likelihood']
        """
        if coords is None:
            coords = self.coords
        
        if percentile is not None:
            computed_tresh = coords["likelihood"].quantile(percentile / 100.0)
        else:
            computed_tresh = thresh

        return computed_tresh


    def filter_likelihood(self, coords: pd.DataFrame, thresh: float, percentile: float = None) -> tuple[pd.DataFrame, float] : 
        """
        Returns the filtered coordinates based on the threshold of low likelihood coordinates.
        Coords must be a dataframe containing the following columns ['x', 'y', 'likelihood']
        Low confidance point will be set to NaN
        """
        if coords is None:
            coords = self.coords

        computed_thresh = self._define_likelihood_threshold(coords, thresh, percentile)

        # print(f"\nthreshold set to: {computed_thresh}")
        mask = coords["likelihood"] > computed_thresh
        filtered_coords: pd.DataFrame = coords.copy()
        filtered_coords.loc[~mask, ["x", "y"]] = np.nan
        
        return filtered_coords, computed_thresh

    @staticmethod
    def _remove_consecutiv_outliers(mask):
        """True = outlier detected"""
        clean_mask = mask.copy()
        mask_size = len(mask)

        is_out = False

        for i in range(mask_size): 

            if clean_mask[i] and not is_out : # init outlier list
                print(i, "IS OUT")
                is_out = True
                continue

            elif not clean_mask[i] and is_out :  # add outlier
                print(i, "IS OUT AGAIN")
                clean_mask[i] = True
                continue

            elif clean_mask[i] and is_out :  # back to "normality"
                print(i, "BACK TO NORMALITY")
                is_out = False
                continue

        print(clean_mask)

        return clean_mask

    @staticmethod
    def outlier_regression(self, t: float, x: float, y: float) : 
        """Return a mask of outliers. True = is an outlier"""
        s_factor=3.0
        k=2
        s=1000

        # Fit splines 
        tck_x = make_splrep(t, x, k=k, s=s)
        tck_y = make_splrep(t, y, k=k, s=s)

        # Predicted smooth trajectory
        x_pred = splev(t, tck_x)
        y_pred = splev(t, tck_y)

        # Compute dists distance
        dists = np.sqrt((x - x_pred)**2 + (y - y_pred)**2)
        thresh = np.nanmean(dists) + s_factor * np.nanstd(dists)
        return dists < thresh


    def outlier_euclidian_dist(self, coords, threshold: float): 
        """Return a mask of outliers. True = is an outlier
        Follows euclidian distances"""

        diffs = coords[["x", "y"]].diff()
        dists = np.sqrt((diffs**2).sum(axis=1))

        # remove consecutive outliers
        is_outlier = dists >= threshold + 0.1
        return self._remove_consecutiv_outliers(is_outlier)


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

        last_good_idx = 0
        for i in range(1, n):
            if np.isnan(xy[i]).any():
                continue
            gap = i - last_good_idx
            local_threshold = threshold + gap_scale * (gap - 1)
            dist = np.linalg.norm(xy[i] - xy[last_good_idx])

            if dist < local_threshold:
                last_good_idx = i      # accept -> becomes new anchor
            else:
                is_outlier[i] = True   # stays flagged; anchor NOT updated
        return is_outlier


    def filter_outliers(self, coords: pd.DataFrame, method: str = 'eucli', thresh: float=0.6, gap_scale: float= 0.0) -> pd.DataFrame : 
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

        if method == "regression": 

            t = coords["t"].to_numpy(dtype=float)
            x = coords["x"].to_numpy(dtype=float)
            y = coords["y"].to_numpy(dtype=float)

            outlier_mask = self.outlier_regression(t, x, y)

        elif method == "eucli"  : 
            outlier_mask = self.outlier_euclidian_dist(coords, threshold=thresh)


        elif method == "eucli_anchored"  : 
            outlier_mask = self.outlier_euclidian_dist_anchored(coords, threshold=thresh, gap_scale=gap_scale)

        filtered_coords.loc[outlier_mask, ["x", "y"]] = np.nan

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

    


    ####################### plotting methods ########################


    def plot_preprocess(self, 
                        interpolated_coords, 
                        # likelihood_filtered_coords,
                        outlier_filtered_coords,
                        raw_coords,
                        time_pad_off,
                        title, 
                        save_as):

        def _plot_traj(coord, offset, label, color, ax: plt.axes = None, marker: str = None):

            x = coord["x"] - offset
            y = coord["y"] - offset

            if ax is not None : 
                ax.plot(x, y, label=label, color=color)
                if marker : 
                    ax.scatter(x, y,marker=marker)
            else : 
                plt.plot(x, y, label=label, color=color)
                if marker : 
                    ax.scatter(x, y,marker=marker)


        def _plot_xy(axes, coords, offset, color,  marker, label, time_pad_off= None) -> None :
    
            t = coords["t"]
            x = coords["x"] + offset
            y = coords["y"] - offset

            axes[0].plot(t, x, marker=marker, color=color, label=label)
            axes[1].plot(t, y, marker=marker, color=color)

            if time_pad_off : 
                axes[0].axvline(time_pad_off, color='k', lw=0.8, ls='--', label="time pad off")
                axes[1].axvline(time_pad_off, color='k', lw=0.8, ls='--')



        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(3, 2)

        offset = 0.2  # cm

        ax_xt = fig.add_subplot(gs[0,0])      # x(t)
        ax_yt = fig.add_subplot(gs[1,0])      # y(t)
        ax_dist = fig.add_subplot(gs[2,0])      # distance
        ax_traj = fig.add_subplot(gs[:,1])    # trajectory spans both rows

        _plot_xy([ax_xt, ax_yt], interpolated_coords, 4*offset,"#0570b0", "|", "3.interpolate")
        # _plot_xy([ax_xt, ax_yt], likelihood_filtered_coords, 2*offset,"#74a9cf", "|", "2.likelihood")
        _plot_xy([ax_xt, ax_yt], outlier_filtered_coords, 2*offset, "#bdc9e1","|", "1.outlier")
        _plot_xy([ax_xt, ax_yt], raw_coords, 0*offset, "#d1cbdc","|", "0.raw", time_pad_off)

        # raw distances
        ax_dist.plot(raw_coords["t"], raw_coords["distances"], color="#d1cbdc")
        ax_dist.scatter(raw_coords["t"], raw_coords["distances"], color="#d1cbdc", marker="|")

        # interpolated distances
        ax_dist.plot(interpolated_coords["t"], interpolated_coords["distances"] , color="#0570b0")
        ax_dist.scatter(interpolated_coords["t"], interpolated_coords["distances"] , color="#0570b0", marker="|")
        ax_dist.axhline(y=0.55, linestyle="--", color="red", label="threshold", lw=0.5)
        ax_dist.axvline(x=time_pad_off, linestyle="--", color="k", label="pad_off", lw=0.5)
        
        # pad_off_frame = int((time_pad_off - 0.1)* 125)
        # pad_off_frame = pad_off_frame if pad_off_frame >=0 else 0
        # off_frame = int((time_pad_off + 0.4) * 125)
        
        # _plot_traj(raw_coords[pad_off_frame : off_frame], 0*offset, "1.raw", "#d1cbdc", ax_traj)
        # _plot_traj(outlier_filtered_coords[pad_off_frame : off_frame], 0*offset, "2.outlier", "#bdc9e1" ,ax_traj, "")
        # _plot_traj(likelihood_filtered_coords[pad_off_frame : off_frame], 0*offset, "3.likelihood", "#74a9cf" ,ax_traj, "")
        # _plot_traj(interpolated_coords[pad_off_frame : off_frame], 0*offset, "4.interpolate", "#0570b0" ,ax_traj, "|")

        _plot_traj(raw_coords, 0*offset, "1.raw", "#d1cbdc", ax_traj)
        _plot_traj(outlier_filtered_coords, 0*offset, "2.outlier", "#bdc9e1" ,ax_traj, "")
        _plot_traj(interpolated_coords, 0*offset, "4.interpolate", "#0570b0" ,ax_traj, "|")


        ax_xt.set(
            ylabel=("x (cm)"),
            # xlim=(time_pad_off - 0.1, time_pad_off + 0.4),
            )
        # ax_xt.set_xticks([])

        ax_yt.set(
            ylabel=("y (cm)"),
            # xlim=(time_pad_off - 0.1, time_pad_off + 0.4),
            )
        ax_yt.invert_yaxis()
        # ax_yt.set_xticks([])

        ax_dist.set(
            ylabel=("distance (cm)"),
            # xlim=(time_pad_off - 0.1, time_pad_off + 0.4),
            xlabel=("time (s)")
            )

        ax_traj.set(
            xlabel=("x (cm)"),
            ylabel=("y (cm)"),
            xlim=(0, self.frame_height * self.cm_per_pixel),
            ylim=(0, self.frame_height * self.cm_per_pixel)
            )
        ax_traj.legend()

        title = title[:len(title)//2] + "\n" + title[len(title)//2:]
        fig.suptitle(title, wrap=True)

        plt.tight_layout()
        fig.savefig(save_as)
        plt.close()

