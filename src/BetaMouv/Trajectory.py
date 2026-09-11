# src/BetaMouv/Trajectory.py


from src.Base.Trajectory import Trajectory as BaseTrajectory
from .File import File
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
    def _remove_consecutiv_outliers(mask, max_len=2):
        clean_mask = mask.copy()
        mask_size = len(mask)

        i = 0
        while i < mask_size:
            if clean_mask[i]:
                start = i

                while i < mask_size and clean_mask[i]:
                    i += 1

                end = i
                length = end - start

                left_bad = start > 0 and not clean_mask[start-1]
                right_bad = end < mask_size and not clean_mask[end]

                if left_bad and right_bad and length <= max_len:
                    clean_mask[start:end] = False
            else:
                i += 1

        return clean_mask



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



    def filter_outliers(self, coords: pd.DataFrame, stat_method: str = 'mad') -> pd.DataFrame : 
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

        t = coords["t"].to_numpy(dtype=float)
        x = coords["x"].to_numpy(dtype=float)
        y = coords["y"].to_numpy(dtype=float)


        if stat_method == "regression" : 
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
            mask = dists < thresh

            params = (dists, thresh, mask)


        if stat_method == "eucli" : 
            threshold = 40 # pixel

            # compute displacement
            diffs = coords[["x","y"]].diff()
            dists = np.sqrt((diffs**2).sum(axis=1))

            # remove consecutive outliers
            is_outlier = dists >= threshold
            is_outlier = self._remove_consecutiv_outliers(is_outlier, max_len=3)
            mask = is_outlier   

            params = (dists, threshold, mask)

        else:
            raise ValueError(f"Unknown stat_method '{stat_method}'")
        
        filtered_coords.loc[mask, ["x", "y"]] = np.nan

        return filtered_coords, params



    def interpolate_data(self, coords: pd.DataFrame, method: str, max_gap: int, displacement_threshold: float | None = None) -> pd.DataFrame:
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

        # Revert large displacements to NaN if threshold is set
        if displacement_threshold is not None:
            dx = coords_interpolated["x"].diff()
            dy = coords_interpolated["y"].diff()
            displacement = (dx ** 2 + dy ** 2) ** 0.5
            exceed = displacement > displacement_threshold
            coords_interpolated.loc[exceed, "x"] = float('nan')
            coords_interpolated.loc[exceed, "y"] = float('nan')
            print(f"{exceed.sum()} frames exceeded displacement threshold and were reverted to NaN")

        return coords_interpolated


    ####################### plotting methods ########################

    def make_interpolation_figures(self, 
                                    interpolated_coords, 
                                    likelihood_filtered_coords,
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
        gs = fig.add_gridspec(2, 2)

        offset = 0.2  # cm

        ax_xt = fig.add_subplot(gs[0,0])      # x(t)
        ax_yt = fig.add_subplot(gs[1,0])      # y(t)
        ax_traj = fig.add_subplot(gs[:,1])    # trajectory spans both rows

        _plot_xy([ax_xt, ax_yt], interpolated_coords, 3*offset,"#0570b0", "|", "3.interpolate")
        _plot_xy([ax_xt, ax_yt], likelihood_filtered_coords, 2*offset,"#74a9cf", "|", "2.likelihood")
        _plot_xy([ax_xt, ax_yt], outlier_filtered_coords, 1*offset, "#bdc9e1","|", "1.outlier")
        _plot_xy([ax_xt, ax_yt], raw_coords, 0*offset, "#d1cbdc","|", "0.raw", time_pad_off)
        
        pad_off_frame = int((time_pad_off - 0.1)* 125)
        pad_off_frame = pad_off_frame if pad_off_frame >=0 else 0
        off_frame = int((time_pad_off + 0.4) * 125)
        
        _plot_traj(raw_coords[pad_off_frame : off_frame], 0*offset, "1.raw", "#d1cbdc", ax_traj)
        _plot_traj(outlier_filtered_coords[pad_off_frame : off_frame], 0*offset, "2.outlier", "#bdc9e1" ,ax_traj, "")
        _plot_traj(likelihood_filtered_coords[pad_off_frame : off_frame], 0*offset, "3.likelihood", "#74a9cf" ,ax_traj, "")
        _plot_traj(interpolated_coords[pad_off_frame : off_frame], 0*offset, "4.interpolate", "#0570b0" ,ax_traj, "|")

        ax_xt.set(
            ylabel=("x (cm)"),
            xlim=(time_pad_off - 0.1, time_pad_off + 0.4),
            )

        ax_yt.set(
            ylabel=("y (cm)"),
            xlim=(time_pad_off - 0.1, time_pad_off + 0.4),
            xlabel=("time (s)"),
            )
        ax_yt.invert_yaxis()

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

