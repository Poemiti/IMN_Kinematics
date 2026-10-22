# src/BetaMouv/Trajectory.py


from src.Base.Trajectory import TrajectoryDLC as BaseTrajectory
from src.Base.Trajectory import DLCTracking as BaseDLCTracking
from .File import File

import pandas as pd
import numpy as np
from scipy.interpolate import make_splrep, splev

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import src.utils as u


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("talk", style="ticks", rc=custom_params, palette="pastel")


# load project_info.yaml to get constant related to the protocole
cfg = u.load_config("./config/BetaMouv/project_info.yaml")


class TrajectoryDLC(BaseTrajectory): 

    file_cls = File 
    STAGES = ("outlier", "validation")

    def __init__(self, 
                 coords: pd.DataFrame,
                view: str,
                bodypart: str,
                cm_per_pixel: float,
                time_pad_off: float,
                shift: tuple[float],):
        super().__init__(coords, view, bodypart, cm_per_pixel, time_pad_off, shift, 
                         cfg["fps"], cfg["frame_height_px"], cfg["frame_width_px"])

        # some constants from cfg
        self.dt = 1 / cfg["fps"]
        self.LEVER_POS = self._point_to_scaled_cartesian(cfg["lever_position"][0], cfg["lever_position"][1])
        self.PAD_POS = self._point_to_scaled_cartesian(cfg["pad_position"][0], cfg["pad_position"][0])
        self.FRAME_WIDTH_CM =  cfg["frame_width_px"] * self.cm_per_pixel
        self.FRAME_HEIGHT_CM =  cfg["frame_height_px"] * self.cm_per_pixel

    # ----------- metrics computation -----------

    def compute_instant_metrics(self, coords: pd.DataFrame | None = None) -> pd.DataFrame:
        if coords is None:
            coords = self.coords

        coords = coords.copy()
        coords["instant_velocity"] = self.instant_velocity(coords)
        coords["instant_speed"] = self.instant_speed(coords)
        coords["angle"] = self.angle(coords)
        coords["lever_distance"] = self.obj_bodypart_distance(coords, self.LEVER_POS)
        coords["distances"] = self.distances(coords)
        coords["acc"] = self.acceleration(coords)
        coords["signed_acc"] = self.signed_acceleration(coords)

        return coords


    def compute_scalar_metrics(self, coords: pd.DataFrame | None = None) -> pd.DataFrame: 
        if coords is None:
            coords = self.coords
        
        scalar_metrics = {}
        scalar_metrics["avg_velocity_laser_period"] = self.avg_velocity(coords, start=self.time_pad_off, end=self.time_pad_off + 0.325)
        scalar_metrics["tortuosity_laser_period"] = self.tortuosity_fixed_period(coords, start=self.time_pad_off, end=self.time_pad_off + 0.325)
        scalar_metrics["tortuosity_pad_to_lever"] = self.tortuosity_from_obj(coords, obj_coords=self.LEVER_POS, time_pad_off=self.time_pad_off)

        return scalar_metrics


    def plot_tendency(self, coords: pd.DataFrame, value , save_as, group, 
                      laser_state: str= "LaserOff", show_angle: bool=False):
        """
        Plot velocity tendency with error span around the average.
        Custome error functions available : 
            - "sem" : Standard Error of the Mean (https://statisticsbyjim.com/hypothesis-testing/standard-error-mean/)
            How to interprete SEM : 'For a SEM of 3, we know that the typical 
            difference between a sample mean and the population mean is 3'
        """
        from scipy.stats import sem

        if coords is None: 
            coords = self.coords.copy()

        coords["relative_t"] = (coords["t"] - self.time_pad_off).round(3)

        fig, ax = plt.subplots(figsize=[8, 8])
        
        ax.plot(coords["relative_t"], coords[value], color="k", marker=".")
        
        ax.axvline(x=0, color="k", label="time pad off", linestyle="--", lw=1)
        ax.axhline(y=0, color="k", linestyle="--", lw=1)
        if show_angle: 
            ax.axhline(y=90, color="red", linestyle="--", lw=1)
            ax.axhline(y=-90, color="red", linestyle="--", lw=1)
            ax.axhline(y=0, color="red", linestyle="--", lw=1)

        if laser_state == "LaserOn":
            ax.axvspan(xmin=0.025, xmax=0.3, label="laser period", color="tomato", alpha=0.3)
        else: 
            ax.axvspan(xmin=0.025, xmax=0.3, label="laser period", color="gray", alpha=0.3)

        ax.set_xlabel("Time (sec)")
        ax.set_ylabel(value)
        ax.legend(loc="upper right")

        title = group[:len(group)//2] + "\n" + group[len(group)//2:]
        ax.figure.suptitle(f"{title}\n {value} - {self.bodypart} - {laser_state}", ha='center')
        ax.figure.subplots_adjust(top=0.8)

        fig.savefig(save_as)
        
        # plt.show()
        plt.close()



class DLCTracking(BaseDLCTracking): 

    def __init__(self, 
                 path: Path,
                 view: str, 
                 bodyparts: list[str], 
                 cm_per_pixel: float, 
                 time_pad_off: float, 
                 shift: tuple[float, float], 
                 status: dict | None = None):
        
        super().__init__(path, view, bodyparts, cm_per_pixel, time_pad_off, shift, status, 
                         cfg["fps"], cfg["frame_height_px"], cfg["frame_width_px"])

        # some constants from cfg
        self.dt = 1 / cfg["fps"]
        self.LEVER_POS = self._point_to_scaled_cartesian(cfg["lever_position"][0], cfg["lever_position"][1])
        self.PAD_POS = self._point_to_scaled_cartesian(cfg["pad_position"][0], cfg["pad_position"][0])
        self.FRAME_WIDTH_CM =  cfg["frame_width_px"] * self.cm_per_pixel
        self.FRAME_HEIGHT_CM =  cfg["frame_height_px"] * self.cm_per_pixel


    def plot_preprocess(self, 
                        interpolated_coords, 
                        interpolated_dist,
                        outlier_filtered_coords,
                        raw_coords,
                        raw_dist,
                        local_thresholds,
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



        fig = plt.figure(figsize=(20,10))
        gs = fig.add_gridspec(3, 2)

        offset = 0.2  # cm

        ax_xt = fig.add_subplot(gs[0,0])      # x(t)
        ax_yt = fig.add_subplot(gs[1,0])      # y(t)
        ax_dist = fig.add_subplot(gs[2,0])      # distance
        ax_traj = fig.add_subplot(gs[:,1])    # trajectory spans both rows

        _plot_xy([ax_xt, ax_yt], interpolated_coords, 4*offset,"#0570b0", "|", "3.interpolate")
        _plot_xy([ax_xt, ax_yt], outlier_filtered_coords, 2*offset, "#bdc9e1","|", "1.outlier")
        _plot_xy([ax_xt, ax_yt], raw_coords, 0*offset, "#d1cbdc","|", "0.raw", time_pad_off)

        # raw distances
        ax_dist.plot(raw_coords["t"], raw_dist, color="#d1cbdc")
        ax_dist.scatter(raw_coords["t"], raw_dist, color="#d1cbdc", marker="|")

        # interpolated distances
        ax_dist.plot(interpolated_coords["t"], interpolated_dist , color="#0570b0")
        ax_dist.scatter(interpolated_coords["t"], interpolated_dist, color="#0570b0", marker="|")

        ax_dist.plot(raw_coords["t"], local_thresholds, color="green", label="local threshold", lw=0.5, linestyle="--")
        ax_dist.axhline(y=0.55, linestyle="--", color="red", label="threshold", lw=0.5)
        ax_dist.axvline(x=time_pad_off, linestyle="--", color="k", label="pad_off", lw=0.5)
        
        _plot_traj(raw_coords, 0*offset, "1.raw", "#d1cbdc", ax_traj)
        _plot_traj(outlier_filtered_coords, 0*offset, "2.outlier", "#bdc9e1" ,ax_traj, "")
        _plot_traj(interpolated_coords, 0*offset, "4.interpolate", "#0570b0" ,ax_traj, "|")

        ax_xt.set(ylabel=("x (cm)"),)

        ax_yt.set(ylabel=("y (cm)"),)

        ax_dist.set(
            ylabel=("distance (cm)"),
            xlabel=("time (s)")
            )

        ax_traj.set(
            xlabel=("x (cm)"),
            ylabel=("y (cm)"),
            xlim=(0, self.FRAME_HEIGHT_CM),
            ylim=(0, self.FRAME_WIDTH_CM)
            )
        ax_traj.legend()
        fig.suptitle(title, wrap=True)

        plt.tight_layout()
        fig.savefig(save_as)
        plt.close()