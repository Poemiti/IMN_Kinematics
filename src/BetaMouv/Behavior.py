# src/BetaMouv/Behavior.py

import pandas as pd
import numpy as np

from src.Base.Trajectory import Trajectory
from src.BetaMouv.Trajectory import TrajectoryDLC

import src.utils as u

# load project_info.yaml to get constant related to the protocole
cfg = u.load_config("./config/BetaMouv/project_info.yaml")

class Behavior:

    """Define the behavior during a trial"""

    def __init__(self, 
                 coords_set: dict[str, Trajectory], 
                 time_pad_off: float, 
                 ) : 
        """
        Return behavior label during trial time:
        press, grasp, open, reach, lift
        """

        self.coords_set = coords_set
        self.time_pad_off = time_pad_off

        finger3_traj: TrajectoryDLC = self.coords_set["finger_3"]
        self.reach_bottom = finger3_traj.clean_coords.loc[finger3_traj.clean_coords["t"] == self.time_pad_off]["y"]

        # constants in cm and cartesian plane
        self.LEVER_POS = finger3_traj._point_to_scaled_cartesian(cfg["lever_position"][0], cfg["lever_position"][1])
        self.PAD_POS = finger3_traj._point_to_scaled_cartesian(cfg["pad_position"][0], cfg["pad_position"][0])
        self.FRAME_WIDTH_CM =  cfg["frame_width_px"] * self.cm_per_pixel
        self.FRAME_HEIGHT_CM =  cfg["frame_height_px"] * self.cm_per_pixel


        # features building
        centroid = Trajectory(self.centroid())
        features = self.centroid()              # initilisation with centroid
        features["area"] = self.area_triangle()
        features["centroid_direction"] = centroid.angle()
        features["centroid_velocity"] = centroid.instant_velocity()
        features["centroid_speed"] = centroid.instant_speed()
        features["centroid_acc"] = centroid.signed_acceleration()
        features["f3_sp_distance"] = self.bodypart_distance()
        features["f3_sp_angle"] = self.bodypart_angle()
        features["centroid_lever_dist"] = centroid.obj_bodypart_distance(obj_coord=self.LEVER_POS)


    def bodypart_angle(self, bp1: str = "soft_pad", bp2: str = "finger_3", 
                               coords_set: dict[str, Trajectory] | None = None) -> np.ndarray:
        """Compute the orientation angle (in degrees) of the segment
        going from bp1 -> bp2, relative to the horizontal axis.
        """
        if coords_set is None:
            coords_set = self.coords_set

        bodypart1 = coords_set[bp1].clean_coords
        bodypart2 = coords_set[bp2].clean_coords

        dx = bodypart2["x"].to_numpy() - bodypart1["x"].to_numpy()
        dy = bodypart2["y"].to_numpy() - bodypart1["y"].to_numpy()

        angle = np.degrees(np.arctan2(dy, -dx))  # range [-180, 180]

        return angle
    
    
    def bodypart_distance(self, bp1: str = "soft_pad", bp2: str = "finger_3", 
                               coords_set: dict[str, Trajectory] | None = None) -> np.ndarray: 
        """Compute the orientation angle (in degrees) of the segment
        going from bp1 -> bp2, relative to the horizontal axis.
        """
        if coords_set is None:
            coords_set = self.coords_set

        bodypart1 = coords_set[bp1].clean_coords
        bodypart2 = coords_set[bp2].clean_coords

        dx = bodypart2["x"].to_numpy() - bodypart1["x"].to_numpy()
        dy = bodypart2["y"].to_numpy() - bodypart1["y"].to_numpy()

        return np.sqrt(dx**2 + dy**2)

    def centroid(self, bp1: str = "soft_pad", bp2: str = "finger_3", bp3: str = "finger_2", 
                 coords_set: dict[str, Trajectory] = None) -> pd.DataFrame: 
        """Compute the position of a centroid point between 3 bodyparts"""
        if coords_set is None:
            coords_set = self.coords_set
        
        bodypart1 = coords_set[bp1].clean_coords
        bodypart2 = coords_set[bp2].clean_coords
        bodypart3 = coords_set[bp3].clean_coords

        centroid_x = bodypart1["x"] + bodypart2["x"] + bodypart3["x"] / 3
        centroid_y = bodypart1["y"] + bodypart2["y"] + bodypart3["y"] / 3

        return pd.DataFrame({
            "x": centroid_x, 
            "y": centroid_y, 
            "t": bodypart1["t"]
        })


    def area_triangle(self, bp1: str = "soft_pad", bp2: str = "finger_3", bp3: str = "finger_2", 
                 coords_set: dict[str, Trajectory] = None) -> np.array:
        """Return area accros time, between 3 bodyparts
        Based on this tuto : https://www.geeksforgeeks.org/python/python-program-to-calculate-the-area-of-a-triangle/"""

        c1 = coords_set[bp1].clean_coords
        c2 = coords_set[bp2].clean_coords
        c3 = coords_set[bp3].clean_coords

        return  0.5 * abs(c1["x"]*c2["y"] + c2["x"]*c3["y"] + c3["x"]*c1["y"] - c1["y"]*c2["x"] - c2["y"]*c3["x"] - c3["y"]*c1["x"])



    # def build_adjusted_boxes(self):
    #     bp_angle: np.array = self.bodypart_angle()
    #     bp_distance: np.array = self.bodypart_distance()
    #     finger3_traj: Trajectory = self.coords_set["finger_3"]
    #     softpad_traj: Trajectory = self.coords_set["soft_pad"]
    #     t = self.coords_set["finger_3"].clean_coords["t"]

    #     fig, ax = plt.subplots(2, 1, figsize=[8, 11])
    #     ax[0] = finger3_traj.plot_traj(finger3_traj.clean_coords, ax[0], fig, "red")
    #     ax[0] = softpad_traj.plot_traj(softpad_traj.clean_coords, ax[0], fig, "orange")
    #     # self.coords_set["finger_3"].show_traj(self.coords_set["finger_3"].clean_coords)
    #     ax[0].set_title("finger3 (red) - soft pad (orange)")
    #     ax[0].set_ylabel("y (cm)")
    #     ax[0].set_xlabel("x (cm)")

    #     sc = ax[1].scatter(bp_angle, bp_distance, c=t, cmap="viridis")
    #     cbar = fig.colorbar(sc, ax=ax[1])
    #     cbar.set_label("time")

    #     ax[1].set_ylabel("distance")
    #     ax[1].set_xlabel("angle")
    #     plt.savefig(f"./data/BetaMouv/results/CONTRA_CHR_#517_#531/behavior_box_testing/{self.name}.png")
    #     plt.close()