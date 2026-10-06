from dataclasses import dataclass
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .Trajectory import Trajectory

class BehaviorBox:

    # BOX DEFINITION 
    # Ref image : "/home/ninjayu/IMN_Kinematics/data/BetaMouv/camera_shift/raw_frames/#517_CHR_CONTRA_Beta_RightHemi_leftView_high_LaserOff_2024-07-05.png"

    XL, YL = (55, 230)   # lever position (px)
    XP, YP = (315, 348)   # pad position (px)

    def __init__(self, 
                 coords: dict[str, Trajectory],
                 time_pad_off: float,
                 shift: tuple[float] = (0.0, 0.0),
                frame_width=512, name: str="lol") : 
        """
        Return absolute behavior boxes:
        press, grasp, open, reach

        Returns
        -------
        Dict[boxes: (x_top_left, y_top_left, x_bottom_left, y_bottom_left)]
        """
        self.dx, self.dy = shift

        self.coords = coords
        self.frame_width = frame_width
        self.time_pad_off = time_pad_off
        self.name = name

        # constants
        self.lever_right = 40 + self.dx
        self.lever_upper = 10 + self.dy
        self.lever_lower = 18 + self.dy

        finger3_traj = self.coords["finger_3"]
        print(finger3_traj.is_valid(), self.coords["soft_pad"].is_valid())
        self.reach_bottom = finger3_traj.clean_coords.loc[finger3_traj.clean_coords["t"] == self.time_pad_off]["y"]

        self.spatial_boxes = self.build_spatial_boxes()
        self.adjusted_boxes = self.build_adjusted_boxes()



    def build_spatial_boxes(self): 
        return {
            "reach": (
                self.XL + self.lever_right,
                self.frame_width,
                self.XP,
                self.reach_bottom,
                "yellow"
            ),
            "open": (
                0,
                self.frame_width,
                self.XL + self.lever_right,
                self.YL + self.lever_upper,
                "orange"
            ),
            "grasp": (
                0,
                self.YL + self.lever_upper,
                self.XL + self.lever_right,
                self.YL - self.lever_lower,
                "blue"
            ),
        }

    def bodypart_angle(self, bp1: str = "soft_pad", bp2: str = "finger_3", 
                               coords: dict[str, Trajectory] | None = None) -> np.ndarray:
        """Compute the orientation angle (in degrees) of the segment
        going from bp1 -> bp2, relative to the horizontal axis.
        """
        if coords is None:
            coords = self.coords

        bodypart1 = coords[bp1].clean_coords
        bodypart2 = coords[bp2].clean_coords

        dx = bodypart2["x"].to_numpy() - bodypart1["x"].to_numpy()
        dy = bodypart2["y"].to_numpy() - bodypart1["y"].to_numpy()

        angle = np.degrees(np.arctan2(dy, -dx))  # range [-180, 180]

        # remove artificial jumps of 360° when the angle crosses 180°
        return angle
        # return np.degrees(np.unwrap(np.radians(angle)))
    
    def bodypart_distance(self, bp1: str = "soft_pad", bp2: str = "finger_3", 
                               coords: dict[str, Trajectory] | None = None) -> np.ndarray: 
        """Compute the orientation angle (in degrees) of the segment
        going from bp1 -> bp2, relative to the horizontal axis.
        """
        if coords is None:
            coords = self.coords

        bodypart1 = coords[bp1].clean_coords
        bodypart2 = coords[bp2].clean_coords

        dx = bodypart2["x"].to_numpy() - bodypart1["x"].to_numpy()
        dy = bodypart2["y"].to_numpy() - bodypart1["y"].to_numpy()

        return np.sqrt(dx**2 + dy**2)


    def build_adjusted_boxes(self):
        bp_angle: np.array = self.bodypart_angle()
        bp_distance: np.array = self.bodypart_distance()
        finger3_traj: Trajectory = self.coords["finger_3"]
        softpad_traj: Trajectory = self.coords["soft_pad"]
        t = self.coords["finger_3"].clean_coords["t"]

        fig, ax = plt.subplots(2, 1, figsize=[8, 11])
        ax[0] = finger3_traj.plot_traj(finger3_traj.clean_coords, ax[0], fig, "red")
        ax[0] = softpad_traj.plot_traj(softpad_traj.clean_coords, ax[0], fig, "orange")
        # self.coords["finger_3"].show_traj(self.coords["finger_3"].clean_coords)
        ax[0].set_title("finger3 (red) - soft pad (orange)")
        ax[0].set_ylabel("y (cm)")
        ax[0].set_xlabel("x (cm)")

        sc = ax[1].scatter(bp_angle, bp_distance, c=t, cmap="viridis")
        cbar = fig.colorbar(sc, ax=ax[1])
        cbar.set_label("time")

        ax[1].set_ylabel("distance")
        ax[1].set_xlabel("angle")
        plt.savefig(f"./data/BetaMouv/results/CONTRA_CHR_#517_#531/behavior_box_testing/{self.name}.png")
        plt.close()
         
        return {
            "reach": (  # does not change
                self.XL + self.lever_right,
                self.frame_width,
                self.XP,
                self.reach_bottom,
                "yellow"
            ),
            "open": (
                0,
                self.frame_width,
                self.XL + self.lever_right,
                self.YL + self.lever_upper,
                "orange"
            ),
            "grasp": (
                0,
                self.YL + self.lever_upper,
                self.XL + self.lever_right,
                self.YL - self.lever_lower,
                "blue"
            ),
        }
    


    def _contains(self, name: str, x: float, y: float) -> bool: 
        """Return if the coordinates are in the boxe"""
        xmin, ymin, xmax, ymax, _ = self.spatial_boxes[name]
        return xmin <= x <= xmax and ymin <= y <= ymax
        

    def classify_behavior(self, x: float, y: float) -> str:
        behavior = "none"
        for name in self.spatial_boxes:
            if self._contains(name, x, y):
                behavior = name

        return behavior
    

    def classify_trajectory(self, coords: pd.DataFrame) -> pd.DataFrame :
        labels = []

        for row in coords.itertuples(index=False):
            x, y, t = row.x, row.y, row.t
            labels.append(self.classify_behavior(x, y))

        coords["behavior"] = labels

        return coords
    
    def draw_boxes(self, ax) :
        for box_name, coords in self.spatial_boxes.items() :
            xmin, ymin, xmax, ymax, color = coords 

            rect = Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor=color,
                edgecolor=None,
                lw=1,
                alpha=0.3,
                label=box_name,
            )
            ax.add_patch(rect)




# ---------------------------------------------------------------------------



if __name__ == "__main__" : 
    import skimage as ski

    Boxes = BehaviorBox(
        xy_lever=(55, 230),
        xy_pad=(315, 325),
        view="left",
    )

    # classification of one points
    pt = (40, 245)
    label = Boxes.classify_behavior(pt[0], pt[1])
    print(label)       

    # classification of a whole trajectory
    traj = [
        (210, 260),
        (200, 100),
        (70, 120),
        (60, 235),
        (35, 260),
    ]

    classified_traj = []

    for (x, y) in traj: 
        label = Boxes.classify_behavior(x, y)
        classified_traj.append(label)
    
    print(classified_traj)

    # display of the boxes and trajectory
    filename = '/home/poemiti/Rats-Kinematics/data_V1/rat_image2.png'
    raw_img = ski.io.imread(filename)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)   # image coordinates
    ax.set_aspect("equal")

    Boxes.draw_boxes(ax)

    # trajectory
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    ax.plot(xs, ys, "-o", c="black")

    ax.imshow(raw_img, cmap="gray")

    ax.legend()
    plt.show()