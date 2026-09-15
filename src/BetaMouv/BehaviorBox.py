from dataclasses import dataclass
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd

class BehaviorBox:
    def __init__(self, 
                xy_lever=(55, 230),
                xy_pad=(315, 325),
                view="left",
                frame_width=512,) : 
        """
        Return absolute behavior boxes:
        press, grasp, open, reach

        Returns
        -------
        Dict[boxes: (x_top_left, y_top_left, x_bottom_left, y_bottom_left)]
        """
        
        self.xl, self.yl = xy_lever
        self.xp, self.yp = xy_pad

        # constants
        self.lever_right = 40
        self.lever_upper = 10
        self.lever_lower = 18
        self.pad_upper = 25

        self.frame_width = frame_width

        if view == "left" :
            self.boxes = self._build_left_boxes()

        elif view == "right" : 
            self.boxes = self._build_right_boxes()

        else:
            raise ValueError("view must be 'left' or 'right'")


    def _build_left_boxes(self): 
        return {
            "reach": (
                self.xl + self.lever_right,
                0,
                self.xp,
                self.yp - self.pad_upper,
                "yellow"
            ),
            "open": (
                0,
                0,
                self.xl + self.lever_right,
                self.yl - self.lever_upper,
                "orange"
            ),
            "grasp": (
                0,
                self.yl - self.lever_upper,
                self.xl + self.lever_right,
                self.yl + self.lever_lower,
                "blue"
            ),

            "press": (
                0,
                self.yl + self.lever_lower,
                self.xl + self.lever_right,
                self.yl + int(self.lever_lower * 2.5),
                "green"
            ),
        }
    

    def _build_right_boxes(self) : 
        return {
            "reach": (
                self.xp,
                0,
                self.xl - self.lever_right,
                self.yp - self.pad_upper,
                "yellow"
            ),
            "open": (
                self.xl - self.lever_right,
                0,
                self.frame_width,
                self.yl - self.lever_upper,
                "orange"
            ),
            "grasp": (
                self.xl - self.lever_right,
                self.yl - self.lever_upper,
                self.frame_width,
                self.yl + self.lever_lower,
                "blue"
            ),
            "press": (
                self.xl - self.lever_right,
                self.yl + self.lever_lower,
                self.frame_width,
                self.yl + int(self.lever_lower * 2.5),
                "green"
            ),
        }
    


    def _contains(self, name: str, x: float, y: float) -> bool: 
        """Return if the coordinates are in the boxe"""
        xmin, ymin, xmax, ymax, _ = self.boxes[name]
        return xmin <= x <= xmax and ymin <= y <= ymax
        

    def classify_behavior(self, x: float, y: float) -> str:
        behavior = "none"
        for name in self.boxes:
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
        for box_name, coords in self.boxes.items() :
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