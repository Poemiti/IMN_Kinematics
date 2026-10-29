# src/BetaMouv/Behavior.py

import pandas as pd
import numpy as np

from src.Base.Trajectory import Trajectory
from src.BetaMouv.Trajectory import TrajectoryDLC

import src.utils as u
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("talk", style="ticks", rc=custom_params, palette="pastel")


# load project_info.yaml to get constant related to the protocole
cfg = u.load_config("./config/BetaMouv/project_info.yaml")

class Behavior:

    """Define the behavior during a trial"""

    def __init__(self, 
                 coords_set: dict[str, Trajectory], 
                 time_pad_off: float, cm_per_pixel: float
                 ) : 
        """
        Return behavior label during trial time:
        press, grasp, open, reach, lift
        """

        self.coords_set = coords_set
        self.time_pad_off = time_pad_off
        self.cm_per_pixel = cm_per_pixel

        finger3_traj: TrajectoryDLC = self.coords_set["finger_3"]
        self.reach_bottom = finger3_traj.coords.loc[finger3_traj.coords["t"] == self.time_pad_off, "y"].item()

        # constants in cm and cartesian plane
        self.LEVER_POS = finger3_traj._point_to_scaled_cartesian(cfg["lever_position"][0], cfg["lever_position"][1])
        self.PAD_POS = finger3_traj._point_to_scaled_cartesian(cfg["pad_position"][0], cfg["pad_position"][0])
        self.FRAME_WIDTH_CM =  cfg["frame_width_px"] * self.cm_per_pixel
        self.FRAME_HEIGHT_CM =  cfg["frame_height_px"] * self.cm_per_pixel

        # features building
        self.centroid = Trajectory(self.compute_centroid())
        self.features = self.compute_centroid()              # initilisation with centroid
        self.features["area"] = self.area_triangle()
        self.features["centroid_direction"] = self.centroid.angle()
        self.features["centroid_velocity"] = self.centroid.instant_velocity()
        self.features["centroid_speed"] = self.centroid.instant_speed()
        self.features["centroid_acc"] = self.centroid.signed_acceleration()
        self.features["f3_sp_distance"] = self.bodypart_distance()
        self.features["f3_sp_angle"] = self.bodypart_angle()
        self.features["centroid_lever_dist"] = self.centroid.obj_bodypart_distance(obj_coord=self.LEVER_POS)

        # spatial zones based on centroid position 
        self.lever_right = 40 * self.cm_per_pixel
        self.lever_upper = 10 * self.cm_per_pixel
        self.lever_lower = 15 * self.cm_per_pixel
        self.spatial_boxes = {
                "reach": (
                    self.LEVER_POS[0] + self.lever_right,
                    self.FRAME_WIDTH_CM,
                    self.PAD_POS[0],
                    self.reach_bottom,
                    "yellow"
                ),
                "open": (
                    0,
                    self.FRAME_WIDTH_CM,
                    self.LEVER_POS[0] + self.lever_right,
                    self.LEVER_POS[1] + self.lever_upper,
                    "orange"
                ),
                "grasp": (
                    0,
                    self.LEVER_POS[1] + self.lever_upper,
                    self.LEVER_POS[0] + self.lever_right,
                    self.LEVER_POS[1] - self.lever_lower,
                    "blue"
                ),
                "press": (
                    0,
                    self.LEVER_POS[1] - self.lever_lower,
                    self.LEVER_POS[0] + self.lever_right,
                    self.LEVER_POS[1] - (self.lever_lower * 2.5),
                    "green"
                ),
            }
        
        self.features["spatial_behavior"] = self.classify_spatial_trajectory(self.centroid.coords)


    def bodypart_angle(self, bp1: str = "soft_pad", bp2: str = "finger_3", 
                               coords_set: dict[str, Trajectory] | None = None) -> np.ndarray:
        """Compute the orientation angle (in degrees) of the segment
        going from bp1 -> bp2, relative to the horizontal axis.
        """
        if coords_set is None:
            coords_set = self.coords_set

        bodypart1 = coords_set[bp1].coords
        bodypart2 = coords_set[bp2].coords

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

        bodypart1 = coords_set[bp1].coords
        bodypart2 = coords_set[bp2].coords

        dx = bodypart2["x"].to_numpy() - bodypart1["x"].to_numpy()
        dy = bodypart2["y"].to_numpy() - bodypart1["y"].to_numpy()

        return np.sqrt(dx**2 + dy**2)

    def compute_centroid(self, bp1: str = "soft_pad", bp2: str = "finger_3", bp3: str = "finger_2", 
                 coords_set: dict[str, Trajectory] = None) -> pd.DataFrame: 
        """Compute the position of a centroid point between 3 bodyparts"""
        if coords_set is None:
            coords_set = self.coords_set
        
        bodypart1 = coords_set[bp1].coords
        bodypart2 = coords_set[bp2].coords
        bodypart3 = coords_set[bp3].coords

        # print(len(bodypart1), len(bodypart2), len(bodypart3), len(bodypart1["t"]))


        centroid_x = (bodypart1["x"] + bodypart2["x"] + bodypart3["x"]) / 3
        centroid_y = (bodypart1["y"] + bodypart2["y"] + bodypart3["y"]) / 3

        return pd.DataFrame({
            "x": centroid_x, 
            "y": centroid_y, 
            "t": bodypart1["t"]
        })


    def area_triangle(self, bp1: str = "soft_pad", bp2: str = "finger_3", bp3: str = "finger_2", 
                 coords_set: dict[str, Trajectory] = None) -> np.array:
        """Return area accros time, between 3 bodyparts
        Based on this tuto : https://www.geeksforgeeks.org/python/python-program-to-calculate-the-area-of-a-triangle/"""
        if coords_set is None:
            coords_set = self.coords_set

        c1 = coords_set[bp1].coords
        c2 = coords_set[bp2].coords
        c3 = coords_set[bp3].coords

        return  0.5 * abs(c1["x"]*c2["y"] + c2["x"]*c3["y"] + c3["x"]*c1["y"] - c1["y"]*c2["x"] - c2["y"]*c3["x"] - c3["y"]*c1["x"])



    # def build_adjusted_boxes(self):
    #     bp_angle: np.array = self.bodypart_angle()
    #     bp_distance: np.array = self.bodypart_distance()
    #     finger3_traj: Trajectory = self.coords_set["finger_3"]
    #     softpad_traj: Trajectory = self.coords_set["soft_pad"]
    #     t = self.coords_set["finger_3"].coords["t"]

    #     fig, ax = plt.subplots(2, 1, figsize=[8, 11])
    #     ax[0] = finger3_traj.plot_traj(finger3_traj.coords, ax[0], fig, "red")
    #     ax[0] = softpad_traj.plot_traj(softpad_traj.coords, ax[0], fig, "orange")
    #     # self.coords_set["finger_3"].show_traj(self.coords_set["finger_3"].coords)
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


    ########################  methods for spatial classification  ######################""

    def _contains(self, name: str, x: float, y: float) -> bool: 
        """Return if the coordinates are in the boxe"""
        xmin, ymin, xmax, ymax, _ = self.spatial_boxes[name]
        return xmin <= x <= xmax and ymax <= y <= ymin
        

    def classify_spatial_behavior(self, x: float, y: float) -> str:
        behavior = "none"
        for name in self.spatial_boxes:
            if self._contains(name, x, y):
                behavior = name

        return behavior
    
    def classify_spatial_trajectory(self, coords: pd.DataFrame) -> pd.DataFrame :
        labels = []

        for i, row in enumerate(coords.itertuples(index=False)):
            x, y, t = row.x, row.y, row.t
            labels.append(self.classify_spatial_behavior(x, y))

        return labels

    def draw_boxes(self, ax) :
        from matplotlib.patches import Rectangle
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

        ax.scatter(self.LEVER_POS[0],self.LEVER_POS[1], color = "red")




## function for behavior clustering



def HMM(data: pd.DataFrame, feature_cols: list[str], n_state: int) -> pd.DataFrame:
    """Run HMM on data (must only contain qualitative data) with n_state
    Returns the same data dataframe, with the column 'pred_state' containing the n states 
    predicted by the model""" 

    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(data[feature_cols].to_numpy())
    lengths = data.groupby("name", sort=True).size().to_numpy()   # frames per trial
    assert lengths.sum() == len(X)

    # several random starts, keep the best (EM finds local optima)
    best, best_score = None, -np.inf
    for seed in range(5):
        m = GaussianHMM(n_components=n_state, covariance_type="diag",
                        n_iter=200, random_state=seed)
        m.fit(X, lengths)
        score = m.score(X, lengths)

        print(f"    seed: {seed}   score: {score}")
        if score > best_score:
            best, best_score = m, score
    model = best

    data["pred_state"] = model.predict(X, lengths)
    print(np.round(model.transmat_, 2))

    return data




def HMM_exploration(data, analysis_dir, load: Path | None = None): 
    """Run full HMM with variing parameters (n_state, features...)
    init = True run the HMM, set it to False to load precomputed data"""

    for n in range(4, 9): 
        output_dir = analysis_dir / "behavior" / "HMM" / f"{n}_componant"
        save_path = u.make_path(output_dir, "HMM_results.csv")
    
        if load is None : # run full computation

            # rows must be contiguous per trial and ordered in time within a trial
            data = data.sort_values(["name", "relative_t"]).reset_index(drop=True)

            behavior_num = {"none": 0, "reach": 1, "open": 2, "grasp": 3, "press": 4}
            data["spatial_state"] = data["spatial_behavior"].map(behavior_num)

            # # circular variables -> sin/cos (adapt if your angles are in degrees)
            # for col in ["centroid_direction", "f3_sp_angle"]:
            #     data[col + "_sin"] = np.sin(data[col])
            #     data[col + "_cos"] = np.cos(data[col])

            feature_cols = [
                        "relative_t",
                        "x", 
                        "y",
                        "area", 
                        "centroid_direction", 
                        "centroid_velocity", 
                        "centroid_speed", 
                        "centroid_acc", 
                        "f3_sp_distance", 
                        "f3_sp_angle", 
                        "centroid_lever_dist",
                        "spatial_state"
            ]

            data = data.sort_values(["name", "relative_t"]).reset_index(drop=True)

            # 1) the 6 NaNs in x/y: see where they are
            bad = data[data["x"].isna()]
            print(bad[["name", "relative_t",]])
            bad.to_csv("bad_df.csv")

            # 2) drop the remaining NaNs (derivative edges), then recompute lengths
            data = data.dropna(subset=feature_cols).reset_index(drop=True)

            print(f"\nFitting best HMM: n_componant = {n}")
        
            data = HMM(data, feature_cols, n)
            data.to_csv(save_path)

        else :   # load data directly
            data = pd.read_csv(save_path)
            print("Loading:", load)

        print("Display matrix")

        order = ["none", "reach", "open", "grasp", "press"]

        # trials in which a (true, pred) pair occurs at least once
        pairs = data[["name", "spatial_behavior", "pred_state"]].drop_duplicates()
        ct_trials = pd.crosstab(pairs["spatial_behavior"], pairs["pred_state"]).reindex(order)

        # frames, normalised per true phase (rows sum to 1)
        ct_frames = pd.crosstab(data["spatial_behavior"], data["pred_state"],
                                normalize="index").reindex(order)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.heatmap(ct_trials, annot=True, fmt="d", cmap="viridis", ax=axes[0])
        axes[0].set_title("Number of trials (pair occurs ≥ once)")

        sns.heatmap(ct_frames, annot=True, fmt=".2f", cmap="viridis", ax=axes[1])
        axes[1].set_title("Fraction of frames per true phase")
        for ax in axes:
            ax.set_xlabel("Predicted HMM state")
            ax.set_ylabel("Spatial behavior")

        plt.tight_layout()
        plt.savefig(u.make_path(output_dir, "HMM.png"))
        plt.savefig(u.make_path(output_dir, "HMM.svg"))
        plt.close()

        print("Display scatterplot")

        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme("talk", style="ticks", rc=custom_params, palette="pastel")

        BEHAVIOR_PALETTE = {
                "reach": "yellow", 
                "open": "orange", 
                "grasp": "blue", 
                "press": "green",
                "none": "gray"
            }
                    
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        sns.scatterplot(data, ax=axes[0], x="x", y="y", hue="pred_state", palette="bright", alpha=0.5)
        sns.scatterplot(data, ax=axes[1], x="x", y="y", hue="spatial_behavior", palette=BEHAVIOR_PALETTE, alpha=0.5)

        axes[0].set_title("Predicted HMM state")
        axes[1].set_title("Spatial behavior state")

        for ax in axes:
                    ax.set_xlabel("centroid x (cm)")
                    ax.set_ylabel("centroid y (cm)")

        plt.tight_layout()
        plt.savefig(u.make_path(output_dir, "HMM_states_xy.svg"))
        plt.savefig(u.make_path(output_dir, "HMM_states_xy.png"))
        plt.close()

        print("Display countplot")

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        sns.countplot(data, ax=axes[0], y="pred_state", hue="pred_state", palette="bright")
        sns.countplot(data, ax=axes[1], y="spatial_behavior", hue="spatial_behavior", palette=BEHAVIOR_PALETTE)

        axes[0].set_title("Predicted HMM state")
        axes[1].set_title("Spatial behavior state")

        for ax in axes:
                    ax.set_xlabel("states")
                    ax.set_ylabel("count")

        plt.tight_layout()
        plt.savefig(u.make_path(output_dir, "HMM_states_counts.svg"))
        plt.close()



############# KEYPOINTS MOSEQ



def find_Kappa(data, pca, config, project_dir, metadata): 
    import keypoint_moseq as kpms

     
    kappas = np.logspace(3,7,5)
    decrease_kappa_factor = 10
    num_ar_iters = 50
    num_full_iters = 200

    prefix = 'my_kappa_scan'

    for kappa in kappas:
        print(f"Fitting model with kappa={kappa}")
        model_name = f'{prefix}-{kappa}'
        model = kpms.init_model(data, pca=pca, **config())

        # stage 1: fit the model with AR only
        model = kpms.update_hypparams(model, kappa=kappa)
        model = kpms.fit_model(
            model,
            data,
            metadata,
            project_dir,
            model_name,
            ar_only=True,
            num_iters=num_ar_iters,
            save_every_n_iters=25
        )[0];

        # stage 2: fit the full model
        model = kpms.update_hypparams(model, kappa=kappa/decrease_kappa_factor)
        kpms.fit_model(
            model,
            data,
            metadata,
            project_dir,
            model_name,
            ar_only=False,
            start_iter=num_ar_iters,
            num_iters=num_full_iters,
            save_every_n_iters=25
        );

    kpms.plot_kappa_scan(kappas, project_dir, prefix)

        
def Keypoint_MoSeq_exploration(project_dir, dlc_dir, raw_video_dir, ): 
    import keypoint_moseq as kpms

    config = lambda: kpms.load_config(project_dir)

    dlc_config = dlc_dir / "config.yaml"
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config)


    kpms.update_config(
        project_dir,
        video_dir=raw_video_dir,
        anterior_bodyparts=["nose"],
        posterior_bodyparts=["spine4"],
        use_bodyparts=["spine4", "spine3", "spine2", "spine1", "head", "nose", "right ear", "left ear"],
        fps=30,
    )


    # load data (e.g. from DeepLabCut)
    keypoint_data_path = "dlc_project/videos/"  # can be a file, a directory, or a list of files
    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, "deeplabcut")

    kpms.update_config(project_dir, outlier_scale_factor=6.0)

    coordinates, confidences = kpms.outlier_removal(
        coordinates,
        confidences,
        project_dir,
        overwrite=False,
        **config()
    )

    # format data
    data, metadata = kpms.format_data(coordinates, confidences, **config())

    # run pca
    pca = kpms.fit_pca(**data, **config())
    kpms.save_pca(pca, project_dir)

    kpms.print_dims_to_explain_variance(pca, 0.9)
    kpms.plot_scree(pca, project_dir=project_dir)
    kpms.plot_pcs(pca, project_dir=project_dir, **config())

    # use the following to load an already fit model
    # pca = kpms.load_pca(project_dir)

    kpms.update_config(project_dir, latent_dim=4)
    kpms.update_config(
        project_dir,
        sigmasq_loc=kpms.estimate_sigmasq_loc(data["Y"], data["mask"], filter_size=config()["fps"])
    )

    # init model with modified KAPPA
    KAPPA = find_Kappa(data, pca, config, project_dir, metadata)

    model = kpms.init_model(data, pca=pca, **config(), 
                            location_aware=True # setting when spatiality is important
                            )  
    model = kpms.update_hypparams(model, kappa=KAPPA)

    # fitting of the AR-HMM
    num_ar_iters = 50

    model, model_name = kpms.fit_model(model, data, metadata, 
                                        project_dir, ar_only=True, num_iters=num_ar_iters,
                                        location_aware=True # setting when spatiality is important
                                        )

    # load model checkpoint
    model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name, iteration=num_ar_iters)
    model = kpms.update_hypparams(model, kappa=KAPPA)

    # run fitting for an additional 500 iters
    model = kpms.fit_model(model, data, metadata, 
                            project_dir, model_name, ar_only=False, 
                            start_iter=current_iter, num_iters=current_iter + 500,
                            location_aware=True # setting when spatiality is important
                            )[0]

    # extract model results
    # load the most recent model checkpoint
    model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)

    # extract results
    results = kpms.extract_results(model, metadata, project_dir, model_name)
    kpms.save_results_as_csv(results, project_dir, model_name)


    # DISPLAY RESULTS

    kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config())

    kpms.generate_grid_movies(results, project_dir, model_name, coordinates=coordinates, **config())

    kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, **config())