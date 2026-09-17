# src/BetaMouv/TrialGroup.py


from src.Base.TrialGroup import TrialGroup as BaseTrialGroup
from .Trial import Trial
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("talk", style="ticks", rc=custom_params, palette="pastel")

LASER_STATE_PALETTE = {
    "NOstim": "slategray",
    "LaserOff" :"slategray",
    "LaserOn" : "tomato"
}

LASER_STATE_DASH = {
    "LaserOff" : (4,2),
    "LaserOn" : ""
}

LASER_PERIOD_COLOR = "royalblue"


class TrialGroup(BaseTrialGroup): 

    trial_cls = Trial

    def __init__(self, filenames, condition):
        super().__init__(filenames, condition)

        self._success_df = None
        self._scalar_df = None
        self._timeseries_df = None


    def success_df(self) -> pd.DataFrame:
        """One row per trial, including failures — for QC / yield figures
        (e.g. success rate by condition, why trials were excluded)."""

        if self._success_df is None:
            records = []

            for trial in self.trials:
                records.append(trial.identity() | {
                    "task_success": getattr(trial, "task_success", None),
                    "task_success_reason": getattr(trial, "task_success_reason", None),
                    "model_success": getattr(trial, "model_success", None),
                    "model_success_reason": getattr(trial, "model_success_reason", None),
                })

            self._success_df = pd.DataFrame(records)

        return self._success_df


    def scalar_df(self, metrics: list[str] = None) -> pd.DataFrame:
        """One row per trial — for distributions/scatter (avg velocity, tortuosity, ...)."""

        if self._scalar_df is None:
            records = []

            for trial in self.trials:
                if not trial._is_successful() :
                    continue

                records.append(trial.identity() | {
                    f: getattr(trial, f, None) for f in (metrics or trial.SCALAR_METRIC_FIELDS)
                })

            self._scalar_df = pd.DataFrame(records)

        return self._scalar_df


    def timeseries_df(self, value_cols: list[str] = ["x", "y", "instant_velocity"],
                      crop_laser_period: bool = False) -> pd.DataFrame:
        """Many rows per trial (one per frame/timestamp) : for plots over time."""

        if self._timeseries_df is None:
            frames = []

            for trial in tqdm(self.trials, desc="Building timeseries_df"):
                if not trial._is_successful() :
                    continue

                df = trial.coords[["t", *value_cols]].copy()
                if crop_laser_period: 
                    df = df[(df["t"] >= trial.time_pad_off) &
                            (df["t"] <= trial.time_pad_off + 0.325)]

                for col, val in trial.identity().items():
                    df[col] = val
                frames.append(df)

                # if len(df) < 375: 
                #     print( trial.name, len(df))

            self._timeseries_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        return self._timeseries_df
    
    

    
    ####################### plotting methods ###########################

    def lineplot_all_traj(self, save_as):
        data = self.timeseries_df(value_cols=["x", "y"], crop_laser_period=True)

        # order traj in time + crop around pad off
        data = data.sort_values(["name", "t"])
        print(data)

        plt.figure(figsize=(8, 8))

        # Individual trajectories
        sns.lineplot(
            data=data,
            x="x",
            y="y",
            units="name",
            estimator=None,
            sort=False,
            color="gray",
            alpha=0.2,
            linewidth=1,
            legend=False,
        )

        # Mean trajectory
        mean_data = (
            data.groupby("t", as_index=False)[["x", "y"]]
            .mean()
            .sort_values("t")
        )

        sns.lineplot(
            data=mean_data,
            x="x",
            y="y",
            color="red",
            linewidth=3,
            sort=False,
            label="Mean trajectory",
        )

        plt.title(f"{self.group_name}\n n_trial={data['name'].nunique()}")
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.axis("equal")
        plt.legend()

        plt.savefig(save_as, bbox_inches="tight", dpi=300)
        plt.show()
        plt.close()



    def distri_outlier(self, data: pd.DataFrame, save_as):
        """Visualize the distribution of outliers."""

        n_trial = data["trial"].nunique()

        # Create a categorical variable separating zero from non-zero
        plot_data = data.copy()
        plot_data["outlier_status"] = np.where(
                plot_data["n_outlier"] == 0,
                "No outlier",
                "≥1 outlier"
            )

        fig, axes = plt.subplots(
            1, 2,
            figsize=(12, 5),
            gridspec_kw={"width_ratios": [1, 2]}
        )

        # 1. Zero vs non-zero outliers
        sns.countplot(
            data=plot_data,
            x="outlier_status",
            hue="type",
            ax=axes[0], legend=False
        )

        axes[0].set_title("Trials with / without outliers")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Number of trials")

        # 2. Distribution among trials with outliers
        non_zero = plot_data[plot_data["n_outlier"] > 0]

        sns.histplot(
            data=non_zero,
            x="n_outlier",
            hue="type",
            discrete=True,
            multiple="dodge",
            shrink=0.8,
            ax=axes[1]
        )

        axes[1].set_title("Number of outliers\n(trials with ≥1 outlier)")
        axes[1].set_xlabel("Number of outliers")
        axes[1].set_ylabel("Number of trials")

        fig.suptitle(
            f"Outlier distribution - {self.group_name}\n"
            f"n_trial={n_trial}")

        plt.tight_layout()
        plt.savefig(save_as, bbox_inches="tight", dpi=300)
        plt.show()
        plt.close()


    def plot_tendency(self, value , save_as): 
        """
        Plot velocity tendency with error span around the average.
        Custome error functions available : 
            - "sem" : Standard Error of the Mean (https://statisticsbyjim.com/hypothesis-testing/standard-error-mean/)
            How to interprete SEM : 'For a SEM of 3, we know that the typical 
            difference between a sample mean and the population mean is 3'
        """
        from scipy.stats import sem

        data = self.timeseries_df(value_cols=[value])

        # align to pad off
        data["relative_t"] = data["t"] - data["time_pad_off"]

        g = sns.FacetGrid(
            data=data,
            margin_titles=True,
            col="laser_type", 
            row="laser_intensity", 
            height=6
            )

        g.map_dataframe(
            sns.lineplot,
            x="relative_t", y=value, 
            hue="laser_state", style="laser_state" ,
            palette=LASER_STATE_PALETTE, dashes=LASER_STATE_DASH,
            estimator="mean",
            # errorbar = ("pi", 50),
        )

        for row_i, laser_intensity in enumerate(g.row_names):
            for col_j, laser_type in enumerate(g.col_names):

                ax = g.axes[row_i, col_j]

                # ax.set_ylim(0, 3)
                # ax.set_xlim(-0.2, 1.5)

                # Vertical line at pad off
                ax.axvline(
                    0,
                    color="k",
                    alpha=0.5,
                    lw=0.8,
                    ls="--",
                )

                # Laser period annotation
                y = ax.get_ylim()[1] * 0.95

                ax.hlines(
                    y=y,
                    xmin=0.025,
                    xmax=0.325,
                    color=LASER_PERIOD_COLOR,
                    linewidth=3
                )

                ax.text(
                    0.175,
                    y,
                    "Laser period",
                    ha="center",
                    va="bottom",
                    color=LASER_PERIOD_COLOR,
                    fontsize=10
                )

                # Manual SEM shading
                # subset data for this facet
                facet_df = data[
                    (data["laser_type"] == laser_type) &
                    (data["laser_intensity"] == laser_intensity)
                ]

                # do SEM separately for each hue group
                for laser_state, sub in facet_df.groupby("laser_state"):

                    grouped = (
                        sub.groupby("relative_t")[value]
                        .agg(
                            mean="mean",
                            sem=lambda x: sem(x, nan_policy="omit")
                        )
                        .reset_index()
                        .sort_values("relative_t")
                    )

                    lower = grouped["mean"] - grouped["sem"]
                    upper = grouped["mean"] + grouped["sem"]

                    color = LASER_STATE_PALETTE[laser_state]

                    ax.fill_between(
                        grouped["relative_t"].values,
                        lower.values,
                        upper.values,
                        color=color,
                        alpha=0.20
                    )

        g.add_legend()

        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.set_axis_labels("Time (sec)", value)
        g.figure.suptitle(f"{value[1]} over time\nNumber of trials: {len(data.groupby('name'))}", ha='center')
        g.figure.subplots_adjust(top=0.8)

        g.savefig(save_as)
        
        plt.show()
        plt.close()
