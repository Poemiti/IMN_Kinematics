# src/BetaMouv/TrialGroup.py


from src.Base.TrialGroup import TrialGroup as BaseTrialGroup
from .Trial import Trial
from .Trajectory import Trajectory

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("talk", style="ticks", rc=custom_params, palette="pastel")

BOOL_PALETTE = {True: "yellowgreen", False: "tomato"}


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
        self._bodypart_success_df = None

        self.crop_laser_period = False


    def crop_coords(self, bool_val: bool): 
        self.crop_laser_period = bool_val
        print("Coordinates cropped from pad_off to end of laser stimulation !")


    def buils_timeseries_df(self, save_as, bodypart: str = "finger_3", init: bool = False):
        if init: 
            self._timeseries_df = self.timeseries_df(bodypart)
            self._timeseries_df.to_csv(save_as)
            return 
              
        if save_as.exists() : 
            print(f"Loading timeseries_df from : '{save_as}'")
            self._timeseries_df = pd.read_csv(save_as)
            return

        self._timeseries_df = self.timeseries_df(bodypart)
        self._timeseries_df.to_csv(save_as)


    def success_df(self) -> pd.DataFrame:
        """One row per trial, including failures — for QC / yield figures
        (e.g. success rate by condition, why trials were excluded)."""

        if self._success_df is None:
            records = []

            for trial in self.trials:

                for stage, outcome in trial.trial_outcomes.items():
                    records.append(trial.identity() | {
                            "stage": stage,
                            "success": outcome.success,
                            "reason": outcome.reason,
                        })

            self._success_df = pd.DataFrame(records)

        return self._success_df


    def bodypart_success_df(self) -> pd.DataFrame:
        """One row per (trial, bodypart, stage) — for per-bodypart QC figures."""

        if self._bodypart_success_df is None:
            records = []
            for trial in self.trials:
                for bodypart, traj in trial.trajectories.items():
                    for stage, outcome in traj.stage_outcomes.items():

                        records.append(trial.identity() | {
                            "bodypart": bodypart,
                            "stage": stage,
                            "success": outcome.success,
                            "reason": outcome.reason,
                        })
                        
            self._bodypart_success_df = pd.DataFrame(records)
        return self._bodypart_success_df


    def scalar_df(self, metrics: list[str] = None) -> pd.DataFrame:
        """One row per trial — for distributions/scatter (avg velocity, tortuosity, ...)."""

        if self._scalar_df is None:
            records = []

            for trial in self.trials:
                if not trial.is_valid() :
                    continue

                records.append(trial.identity() | {
                    f: getattr(trial, f, None) for f in (metrics or trial.SCALAR_METRIC_FIELDS)
                })

            self._scalar_df = pd.DataFrame(records)

        return self._scalar_df


    def timeseries_df(self, bodypart: str = "finger_3") -> pd.DataFrame:
        """Many rows per trial (one per frame/timestamp) : for plots over time."""

        if self._timeseries_df is None:
            frames = []

            for trial in tqdm(self.trials, desc="Building timeseries_df"):
                if not trial.is_valid() :
                    continue

                traj: Trajectory = trial.trajectories.get(bodypart)

                if not traj.is_valid(): 
                    continue

                df = traj.clean_coords

                if self.crop_laser_period: 
                    df = traj.crop_xy(traj.clean_coords, trial.time_pad_off - 0.1, trial.time_pad_off + 0.4)

                for col, val in trial.identity().items():
                    df[col] = val

                df = df.loc[df["laser_intensity"] != "incompatible"]
                df["index_order"] = range(len(df))
                df["relative_t"] = (df["t"] - trial.time_pad_off).round(2)

                frames.append(df)

            self._timeseries_df = pd.concat(frames, ignore_index=True)

        return self._timeseries_df
    
    

    
    ####################### plotting methods ###########################

    def trial_success_rate(self, save_as):
        """Display success rate and failure reasons per stage, and save the report."""

        df = self.success_df()

        g = sns.catplot(
            data=df, kind="count",
            x="reason",
            col="stage", hue="success", palette=BOOL_PALETTE,
            sharex=False,
        )
        
        g.set_xticklabels(rotation=45, ha="right")
        g.set_axis_labels("", "Number of trials")
        g.set_titles(col_template="{col_name}")
        g.figure.suptitle("Trial success rate by stage")
        g.figure.subplots_adjust(top=0.80)

        # annotate each bar with its % of that stage's total trial count
        for stage, ax in zip(g.col_names, g.axes.flat):
            stage_total = (df["stage"] == stage).sum()
            if stage_total == 0:
                continue

            for bar in ax.patches:
                height = bar.get_height()
                if height <= 0:
                    continue

                rate = 100 * height / stage_total
                ax.annotate(
                    f"{rate:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=12
                )

        g.figure.savefig(save_as, bbox_inches="tight")
        plt.show()
        plt.close(g.figure)

        print(f"Report saved to: {save_as}")


    def trajectories_success_rate(self, save_as):

        df = self.bodypart_success_df()

        g = sns.catplot(
            data=df, kind="count",
            x="reason",
            col="stage", row="bodypart",
            hue="success", palette=BOOL_PALETTE,
            sharex=False
        )
        
        g.set_xticklabels(rotation=45, ha="right")
        g.set_axis_labels("", "Number of trials")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.figure.suptitle("Trajectory success rate by stage")
        g.figure.subplots_adjust(top=0.80)

        g.figure.savefig(save_as, bbox_inches="tight")
        plt.show()
        plt.close(g.figure)

        print(f"Report saved to: {save_as}")


    def lineplot_all_traj(self, save_as, bodypart: str = "finger_3"):
        data = self.timeseries_df(bodypart)

        fig, ax = plt.subplots()

        # Individual trajectories
        sns.lineplot(
            data=data,
            x="x", y="y",
            units="name",
            estimator=None,
            sort=False,
            color="gray",
            alpha=0.2,
            legend=False, ax=ax
        )

        XL, YL = (55, 230)   # lever position (px)
        XP, YP = (315, 348)   # pad position (px)
        
        plt.plot()

        mean_data =(data
                    .groupby("index_order", as_index=False)[["x", "y"]]
                    .mean())

        sns.lineplot(
            data=mean_data,
            x="x", y="y",
            color="red",
            linewidth=3,
            sort=False,
            label="Mean trajectory", estimator=None
        )

        plt.title(f"{self.group_name}\n bp={bodypart}, n_trial={data['name'].nunique()}")
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


    def plot_tendency(self, value , save_as, bodypart: str = "finger_3"): 
        """
        Plot velocity tendency with error span around the average.
        Custome error functions available : 
            - "sem" : Standard Error of the Mean (https://statisticsbyjim.com/hypothesis-testing/standard-error-mean/)
            How to interprete SEM : 'For a SEM of 3, we know that the typical 
            difference between a sample mean and the population mean is 3'
        """
        from scipy.stats import sem

        data = self._timeseries_df

        if data is None: 
            data = self.timeseries_df(bodypart)            

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
            errorbar = "se",  # SEM
        )

        g.add_legend()

        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.set_axis_labels("Time (sec)", value)
        g.figure.suptitle(f"{value} over time\n bodypart= {bodypart}, n trials: {len(data.groupby('name'))}", ha='center')
        g.figure.subplots_adjust(top=0.8)

        g.savefig(save_as)
        
        plt.show()
        plt.close()
