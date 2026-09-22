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

        self.crop_laser_period = False


    def crop_coords(self, bool_val: bool): 
        self.crop_laser_period = bool_val
        print("Coordinates cropped from pad_off to end of laser stimulation !")

    def success_df(self) -> pd.DataFrame:
        """One row per trial, including failures — for QC / yield figures
        (e.g. success rate by condition, why trials were excluded)."""

        if self._success_df is None:
            records = []

            for trial in self.trials:
                records.append(trial.identity() | {
                    "task_success": trial.task_success ,
                    "task_success_reason": trial.task_success_reason ,
                    "coords_success": trial.coords_success,
                    "coords_success_reason": trial.coords_success_reason,
                    "validation_success": trial.validation_success,
                    "validation_success_reason": trial.validation_success_reason,
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


    def timeseries_df(self, value_cols: list[str] = ["x", "y", "instant_velocity"]) -> pd.DataFrame:
        """Many rows per trial (one per frame/timestamp) : for plots over time."""

        if self._timeseries_df is None:
            frames = []

            for trial in tqdm(self.trials, desc="Building timeseries_df"):
                if not trial._is_successful() :
                    continue

                df = trial.coords[["t", *value_cols]].copy()
                if self.crop_laser_period: 
                    df = trial.traj.crop_xy(df, trial.time_pad_off, trial.time_pad_off + 0.4)

                for col, val in trial.identity().items():
                    df[col] = val
                frames.append(df)

            self._timeseries_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        return self._timeseries_df
    
    

    
    ####################### plotting methods ###########################

    def success_rate_report(self, save_as):
        """Display success rates and failure reasons, and save the report."""

        df = self.success_df()

        n_trials = len(df)

        if n_trials == 0:
            report = "No trials found.\n"
            print(report)
            with open(save_as, "w") as f:
                f.write(report)
            return

        # Store everything that will be printed
        report_lines = []

        report_lines.append(f"Total trials: {n_trials}\n")

        # Success rates
        success_columns = [
            "task_success",
            "coords_success",
            "validation_success",
        ]

        report_lines.append("SUCCESS RATES")
        report_lines.append("-" * 40)

        for column in success_columns:
            # Treat None/NaN as False
            success = df[column].fillna(False).astype(bool)

            n_success = success.sum()
            rate = n_success / n_trials * 100

            report_lines.append(
                f"{column:20s}: "
                f"{n_success:4d}/{n_trials:<4d} "
                f"({rate:5.1f}%)"
            )

        # Failure reasons
        reason_columns = [
            ("task_success", "task_success_reason"),
            ("coords_success", "coords_success_reason"),
            ("validation_success", "validation_success_reason"),
        ]

        for success_col, reason_col in reason_columns:

            failures = df[reason_col].dropna()

            report_lines.append(f"\n{reason_col}")
            report_lines.append("-" * 40)

            if failures.empty:
                report_lines.append("No failures")
                continue

            counts = failures.value_counts()

            for reason, count in counts.items():
                percentage = count / n_trials * 100

                report_lines.append(
                    f"{str(reason):30s}: "
                    f"{count:4d} "
                    f"({percentage:5.1f}%)"
                )

        report = "\n".join(report_lines) + "\n"
        print(report)

        with open(save_as, "w") as f:
            f.write(report)

        print(f"Report saved to: {save_as}")



    def lineplot_all_traj(self, save_as):
        data = self.timeseries_df(value_cols=["x", "y"])
        data = data.sort_values(["name", "t"])

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

        data["index_order"] = data.groupby("name").cumcount()
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
        data["index_order"] = data.groupby("name").cumcount()

        # align to pad off
        data["relative_t"] = data["index_order"] * 0.08

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
