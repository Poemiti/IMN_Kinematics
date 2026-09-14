# src/BetaMouv/TrialGroup.py


from src.Base.TrialGroup import TrialGroup as BaseTrialGroup
from .Trial import Trial
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


    def timeseries_df(self, value_cols: list[str] = ["x", "y", "instant_velocity"]) -> pd.DataFrame:
        """Many rows per trial (one per frame/timestamp) : for plots over time."""

        if self._timeseries_df is None:
            frames = []

            for trial in self.trials:
                if not trial._is_successful() :
                    continue

                df = trial.coords[["t", *value_cols]].copy()
                for col, val in trial.identity().items():
                    df[col] = val
                frames.append(df)

                if len(df) < 375: 
                    print( trial.name)

            self._timeseries_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        return self._timeseries_df
    
    

    
    ####################### plotting methods ###########################


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
