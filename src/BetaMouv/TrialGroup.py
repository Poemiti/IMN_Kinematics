# src/BetaMouv/TrialGroup.py


from src.Base.TrialGroup import TrialGroup as BaseTrialGroup
from .Trial import Trial
from .Trajectory import TrajectoryDLC

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
from pathlib import Path


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

LASER_STATE_HATCH = {          
    "LaserOff": "oo",
    "LaserOn": "",
}

LASER_PERIOD_COLOR = "royalblue"

BEHAVIOR_PALETTE = {
    "reach": "yellow", 
    "open": "orange", 
    "grasp": "blue", 
    "press": "green"
}




class TrialGroup(BaseTrialGroup): 

    trial_cls = Trial

    def __init__(self, filenames, condition):
        super().__init__(filenames, condition)

        self._success_df = None
        self._scalar_df = None
        self._timeseries_df = None
        self._bodypart_success_df = None
        self._behavior_features_df = None

        self.crop_laser_period = False
        self.remove_NOstim = False

    def buils_timeseries_df(self, save_as, bodypart: str = "finger_3", init: bool = True):
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
                            "is_valid": "valid" if trial.is_valid() else "not_valid",
                            "reason": trial.failure()                        
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

                if self.remove_NOstim: 
                    if trial.laser_type == "NOstim": 
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
                if not trial.is_valid() or trial.laser_intensity == "incompatible":
                    continue

                traj: TrajectoryDLC = trial.trajectories.get(bodypart)

                if not traj.is_valid(): 
                    continue

                if self.remove_NOstim: 
                    if trial.laser_type == "NOstim": 
                        continue

                df = traj.clean_coords

                if self.crop_laser_period: 
                    df = traj.crop_xy(traj.clean_coords, trial.time_pad_off - 0.1, trial.time_pad_off + 0.4)

                for col, val in trial.identity().items():
                    df[col] = [val] * len(df)

                df["index_order"] = range(len(df))
                df["relative_t"] = (df["t"] - trial.time_pad_off).round(3)

                frames.append(df)

            self._timeseries_df = pd.concat(frames, ignore_index=True)

        return self._timeseries_df


    def behavior_features_df(self): 
        if self._behavior_features_df is None:
            frames = []

            for trial in tqdm(self.trials, desc="Building behavior_features_df"):
                if not trial.is_valid() or trial.laser_intensity == "incompatible":
                    continue

                if trial.behavior is None: 
                    continue
 
                if self.remove_NOstim: 
                    if trial.laser_type == "NOstim": 
                        continue

                df = trial.behavior.features
                traj = trial.trajectories["finger_3"]

                if self.crop_laser_period: 
                    df = traj.crop_xy(df, trial.time_pad_off - 0.1, trial.time_pad_off + 0.4)

                for col, val in trial.identity().items():
                    df[col] = [val] * len(df)

                df["index_order"] = range(len(df))
                df["relative_t"] = (df["t"] - trial.time_pad_off).round(3)

                frames.append(df)

            self._behavior_features_df = pd.concat(frames, ignore_index=True)

        return self._behavior_features_df
    

    
    ####################### plotting methods ###########################

    def trial_success_rate(self, save_as, show: bool = False): 
        return self.success_rate(self.success_df(), None, save_as, show)

    def trajectories_success_rate(self, save_as, show: bool = False): 
            return self.success_rate(self.bodypart_success_df(), "bodypart", save_as, show)


    def success_rate(self, df, col, save_as, show: bool = False):
        """Display success rate and failure reasons per stage, and save the report."""

        g = sns.catplot(
            data=df, kind="count",
            y="reason",
            row="stage", col=col,
            hue="success", palette=BOOL_PALETTE,
            sharey=False, height=4, aspect=3,
        )
        
        # g.set_xticklabels(rotation=45, ha="right")
        g.set_axis_labels("Number of trials", "")
        g.set_titles(row_template="{row_name}")
        g.figure.suptitle(f"Success rate by stage - n_trials: {len(self.trials)}\n{self.group_name}")
        g.figure.subplots_adjust(top=0.90)

        # Annotate each bar with its % of that stage's total trial count
        for stage, ax in zip(g.row_names, g.axes.flat):
            stage_total = (df["stage"] == stage).sum()

            if stage_total == 0:
                continue

            for bar in ax.patches:
                width = bar.get_width()

                if width <= 0:
                    continue

                rate = 100 * width / stage_total

                ax.annotate(
                    f"{rate:.1f}%",
                    xy=(
                        width,
                        bar.get_y() + bar.get_height() / 2,
                    ),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=12,
                )

        g.figure.savefig(save_as, bbox_inches="tight")
        if show: 
            plt.show()
        plt.close(g.figure)

        print(f"Report saved to: {save_as}")


    def lineplot_traj(self, save_as, bodypart: str = "finger_3", Beha = None): 
        data = self.timeseries_df(bodypart)
        return self._lineplot_traj(data=data, save_as=save_as, Beha=Beha,
                    title=f"{self.group_name}\n bp={bodypart}, n_trial={data['name'].nunique()}")


    def lineplot_traj_per_identity(self, save_as, bodypart: str = "finger_3"): 
        data = self.timeseries_df(bodypart)
        return self._lineplot_traj_per_indentity(data=data, save_as=save_as,
                    title=f"{self.group_name}\n bp={bodypart}, n_trial={data['name'].nunique()}")

 
    def lineplot_traj_centroid(self, save_as, Beha = None): 
        data = self.behavior_features_df()
        return self._lineplot_traj(data=data, save_as=save_as, Beha=Beha,
                    title=f"{self.group_name}\n centroid, n_trial={data['name'].nunique()}")


    def _lineplot_traj(self, data, save_as, title: str, Beha = None):

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
            legend=False, ax=ax, zorder=-1
        )

        mean_data =(data
                    .groupby("index_order", as_index=False)[["x", "y"]]
                    .mean())

        sns.lineplot(
            data=mean_data,
            x="x", y="y",
            color="red",
            linewidth=3,
            sort=False,
            label="Mean trajectory", estimator=None, 
        )

        # plot lever + pad

        XL, YL = (55, 282)   # lever position (px)
        XP, YP = (250, 164)  # pad position (px)
        
        frame_width_px = 512
        frame_width_cm = 8.7
        cm_per_pixel = frame_width_cm / frame_width_px

        ax.scatter(x=XL*cm_per_pixel, y=YL*cm_per_pixel, color="k", zorder=1, lw=5)
        ax.hlines(y=YP*cm_per_pixel, xmin=90*cm_per_pixel, xmax=XP*cm_per_pixel, color="k", lw=5)

        if Beha is not None: 
            Beha.draw_boxes(ax)
            # ax.set_xlim(0, 6)
            ax.set_ylim(2, 6)

        plt.title(title)
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")

        plt.savefig(save_as)
        plt.close()


    def _lineplot_traj_per_indentity(data, save_as, title: str): 

        def plot_mean(data, **kwargs):
            m = (data
                .groupby("index_order", as_index=False)[["x", "y"]]
                .mean()
                .sort_values("index_order"))
            plt.gca().plot(m["x"], m["y"], 
                           color="black", linewidth=4, label="Mean trajectory")


        g = sns.FacetGrid(
                    data=data,
                    margin_titles=True,
                    col="laser_type", 
                    row="laser_intensity", 
                    height=6
                    )

        # Individual trajectories
        g.map_dataframe(sns.lineplot,
            x="x", y="y",
            units="name",
            estimator=None, sort=False,
            alpha=0.6, 
            hue="laser_state", style="laser_state",
            palette=LASER_STATE_PALETTE, dashes=LASER_STATE_DASH,
            legend=False,
        )

        # Mean trajectories 
        g.map_dataframe(plot_mean)

        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.figure.suptitle(title)
        g.set_axis_labels("x (cm)", "y (cm)")
        g.figure.subplots_adjust(top=0.90)

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




    def plot_tendency_beha_features(self, save_as: str, x="t", y="area", show_units: bool = False, show_beha: str = "spatial_behavior"): 
        data = self.behavior_features_df()
        data.to_csv("data.csv")
        return self.plot_tendency(data=data, save_as=save_as, x=x, y=y, show_units=show_units, show_beha=show_beha,
                                  title=f"{y} - n trials: {len(data.groupby('name'))}",)

    def plot_tendency_metrics(self, value , save_as, bodypart: str = "finger_3", show_units: bool=False, show_beha: str = "spatial_behavior"): 
        data = self.timeseries_df(bodypart)
        return self.plot_tendency(data=data, save_as=save_as, x='relative_t', y=value, show_units=show_units, show_beha=show_beha,
                                  title=f"{value} over time\n bodypart= {bodypart}, n trials: {len(data.groupby('name'))}",)

    def plot_tendency(self, data, title, save_as: str, x="t", y="area", show_units: bool = False, show_beha: str="spatial_behavior"):


        g = sns.FacetGrid(
            data=data,
            margin_titles=True,
            col="laser_type", 
            row="laser_intensity", 
            height=6
            )

        g.map_dataframe(
            sns.lineplot,
            x=x, y=y, 
            hue="laser_state", style="laser_state" ,
            palette=LASER_STATE_PALETTE, dashes=LASER_STATE_DASH,
            estimator="mean",
            errorbar = "se",  # SEM
        )

        if show_units:
            g.map_dataframe(
                sns.lineplot,
                x=x, y=y, 
                units="name",
                hue="laser_state", style="laser_state" , linewidth=0.5,
                palette=LASER_STATE_PALETTE, 
                alpha=0.3, estimator=None, sort=None,
            )

        if show_beha: 
            self.draw_behavior_proportion(data, x, g, show_beha)

        self.add_trial_count_to_facet(data, g, hue_colors=LASER_STATE_PALETTE)

        g.add_legend()
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.set_axis_labels(x, y)
        g.figure.suptitle(title,  ha='center')
        g.figure.subplots_adjust(top=0.9)

        g.savefig(save_as)
        
        plt.close()



    def draw_behavior_proportion(self, data, x, g, behavior="spatial_behavior",
                                hue="laser_state"):
        facet_cols = ["laser_type", "laser_intensity"]   # col, row
        group_cols = facet_cols + [hue]                  # one dominant-behavior track per group
        keys = group_cols + [x]

        # 1-3. proportions, now per laser_state as well
        counts = (data.groupby(keys + [behavior])["name"].nunique()
                    .rename("n_trials").reset_index())
        totals = (data.groupby(keys)["name"].nunique()
                    .rename("total_trials").reset_index())
        props = counts.merge(totals, on=keys)
        props["proportion"] = props["n_trials"] / props["total_trials"]

        # 4. dominant behavior (deterministic on ties)
        props: pd.DataFrame = props.sort_values(keys + [behavior])
        dominant: pd.DataFrame = (props.loc[props.groupby(keys)["proportion"].idxmax()]
                        .sort_values(keys)
                        .reset_index(drop=True))

        # 5. runs of identical behavior, never crossing a group boundary
        prev: pd.DataFrame = dominant.groupby(group_cols)[behavior].shift()
        dominant["segment"] = dominant[behavior].ne(prev).cumsum()

        step = dominant.groupby(group_cols)[x].diff().median()
        nxt = dominant.groupby(group_cols)[x].shift(-1)
        dominant["x_end"] = nxt.fillna(dominant[x] + step)

        spans = (dominant.groupby("segment")
                        .agg(laser_type=("laser_type", "first"),
                            laser_intensity=("laser_intensity", "first"),
                            state=(hue, "first"),
                            beh=(behavior, "first"),
                            start=(x, "min"),
                            end=("x_end", "max"))
                        .reset_index())

        # 6. one band per hue level inside each facet
        # hue_levels = list(g.hue_names)                   # same order as the legend
        hue_levels = ["LaserOff", "LaserOn"]
        n = len(hue_levels)
        for row in spans.itertuples():
            ax = g.axes_dict[(row.laser_intensity, row.laser_type)]   # (row_name, col_name)
            i = hue_levels.index(row.state)
            ax.axvspan(row.start, row.end,
                    ymin=i / n, ymax=(i + 1) / n,     # axes fractions
                    color=BEHAVIOR_PALETTE.get(row.beh, "gray"),
                    hatch=LASER_STATE_HATCH.get(row.state, ""),
                    alpha=0.15, zorder=0, lw=0)



    def add_trial_count_to_facet(self, data, g, hue="laser_state", hue_colors=None,
                            row_col="laser_intensity", col_col="laser_type", loc=(0.90, 0.81), fontsize=12):
        """Write 'n = X' on each facet, one line per hue level."""
        
        counts = (data.groupby([row_col, col_col, hue])["name"]
                    .nunique().rename("n").reset_index())

        hue_levels = sorted(data[hue].unique())
        hue_colors = hue_colors or {}

        for (row_name, col_name), ax in g.axes_dict.items():   # keys are (row, col)
            sub = counts[(counts[row_col] == row_name) & (counts[col_col] == col_name)]
            
            for i, level in enumerate(hue_levels):

                n = int(sub.loc[sub[hue] == level, "n"].sum())
                ax.text(loc[0], loc[1] - i * 0.08,
                        f"n = {n}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=fontsize, color=hue_colors.get(level, "black"),
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.5),
                        zorder=5)



    def sunburst_metadata(self, output_dir: Path, title: str,
                        subfig_group: str = None,
                        groups: list[str] = ["subject", "view"],
                        show_total: bool = True):
        """Generate a sunburst figure in the order of the groups given. 
        If a subfig_group is given, it will split into has many sunburst as they are subfig.
        The outpath will follow the groups order."""
        import plotly.express as px
        from plotly.subplots import make_subplots

        data = self.success_df()
        data = data.drop_duplicates(subset="name", keep="first")

        filename = "_".join(groups)
        if subfig_group: 
            filename = subfig_group.upper() + "_" + filename
        save_as = output_dir / filename 

        # ---------------- CASE 1: no subfig_groups

        if subfig_group is None:
            counts = (
                data
                .groupby(groups)
                .size()
                .reset_index(name="count")
            )

            fig = px.sunburst(
                counts,
                path=groups,
                values="count",
                color_discrete_sequence=px.colors.qualitative.D3   
            )

            fig.update_layout(title=title)

            fig.update_traces(
                branchvalues="total" if show_total else "remainder",
                texttemplate="<b>%{label}</b><br>%{value} (%{percentParent:.0%})",
                textfont_size=25
            )

            fig.write_html(str(save_as.with_suffix(".html")))
            fig.show()
            return

        # ---------------- CASE 2: subfig_groups

        subgroups = sorted(data[subfig_group].dropna().unique())

        fig = make_subplots(
            rows=1,
            cols=len(subgroups),
            specs=[[{"type": "domain"}] * len(subgroups)],
            subplot_titles=subgroups
        )

        for i, g in enumerate(subgroups, start=1):

            tmp = data[data[subfig_group] == g]

            counts = (
                tmp
                .groupby(groups)
                .size()
                .reset_index(name="count")
            )

            pie = px.sunburst(
                counts,
                path=groups,
                values="count",
                color=groups[0],
                color_discrete_sequence=px.colors.qualitative.D3     
            )

            fig.add_trace(
                pie.data[0],
                row=1,
                col=i
            )

        fig.update_traces(
            branchvalues="total",
            texttemplate="<b>%{label}</b><br>%{value} (%{percentParent:.0%})",
            textfont_size=25
        )

        fig.update_layout(title=title)

        fig.write_html(str(save_as.with_suffix(".html")))
        fig.show()