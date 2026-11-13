# src/Base/TrialGroup.py

from .Trial import Trial as BaseTrial
import joblib, os, shutil
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt



custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("talk", style="ticks", rc=custom_params, palette="pastel")

BOOL_PALETTE = {True: "yellowgreen", False: "tomato"}



class TrialGroup:
    """Manager to filter trials to use by their group name"""

    trial_cls = BaseTrial  

    def __init__(self, filenames: list[Path], condition: dict):
        self.filenames = filenames
        self.condition = condition

        self.keep_val = []
        for cond, criteria in self.condition.items():
            self.keep_val.extend(value for value, keep in criteria.items() if keep)
        self.group_name = "_".join(self.keep_val)
        print("\n",self.group_name)

        self.keep_joblib: list[str] = None
        self.trials: list[BaseTrial] = None

        self._trial_success_df = None
        self._bodypart_success_df = None
        self._scalar_df = None
        self._kinematics_df = None
        self._behavior_df = None

        

    def load_trials(self) -> None : 
        """Read joblib files and save trials in self.trials.
        Display the number of joblib open and total number of trials"""
        if self.trials is not None: 
            return
        self.keep_joblib = []
        self.trials = self._filter_joblib()

        print(f"--> {len(self.keep_joblib)}/{len(self.filenames)} FILES LOADED")
        print(f"--> {len(self.trials)} trials")


    def _filter_joblib(self) -> list[dict]:
        trials = []

        for filename in tqdm(self.filenames, desc="Loading joblib"):
            if self._keep_file(filename):

                self.keep_joblib.append(filename)
                records: list = joblib.load(filename)   # list of Trial objects
                # print(f"{len(records)} added - {filename.stem}")

                trials.extend(records)
            # else:
            #     print(f"Not Keep: {filename.name}")

        return trials

    def _keep_file(self, filename: Path) -> bool:
        name = filename.name

        for condition, criteria in self.condition.items():
            val_to_keep = [value for value, keep in criteria.items() if keep]

            if val_to_keep and not any(value in name for value in val_to_keep):
                return False

            not_keep_val = [value for value, keep in criteria.items() if not keep]
            if any(value in name for value in not_keep_val):
                return False

        return True

    def save(self, output_dir: Path, trial_list: list = None):
        by_group = {}

        if trial_list is None : 
            print("Saving trials from trialgroup")
            trial_list = self.trials

        for trial in trial_list:
            by_group.setdefault(trial.group, []).append(trial)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for group, records in by_group.items():
            path = output_dir / f"{group}.joblib"
            tmp = path.with_name(path.name + ".tmp")
            try:
                joblib.dump(records, tmp)
                if tmp.stat().st_size == 0:
                    raise IOError(f"Refusing to write empty file: {path}")
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)


    def save_backup(self, dir_to_copy: Path, output_dir: Path, step: str) -> None:
        """Save trial metadata backup""" 
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output_dir / f"{step}_{stamp}"
        shutil.copytree(dir_to_copy, backup)
        print(f"Backup: {backup}")


    ####################### data preprocess before plotting #############################


    def extract_bodypart_data(self, df: pd.MultiIndex, bodypart) -> pd.DataFrame: 
        """Take a Multindex and return a clean dataframe for the specified bodypart"""
        if bodypart not in df.columns.get_level_values(0):
            raise KeyError(f"Body part '{bodypart}' not found in DataFrame")
        
        bodypart_data = df[bodypart]
        mask = bodypart_data.notna().all(axis=1)

        return pd.concat([df["trial"], bodypart_data], axis=1).loc[mask]


    def trial_success_df(self) -> pd.DataFrame:
        """One row per trial, including failures — for QC / yield figures
        (e.g. success rate by condition, why trials were excluded)."""

        if self._trial_success_df is not None:
            return self._trial_success_df
        
        records = []

        for trial in self.trials:

            for stage, outcome in trial.trial_outcomes.items():
                records.append(trial.identity() | {
                        "stage": stage,
                        "success": outcome.success,
                        "reason": outcome.reason,
                        "is_valid": "valid" if trial.is_valid() else "not_valid",
                        "failing_step": trial.failure()                        
                        })

        res = pd.DataFrame(records)
        self._success_df = res

        return res


    def bodypart_success_df(self) -> pd.DataFrame:
        """One row per (trial, bodypart, stage) — for per-bodypart QC figures."""

        if self._bodypart_success_df is not None:
            return self._bodypart_success_df

        records = []
        for trial in self.trials:
            for bodypart, stat in trial.trajectories.items():
                    for stage, outcome in stat["stage_outcomes"].items():

                        records.append(trial.identity() | {
                            "bodypart": bodypart,
                            "stage": stage,
                            "success": outcome["success"],
                            "reason": outcome["reason"],
                        })

        res = pd.DataFrame(records)
        self._bodypart_success_df = res
            
        return res


    ####################### plotting ###################################################



    def _countplot_success_rate(self, df, col, save_as, show: bool = False,):
        """Display success rate and failure reasons per stage, and save the report."""

        g = sns.catplot(
            data=df, kind="count",
            y="reason",
            row="stage", col=col,
            hue="success", palette=BOOL_PALETTE,
            sharey=False, height=4, aspect=3,
        )

        # add count on bars
        for ax in g.axes.flat: 
            for container in ax.containers: 
                ax.bar_label(container, fmt="%d", padding=3)
        
        # g.set_xticklabels(rotation=45, ha="right")
        g.set_axis_labels("Number of trials", "")
        g.set_titles(row_template="{row_name}")
        g.figure.suptitle(f"Success rate by stage - n_trials: {len(df['name'].unique())}\n{self.group_name}")
        g.figure.subplots_adjust(top=0.90)

        g.figure.savefig(save_as, bbox_inches="tight")
        if show: 
            plt.show()
        plt.close()

        print(f"Report saved to: {save_as}")




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
                    .groupby("relative_t", as_index=False)[["x", "y"]]
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




    def sunburst_metadata(self, output_dir: Path, title: str,
                        subfig_group: str = None,
                        groups: list[str] = ["subject", "view"],
                        show_total: bool = True):
        """Generate a sunburst figure in the order of the groups given. 
        If a subfig_group is given, it will split into has many sunburst as they are subfig.
        The outpath will follow the groups order."""
        import plotly.express as px
        from plotly.subplots import make_subplots

        data = self.trial_success_df()
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

        fig.write_html(str(save_as.with_suffix(".html")), openauto=True)
        fig.show()
