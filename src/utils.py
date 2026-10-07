# src/utils.py

from pathlib import Path
import time, os, datetime as dt, csv, yaml
from collections import Counter
from functools import wraps


def load_config(filename: Path):
        with open(filename, "r") as cfg_file:
            cfg = yaml.safe_load(cfg_file)
        return cfg


def match_rule(meta, rules):
    best_match = None
    best_score = -1

    for rule in rules.get("rules"):
        conditions = rule.get("when", {})
    
        if all(meta.get(k) == v for k, v in conditions.items()):
            score = len(conditions)
            if score >= best_score:             # >= because we follow the order of the rules document, and the last one has higher priority
                best_match = rule["value"]
                best_score = score

    return best_match if best_match else rules.get("default")



def make_path(base_dir: Path, file_name: str) ->  Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / file_name


def make_dir(dir: Path) -> Path: 
    dir.mkdir(parents=True, exist_ok=True)
    return dir


def save(path: Path, write_fn):
    tmp = path.with_name(path.name + ".tmp")
    try:
        write_fn(tmp)
        if tmp.stat().st_size == 0:
            raise IOError(f"Refusing to write empty file: {path}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)   # cleans up if anything failed



def _save_error_report(output_dir, errors: dict, update: bool):
    if not errors:
        return None

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"errors_{'update' if update else 'init'}_{stamp}"

    # CSV: one line per clip, sorted by error type (open in Excel/pandas)
    with open(base.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "error_type", "message"])
        for path, e in sorted(errors.items(), key=lambda kv: (kv[1]["type"], kv[1]["message"])):
            writer.writerow([path, e["type"], e["message"]])

    # TXT: full tracebacks, to find where the error comes from
    with open(base.with_suffix(".txt"), "w") as f:
        for path, e in errors.items():
            f.write(f"{'=' * 80}\n{path}\n{e['traceback']}\n")

    # short summary in the terminal: most frequent errors first
    counts = Counter((e["type"], e["message"][:80]) for e in errors.values())
    print(f"\n{len(errors)} errors, most frequent causes:")
    for (etype, msg), n in counts.most_common(5):
        print(f"  {n:4d} x {etype}: {msg}")
    print(f"Report saved: {base}.csv / .txt")
    return base



###################### decorator ###############################

def process_time(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        elapsed = time.perf_counter() - start

        print(f"\n[{func.__name__}] time: {_format_duration(elapsed)}")

        return result
    return wrapper

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    
    return f"{seconds/3600:.1f} h"




# TODO 
# refaire cette option

def _plot_metadata_report(
            data,
            output_path: Path,
            subfig_group: str = None,
            groups: list[str] = ["condition", "view", "stim", "cue", "laser", "intensity"],
            rat_name: str = None,
            rat_type: str = None,
            show_total: bool = True
        ):
    
    import plotly.express as px
    from plotly.subplots import make_subplots

    def build_title():
        if rat_name is not None and rat_type is not None:
            return f"Trial metadata of rat {rat_name}, {rat_type}"
        elif rat_name is None and rat_type is not None:
            return f"Trial metadata of rat {rat_type}"
        return "Trial metadata"

    title = build_title()

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
            values="count"
        )

        fig.update_layout(title=title)

        fig.update_traces(
            branchvalues="total" if show_total else "remainder",
            texttemplate="<b>%{label}</b><br>%{value} (%{percentParent:.0%})",
            textfont_size=14
        )

        fig.write_html(str(output_path.with_suffix(".html")))
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
            # color="condition",
            # color_discrete_sequence=px.colors.qualitative.D3     
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

    fig.write_html(str(output_path.with_suffix(".html")))
    fig.show()
