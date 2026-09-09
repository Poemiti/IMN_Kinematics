# src/BetaMouv/Config.py


from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathConfig:

    project_name: str
    subject_name: str | None = None
    bodypart: str | None = None

    # Input data
    model: Path = Path("/media/filer2/T4b/UserFolders/Poemiti/Reaching-DLC-model-main/data/model/DLC-Poe-2026-03-27/")
    raw_videos: Path = Path("/media/filer2/T4b/UserFolders/Raphael/Raphael_M2_2024/Opto_SkilledReaching_2024_experiments/MOVIES/")

    # Output folders
    @property
    def data_root(self) -> Path:
        return Path("./data") / self.project_name

    @property
    def database(self) -> Path:
        return self.data_root / "database"

    @property
    def raw(self) -> Path:
        return self.data_root / "raw"

    @property
    def trials_metadata(self) -> Path:
        return self.data_root / "trials_metadata"

    @property
    def results_root(self) -> Path:
        return self.data_root / "results"

    @property
    def raw_subject(self) -> Path:
        if self.subject_name is None:
            raise ValueError("subject_name has not been defined")

        return self.raw / f"subject_{self.subject_name}"

    def analysis(self, filters_name: list[str]) -> Path:
        results_name = "_".join(filters_name)

        return self.results_root / results_name