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

    # Output folder
    data_root: Path = Path(f"./data/{project_name}")
    database: Path = Path(f"{data_root}/database")
    raw: Path = Path(f"{data_root}/raw")
    trials_metadata: Path = Path(f"{data_root}/trials_metadata")
    results_root: Path = Path(f"{data_root}/results")

    @property
    def raw_subject(self) -> Path:
        if self.subject_name is None:
            raise ValueError("subject_name has not been defined")

        return self.raw / f"subject_{self.subject_name}"

    @property
    def analysis(self, filters_name: list) -> Path:
        
        results_name = filters_name.joint("_")

        return self.results_root / results_name