# src/Base/Outcome.py
from dataclasses import dataclass, asdict

@dataclass
class Outcome:
    stage: str
    order: int                     # position in the pipeline — enables sorting/comparison
    success: bool | None = False    # None = not yet run
    reason: str = "not_run"

    def __bool__(self):
        return bool(self.success)

    def __lt__(self, other):
        return self.order < other.order

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Outcome":
        return cls(**data)