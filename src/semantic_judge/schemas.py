"""Input and output schemas for Semantic Judge."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Option:
    id: str
    description: str

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.description.strip():
            raise ValueError("option id and description must be nonempty")


@dataclass(frozen=True)
class ClassificationRequest:
    evidence: str
    question: str
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        if not self.evidence.strip():
            raise ValueError("evidence must be nonempty")
        if not self.question.strip():
            raise ValueError("question must be nonempty")
        if len(self.options) < 2:
            raise ValueError("at least two options are required")
        ids = [option.id for option in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("option ids must be unique")


@dataclass(frozen=True)
class ClassificationResult:
    choice: str
    scores: dict[str, float]
    selected_logits: dict[str, float]
    model_name: str
    model_revision: str
    prompt_version: str
    prompt_hash: str
    input_token_count: int
    execution_time_ms: float
    score_notice: str = "Scores are uncalibrated relative model-preference scores."

    def as_dict(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "scores": self.scores,
            "selected_logits": self.selected_logits,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "input_token_count": self.input_token_count,
            "execution_time_ms": self.execution_time_ms,
            "score_notice": self.score_notice,
        }
