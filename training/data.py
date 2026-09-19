"""Public classification dataset adapters for Semantic Judge training.

The adapters intentionally emit the same evidence/question/options/target shape
used by the runtime classifier. Raw datasets are downloaded by ``datasets`` at
execution time and are not checked into this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from semantic_judge.prompt import build_prompt, labels_for
from semantic_judge.schemas import ClassificationRequest, Option


@dataclass(frozen=True)
class TrainingExample:
    """One example whose target is a runtime option label such as ``A``."""

    example_id: str
    request: ClassificationRequest
    target: str
    source: str

    def __post_init__(self) -> None:
        labels = labels_for(len(self.request.options))
        if self.target not in labels:
            raise ValueError(f"target {self.target!r} is not a valid option label")

    def prompt(self) -> str:
        return build_prompt(self.request, labels_for(len(self.request.options)))

    def reordered(self, order: list[int]) -> "TrainingExample":
        """Return a copy with options reordered and the target label remapped."""
        if sorted(order) != list(range(len(self.request.options))):
            raise ValueError("order must be a permutation of option positions")
        options = tuple(self.request.options[index] for index in order)
        old_target_index = labels_for(len(self.request.options)).index(self.target)
        new_target_index = order.index(old_target_index)
        return TrainingExample(
            example_id=self.example_id,
            request=ClassificationRequest(self.request.evidence, self.request.question, options),
            target=labels_for(len(options))[new_target_index],
            source=self.source,
        )


def _example(
    example_id: str,
    source: str,
    evidence: str,
    question: str,
    option_descriptions: Iterable[str],
    target_index: int,
) -> TrainingExample:
    descriptions = tuple(option_descriptions)
    if not 0 <= target_index < len(descriptions):
        raise ValueError("target index is outside the option list")
    # IDs are deliberately stable but are not exposed as training labels.
    options = tuple(Option(f"option_{index}", description) for index, description in enumerate(descriptions))
    return TrainingExample(
        example_id=example_id,
        request=ClassificationRequest(evidence, question, options),
        target=labels_for(len(options))[target_index],
        source=source,
    )


def from_ag_news(row: dict[str, Any], index: int) -> TrainingExample:
    labels = ("World news", "Sports news", "Business news", "Science and technology news")
    return _example(
        f"ag_news-{index}",
        "ag_news",
        str(row["text"]),
        "Which topic best describes this article?",
        labels,
        int(row["label"]),
    )


def from_mnli(row: dict[str, Any], index: int) -> TrainingExample:
    labels = ("The premise entails the hypothesis", "The premise contradicts the hypothesis", "The relationship is neutral")
    return _example(
        f"mnli-{index}",
        "mnli",
        str(row["premise"]),
        f"What is the relationship between the premise and this hypothesis?\nHypothesis: {row['hypothesis']}",
        labels,
        int(row["label"]),
    )


def from_boolq(row: dict[str, Any], index: int) -> TrainingExample:
    return _example(
        f"boolq-{index}",
        "boolq",
        str(row["passage"]),
        str(row["question"]),
        ("Yes", "No"),
        0 if bool(row["answer"]) else 1,
    )


ADAPTERS = {"ag_news": from_ag_news, "mnli": from_mnli, "boolq": from_boolq}


def load_huggingface_examples(
    name: str,
    *,
    split: str = "train",
    limit: int | None = None,
) -> list[TrainingExample]:
    """Download and convert one supported public HF dataset split."""
    if name not in ADAPTERS:
        raise ValueError(f"unsupported dataset {name!r}; choose from {sorted(ADAPTERS)}")
    from datasets import load_dataset

    dataset_name = "glue" if name == "mnli" else name
    config = "mnli" if name == "mnli" else None
    if split == "test" and name == "boolq":
        split = "validation"
    if split == "test" and name == "mnli":
        split = "validation_matched"
    raw = load_dataset(dataset_name, config, split=split)
    rows = raw if limit is None else raw.select(range(min(limit, len(raw))))
    return [ADAPTERS[name](dict(row), index) for index, row in enumerate(rows)]


def write_jsonl(examples: Iterable[TrainingExample], path: str) -> int:
    """Write converted examples without requiring the datasets package."""
    import json
    from pathlib import Path

    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = {
                "example_id": example.example_id,
                "source": example.source,
                "evidence": example.request.evidence,
                "question": example.request.question,
                "options": [{"id": option.id, "description": option.description} for option in example.request.options],
                "target": example.target,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count
