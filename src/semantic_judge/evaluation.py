"""Load labeled evaluation cases and score classifier predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from .schemas import ClassificationRequest, ClassificationResult, Option


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    request: ClassificationRequest
    expected_choice: str
    rationale: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        option_ids = {option.id for option in self.request.options}
        if not self.case_id.strip():
            raise ValueError("case_id must be nonempty")
        if self.expected_choice not in option_ids:
            raise ValueError("expected_choice must match an option id")
        if not self.rationale.strip():
            raise ValueError("rationale must be nonempty")


@dataclass(frozen=True)
class EvaluationReport:
    total: int
    correct: int
    accuracy: float
    macro_f1: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    per_label: dict[str, dict[str, int | float]]
    confusion_matrix: dict[str, dict[str, int]]
    predictions: tuple[dict[str, str | float], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "mean_latency_ms": self.mean_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "per_label": self.per_label,
            "confusion_matrix": self.confusion_matrix,
            "predictions": list(self.predictions),
        }


def load_corpus(path: str | Path) -> list[EvaluationCase]:
    """Load one JSON object per line from a labeled evaluation corpus."""
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                raw = json.loads(line)
                case = _case_from_dict(raw)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid evaluation case on line {line_number}: {exc}") from exc
            if case.case_id in seen_ids:
                raise ValueError(f"duplicate evaluation case id: {case.case_id}")
            seen_ids.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError("evaluation corpus must contain at least one case")
    return cases


def evaluate(
    judge: Any,
    cases: Iterable[EvaluationCase],
    *,
    on_result: Callable[[EvaluationCase, ClassificationResult], None] | None = None,
) -> EvaluationReport:
    """Run cases through a judge and compute exact-match classification metrics."""
    cases = list(cases)
    if not cases:
        raise ValueError("at least one evaluation case is required")
    predictions: list[dict[str, str | float]] = []
    latencies = []
    actual = []
    predicted = []
    for case in cases:
        result = judge.classify(case.request)
        if on_result:
            on_result(case, result)
        actual.append(case.expected_choice)
        predicted.append(result.choice)
        latencies.append(result.execution_time_ms)
        predictions.append(
            {
                "case_id": case.case_id,
                "expected": case.expected_choice,
                "predicted": result.choice,
                "execution_time_ms": result.execution_time_ms,
            }
        )

    labels = sorted(set(actual) | set(predicted))
    confusion = {label: {other: 0 for other in labels} for label in labels}
    for expected, choice in zip(actual, predicted):
        confusion.setdefault(expected, {}).setdefault(choice, 0)
        confusion[expected][choice] += 1
        confusion.setdefault(choice, {}).setdefault(expected, 0)
    per_label: dict[str, dict[str, int | float]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = confusion[label].get(label, 0)
        fp = sum(confusion[other].get(label, 0) for other in confusion if other != label)
        fn = sum(value for other, value in confusion[label].items() if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "support": tp + fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    correct = sum(expected == choice for expected, choice in zip(actual, predicted))
    ordered_latencies = sorted(latencies)
    return EvaluationReport(
        total=len(cases),
        correct=correct,
        accuracy=correct / len(cases),
        macro_f1=sum(f1_values) / len(f1_values),
        mean_latency_ms=mean(latencies),
        p50_latency_ms=_percentile(ordered_latencies, 0.50),
        p95_latency_ms=_percentile(ordered_latencies, 0.95),
        per_label=per_label,
        confusion_matrix=confusion,
        predictions=tuple(predictions),
    )


def _percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] + (values[upper] - values[lower]) * weight


def _case_from_dict(raw: dict[str, Any]) -> EvaluationCase:
    options = tuple(Option(item["id"], item["description"]) for item in raw["options"])
    request = ClassificationRequest(raw["evidence"], raw["question"], options)
    return EvaluationCase(
        case_id=raw["case_id"],
        request=request,
        expected_choice=raw["expected_choice"],
        rationale=raw["rationale"],
        tags=tuple(raw.get("tags", ())),
    )
