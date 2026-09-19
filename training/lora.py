"""LoRA fine-tuning and label-level evaluation for Semantic Judge."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .data import TrainingExample


@dataclass(frozen=True)
class TrainingMetrics:
    nll: float
    accuracy: float
    macro_f1: float
    count: int


def encode_example(tokenizer: Any, example: TrainingExample, max_length: int = 1024) -> dict[str, list[int]]:
    """Encode one example with loss on only its single correct label token."""
    prompt_ids = tokenizer(example.prompt(), add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(example.target, add_special_tokens=False)["input_ids"]
    if len(target_ids) != 1:
        raise ValueError(f"target {example.target!r} must encode as one token")
    if len(prompt_ids) + 1 > max_length:
        prompt_ids = prompt_ids[: max_length - 1]
    input_ids = prompt_ids + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def collate(features: list[dict[str, list[int]]], tokenizer: Any) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    width = max(len(item["input_ids"]) for item in features)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in features:
        padding = width - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [pad_id] * padding)
        batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
        batch["labels"].append(item["labels"] + [-100] * padding)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def evaluate_label_metrics(
    model: Any,
    tokenizer: Any,
    examples: Iterable[TrainingExample],
    *,
    device: torch.device | str = "cpu",
    max_length: int = 1024,
) -> TrainingMetrics:
    """Evaluate target-label NLL, accuracy, and macro-F1."""
    examples = list(examples)
    if not examples:
        raise ValueError("at least one example is required")
    model.eval()
    losses: list[float] = []
    actual: list[str] = []
    predicted: list[str] = []
    with torch.inference_mode():
        for example in examples:
            encoded = encode_example(tokenizer, example, max_length)
            inputs = {key: torch.tensor([value], device=device) for key, value in encoded.items() if key != "labels"}
            # Causal LM logits at position t predict the token at t + 1.
            logits = model(**inputs).logits[0, len(encoded["input_ids"]) - 2]
            labels = [chr(ord("A") + index) for index in range(len(example.request.options))]
            label_ids = [tokenizer(label, add_special_tokens=False)["input_ids"] for label in labels]
            if any(len(ids) != 1 for ids in label_ids):
                raise ValueError("all option labels must encode as one token")
            scores = [float(logits[ids[0]].item()) for ids in label_ids]
            probabilities = torch.softmax(torch.tensor(scores, dtype=torch.float64), dim=0)
            losses.append(-math.log(max(float(probabilities[labels.index(example.target)]), 1e-30)))
            actual.append(example.target)
            predicted.append(labels[max(range(len(labels)), key=scores.__getitem__)])
    labels = sorted(set(actual) | set(predicted))
    f1_values = []
    for label in labels:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return TrainingMetrics(sum(losses) / len(losses), sum(a == p for a, p in zip(actual, predicted)) / len(actual), sum(f1_values) / len(f1_values), len(actual))
