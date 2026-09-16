"""Stable, versioned decision prompt construction."""

import hashlib

from .schemas import ClassificationRequest

PROMPT_VERSION = "semantic-judge-v1"


def labels_for(count: int) -> list[str]:
    if count > 26:
        raise ValueError("at most 26 options are supported")
    return [chr(ord("A") + index) for index in range(count)]


def build_prompt(request: ClassificationRequest, labels: list[str]) -> str:
    lines = [
        "You are a careful classification system.",
        "Choose exactly one option label based only on the evidence.",
        "Return the option label and nothing else.",
        "",
        f"Evidence:\n{request.evidence.strip()}",
        f"\nQuestion:\n{request.question.strip()}",
        "\nOptions:",
    ]
    lines.extend(
        f"{label}. {option.description.strip()}"
        for label, option in zip(labels, request.options)
    )
    lines.append("\nAnswer:")
    return "\n".join(lines)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
