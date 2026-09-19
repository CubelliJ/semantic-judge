"""Stable, versioned decision prompt construction."""

import hashlib

from .schemas import ClassificationRequest

PROMPT_VERSION = "semantic-judge-v2-label-pool"


LABEL_POOL = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def labels_for(count: int) -> list[str]:
    """Return positional labels for an option set.

    Labels are deliberately position-based. The runtime additionally verifies
    that the selected labels are single tokenizer tokens before scoring them.
    Stage 1 supports up to 36 labels; larger sets need sequence or hierarchical
    scoring rather than a single next-token distribution.
    """
    if count < 2:
        raise ValueError("at least two labels are required")
    if count > len(LABEL_POOL):
        raise ValueError(f"at most {len(LABEL_POOL)} options are supported")
    return list(LABEL_POOL[:count])


def tokenizer_labels(tokenizer: object, count: int) -> list[str]:
    """Select ``count`` labels that are exactly one tokenizer token each."""
    labels: list[str] = []
    for label in LABEL_POOL:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            labels.append(label)
        if len(labels) == count:
            return labels
    raise ValueError(f"tokenizer has fewer than {count} usable labels encoded as exactly one token")


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
