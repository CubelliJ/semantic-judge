import hashlib

import pytest
import torch

from semantic_judge.classifier import SemanticJudge
from semantic_judge.prompt import build_prompt, labels_for, prompt_hash
from semantic_judge.schemas import ClassificationRequest, Option


class FakeTokenizer:
    def __init__(self, *, split_labels=False):
        self.split_labels = split_labels

    def __call__(self, value, return_tensors=None, add_special_tokens=True):
        if isinstance(value, str) and len(value) == 1 and value.isalnum():
            if self.split_labels:
                return {"input_ids": [ord(value), ord(value) + 1]}
            return {"input_ids": [ord(value)]}
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([[10, 11, 12]])}
        return {"input_ids": [10]}


class FakeModel:
    def __init__(self, label_logits):
        self.label_logits = label_logits

    def eval(self):
        return self

    def __call__(self, **kwargs):
        logits = torch.zeros((1, 3, 128), dtype=torch.float32)
        for token_id, value in self.label_logits.items():
            logits[0, -1, token_id] = value
        return type("Output", (), {"logits": logits})()


def request(*options):
    return ClassificationRequest(
        evidence="The account owner lost access after replacing their phone.",
        question="Which route should handle this case?",
        options=tuple(Option(*option) for option in options),
    )


def test_request_validation():
    with pytest.raises(ValueError, match="at least two"):
        ClassificationRequest("evidence", "question", (Option("a", "A"),))
    with pytest.raises(ValueError, match="unique"):
        request(("same", "one"), ("same", "two"))


def test_prompt_is_stable_and_hashed():
    req = request(("recovery", "Account recovery"), ("billing", "Billing"))
    prompt = build_prompt(req, labels_for(2))
    assert "A. Account recovery" in prompt
    assert "B. Billing" in prompt
    assert prompt_hash(prompt) == hashlib.sha256(prompt.encode()).hexdigest()


def test_single_token_labels_are_required():
    judge = SemanticJudge(FakeModel({}), FakeTokenizer(split_labels=True))
    with pytest.raises(ValueError, match="exactly one token"):
        judge.classify(request(("a", "First"), ("b", "Second")))


def test_classification_maps_winner_to_original_id_and_normalizes_scores():
    # A/B token IDs are their character code in FakeTokenizer.
    judge = SemanticJudge(
        FakeModel({ord("A"): 4.0, ord("B"): 1.0}),
        FakeTokenizer(),
        model_revision="test-revision",
    )
    result = judge.classify(request(("account_access", "Account recovery"), ("billing", "Billing")))
    assert result.choice == "account_access"
    assert set(result.scores) == {"account_access", "billing"}
    assert sum(result.scores.values()) == pytest.approx(1.0)
    assert result.selected_logits == {"account_access": 4.0, "billing": 1.0}
    assert result.model_revision == "test-revision"
    assert len(result.prompt_hash) == 64


@pytest.mark.parametrize(
    ("logits", "expected"),
    [
        ({"A": 5.0, "B": 0.0, "C": -1.0}, "urgent"),
        ({"A": -1.0, "B": 4.0, "C": 0.0}, "routine"),
        ({"A": 0.0, "B": 0.0, "C": 6.0}, "insufficient"),
    ],
)
def test_classification_scenarios(logits, expected):
    judge = SemanticJudge(
        FakeModel({ord(label): value for label, value in logits.items()}), FakeTokenizer()
    )
    result = judge.classify(
        request(
            ("urgent", "Immediate action required"),
            ("routine", "Handle through the normal process"),
            ("insufficient", "Insufficient evidence; request more information"),
        )
    )
    assert result.choice == expected
