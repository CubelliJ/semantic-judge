"""Model-backed constrained logit classifier."""

from __future__ import annotations

import time
from typing import Any

import torch

from .prompt import PROMPT_VERSION, build_prompt, labels_for, prompt_hash, tokenizer_labels
from .schemas import ClassificationRequest, ClassificationResult

MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class SemanticJudge:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_name: str = MODEL_NAME,
        model_revision: str = "unknown",
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.model_revision = model_revision

    @classmethod
    def from_pretrained(cls, revision: str | None = MODEL_REVISION) -> "SemanticJudge":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"revision": revision} if revision else {}
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
        resolved_revision = revision or getattr(model.config, "_commit_hash", None) or "unknown"
        return cls(model, tokenizer, model_revision=resolved_revision)

    def classify(self, request: ClassificationRequest) -> ClassificationResult:
        labels = tokenizer_labels(self.tokenizer, len(request.options))
        prompt = build_prompt(request, labels)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        label_ids = self._label_token_ids(labels)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model(**encoded)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logits = output.logits[0, -1]
        selected = {label: float(logits[token_id].item()) for label, token_id in zip(labels, label_ids)}
        values = torch.tensor(list(selected.values()), dtype=torch.float64)
        probabilities = torch.softmax(values, dim=0).tolist()
        scores = {option.id: float(score) for option, score in zip(request.options, probabilities)}
        selected_logits = {option.id: selected[label] for option, label in zip(request.options, labels)}
        choice = max(scores, key=scores.__getitem__)
        return ClassificationResult(
            choice=choice,
            scores=scores,
            selected_logits=selected_logits,
            model_name=self.model_name,
            model_revision=self.model_revision,
            prompt_version=PROMPT_VERSION,
            prompt_hash=prompt_hash(prompt),
            input_token_count=int(input_ids.shape[-1]),
            execution_time_ms=elapsed_ms,
        )

    def _label_token_ids(self, labels: list[str]) -> list[int]:
        token_ids: list[int] = []
        for label in labels:
            ids = self.tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"label {label!r} must encode as exactly one token")
            token_ids.append(int(ids[0]))
        return token_ids
