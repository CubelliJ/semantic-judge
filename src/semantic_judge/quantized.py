"""Lazy, process-wide Qwen3 GGUF backend with constrained logit scoring."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .prompt import LABEL_POOL, PROMPT_VERSION, build_prompt, prompt_hash
from .schemas import ClassificationRequest, ClassificationResult

QUANTIZED_MODEL_NAME = "Qwen/Qwen3-0.6B-Q4_K_M-GGUF"
QUANTIZED_MODEL_REPO = "rippertnt/Qwen3-0.6B-Q4_K_M-GGUF"
QUANTIZED_MODEL_FILE = "qwen3-0.6b-q4_k_m.gguf"
QUANTIZED_MODEL_REVISION = "fa72ebc1225f63d0941770a1badf04594ddff6b7"

_model_cache: dict[tuple[str, int, int], Any] = {}
_model_locks: dict[tuple[str, int, int], threading.Lock] = {}
_model_lock = threading.Lock()


def get_quantized_model(
    *,
    model_path: str | Path | None = None,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
) -> Any:
    """Return one cached llama.cpp model instance per configuration in this process."""
    key = (str(model_path or "huggingface://" + QUANTIZED_MODEL_FILE), n_ctx, n_gpu_layers)
    if key not in _model_cache:
        with _model_lock:
            if key not in _model_cache:
                if model_path is None:
                    from huggingface_hub import hf_hub_download

                    model_path = hf_hub_download(
                        repo_id=QUANTIZED_MODEL_REPO,
                        filename=QUANTIZED_MODEL_FILE,
                        revision=QUANTIZED_MODEL_REVISION,
                    )
                from llama_cpp import Llama

                _model_cache[key] = Llama(
                    model_path=str(model_path),
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    logits_all=True,
                    verbose=False,
                )
                _model_locks[key] = threading.Lock()
    return _model_cache[key]


def clear_quantized_model_cache() -> None:
    """Release cached quantized models; mainly useful for tests and benchmarks."""
    with _model_lock:
        _model_cache.clear()
        _model_locks.clear()


class QuantizedSemanticJudge:
    """Semantic Judge backend for a loaded Qwen3 GGUF model."""

    def __init__(
        self,
        model: Any,
        *,
        model_revision: str = "unknown",
        inference_lock: threading.Lock | None = None,
    ) -> None:
        self.model = model
        self.model_name = QUANTIZED_MODEL_NAME
        self.model_revision = model_revision
        self._inference_lock = inference_lock

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_path: str | Path | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ) -> "QuantizedSemanticJudge":
        model = get_quantized_model(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        key = (str(model_path or "huggingface://" + QUANTIZED_MODEL_FILE), n_ctx, n_gpu_layers)
        return cls(model, inference_lock=_model_locks[key])

    def classify(self, request: ClassificationRequest) -> ClassificationResult:
        labels = self._labels_for(len(request.options))
        prompt = build_prompt(request, labels)
        prompt_tokens = self.model.tokenize(prompt.encode("utf-8"), add_bos=True)
        label_ids = self._label_token_ids(labels)
        start = time.perf_counter()
        lock = self._inference_lock
        if lock is None:
            self.model.reset()
            self.model.eval(prompt_tokens)
        else:
            with lock:
                self.model.reset()
                self.model.eval(prompt_tokens)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logits = np.asarray(self.model.scores[len(prompt_tokens) - 1])
        selected = {label: float(logits[token_id]) for label, token_id in zip(labels, label_ids)}
        values = torch.tensor(list(selected.values()), dtype=torch.float64)
        probabilities = torch.softmax(values, dim=0).tolist()
        scores = {option.id: float(score) for option, score in zip(request.options, probabilities)}
        selected_logits = {option.id: selected[label] for option, label in zip(request.options, labels)}
        return ClassificationResult(
            choice=max(scores, key=scores.__getitem__),
            scores=scores,
            selected_logits=selected_logits,
            model_name=self.model_name,
            model_revision=self.model_revision,
            prompt_version=PROMPT_VERSION,
            prompt_hash=prompt_hash(prompt),
            input_token_count=len(prompt_tokens),
            execution_time_ms=elapsed_ms,
        )

    def _labels_for(self, count: int) -> list[str]:
        labels: list[str] = []
        for label in LABEL_POOL:
            if len(self.model.tokenize(label.encode("utf-8"), add_bos=False)) == 1:
                labels.append(label)
            if len(labels) == count:
                return labels
        raise ValueError(f"model tokenizer has fewer than {count} usable single-token labels")

    def _label_token_ids(self, labels: list[str]) -> list[int]:
        token_ids = []
        for label in labels:
            ids = self.model.tokenize(label.encode("utf-8"), add_bos=False)
            if len(ids) != 1:
                raise ValueError(f"label {label!r} must encode as exactly one token")
            token_ids.append(int(ids[0]))
        return token_ids
