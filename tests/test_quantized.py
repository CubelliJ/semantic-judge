import numpy as np
import pytest

from semantic_judge.quantized import QuantizedSemanticJudge, clear_quantized_model_cache
from semantic_judge.schemas import ClassificationRequest, Option


class FakeLlama:
    def __init__(self, logits):
        self.logits = logits
        self.scores = []
        self.calls = 0

    def tokenize(self, value, add_bos=True):
        if isinstance(value, bytes) and len(value) == 1 and value.isalpha():
            return [ord(value)]
        return [10, 11, 12]

    def reset(self):
        self.scores = []

    def eval(self, tokens):
        self.calls += 1
        self.scores = np.zeros((4096, 128), dtype=np.float32)
        self.scores[len(tokens) - 1] = self.logits


def request():
    return ClassificationRequest(
        "The customer reports an unauthorized withdrawal.",
        "Which route should handle this?",
        (Option("fraud", "Fraud"), Option("billing", "Billing")),
    )


def test_quantized_backend_scores_labels_and_latency():
    model = FakeLlama(np.array([0.0] * 128))
    model.logits[ord("A")] = 3.0
    model.logits[ord("B")] = 1.0
    result = QuantizedSemanticJudge(model).classify(request())
    assert result.choice == "fraud"
    assert result.selected_logits == {"fraud": 3.0, "billing": 1.0}
    assert sum(result.scores.values()) == pytest.approx(1.0)
    assert result.execution_time_ms >= 0.0


def test_quantized_model_cache_reuses_one_instance(monkeypatch):
    import semantic_judge.quantized as quantized

    clear_quantized_model_cache()
    model = FakeLlama(np.zeros(128))
    created = []

    class FakeLlamaConstructor:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.model = model

        def __getattr__(self, name):
            return getattr(self.model, name)

    monkeypatch.setitem(__import__("sys").modules, "llama_cpp", type("Module", (), {"Llama": FakeLlamaConstructor}))
    first = quantized.get_quantized_model(model_path="/tmp/qwen.gguf")
    second = quantized.get_quantized_model(model_path="/tmp/qwen.gguf")
    assert first is second
    assert len(created) == 1
    clear_quantized_model_cache()
