import torch

from training.data import from_ag_news, from_boolq, from_mnli
from training.lora import candidate_loss, encode_example, evaluate_label_metrics


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        if add_special_tokens:
            return {"input_ids": list(range(10, 10 + len(text.split())))}
        return {"input_ids": [ord(text) - ord("A")]}


class FakeModel:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        width = kwargs["input_ids"].shape[1]
        logits = torch.zeros((1, width, 32), dtype=torch.float32)
        logits[:, :, 0] = 2.0  # A
        logits[:, :, 1] = 0.0  # B
        return type("Output", (), {"logits": logits})()


def test_adapters_use_runtime_options_and_labels():
    ag = from_ag_news({"text": "A report about markets", "label": 2}, 4)
    mnli = from_mnli({"premise": "A person runs", "hypothesis": "Someone moves", "label": 0}, 5)
    boolean = from_boolq({"passage": "Water is wet.", "question": "Is water wet?", "answer": True}, 6)
    assert ag.target == "C"
    assert len(mnli.request.options) == 3
    assert boolean.target == "A"
    assert ag.prompt().endswith("Answer:")


def test_reordered_examples_remap_target_label():
    example = from_ag_news({"text": "A report about markets", "label": 2}, 0)
    reordered = example.reordered([2, 0, 1, 3])
    assert reordered.request.options[0].description == "Business news"
    assert reordered.target == "A"


def test_encode_masks_prompt_and_trains_only_target_token():
    example = from_boolq({"passage": "Water is wet.", "question": "Is water wet?", "answer": True}, 0)
    encoded = encode_example(FakeTokenizer(), example)
    assert encoded["labels"][:-1] == [-100] * (len(encoded["labels"]) - 1)
    assert encoded["labels"][-1] == 0


def test_candidate_loss_uses_only_current_options():
    from training.lora import collate

    tokenizer = FakeTokenizer()
    example = from_boolq({"passage": "Water is wet.", "question": "Is water wet?", "answer": True}, 0)
    feature = encode_example(tokenizer, example)
    batch = collate([feature], tokenizer)
    loss = candidate_loss(FakeModel(), tokenizer, batch, [example], [feature])
    assert float(loss) == torch.nn.functional.cross_entropy(torch.tensor([[2.0, 0.0]]), torch.tensor([0])).item()


def test_evaluation_reports_nll_accuracy_and_macro_f1():
    tokenizer = FakeTokenizer()
    first = from_boolq({"passage": "Water is wet.", "question": "Is water wet?", "answer": True}, 0)
    second = from_boolq({"passage": "Fire is cold.", "question": "Is fire cold?", "answer": False}, 1)
    # A is the highest score for both examples; one of two labels is correct.
    model = FakeModel()
    metrics = evaluate_label_metrics(model, tokenizer, [first, second])
    assert metrics.count == 2
    assert metrics.accuracy == 0.5
    assert metrics.macro_f1 == 1 / 3
    assert metrics.nll > 0
