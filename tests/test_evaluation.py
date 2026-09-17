from pathlib import Path

import pytest

from semantic_judge.evaluation import EvaluationCase, evaluate, load_corpus
from semantic_judge.schemas import ClassificationRequest, ClassificationResult, Option


CORPUS = Path(__file__).parents[1] / "data" / "evaluation_corpus.jsonl"


class FakeJudge:
    def __init__(self, predictions):
        self.predictions = predictions

    def classify(self, request):
        choice = self.predictions.pop(0)
        return ClassificationResult(
            choice=choice,
            scores={option.id: (1.0 if option.id == choice else 0.0) for option in request.options},
            selected_logits={},
            model_name="fake",
            model_revision="test",
            prompt_version="test",
            prompt_hash="0" * 64,
            input_token_count=1,
            execution_time_ms=0.0,
        )


def test_loads_labeled_corpus_with_answers():
    cases = load_corpus(CORPUS)
    assert len(cases) == 10
    assert cases[0].expected_choice == "account_access"
    assert "high_risk" in cases[2].tags
    assert all(case.rationale for case in cases)
    assert all(
        option.id != "insufficient_evidence"
        for case in cases
        for option in case.request.options
    )


def test_corpus_expected_choice_must_be_an_option():
    request = ClassificationRequest(
        "evidence",
        "question",
        (Option("a", "A"), Option("b", "B")),
    )
    with pytest.raises(ValueError, match="expected_choice"):
        EvaluationCase("case", request, "missing", "reason")


def test_evaluation_report_metrics_and_callback():
    cases = load_corpus(CORPUS)[:3]
    seen = []
    judge = FakeJudge(["account_access", "fraud", "billing"])
    report = evaluate(judge, cases, on_result=lambda case, result: seen.append(case.case_id))

    assert report.total == 3
    assert report.correct == 1
    assert report.accuracy == pytest.approx(1 / 3)
    assert report.macro_f1 == pytest.approx(1 / 3)
    assert report.mean_latency_ms == pytest.approx(0.0)
    assert report.p50_latency_ms == pytest.approx(0.0)
    assert report.p95_latency_ms == pytest.approx(0.0)
    assert seen == [case.case_id for case in cases]
    assert report.confusion_matrix["account_access"]["account_access"] == 1
    assert report.confusion_matrix["billing"]["fraud"] == 1
    assert report.as_dict()["predictions"][1] == {
        "case_id": cases[1].case_id,
        "expected": "billing",
        "predicted": "fraud",
        "execution_time_ms": 0.0,
    }


def test_empty_corpus_is_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        load_corpus(path)
