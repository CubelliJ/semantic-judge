"""Semantic Judge: constrained next-token semantic classification."""

from .classifier import SemanticJudge
from .evaluation import EvaluationCase, EvaluationReport, evaluate, load_corpus
from .schemas import ClassificationRequest, Option, ClassificationResult

__all__ = [
    "ClassificationRequest",
    "ClassificationResult",
    "EvaluationCase",
    "EvaluationReport",
    "Option",
    "SemanticJudge",
    "evaluate",
    "load_corpus",
]
