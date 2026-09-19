.DEFAULT_GOAL := help

PYTHON ?= python3.12
VENV ?= .venv
VENV_BIN := $(VENV)/bin
PY := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip

.PHONY: help init install install-quantized test compile check evaluate clean

help:
	@echo "Available targets:"
	@echo "  make init              Create the virtual environment and install dependencies"
	@echo "  make install           Install the package and test dependencies"
	@echo "  make install-quantized Install llama.cpp with Metal support on Apple Silicon"
	@echo "  make test              Run the test suite"
	@echo "  make compile           Compile-check source and tests"
	@echo "  make check             Run tests, compilation, and whitespace checks"
	@echo "  make evaluate          Run the labeled corpus with the default quantized backend"
	@echo "  make clean             Remove local Python/test caches"

init:
	@test -x "$(PY)" || $(PYTHON) -m venv "$(VENV)"
	@$(MAKE) install

install:
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@$(PIP) install --no-deps -e .

install-quantized:
	@CMAKE_ARGS="-DGGML_METAL=on" $(PIP) install 'llama-cpp-python==0.3.16'

test:
	@$(PY) -m pytest -q

compile:
	@$(PY) -m compileall -q src tests

check: test compile
	@git diff --check

evaluate:
	@PYTHONPATH=src $(PY) -c '\
from semantic_judge import QuantizedSemanticJudge, evaluate, load_corpus; \
report = evaluate(QuantizedSemanticJudge.from_pretrained(), load_corpus("data/evaluation_corpus.jsonl")); \
print(report.as_dict())'

clean:
	@find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
