.DEFAULT_GOAL := help

PYTHON ?= python3
ifneq ($(COLAB_RELEASE_TAG),)
VENV ?= system
else
VENV ?= .venv
endif
ifeq ($(VENV),system)
PY := $(PYTHON)
PIP := $(PYTHON) -m pip
else
VENV_BIN := $(VENV)/bin
PY := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip
endif

.PHONY: help init install install-training install-quantized train plot-loss test compile check evaluate clean

TRAIN_DATASETS ?= ag_news mnli boolq
TRAIN_LIMIT ?= 2000
TRAIN_EVAL_LIMIT ?= 500
TRAIN_EPOCHS ?= 10
TRAIN_BATCH_SIZE ?= 4
TRAIN_GRADIENT_ACCUMULATION ?= 2
TRAIN_MAX_LENGTH ?= 256
TRAIN_OUTPUT ?= artifacts/mixed-classification

help:
	@echo "Available targets:"
	@echo "  make init              Create the virtual environment and install dependencies"
	@echo "  make install           Install the package and test dependencies"
	@echo "  make install-training  Install LoRA training dependencies"
	@echo "  make train             Install, train mixed datasets, and plot loss"
	@echo "  make plot-loss         Plot an existing training loss log"
	@echo "  make install-quantized Install llama.cpp with Metal support on Apple Silicon"
	@echo "  make test              Run the test suite"
	@echo "  make compile           Compile-check source and tests"
	@echo "  make check             Run tests, compilation, and whitespace checks"
	@echo "  make evaluate          Run the labeled corpus with the default quantized backend"
	@echo "  make clean             Remove local Python/test caches"

init:
ifeq ($(VENV),system)
	@$(MAKE) install
else
	@test -x "$(PY)" || $(PYTHON) -m venv "$(VENV)"
	@$(MAKE) install
endif

install-training: init
	@$(PIP) install -e '.[training]'

install:
ifeq ($(VENV),system)
	@$(PIP) install --no-deps -e .
else
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@$(PIP) install --no-deps -e .
endif

train: install-training
	@$(PY) -m training.train_lora \
		--dataset $(TRAIN_DATASETS) \
		--limit $(TRAIN_LIMIT) \
		--eval-limit $(TRAIN_EVAL_LIMIT) \
		--epochs $(TRAIN_EPOCHS) \
		--batch-size $(TRAIN_BATCH_SIZE) \
		--gradient-accumulation $(TRAIN_GRADIENT_ACCUMULATION) \
		--max-length $(TRAIN_MAX_LENGTH) \
		--output $(TRAIN_OUTPUT)
	@$(MAKE) plot-loss TRAIN_OUTPUT=$(TRAIN_OUTPUT)

plot-loss:
	@$(PY) -m training.plot_loss \
		$(TRAIN_OUTPUT)/microbatch_losses.jsonl \
		--output $(TRAIN_OUTPUT)/loss.png

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
