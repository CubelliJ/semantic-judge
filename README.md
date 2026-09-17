# Semantic Judge

**Runtime-defined zero-shot classification through constrained next-token logit scoring.**

Semantic Judge is an open, local, reproducible decision engine built with [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B).

It accepts evidence, a natural-language question, and a runtime-defined list of options. It scores short option labels using the model's next-token logits, selects the highest-scoring option, and returns structured metadata for auditing and reproduction.

## Design goals

- Run locally with standard open-source tooling; no proprietary inference API is required.
- Support arbitrary domains such as routing, triage, policy evaluation, classification, and candidate selection.
- Keep the decision output constrained to one short label rather than requiring generated explanations or JSON.
- Make model, tokenizer, prompt, input, and inference configuration explicit and reproducible.
- Clearly distinguish relative model preference scores from calibrated probabilities of correctness.

## How scoring works

For each request, the system:

1. Validates nonempty evidence and questions, at least two options, unique option IDs, and JSON-compatible input.
2. Assigns short labels (`A`, `B`, `C`, ...) to the supplied options.
3. Builds a versioned decision prompt containing the evidence, question, and option descriptions.
4. Runs one causal-language-model forward pass.
5. Extracts the next-token logits for the labels, after verifying that every label is exactly one tokenizer token.
6. Applies softmax only across the valid option-label logits.
7. Maps the winning label back to the original option ID.

The scores are conditional, relative scores over the options supplied in that request. A score of `0.90` means the model strongly preferred that option over the alternatives presented; it is not automatically a 90% probability that the answer is correct. Calibrate scores on held-out labeled data before using them as confidence thresholds.

## Example request

```json
{
  "evidence": "The customer cannot sign in after changing phones and has lost access to the old authenticator.",
  "question": "What is the most appropriate support route?",
  "options": [
    {"id": "account_access", "description": "Account access recovery"},
    {"id": "billing", "description": "Billing and payments"},
    {"id": "sales", "description": "New purchase or sales inquiry"},
    {"id": "human_review", "description": "Escalate for human review"}
  ]
}
```

## Example result

```json
{
  "request_id": "...",
  "choice": "account_access",
  "scores": {
    "account_access": 0.90,
    "billing": 0.04,
    "sales": 0.02,
    "human_review": 0.04
  },
  "selected_logits": {
    "account_access": 4.2,
    "billing": 1.1,
    "sales": 0.5,
    "human_review": 1.1
  },
  "model_name": "Qwen/Qwen3-0.6B",
  "model_revision": "<pinned revision>",
  "prompt_version": "<version>",
  "prompt_hash": "<sha256>",
  "input_token_count": 42,
  "execution_time_ms": 123.4,
  "score_notice": "Scores are uncalibrated relative model-preference scores."
}
```

## Quick start

Create an isolated virtual environment, install the pinned dependencies, and install this package in editable mode:

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

The Transformers model is downloaded from Hugging Face on the first real inference call and cached locally. If Hugging Face requests authentication, run `huggingface-cli login` first.

### Quantized Qwen3-0.6B backend

For lower-latency classification on Apple Silicon, install the optional GGUF backend:

```bash
CMAKE_ARGS="-DGGML_METAL=on" python -m pip install 'llama-cpp-python==0.3.16'
```

Then load the Qwen3-0.6B Q4_K_M model. The pinned community GGUF file is downloaded lazily from [`rippertnt/Qwen3-0.6B-Q4_K_M-GGUF`](https://huggingface.co/rippertnt/Qwen3-0.6B-Q4_K_M-GGUF):

```text
Revision: fa72ebc1225f63d0941770a1badf04594ddff6b7
File: qwen3-0.6b-q4_k_m.gguf
```

```python
from semantic_judge import QuantizedSemanticJudge

judge = QuantizedSemanticJudge.from_pretrained(
    n_ctx=4096,
    n_gpu_layers=-1,  # offload all possible layers to Metal
)
result = judge.classify(request)
```

`get_quantized_model()` maintains one model instance per `(model_path, context size, GPU-layer setting)` in the process. Repeated `QuantizedSemanticJudge.from_pretrained()` calls with the same settings reuse that instance, avoiding duplicate model memory. Inference on the shared llama.cpp context is serialized with a lock; use multiple worker processes only if you intentionally want multiple model copies.

```python
from semantic_judge import ClassificationRequest, Option, SemanticJudge

request = ClassificationRequest(
    evidence="The customer lost access to the old authenticator after changing phones.",
    question="What is the appropriate support route?",
    options=(
        Option("account_access", "Account access recovery"),
        Option("billing", "Billing and payments"),
        Option("human_review", "Escalate for human review"),
    ),
)

judge = SemanticJudge.from_pretrained()
# Default revision: c1899de289a04d12100db370d81485cdf75e47ca
result = judge.classify(request)
print(result.as_dict())
```

The test scenarios exercise common routing outcomes without downloading model weights:

- urgent versus routine handling
- account access, billing, fraud, and technical support routing
- deterministic option mapping and relative scoring

These tests use deterministic fake logits to verify mapping and scoring mechanics. They do not measure Qwen quality; evaluate the real model on labeled scenarios before deployment.

## Labeled evaluation corpus

`data/evaluation_corpus.jsonl` contains 10 human-authored cases with expected option IDs and short rationales. It includes:

- account access, billing, fraud, sales, technical support, privacy, and safety routing
- high-risk safety and fraud cases
- clear evidence cases with substantive routing labels

The primary corpus intentionally does not include an `insufficient_evidence` option: every case forces a substantive route so option-selection quality can be measured directly. If a deployment needs abstention, callers should add an explicit policy option such as `human_review` or `escalate` to that request and evaluate it in a separate corpus.

The `expected_choice` values are the gold labels for this corpus. They are evaluation targets, not model instructions embedded at runtime. Review and version the corpus when policy or routing definitions change.

Run it against a loaded judge with:

```python
from semantic_judge import SemanticJudge, evaluate, load_corpus

cases = load_corpus("data/evaluation_corpus.jsonl")
report = evaluate(SemanticJudge.from_pretrained(), cases)
print(report.as_dict())
```

The report includes exact-match accuracy, macro-F1, per-label precision/recall/F1, a confusion matrix, every case's expected and predicted IDs, and `mean_latency_ms`, `p50_latency_ms`, and `p95_latency_ms`. Each prediction also records its individual `execution_time_ms`. The corpus is intentionally small and illustrative; do not treat its score as a production quality claim. Add a separate held-out corpus before tuning prompts or thresholds against these cases.

## Project status

This repository is being built incrementally. The first implementation will focus on the direct scoring path:

1. Load the model and tokenizer.
2. Validate requests.
3. Construct the stable decision prompt.
4. Extract and normalize label logits.
5. Return reproducible structured output.

Evaluation, calibration, batching, caching, quantization, acceleration, and an HTTP API are later phases.

## Reproducibility

Results should be reproducible when the following are fixed:

- model name and exact model revision
- tokenizer and library versions
- prompt template and prompt version
- evidence, question, option IDs, descriptions, and option order
- inference configuration and numerical environment

Prompt wording, formatting, labels, and option ordering can affect model behavior. The implementation must record the model revision, prompt hash/version, token count, logits, scores, and execution metadata.

## Safety and limitations

The model is forced to choose among the supplied options. It may appear confident when evidence is incomplete, ambiguous, contradictory, or when none of the options is correct. Include an option such as `Insufficient evidence` or `Escalate for human review` when abstention matters.

Do not treat the selected option or its relative score as proof of correctness. Consequential use requires labeled evaluation, robustness testing, and empirical calibration.

## Evaluation plan

A labeled evaluation set should measure:

- accuracy, balanced accuracy, and macro F1
- confusion matrices and latency
- calibration on held-out data
- stability under option-order reversal
- robustness to irrelevant context
- behavior under rephrased questions and changed option wording
- behavior under missing, weakened, or ambiguous evidence

## License and model terms

Project licensing and dependency details will be added with the implementation. The Qwen model is subject to its own license and usage terms; review the model card and applicable terms before distribution or deployment.
