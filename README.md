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

The model is downloaded from Hugging Face on the first real inference call and cached locally. If Hugging Face requests authentication, run `huggingface-cli login` first.

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

The test scenarios exercise three common outcomes without downloading model weights:

- urgent versus routine handling
- routine handling when evidence is sufficient
- abstention through an `Insufficient evidence` option

These tests use deterministic fake logits to verify mapping and scoring mechanics. They do not measure Qwen quality; evaluate the real model on labeled scenarios before deployment.

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
