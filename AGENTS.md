# Agent guidance

This project is a local semantic decision system built around `google/gemma-3-1b-it`.

## Rules

- Keep the implementation small and local; do not add proprietary inference APIs.
- Pin the model revision and treat the prompt template as a versioned interface.
- Validate nonempty evidence and questions, at least two options, unique option IDs, context length, and single-token labels.
- Show option descriptions to the model, but return the original option IDs to callers.
- Score only the next-token logits for the option labels, then apply softmax across those labels.
- Do not require generated explanations or treat relative scores as calibrated probabilities.
- Record the model revision, prompt version/hash, token count, logits, scores, and execution metadata.
- Support an abstention option such as `Insufficient evidence` when appropriate.

## Changes and checks

- Prefer focused changes and update `README.md` when behavior or limitations change.
- Add tests for validation, prompt stability, label tokenization, scoring, option mapping, and metadata.
- Run the relevant tests and configured formatting, linting, or type checks before submitting changes.
