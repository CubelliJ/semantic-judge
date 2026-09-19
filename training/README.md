# LoRA training

This directory trains adapters against the same prompt and next-token label
interface used by `semantic_judge.SemanticJudge`. It does not train the GGUF
runtime model directly.

## Install

```bash
python -m pip install -e '.[training]'
```

## Supported public datasets

The adapters currently support:

- `ag_news` — four-way news topic classification
- `mnli` — entailment, contradiction, or neutral
- `boolq` — yes/no reading comprehension

A run can mix datasets with different numbers of options:

```bash
python -m training.train_lora \
  --dataset ag_news mnli boolq \
  --limit 2000 \
  --eval-limit 500 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --max-length 256 \
  --output artifacts/mixed-classification
```

Datasets are downloaded and cached by Hugging Face when the command runs. They
are not committed to this repository. Each record is rendered with the
canonical `semantic_judge.prompt.build_prompt` function.

## Train

The Makefile provides one command that creates the environment, installs the
runtime and training dependencies, runs a mixed-dataset 10-epoch job, and
writes a loss graph to `loss.png`:

```bash
make train
```

Defaults are 2,000 examples per source, 500 validation examples per source,
10 epochs, 4 examples per microbatch, gradient accumulation of 2, and a
maximum sequence length of 256. Override them without editing files:

```bash
make train TRAIN_LIMIT=200 TRAIN_EPOCHS=1 TRAIN_OUTPUT=artifacts/smoke
```

The default four-example microbatch reduces the variance of each gradient
estimate compared with single-example updates. On a T4, reduce it to 1 or 2
if memory is limited and increase `TRAIN_GRADIENT_ACCUMULATION` to preserve an
effective batch size.

Start with a small smoke run before spending compute:

```bash
python -m training.train_lora \
  --dataset ag_news \
  --limit 200 \
  --eval-limit 100 \
  --epochs 1 \
  --output artifacts/ag-news-smoke
```

A larger initial run can use 5,000 training records and two epochs. The default
objective masks every prompt token and computes cross-entropy only over the
current example's candidate labels, rather than over the full vocabulary. Option
order is shuffled each epoch and the target label is remapped safely. The
printed validation metrics are:

- `nll`: mean negative log likelihood of the correct option label
- `accuracy`: exact option-label accuracy
- `macro_f1`: unweighted macro F1 across labels

For comparisons, keep the model revision, dataset split, limits, seed, prompt
version, and hyperparameters fixed. Report NLL and macro-F1 together; a lower
NLL can coexist with unchanged F1 when probabilities improve without changing
the argmax prediction.

The produced adapter can be loaded with PEFT for Transformers evaluation. Do
not convert it to GGUF until its unquantized adapter evaluation is complete.

Every microbatch is also printed and appended immediately to
`microbatch_losses.jsonl` in the output directory. Each record contains the
epoch, microbatch number, unscaled `nll`, gradient-accumulation-scaled
`backward_loss`, whether that microbatch triggered an optimizer step, and the
selected device. The unscaled `nll` is the value to use for plotting training
loss.
