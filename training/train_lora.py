"""Command-line LoRA training entry point.

Example:
    python -m training.train_lora --dataset ag_news --limit 2000 --output artifacts/ag-news
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .data import load_huggingface_examples
from .data import TrainingExample
from .lora import candidate_loss, collate, encode_example, evaluate_label_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ag_news", "mnli", "boolq"), nargs="+", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[training]'") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"revision": args.revision} if args.revision else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.train()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    loss_log_path = output_path / "microbatch_losses.jsonl"
    train: list[TrainingExample] = []
    validation: list[TrainingExample] = []
    for dataset_name in args.dataset:
        train.extend(load_huggingface_examples(dataset_name, split="train", limit=args.limit))
        validation.extend(load_huggingface_examples(dataset_name, split="test", limit=args.eval_limit))
    if len(args.dataset) > 1:
        print({"datasets": args.dataset, "train_examples": len(train), "validation_examples": len(validation)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    with loss_log_path.open("w", encoding="utf-8") as loss_log:
        for epoch in range(args.epochs):
            random.shuffle(train)
            reordered_train = []
            for example in train:
                order = list(range(len(example.request.options)))
                random.shuffle(order)
                reordered_train.append(example.reordered(order))
            steps = (len(reordered_train) + args.batch_size - 1) // args.batch_size
            for batch_index, start in enumerate(range(0, len(reordered_train), args.batch_size)):
                examples = reordered_train[start : start + args.batch_size]
                features = [encode_example(tokenizer, item, args.max_length) for item in examples]
                batch = {key: value.to(device) for key, value in collate(features, tokenizer).items()}
                raw_loss = candidate_loss(model, tokenizer, batch, examples, features)
                loss = raw_loss / args.gradient_accumulation
                loss.backward()
                is_update = (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == steps
                record = {
                    "epoch": epoch + 1,
                    "microbatch": batch_index + 1,
                    "microbatches_total": steps,
                    "nll": float(raw_loss.detach().cpu().item()),
                    "backward_loss": float(loss.detach().cpu().item()),
                    "optimizer_step": is_update,
                    "device": str(device),
                }
                loss_log.write(json.dumps(record) + "\n")
                loss_log.flush()
                print(record)
                if is_update:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            metrics = evaluate_label_metrics(model, tokenizer, validation, device=device, max_length=args.max_length)
            print({"epoch": epoch + 1, "nll": metrics.nll, "accuracy": metrics.accuracy, "macro_f1": metrics.macro_f1})

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
