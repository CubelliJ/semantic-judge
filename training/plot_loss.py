"""Plot per-microbatch candidate NLL from a training JSONL log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"no loss records found in {args.log}")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[training]'") from exc

    x = list(range(1, len(records) + 1))
    losses = [float(record["nll"]) for record in records]
    window = max(1, min(25, len(losses) // 20))
    smoothed = [sum(losses[max(0, index - window + 1) : index + 1]) / len(losses[max(0, index - window + 1) : index + 1]) for index in range(len(losses))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(x, losses, alpha=0.25, linewidth=0.8, label="microbatch NLL")
    plt.plot(x, smoothed, linewidth=2, label=f"rolling mean ({window})")
    plt.xlabel("Microbatch")
    plt.ylabel("Candidate-label NLL")
    plt.title("LoRA training loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
