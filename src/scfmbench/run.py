"""Command line entry point.

    python -m scfmbench.run --synthetic
    python -m scfmbench.run --data data/human_pancreas_norm_complexBatch.h5ad \
        --batch-key tech --label-key celltype --name pancreas
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_h5ad, make_synthetic
from .embedders import REGISTRY
from .evaluate import run_benchmark, summarise, to_markdown


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark single-cell embeddings.")
    p.add_argument("--data", type=Path, help="Path to an .h5ad file.")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data.")
    p.add_argument("--batch-key", default="tech")
    p.add_argument("--label-key", default="celltype")
    p.add_argument("--name", default=None)
    p.add_argument("--embedders", nargs="+", default=sorted(REGISTRY))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    if args.synthetic:
        dataset = make_synthetic()
    elif args.data:
        dataset = load_h5ad(
            args.data,
            batch_key=args.batch_key,
            label_key=args.label_key,
            name=args.name,
        )
    else:
        p.error("Pass --data PATH or --synthetic.")

    print(f"\n{dataset}\nBatches: {', '.join(dataset.batches)}\n")

    results = run_benchmark(dataset, args.embedders, tuple(args.seeds))
    summary = summarise(results)

    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / f"{dataset.name}_folds.csv", index=False)
    summary.to_csv(args.out / f"{dataset.name}_summary.csv", index=False)
    table = to_markdown(summary)
    (args.out / f"{dataset.name}_summary.md").write_text(table + "\n")

    print(f"\nLeave-one-batch-out results for {dataset.name}:\n")
    print(table)
    print(f"\nWritten to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
