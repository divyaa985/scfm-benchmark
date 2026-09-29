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
from .plots import make_all, plot_umap_panels


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
    p.add_argument("--no-plots", action="store_true", help="Skip figures.")
    p.add_argument(
        "--umap",
        action="store_true",
        help="Also draw UMAP panels. Illustration only, and slow on large data.",
    )
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

    bench = run_benchmark(dataset, args.embedders, tuple(args.seeds))
    results = bench.folds
    summary = summarise(results)

    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / f"{dataset.name}_folds.csv", index=False)
    summary.to_csv(args.out / f"{dataset.name}_summary.csv", index=False)
    bench.per_class.to_csv(args.out / f"{dataset.name}_per_class.csv", index=False)
    bench.confusion.to_csv(args.out / f"{dataset.name}_confusion.csv", index=False)
    table = to_markdown(summary)
    (args.out / f"{dataset.name}_summary.md").write_text(table + "\n")

    if not args.no_plots:
        figures = make_all(
            results,
            args.out / "figures",
            dataset.name,
            per_class=bench.per_class,
            confusion=bench.confusion,
        )
        if args.umap:
            print("  drawing UMAP panels, this takes a while...")
            figures.append(
                plot_umap_panels(dataset, args.embedders, args.out / "figures")
            )
        for fig in figures:
            print(f"  figure: {fig}")

    print(f"\nLeave-one-batch-out results for {dataset.name}:\n")
    print(table)
    print(f"\nWritten to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
