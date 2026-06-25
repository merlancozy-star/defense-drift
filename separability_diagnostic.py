#!/usr/bin/env python
"""Defense Drift — Separability Collapse Diagnostic (Table 1).

Main entry point for computing the separability collapse matrix.

Usage:
    # Single signal on single domain
    python separability_diagnostic.py --signal perplexity --domain nq --ratio 0.1

    # All signals on source domains (M1 checkpoint)
    python separability_diagnostic.py --signal all --domain nq,hotpotqa

    # Single cross-domain comparison (M3)
    python separability_diagnostic.py --signal all --source nq --target bioasq --ratio 0.1

    # Full collapse matrix (M4)
    python separability_diagnostic.py --full-matrix

    # Dry run with synthetic data (no models needed)
    python separability_diagnostic.py --dry-run
"""

import sys
import os
import logging
import click
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import load_config
from utils.results import save_results, summarize_results
from pipeline.build_collapse_matrix import CollapseMatrixBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("separability_diagnostic")


@click.command()
@click.option("--config", "-c", default=None,
              help="Path to separability_collapse.yaml")
@click.option("--signal", "-s", default="perplexity",
              help="Signal name or 'all' (perplexity, drs, cluster_distance, attention_variance)")
@click.option("--domain", "-d", default=None,
              help="Domain(s) to evaluate, comma-separated (nq, hotpotqa, bioasq, finance)")
@click.option("--source", default=None,
              help="Source domain for cross-domain comparison")
@click.option("--target", default=None,
              help="Target domain for cross-domain comparison")
@click.option("--attack", "-a", default="poisonedrag",
              help="Attack type (poisonedrag, corruptrag)")
@click.option("--ratio", "-r", default=0.1, type=float,
              help="Poison ratio (0.05, 0.1, 0.2, 0.4)")
@click.option("--max-queries", default=None, type=int,
              help="Max queries per cell (overrides config)")
@click.option("--full-matrix", is_flag=True,
              help="Run the full Cartesian product matrix")
@click.option("--dry-run", is_flag=True,
              help="Synthetic data + random AUROC (no models, no datasets)")
@click.option("--offline", is_flag=True,
              help="Skip HF dataset download, use synthetic data with REAL models")
@click.option("--output-dir", default=None,
              help="Override output directory")
@click.option("--no-parallel", is_flag=True,
              help="Disable parallel processing")
def main(
    config, signal, domain, source, target, attack, ratio,
    max_queries, full_matrix, dry_run, offline, output_dir, no_parallel,
):
    """Defense Drift — Separability Collapse Diagnostic.

    Compute AUROC for poison/clean separability across signals, domains,
    attacks, and poison ratios. Produces Table 1 of the paper.
    """
    # Load configuration
    cfg = load_config(config)
    if output_dir:
        cfg.output.results_dir = output_dir
    cfg.offline = offline  # Attach offline flag for downstream components

    logger.info("=" * 60)
    logger.info("Defense Drift — Separability Collapse Diagnostic")
    logger.info("=" * 60)

    if offline:
        logger.warning("OFFLINE mode: skipping HF datasets, using synthetic queries+corpus")
        logger.warning("Model-based scoring (PPL/embeddings) is REAL — only the text is synthetic")

    if dry_run:
        logger.warning(
            "DRY RUN mode: using synthetic data only. "
            "Results are for pipeline validation, NOT publication."
        )
        _dry_run(cfg)
        return

    # Initialize
    builder = CollapseMatrixBuilder(cfg)

    try:
        builder.initialize_models()

        if full_matrix:
            # M4: Full Cartesian product
            results = builder.build_full_matrix(
                max_queries=max_queries,
                parallel=not no_parallel,
            )
        elif source and target:
            # M3: Cross-domain comparison
            logger.info(f"Cross-domain: {source} → {target}")
            signals = cfg.signals if signal == "all" else [signal]

            results = builder.build_full_matrix(
                signals=signals,
                domains=[source, target],
                attacks=[attack],
                ratios=[ratio],
                max_queries=max_queries,
            )
        else:
            # Single/multiple domain evaluation
            domains = domain.split(",") if domain else cfg.all_domains
            signals = cfg.signals if signal == "all" else [signal]

            results = builder.build_full_matrix(
                signals=signals,
                domains=domains,
                attacks=[attack],
                ratios=[ratio],
                max_queries=max_queries,
            )

        # Save results
        if results:
            save_path = save_results(
                results,
                cfg.output.results_dir,
                prefix=cfg.output.csv_prefix,
                formats=cfg.output.save_format,
            )
            logger.info(f"Results saved to: {save_path}")

            # Summary
            summary = summarize_results(results)
            logger.info(f"Summary: {summary}")

            # Print results table
            _print_results(results)

    finally:
        builder.shutdown_models()


def _dry_run(cfg):
    """Run a minimal pipeline test with synthetic data.

    No real models or datasets are loaded. Uses:
    - Synthetic queries from DatasetLoader
    - Template-mode poison (no generator needed)
    - Embedder and generator can use CPU fallback or be skipped
    """
    from data.schemas import CellResult
    import random
    import numpy as np

    logger.info("Generating synthetic dry-run results...")

    rng = random.Random(cfg.diagnostic.random_seed)
    results = []

    for domain in cfg.all_domains:
        domain_type = cfg.get_domain_type(domain)
        for signal_name in cfg.signals:
            for attack_name in cfg.attacks:
                for ratio in cfg.poison_ratios:
                    # Generate synthetic AUROC
                    # Source: AUROC ~ 0.80-0.90 (representative of published results)
                    # Target: AUROC ~ 0.60-0.75 (simulating collapse)
                    if domain_type == "source":
                        auroc = rng.uniform(0.80, 0.92)
                    else:
                        auroc = rng.uniform(0.58, 0.76)

                    n_samples = rng.randint(80, 200)

                    result = CellResult(
                        signal=signal_name,
                        domain=domain,
                        domain_type=domain_type,
                        attack=attack_name,
                        poison_ratio=ratio,
                        auroc=round(auroc, 4),
                        n_samples=n_samples,
                        is_source_baseline=(domain_type == "source"),
                    )
                    results.append(result)

    # Compute drops
    source_domains = cfg.source_domains
    source_aurocs = {}
    for r in results:
        if r.domain_type == "source":
            key = (r.signal, r.attack, r.poison_ratio)
            source_aurocs.setdefault(key, []).append(r.auroc)

    source_avg = {}
    for key, aurocs in source_aurocs.items():
        source_avg[key] = sum(aurocs) / len(aurocs)

    for r in results:
        if r.domain_type == "target":
            key = (r.signal, r.attack, r.poison_ratio)
            ref = source_avg.get(key, float("nan"))
            if not np.isnan(ref):
                r.compute_drop(ref)

    # Save
    save_path = save_results(
        results,
        cfg.output.results_dir,
        prefix=f"{cfg.output.csv_prefix}_dryrun",
        formats=cfg.output.save_format,
    )
    logger.info(f"Dry-run results saved to: {save_path}")

    _print_results(results)


def _print_results(results):
    """Print results in a readable table format."""
    from data.schemas import CellResult

    if not results:
        return

    # Header
    header = CellResult.csv_header()
    col_widths = {h: len(h) for h in header}
    for r in results:
        row = r.to_csv_row()
        for h in header:
            val = str(row.get(h, ""))
            col_widths[h] = max(col_widths[h], len(val))

    # Print
    sep = " | "
    header_line = sep.join(h.ljust(col_widths[h]) for h in header)
    print("\n" + "=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))

    source_results = [r for r in results if r.domain_type == "source"]
    target_results = [r for r in results if r.domain_type == "target"]

    for r in source_results:
        row = r.to_csv_row()
        line = sep.join(str(row.get(h, "")).ljust(col_widths[h]) for h in header)
        print(line)

    if source_results and target_results:
        print("-" * len(header_line))

    for r in target_results:
        row = r.to_csv_row()
        line = sep.join(str(row.get(h, "")).ljust(col_widths[h]) for h in header)
        print(line)

    print("=" * len(header_line) + "\n")


if __name__ == "__main__":
    main()
