"""Result persistence: CellResult → CSV / JSON.

Output format matches paper Table 1 columns exactly.
"""

import csv
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List

from data.schemas import CellResult


def save_results(
    results: List[CellResult],
    output_dir: str,
    prefix: str = "collapse_matrix",
    formats: list = None,
) -> str:
    """Save CellResult list to disk in specified formats.

    Args:
        results: List of CellResult objects (one per matrix cell).
        output_dir: Directory to write results to.
        prefix: Filename prefix.
        formats: List of "csv" and/or "json".

    Returns:
        Path to the primary output file (CSV if available, else JSON).
    """
    if formats is None:
        formats = ["csv"]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = f"{prefix}_{timestamp}"

    primary_path = None

    if "csv" in formats:
        csv_path = os.path.join(output_dir, f"{basename}.csv")
        _save_csv(results, csv_path)
        primary_path = csv_path

    if "json" in formats:
        json_path = os.path.join(output_dir, f"{basename}.json")
        _save_json(results, json_path)
        if primary_path is None:
            primary_path = json_path

    return primary_path


def _save_csv(results: List[CellResult], path: str):
    """Save results as CSV with columns matching paper Table 1."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CellResult.csv_header())
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_csv_row())


def _save_json(results: List[CellResult], path: str):
    """Save results as JSON (list of dicts)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)


def load_results(path: str) -> List[CellResult]:
    """Load results from CSV or JSON back into CellResult objects."""
    if path.endswith(".csv"):
        return _load_csv(path)
    elif path.endswith(".json"):
        return _load_json(path)
    else:
        raise ValueError(f"Unsupported format: {path}")


def _load_csv(path: str) -> List[CellResult]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(CellResult(
                signal=row["signal"],
                domain=row["domain"],
                domain_type=row["domain_type"],
                attack=row["attack"],
                poison_ratio=float(row["poison_ratio"]),
                auroc=float(row["auroc"]),
                n_samples=int(row["n_samples"]),
                separability_drop=float(row["separability_drop"]) if row.get("separability_drop", "") else None,
                source_auroc=float(row["source_auroc"]) if row.get("source_auroc", "") else None,
                domain_robust=row.get("domain_robust", "") == "True" if row.get("domain_robust", "") else None,
            ))
    return results


def _load_json(path: str) -> List[CellResult]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [CellResult(**d) for d in data]


def summarize_results(results: List[CellResult]) -> dict:
    """Generate a summary of the collapse matrix for quick inspection.

    Returns a dict with:
        - n_cells: Total number of cells
        - n_significant_collapse: Number of cells with Δ_sep > 0.10
        - domain_robust_signals: List of (signal, domain) pairs where Δ_sep ≈ 0
        - avg_collapse_by_signal: Average Δ_sep per signal
        - avg_collapse_by_domain: Average Δ_sep per target domain
    """
    summary = {
        "n_cells": len(results),
        "n_significant_collapse": 0,
        "domain_robust_signals": [],
        "avg_collapse_by_signal": {},
        "avg_collapse_by_domain": {},
    }

    # Collect drops by signal and domain
    signal_drops = {}
    domain_drops = {}

    for r in results:
        if r.separability_drop is not None and not r.is_source_baseline:
            if r.separability_drop > 0.10:
                summary["n_significant_collapse"] += 1
            if r.domain_robust:
                summary["domain_robust_signals"].append((r.signal, r.domain))

            signal_drops.setdefault(r.signal, []).append(r.separability_drop)
            domain_drops.setdefault(r.domain, []).append(r.separability_drop)

    for signal, drops in signal_drops.items():
        summary["avg_collapse_by_signal"][signal] = round(sum(drops) / len(drops), 4)

    for domain, drops in domain_drops.items():
        summary["avg_collapse_by_domain"][domain] = round(sum(drops) / len(drops), 4)

    return summary
