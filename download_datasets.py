#!/usr/bin/env python
"""Download all 4 datasets for Defense Drift and save as local JSONL files.

Run this on a machine WITH internet access, then copy the output directory
to the GPU server.

Usage:
    python download_datasets.py --output-dir ./defend_data

Output structure:
    defend_data/
        nq_queries.jsonl
        hotpotqa_queries.jsonl
        bioasq_queries.jsonl
        finance_queries.jsonl
"""

import json
import os
import argparse
from pathlib import Path

DATASETS = {
    "nq": {
        "path": "nq_open",
        "subset": None,
        "split": "validation",  # validation has cleaner format
        "question_field": "question",
        "answer_field": "answer",
        "max_queries": 1000,
    },
    "hotpotqa": {
        "path": "hotpot_qa",
        "subset": "fullwiki",
        "split": "validation",
        "question_field": "question",
        "answer_field": "answer",
        "max_queries": 1000,
    },
    "bioasq": {
        "path": "rag-datasets/mini-bioasq",
        "subset": "question-answer-passages",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "max_queries": 500,
    },
    "finance": {
        "path": "Linq-AI-Research/FinanceRAG",
        "subset": "FinQA",
        "split": "queries",  # FinanceRAG has 'corpus' and 'queries' splits
        "question_field": "question",
        "answer_field": "answer",
        "max_queries": 500,
    },
}


def download_datasets(output_dir: str):
    """Download all datasets and save as JSONL."""
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    for name, cfg in DATASETS.items():
        output_path = os.path.join(output_dir, f"{name}_queries.jsonl")
        if os.path.exists(output_path):
            print(f"[SKIP] {name}: already exists at {output_path}")
            continue

        print(f"[DOWNLOAD] {name}: {cfg['path']} (subset={cfg['subset']}, split={cfg['split']})")

        try:
            if cfg["subset"]:
                dataset = load_dataset(cfg["path"], cfg["subset"], split=cfg["split"])
            else:
                dataset = load_dataset(cfg["path"], split=cfg["split"])
        except Exception as e:
            print(f"[ERROR] Failed to load {name}: {e}")
            # Try without split specification
            try:
                if cfg["subset"]:
                    dataset = load_dataset(cfg["path"], cfg["subset"])
                else:
                    dataset = load_dataset(cfg["path"])
                if isinstance(dataset, dict):
                    keys = list(dataset.keys())
                    dataset = dataset.get(cfg["split"], dataset[keys[0]])
            except Exception as e2:
                print(f"[FATAL] Cannot load {name}: {e2}")
                continue

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                if count >= cfg["max_queries"]:
                    break

                question = _extract(item, cfg["question_field"])
                answer = _extract(item, cfg["answer_field"])
                if not question:
                    continue

                record = {
                    "id": f"{name}_{count}",
                    "question": str(question).strip(),
                    "answer": _format_answer(answer),
                    "domain": name,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"[DONE] {name}: {count} queries → {output_path}")
        file_size = os.path.getsize(output_path) / 1024
        print(f"       Size: {file_size:.0f} KB")

    print(f"\nAll datasets saved to: {output_dir}")
    print(f"Copy this directory to your GPU server, then add to separability_collapse.yaml:")
    print(f"  datasets.local_dir: \"{output_dir}\"")


def _extract(item: dict, field: str):
    """Extract field from dataset item, handling nested keys."""
    if field in item:
        return item[field]
    # Handle list answers
    for f in [field, field + "s"]:
        if f in item:
            val = item[f]
            if isinstance(val, list) and len(val) > 0:
                if isinstance(val[0], str):
                    return val[0]
                if isinstance(val[0], dict):
                    return val[0].get("text", str(val[0]))
            return val
    return None


def _format_answer(answer) -> str:
    """Format answer as string."""
    if answer is None:
        return ""
    if isinstance(answer, list):
        if len(answer) == 0:
            return ""
        if isinstance(answer[0], str):
            return answer[0]
        return str(answer[0])
    return str(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Defense Drift datasets")
    parser.add_argument("--output-dir", default="./defend_data",
                        help="Output directory for JSONL files")
    args = parser.parse_args()
    download_datasets(args.output_dir)
