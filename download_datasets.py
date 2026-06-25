#!/usr/bin/env python
"""Download queries + corpus for all 4 Defense Drift datasets.

Run on a machine WITH internet, then copy to GPU server.

Usage:
    python download_datasets.py --output-dir ./defend_data

Output:
    defend_data/
        nq_queries.jsonl         # NQ-Open questions
        nq_corpus.jsonl          # Wikipedia passages for NQ
        hotpotqa_queries.jsonl   # HotpotQA questions
        hotpotqa_corpus.jsonl    # Wikipedia context sentences
        bioasq_queries.jsonl     # BioASQ questions
        bioasq_corpus.jsonl      # PubMed passages
        finance_queries.jsonl    # FinanceQA questions
        finance_corpus.jsonl     # Financial corpus
"""

import json
import os
import argparse
from pathlib import Path

DATASETS = {
    "nq": {
        "query_path": "nq_open",
        "query_subset": None,
        "query_split": "validation",
        "question_field": "question",
        "answer_field": "answer",
        "max_queries": 1000,
        # Use simple English Wikipedia subset as corpus
        "corpus_path": "wikipedia",
        "corpus_subset": "20220301.en",
        "corpus_split": "train",
        "corpus_text_field": "text",
        "max_corpus": 20000,
        "corpus_streaming": True,  # Wikipedia is huge, use streaming
    },
    "hotpotqa": {
        "query_path": "hotpot_qa",
        "query_subset": "fullwiki",
        "query_split": "validation",
        "question_field": "question",
        "answer_field": "answer",
        "max_queries": 1000,
        # Extract Wikipedia sentences from context field (built-in)
        "corpus_path": None,  # corpus comes from the queries' context field
    },
    "bioasq": {
        "query_path": "rag-datasets/mini-bioasq",
        "query_subset": "question-answer-passages",
        "query_split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "max_queries": 500,
        # Built-in passages field
        "corpus_path": None,
    },
    "finance": {
        "query_path": "Linq-AI-Research/FinanceRAG",
        "query_subset": "FinQA",
        "query_split": "queries",
        "question_field": "question",
        "answer_field": "answer",
        "max_queries": 500,
        # FinanceRAG has a separate 'corpus' split
        "corpus_path": "Linq-AI-Research/FinanceRAG",
        "corpus_subset": "FinQA",
        "corpus_split": "corpus",
        "corpus_text_field": "text",
        "max_corpus": 50000,
        "corpus_streaming": False,
    },
}


def download_datasets(output_dir: str):
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    for name, cfg in DATASETS.items():
        # ── Queries ──
        q_path = os.path.join(output_dir, f"{name}_queries.jsonl")
        if os.path.exists(q_path):
            print(f"[SKIP] {name} queries: already exists")
        else:
            print(f"[QUERIES] {name}: {cfg['query_path']}")
            try:
                if cfg["query_subset"]:
                    ds = load_dataset(cfg["query_path"], cfg["query_subset"],
                                      split=cfg["query_split"])
                else:
                    ds = load_dataset(cfg["query_path"], split=cfg["query_split"])
            except Exception:
                ds = _load_fallback(cfg["query_path"], cfg["query_subset"],
                                    cfg["query_split"])

            count = _save_queries(ds, q_path, name, cfg, output_dir)
            print(f"  → {count} queries saved ({_size_kb(q_path):.0f} KB)")

        # ── Corpus ──
        c_path = os.path.join(output_dir, f"{name}_corpus.jsonl")
        if os.path.exists(c_path):
            print(f"[SKIP] {name} corpus: already exists")
            continue

        if cfg["corpus_path"]:
            # External corpus
            _download_external_corpus(name, cfg, c_path)
        elif name == "hotpotqa":
            # Extract from queries' context field
            _extract_hotpotqa_corpus(cfg, q_path, c_path)
        elif name == "bioasq":
            # Extract from queries' passages field
            _extract_bioasq_corpus(cfg, c_path)
        else:
            # Use the same data as corpus
            print(f"[CORPUS] {name}: using queries as corpus fallback")
            import shutil
            shutil.copy(q_path, c_path)

    _print_summary(output_dir)


def _load_fallback(path, subset, split):
    """Try loading without split, then pick first available."""
    from datasets import load_dataset
    if subset:
        ds = load_dataset(path, subset)
    else:
        ds = load_dataset(path)
    if isinstance(ds, dict):
        ds = ds.get(split, list(ds.values())[0])
    return ds


def _save_queries(ds, q_path, name, cfg, output_dir):
    """Save queries as JSONL, return count."""
    count = 0
    with open(q_path, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= cfg["max_queries"]:
                break
            q = _extract(item, cfg["question_field"])
            a = _extract(item, cfg["answer_field"])
            if not q:
                continue
            f.write(json.dumps({
                "id": f"{name}_{count}",
                "question": str(q).strip(),
                "answer": _fmt(a),
                "domain": name,
            }, ensure_ascii=False) + "\n")
            count += 1
    return count


def _download_external_corpus(name, cfg, c_path):
    """Download a separate corpus dataset."""
    from datasets import load_dataset
    print(f"[CORPUS] {name}: {cfg['corpus_path']} (this may take a minute...)")

    streaming = cfg.get("corpus_streaming", False)
    try:
        if cfg.get("corpus_subset"):
            ds = load_dataset(cfg["corpus_path"], cfg["corpus_subset"],
                              split=cfg["corpus_split"], streaming=streaming)
        else:
            ds = load_dataset(cfg["corpus_path"], split=cfg["corpus_split"],
                              streaming=streaming)
    except Exception:
        ds = _load_fallback(cfg["corpus_path"], cfg.get("corpus_subset"),
                            cfg["corpus_split"])

    count = 0
    max_c = cfg.get("max_corpus", 50000)
    text_field = cfg.get("corpus_text_field", "text")
    with open(c_path, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_c:
                break
            t = _extract(item, text_field)
            if not t or len(str(t).strip()) < 50:
                continue
            # For Wikipedia: take first 500 chars (first paragraph) as a passage
            text = str(t).strip()
            if name == "nq" and len(text) > 500:
                text = text[:500] + "..."
            f.write(json.dumps({
                "id": f"{name}_corpus_{count}",
                "text": text,
            }, ensure_ascii=False) + "\n")
            count += 1
            if count % 5000 == 0:
                print(f"  ... {count} passages")
    print(f"  → {count} passages saved ({_size_kb(c_path):.0f} KB)")


def _extract_hotpotqa_corpus(cfg, q_path, c_path):
    """Extract Wikipedia sentences from HotpotQA fullwiki queries.

    Re-download the dataset to get the context field directly
    (queries JSONL only stores question+answer, not full context).
    """
    from datasets import load_dataset
    print(f"[CORPUS] hotpotqa: extracting from fullwiki context...")
    try:
        ds = load_dataset(cfg["query_path"], cfg["query_subset"],
                          split=cfg["query_split"])
    except Exception:
        ds = _load_fallback(cfg["query_path"], cfg["query_subset"],
                            cfg["query_split"])

    seen = set()
    count = 0
    with open(c_path, "w", encoding="utf-8") as f:
        for item in ds:
            ctx = item.get("context", {})
            titles = ctx.get("title", [])
            sentences = ctx.get("sentences", [])
            for title, sents in zip(titles, sentences):
                for sent in sents:
                    if sent in seen or len(sent.strip()) < 30:
                        continue
                    seen.add(sent)
                    f.write(json.dumps({
                        "id": f"hp_corpus_{count}",
                        "text": sent.strip(),
                        "title": title,
                    }, ensure_ascii=False) + "\n")
                    count += 1
            if count >= 50000:
                break
    print(f"  → {count} unique sentences saved ({_size_kb(c_path):.0f} KB)")


def _extract_bioasq_corpus(cfg, c_path):
    """Extract PubMed passages from BioASQ mini dataset."""
    from datasets import load_dataset
    print(f"[CORPUS] bioasq: extracting from passages field...")
    try:
        ds = load_dataset(cfg["query_path"], cfg["query_subset"],
                          split=cfg["query_split"])
    except Exception:
        ds = _load_fallback(cfg["query_path"], cfg["query_subset"],
                            cfg["query_split"])

    seen = set()
    count = 0
    with open(c_path, "w", encoding="utf-8") as f:
        for item in ds:
            passages = item.get("passages", [])
            if isinstance(passages, dict):
                passages = passages.get("text", [])
            if isinstance(passages, str):
                passages = [passages]
            for p in passages:
                p_text = p if isinstance(p, str) else str(p)
                if p_text in seen or len(p_text.strip()) < 30:
                    continue
                seen.add(p_text)
                f.write(json.dumps({
                    "id": f"bio_corpus_{count}",
                    "text": p_text.strip(),
                }, ensure_ascii=False) + "\n")
                count += 1
    print(f"  → {count} passages saved ({_size_kb(c_path):.0f} KB)")


def _print_summary(output_dir):
    print(f"\n{'='*60}")
    print("Download complete. Files:")
    for f in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, f)
        kb = os.path.getsize(path) / 1024
        print(f"  {f:40s} {kb:8.0f} KB")
    print(f"\nCopy to GPU server and set in separability_collapse.yaml:")
    print(f'  datasets.local_dir: "{output_dir}"')


def _extract(item, field):
    if field in item:
        return item[field]
    for f in [field, field + "s"]:
        if f in item:
            v = item[f]
            if isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], str):
                    return v[0]
                if isinstance(v[0], dict):
                    return v[0].get("text", str(v[0]))
            return v
    return None


def _fmt(a):
    if a is None:
        return ""
    if isinstance(a, list):
        if len(a) == 0:
            return ""
        return str(a[0]) if isinstance(a[0], str) else str(a[0])
    return str(a)


def _size_kb(path):
    return os.path.getsize(path) / 1024 if os.path.exists(path) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Defense Drift queries + corpus"
    )
    parser.add_argument("--output-dir", default="./defend_data",
                        help="Output directory")
    args = parser.parse_args()
    download_datasets(args.output_dir)
