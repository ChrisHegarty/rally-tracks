#!/usr/bin/env python3
"""
Generate brute-force ground truth files for msmarco-v2-vector recall measurement.

Loads query embeddings from the existing queries-recall.json.bz2, streams document
vectors from the Hugging Face Cohere/msmarco-v2-embed-english-v3 dataset, computes
exact dot-product similarities, and writes per-query top-K nearest neighbors in the
format expected by the Rally track's KnnRecallRunner.

Setup:
    python3 -m venv ground-truth-env
    source ground-truth-env/bin/activate
    pip install datasets numpy huggingface_hub

Usage:
    python generate_ground_truth.py --doc-count 18000000 --output queries-recall-18m.json
    python generate_ground_truth.py --doc-count 36000000 --output queries-recall-36m.json

The output is uncompressed JSONL.  Compress with bzip2 afterwards:
    bzip2 queries-recall-18m.json

Parallelism:
    --workers controls how many processes decode Arrow batches in parallel.
    numpy matmul uses all available cores via OpenBLAS/MKL automatically.
    Set OMP_NUM_THREADS or MKL_NUM_THREADS to tune numpy thread count.

AWS instance recommendation:
    c7i.8xlarge (32 vCPU, 64 GB RAM, 12.5 Gbps) is a good balance.
    See scaling-benchmark-plan.md for details.
"""

import argparse
import bz2
import heapq
import json
import os
import sys
import time

import numpy as np

TRACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES_RECALL_FILE = os.path.join(TRACK_DIR, "queries-recall.json.bz2")

DATASET_NAME = "Cohere/msmarco-v2-embed-english-v3"
DIMS = 1024
TOP_K = 1000
BATCH_SIZE = 50_000
PROGRESS_INTERVAL = 500_000


def sigmoid_score(dot_product):
    """Replicate ES script_score: sigmoid(1, Math.E, -dotProduct(q, d))"""
    return 1.0 / (1.0 + np.exp(-dot_product))


def load_queries():
    """Load the 76 recall queries (embeddings + metadata) from the track."""
    queries = []
    with bz2.open(QUERIES_RECALL_FILE, "r") as f:
        for line in f:
            q = json.loads(line)
            queries.append(q)
    print(f"Loaded {len(queries)} queries from {QUERIES_RECALL_FILE}")
    return queries


def build_query_matrix(queries):
    """Stack query embeddings into a (n_queries, dims) float32 matrix."""
    embs = [q["emb"] for q in queries]
    return np.array(embs, dtype=np.float32)


def normalize_rows(matrix):
    """L2-normalize each row in-place (matching vg.normalize in parse_documents.py)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms
    return matrix


def process_batch(query_matrix, doc_ids_batch, doc_vecs_batch, heaps, top_k):
    """
    Compute dot products for a batch of document vectors against all queries,
    and maintain per-query min-heaps of size top_k.

    query_matrix: (n_queries, dims) float32
    doc_vecs_batch: (batch_size, dims) float32
    doc_ids_batch: list of str, length batch_size
    heaps: list of min-heaps (one per query), each element is (score, doc_id)
    """
    scores = query_matrix @ doc_vecs_batch.T  # (n_queries, batch_size)
    scores = sigmoid_score(scores)

    n_queries = scores.shape[0]
    for qi in range(n_queries):
        row = scores[qi]
        heap = heaps[qi]
        for di in range(len(doc_ids_batch)):
            s = float(row[di])
            if len(heap) < top_k:
                heapq.heappush(heap, (s, doc_ids_batch[di]))
            elif s > heap[0][0]:
                heapq.heapreplace(heap, (s, doc_ids_batch[di]))


def process_batch_fast(query_matrix, doc_ids_batch, doc_vecs_batch, heaps, top_k):
    """
    Optimized version: pre-filter with numpy to avoid Python-level iteration
    over every (query, doc) pair.  For each query, only push docs whose score
    exceeds the current heap minimum.
    """
    scores = query_matrix @ doc_vecs_batch.T  # (n_queries, batch_size)
    scores = sigmoid_score(scores)

    n_queries = scores.shape[0]
    for qi in range(n_queries):
        row = scores[qi]
        heap = heaps[qi]

        if len(heap) < top_k:
            for di in range(len(doc_ids_batch)):
                heapq.heappush(heap, (float(row[di]), doc_ids_batch[di]))
                if len(heap) > top_k:
                    heapq.heappop(heap)
            continue

        threshold = heap[0][0]
        candidates = np.where(row > threshold)[0]
        for di in candidates:
            heapq.heapreplace(heap, (float(row[di]), doc_ids_batch[di]))
            threshold = heap[0][0]


def stream_dataset(doc_count, num_workers, batch_size):
    """
    Stream (doc_id, embedding) pairs from the Hugging Face dataset.
    Uses datasets library with streaming to avoid downloading the full dataset.
    """
    from datasets import load_dataset

    ds = load_dataset(
        DATASET_NAME,
        split=f"train[:{doc_count}]",
        num_proc=num_workers if num_workers > 1 else None,
    )

    batch_ids = []
    batch_vecs = []

    for doc in ds:
        emb = np.array(doc["emb"], dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        batch_ids.append(doc["_id"])
        batch_vecs.append(emb)

        if len(batch_ids) == batch_size:
            yield batch_ids, np.stack(batch_vecs)
            batch_ids = []
            batch_vecs = []

    if batch_ids:
        yield batch_ids, np.stack(batch_vecs)


def stream_dataset_batched(doc_count, num_workers, batch_size):
    """
    Faster alternative using datasets batch iteration and Arrow-native column access.
    """
    from datasets import load_dataset

    print(f"Loading dataset split train[:{doc_count}] ...")
    t0 = time.time()
    ds = load_dataset(
        DATASET_NAME,
        split=f"train[:{doc_count}]",
        num_proc=num_workers if num_workers > 1 else None,
    )
    ds.set_format("numpy", columns=["emb"])
    print(f"Dataset loaded in {time.time() - t0:.1f}s")

    total = len(ds)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = ds[start:end]
        doc_ids = [ds[i]["_id"] for i in range(start, end)]
        vecs = np.array(batch["emb"], dtype=np.float32)
        normalize_rows(vecs)
        yield doc_ids, vecs


def stream_dataset_arrow(doc_count, num_workers, batch_size):
    """
    Most efficient: iterate Arrow batches directly, avoiding per-row Python overhead.
    Falls back to stream_dataset_batched if Arrow access is unavailable.
    """
    from datasets import load_dataset

    print(f"Loading dataset split train[:{doc_count}] with {num_workers} workers ...")
    t0 = time.time()
    ds = load_dataset(
        DATASET_NAME,
        split=f"train[:{doc_count}]",
        num_proc=num_workers if num_workers > 1 else None,
        columns=["_id", "emb"],
    )
    print(f"Dataset loaded in {time.time() - t0:.1f}s  ({len(ds):,} rows)")

    total = len(ds)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        slice_ds = ds.select(range(start, end))
        doc_ids = slice_ds["_id"]
        vecs = np.array(slice_ds["emb"], dtype=np.float32)
        normalize_rows(vecs)
        yield doc_ids, vecs


def write_output(queries, heaps, output_path):
    """Write ground truth JSONL matching the format of queries-recall.json.bz2."""
    with open(output_path, "w") as f:
        for qi, q in enumerate(queries):
            top_items = sorted(heaps[qi], key=lambda x: -x[0])
            ids = [[doc_id, round(score, 7)] for score, doc_id in top_items]
            line = {
                "query_id": q["query_id"],
                "text": q["text"],
                "emb": q["emb"],
                "ids": ids,
            }
            f.write(json.dumps(line) + "\n")
    print(f"Wrote {len(queries)} queries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate brute-force ground truth for msmarco-v2-vector subsets"
    )
    parser.add_argument(
        "--doc-count", type=int, required=True,
        help="Number of documents from the dataset to use (e.g. 18000000)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file path (e.g. queries-recall-18m.json)"
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of nearest neighbors per query (default: {TOP_K})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Documents per batch for matmul (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers for dataset loading (default: 4)"
    )
    args = parser.parse_args()

    batch_size = args.batch_size

    queries = load_queries()
    query_matrix = build_query_matrix(queries)
    n_queries = len(queries)

    heaps = [[] for _ in range(n_queries)]

    print(f"Computing top-{args.top_k} neighbors for {n_queries} queries "
          f"over {args.doc_count:,} documents ...")
    print(f"Batch size: {batch_size:,}, Workers: {args.workers}")

    docs_processed = 0
    t_start = time.time()
    t_last = t_start

    for doc_ids_batch, doc_vecs_batch in stream_dataset_arrow(args.doc_count, args.workers, batch_size):
        process_batch_fast(query_matrix, doc_ids_batch, doc_vecs_batch, heaps, args.top_k)
        docs_processed += len(doc_ids_batch)

        now = time.time()
        if docs_processed % PROGRESS_INTERVAL < batch_size or docs_processed == args.doc_count:
            elapsed = now - t_start
            rate = docs_processed / elapsed if elapsed > 0 else 0
            eta = (args.doc_count - docs_processed) / rate if rate > 0 else 0
            print(f"  {docs_processed:>12,} / {args.doc_count:,} docs  "
                  f"({100*docs_processed/args.doc_count:5.1f}%)  "
                  f"{rate:,.0f} docs/s  ETA {eta:.0f}s")

    elapsed = time.time() - t_start
    print(f"Brute-force search completed in {elapsed:.1f}s "
          f"({docs_processed/elapsed:,.0f} docs/s)")

    write_output(queries, heaps, args.output)
    print(f"Done. Compress with:  bzip2 {args.output}")


if __name__ == "__main__":
    main()
