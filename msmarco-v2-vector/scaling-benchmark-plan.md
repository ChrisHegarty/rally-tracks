# Scaling Benchmark Plan: msmarco-v2-vector with GPU Indexing

## Goal

Demonstrate linear scale-out of Elasticsearch vector search using the
msmarco-v2-vector Rally track (~138M Cohere 1024-dim embeddings) with GPU
indexing.  Four tiers double data and nodes at each step, keeping
data-per-node roughly constant:

| Tier   | Documents  | Data Nodes | Shards | Corpus Groups (base64) |
|--------|------------|------------|--------|------------------------|
| 1/8    | 18,000,000 | 1          | 1      | 1                      |
| 1/4    | 36,000,000 | 2          | 2      | 1–2                    |
| 1/2    | 72,000,000 | 4          | 4      | 1–4                    |
| Full   |138,364,198 | 8          | 8      | 1–8 + tail             |

Each base64 corpus group holds 18M docs (6 files × 3M), except group 8
which holds 12M (4 × 3M).

## Recall Measurement

Rally's `KnnRecallRunner` compares approximate kNN hits against brute-force
ground truth stored in `queries-recall-*.json.bz2`.  The ground truth
differs per subset because the true nearest neighbors depend on which
documents are in the index.

**Existing ground truth files:**

| File                         | Corpus size | Top-K per query | Queries |
|------------------------------|-------------|-----------------|---------|
| `queries-recall.json.bz2`   | Full (138M) | 1000            | 76      |
| `queries-recall-10m.json.bz2`| 10M         | 100             | 76      |

**New ground truth files to generate:**

| File                         | Corpus size | Top-K per query | Queries |
|------------------------------|-------------|-----------------|---------|
| `queries-recall-18m.json.bz2`| 18M         | 1000            | 76      |
| `queries-recall-36m.json.bz2`| 36M         | 1000            | 76      |

All files use the same 76 queries (from `msmarco-passage-v2/trec-dl-2022/judged`
with Cohere embed-english-v3.0 embeddings).  Only the `ids` (nearest neighbor
lists) change per subset.

The 72M and full tiers use the existing `queries-recall.json.bz2` (full
corpus ground truth).  For 72M this means recall is measured against the full
138M neighbors — this gives a conservative (lower) recall number, which is
acceptable since the 72M index contains roughly half the full corpus and true
neighbors are a subset.  Alternatively, the 72M tier can skip recall if the
18M and 36M data points plus the full dataset are sufficient.

## Ground Truth Generation

### Approach: numpy brute-force over Hugging Face dataset

Rather than standing up an Elasticsearch instance for brute-force script_score
queries, we compute exact dot-product similarities directly in numpy:

1. Load the 76 query embeddings from `queries-recall.json.bz2`
2. Stream document vectors from `Cohere/msmarco-v2-embed-english-v3` on HF
3. L2-normalize each document vector (matching `vg.normalize` in the corpus pipeline)
4. Compute `sigmoid(1, e, -dot(q, d))` to match the ES script_score formula
5. Maintain per-query top-1000 via min-heap
6. Write JSONL output matching the existing ground truth format

### Script

`_tools/generate_ground_truth.py` — see file for usage.

### AWS Instance for Generation

**The bottleneck is dataset download and Arrow decoding, not dot-product compute.**
The actual matrix multiplication for 36M docs × 76 queries × 1024 dims takes
under 2 minutes on a modern CPU.  Downloading and decoding the HuggingFace
dataset (Arrow format, ~50–100 GB for 36M rows selecting only `_id` + `emb`)
takes 10–30 minutes depending on network bandwidth and decode parallelism.

**Recommendation: `c7i.4xlarge`** (16 vCPU, 32 GB RAM, 12.5 Gbps, ~$0.71/hr)

- 32 GB RAM is plenty (working set is ~2 GB per batch; total peaks ~4 GB)
- 16 vCPUs provide enough parallelism for HF dataset decoding (`--workers 8`)
  and numpy uses remaining cores for matmul via OpenBLAS
- 12.5 Gbps network saturates HF download throughput
- Total cost: well under $1 for both runs combined

**Does more vCPUs help?**

Marginally.  The `datasets` library parallelizes Arrow shard decoding across
`num_proc` workers, so more CPUs help with decode/parse.  numpy matmul also
benefits from more cores (OpenBLAS thread pool).  But the gains flatten beyond
~16 cores because:

- HF download bandwidth is the primary constraint, not CPU
- The per-query heap maintenance is single-threaded Python (76 queries × top-1000)
- Arrow decode parallelism has diminishing returns beyond 8–12 workers

A `c7i.8xlarge` (32 vCPU, $1.43/hr) would be ~20–30% faster for decode but
costs twice as much; not worth it for a one-off job that completes in under
an hour regardless.

### Running the Generation

```bash
# On the AWS instance:
pip install datasets numpy huggingface_hub

# 18M ground truth (~15-25 min)
python _tools/generate_ground_truth.py \
    --doc-count 18000000 \
    --output queries-recall-18m.json \
    --workers 8
bzip2 queries-recall-18m.json

# 36M ground truth (~25-45 min)
python _tools/generate_ground_truth.py \
    --doc-count 36000000 \
    --output queries-recall-36m.json \
    --workers 8
bzip2 queries-recall-36m.json
```

### Validation

After generation, verify format consistency:

```bash
python -c "
import bz2, json
for f in ['queries-recall-18m.json.bz2', 'queries-recall-36m.json.bz2']:
    with bz2.open(f, 'r') as fh:
        lines = fh.readlines()
    print(f'{f}: {len(lines)} queries')
    q = json.loads(lines[0])
    print(f'  keys: {list(q.keys())}')
    print(f'  ids len: {len(q[\"ids\"])}')
    print(f'  top score: {q[\"ids\"][0][1]}, bottom: {q[\"ids\"][-1][1]}')
"
```

## Rally Track Changes

### track.py

Add filename constants for the new recall files and extend `KnnRecallRunner`
to select the right file based on `recall-doc-set` parameter.

### operations/default.json

Extend the `recall-doc-set` selection logic to detect the 18M and 36M doc counts.

### Parameter Files

New GPU-scaling parameter files:

| File                               | Docs       | Shards | Index Type    |
|------------------------------------|------------|--------|---------------|
| `params-gpu-scaling-1node-18m.json`| 18,000,000 | 1      | `hnsw` + GPU  |
| `params-gpu-scaling-2node-36m.json`| 36,000,000 | 2      | `hnsw` + GPU  |
| `params-gpu-scaling-4node-72m.json`| 72,000,000 | 4      | `hnsw` + GPU  |
| `params-gpu-scaling-8node-full.json`|138,364,198| 8      | `hnsw` + GPU  |

## Running the Benchmark

```bash
# Tier 1: 1 data node, 18M docs
esrally race --track-path=/path/to/msmarco-v2-vector \
    --track-params=@params-gpu-scaling-1node-18m.json \
    --target-hosts=<1-node-cluster> \
    --pipeline=benchmark-only

# Tier 2: 2 data nodes, 36M docs
esrally race --track-path=/path/to/msmarco-v2-vector \
    --track-params=@params-gpu-scaling-2node-36m.json \
    --target-hosts=<2-node-cluster> \
    --pipeline=benchmark-only

# Tier 3: 4 data nodes, 72M docs
esrally race --track-path=/path/to/msmarco-v2-vector \
    --track-params=@params-gpu-scaling-4node-72m.json \
    --target-hosts=<4-node-cluster> \
    --pipeline=benchmark-only

# Tier 4: 8 data nodes, full dataset
esrally race --track-path=/path/to/msmarco-v2-vector \
    --track-params=@params-gpu-scaling-8node-full.json \
    --target-hosts=<8-node-cluster> \
    --pipeline=benchmark-only
```

## Recall Data Points

Three search configurations measured at each tier:

1. **k=10, num_candidates=100** — default search quality baseline
2. **k=10, num_candidates=400** — higher quality, moderate latency
3. **k=100, num_candidates=1000** — deep retrieval scenario

These appear in the `search_ops` array in each parameter file and produce
both throughput latency metrics (via `knn-search-*`) and recall metrics
(via `knn-recall-*`).
