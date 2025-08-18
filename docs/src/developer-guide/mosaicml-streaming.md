# MosaicML Streaming (MDS) — Practical Guide for Codex Agents

This document explains **what** the MosaicML Streaming library is, **why** it’s useful for large, sharded datasets, and **how** to write and read **MDS** shards locally or to an S3-compatible store (MinIO, R2, etc.). It includes runnable patterns you can copy into data pipelines.

---

## Why Streaming/MDS?

- **Sharded, resumable, deterministic** data loading across workers/nodes, with built-in shuffling and fault tolerance. Useful when the dataset is bigger than local disk/RAM or lives in object storage.
- **MDS** is the on-disk format used by Streaming: a set of shard files (e.g., `shard.00000.mds[.zstd]`) plus an `index.json` that records metadata/offsets for fast random access.

---

## Key Concepts (10,000-ft view)

- **MDSWriter**: builds shards from Python dict samples (columnar schema you define), supports compression and **direct upload to remote storage**.
- **StreamingDataset**: reads from a local cache and/or directly from **remote** (S3, GCS, etc.), handling shuffling, ordering, and resumption deterministically.
- **S3-compatible endpoints**: Use standard AWS creds + `S3_ENDPOINT_URL` to point to MinIO, Cloudflare R2, etc.

---

## Minimal: Write an MDS dataset

```python
from streaming import MDSWriter
import json, io, numpy as np

# 1) Define the schema: map column -> storage type.
# Supported logical types include 'bytes', 'str', numeric scalars, etc. (see docs)
columns = {
    "frame_table": "bytes",   # packed npy/npz/etc.
    "meta_json":   "str",     # utf-8 JSON string
}

# 2) Writer: local out + optional remote (s3://bucket/prefix).
# Compression & shard size are critical for throughput.
with MDSWriter(
    out="data/mds/train",                 # local staging/cache
    columns=columns,
    compression="zstd",                   # fast decode in training
    hashes="sha1",                        # integrity
    size_limit="128MB",                   # shard target
    remote="s3://my-bucket/datasets/ui-trajs"  # optional: upload shards as they close
) as w:
    for episode in make_episodes():
        # Pack the frame table as bytes (e.g., np.save to a BytesIO)
        buf = io.BytesIO()
        np.save(buf, episode["frame_table"].astype("float32"))
        sample = {
            "frame_table": buf.getvalue(),
            "meta_json":   json.dumps(episode["meta"], ensure_ascii=False),
        }
        w.write(sample)

Notes
	•	remote=... enables automatic upload of completed shards to object storage while writing.
	•	Prefer compression="zstd" for read-time throughput; it’s commonly used in training loops.

⸻

Configure S3 / S3-Compatible Remotes

Set standard AWS creds and (for non-AWS) an endpoint override:

export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
# For MinIO / Cloudflare R2 / etc.:
export S3_ENDPOINT_URL="https://<your-endpoint>"   # e.g., https://<accountid>.r2.cloudflarestorage.com

S3_ENDPOINT_URL is the knob for S3-compatible providers.

⸻

Read an MDS dataset (local cache + remote)

from streaming import StreamingDataset
from torch.utils.data import DataLoader

ds = StreamingDataset(
    remote="s3://my-bucket/datasets/ui-trajs",  # remote is optional if data is fully local
    local="data/cache/ui-trajs",                # local cache directory
    shuffle=True, shuffle_seed=1337,            # deterministic across workers
)

def collate(batch):
    # Each item is a dict with your columns.
    return batch

loader = DataLoader(ds, batch_size=8, num_workers=8, collate_fn=collate, persistent_workers=True)
for batch in loader:
    ...

	•	StreamingDataset handles shuffling/resume deterministically across workers and epochs.
	•	Local cache fills on demand; you can pin/cache shards for repeated jobs.

⸻

Practical Tips
	•	Shard size: 64–256 MB is a good default for training throughput; too tiny increases open/seek overhead, too huge hurts parallelism (tune per infra). (General best practice aligned with MDS design. )
	•	Compression: zstd balances ratio & decode speed for training (common choice in large-scale pipelines).
	•	Schema: store big blobs under bytes (e.g., numpy npz) and small metadata as str/scalars—keeps readers simple.
	•	Determinism: set shuffle_seed once for the run; Streaming takes care of global ordering across ranks.

⸻

FAQ

Q: Can I mix multiple datasets?
Yes—construct multiple StreamingDataset streams or concatenate datasets; Streaming supports mixing sources and remote stores. See the docs’ “main concepts” and examples.

Q: How do I resume mid-epoch?
The index and shard metadata let Streaming resume deterministically; you don’t need custom offset logic.

Q: Non-AWS S3?
Use S3_ENDPOINT_URL with your provider’s endpoint (R2/MinIO/etc.).

---
