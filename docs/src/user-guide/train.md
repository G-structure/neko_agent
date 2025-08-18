# Finetuning on Captured Episodes

This guide explains how to finetune the agent’s ShowUI/Qwen2VL model using the
episodes recorded by `src/capture.py` (MosaicML Streaming, aka MDS).

All commands assume you are inside the Nix development shell (`nix develop`).

## What It Trains

The training script turns each annotated action in an episode into a supervised
example:

- Prompt: the same chat template used by the agent (system + task + image and
  optional short action history).
- Target: the JSON action string (e.g., `{"action":"CLICK", ...}`) observed in
  chat as `Action: {...}` during capture.

This directly teaches the model to emit the agent’s expected action schema.

## Environment Variables

Set these in `.env` (already scaffolded in `.env.example`):

```bash
# Data sources (MDS)
TRAIN_LOCAL="./data/mds"              # defaults to CAPTURE_OUT
# TRAIN_REMOTE="s3://bucket/prefix"    # optional remote MDS
TRAIN_CACHE="./data/cache"            # local cache for StreamingDataset

# Output
TRAIN_OUTPUT="./checkpoints"          # where to save model + processor

# Logging
TRAIN_LOGLEVEL=INFO                    # DEBUG|INFO|WARNING|ERROR
TRAIN_LOG_FORMAT=text                  # text|json

# Optimization
TRAIN_EPOCHS=1
TRAIN_BATCH=1
TRAIN_ACCUM=1
TRAIN_LR=5e-6
TRAIN_WD=0.0
TRAIN_MAX_STEPS=0                      # 0 disables cap
TRAIN_MAX_SAMPLES_PER_EPOCH=0          # 0 disables per-epoch cap
TRAIN_HISTORY_STEPS=0                  # include N prior actions in prompt
SEED=1337
```

The model ID and image sizes inherit from the agent defaults:

```bash
REPO_ID="showlab/ShowUI-2B"
SIZE_SHORTEST_EDGE=224
SIZE_LONGEST_EDGE=1344
```

## Running Training

```bash
# Basic run (uses .env)
python src/train.py

# With overrides
TRAIN_BATCH=2 TRAIN_EPOCHS=1 python src/train.py --output ./ckpt/showui-ft

# Using just
just train
```

Checkpoints are written under `TRAIN_OUTPUT` (`epoch-*` and `final/`). Use them
with the agent by pointing `REPO_ID` to the checkpoint folder.

## Notes

- The script streams episodes with `streaming.StreamingDataset` (MDS). It can
  read from a local directory and/or a remote (S3/R2/MinIO) when configured.
- Mixed precision is enabled on CUDA automatically. On Apple MPS it uses the
  default precision.
- This is a minimal reference trainer for reproducible fine-tuning. For large
  jobs, consider adding gradient checkpointing, deepspeed, or PEFT adapters in
  your environment.

