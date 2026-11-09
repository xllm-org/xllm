---
title: "Engram Host Offload"
sidebar:
  order: 83
---

## Overview

DeepSeek V4.1 W8A8 checkpoints can keep their int8 Engram tables in host memory to reduce HBM usage. The memory saving depends on the checkpoint and TP partition. Each rank computes its own hash IDs, gathers and dequantizes the requested rows with `xllm_ops.engram_int8_lookup_cpu`, and uploads those rows to its device.

Offload adds host lookup and transfer costs. ID downloads are batched before a current-stream synchronization; pinned staging storage is reused only after its previous upload completes. Graph entries own their device row buffers, so eager batches cannot replace storage captured by a graph.

## Configuration

Pass a JSON object through the `engram_config` gflag:

```bash
xllm --model /path/to/model --engram_config '{"cpu_offload":true}'
```

| Key | Default | Meaning |
|---|---|---|
| `cpu_offload` | `false` | Store int8 Engram tables in host memory. |
| `dp_shared_memory` | `false` | Share each read-only TP shard among node-local DP replicas. Requires `cpu_offload=true`. |

Omitting the flag, or passing `{}`, retains the HBM path. To enable host table sharing:

```bash
xllm --model /path/to/model --engram_config '{"cpu_offload":true,"dp_shared_memory":true}'
```

Each rank still performs its own lookup and upload because inputs and history differ across DP replicas. Sharing applies to table data, not lookup results.

Values must be JSON booleans. Unknown keys, duplicate keys, invalid JSON, and `dp_shared_memory=true` without offload raise configuration errors.

## Constraints

- Host offload currently supports TP sizes 1, 2, 4, and 8, with PP=1 and DCP=1. These offload-specific checks do not change the resident HBM path's topology constraints.
- TP shards use a ceiling-sized number of hash columns. Extra columns are zero-padded and removed after gathering; the hash-column count need not be divisible by TP.
- Offload accepts safetensors or dummy weights. Only int8 table storage has a CPU lookup path. Other storage keeps the HBM path; requesting shared host storage for it is an error.
- Host table sharing uses a node-local Gloo group and sufficient `/dev/shm` capacity. Replicas must have access to the same shared-memory namespace. A segment token is checked after attachment; matching hostnames alone do not prove that sharing works.
- Shared tables reduce host memory usage, not per-rank lookup or transfer work.

## Implementation

- Configuration: `xllm/python/models/engram_config.py`
- Hashing, staging, and execution metadata: `xllm/python/models/deepseek_v41_engram.py`
- Shared host storage: `xllm/python/models/engram_shared_memory.py`
- CPU lookup: `xllm/core/kernels/npu/xllm_ops/engram_int8_lookup_cpu.cpp`
