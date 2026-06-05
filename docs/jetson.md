# Jetson Orin Nano Super Setup

## Hardware

- **Device:** NVIDIA Jetson Orin Nano Super
- **JetPack:** 6.2
- **CUDA:** 12.6
- **Python:** 3.10 (not 3.12 — JetPack 6.2 ships 3.10)
- **RAM:** 8 GB unified (CPU + GPU share the same pool)

## Local install

PyTorch must come from the Jetson AI Lab index — standard PyPI wheels are x86-only and will not install on aarch64:

```bash
# 1. Install PyTorch first
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# 2. Install other deps
pip install -r requirements-jetson.txt

# 3. Install the package entry points
pip install -e . --no-deps
```

## Docker

Build and run **on the Jetson itself**. The PyTorch wheels are aarch64-only and cannot be installed on x86.

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host.

```bash
# First-time build (slow — ~1.5 GB PyTorch layer, cached after first run)
make build-jetson

# Index
make jetson-index

# Query
make jetson-query Q="What do I know about K3s?"

# Smoke tests
make jetson-test
```

Or directly:

```bash
docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.indexer
docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.query "your question"
```

The Jetson Compose file sets `runtime: nvidia`, `NVIDIA_VISIBLE_DEVICES=all`, and `NVIDIA_DRIVER_CAPABILITIES=compute,utility` for full GPU access inside the container.

## Memory budget

With 8 GB unified RAM shared between CPU and GPU, keep these config values:

| Setting | Value | Reason |
|---|---|---|
| `embedding_batch_size` | `16` | Limits GPU memory per encode call |
| `markdown_workers` | `1` | Sequential MD extraction; avoids parallel RAM spikes |
| `pdf_workers` | `1` | Sequential PDF extraction |

The streaming indexer never accumulates all chunks globally — peak RAM is bounded to one file's chunks at a time.

## Why not `encode_multi_process`

Jetson uses NvSCI IPC instead of CUDA IPC. Cross-process CUDA tensor sharing fails on Jetson. The indexer uses single-process GPU encoding only.

## ChromaDB on aarch64

ChromaDB `0.6.3` publishes `manylinux_2_17_aarch64` wheels — no special handling needed.
