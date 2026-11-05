# Dataset and Split Records

English | [简体中文](README.zh-CN.md)

Original LIBERO dataset entry:
https://github.com/Lifelong-Robot-Learning/LIBERO#Dataset

The experiment trains on PI's LeRobot-formatted data, rather than reading the raw HDF5 files directly:
https://huggingface.co/datasets/physical-intelligence/libero

Pinned revision: `a4336d589d589045d1c56423ffdf3b88a0e19b1f`.

- LIBERO_10 source subset: 379 demonstrations and 101,469 frames.
- `splits/train329.json`: 329 training episode IDs, 87,769 frames.
- `splits/validation50.json`: 50 held-out episode IDs, 13,700 frames.
- IDs are shuffled separately by task with seed 7, holding out 5 per task. Task order and the exact algorithm are recorded in `splits/source_manifest.json`.
- Original upstream episode IDs are preserved.
- The source manifest records SHA256, byte size, task membership, and split assignment for reproducible verification.
- The held-out episodes are excluded from training and normalization. No dedicated validation loss was computed.

This directory contains no Parquet files, images, or model weights. Run data preparation on the cloud host; see [Reproduction](../docs/reproduction.md).
