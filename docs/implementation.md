# Implementation and Data Contract

English | [简体中文](implementation.zh-CN.md)

## Source changes

This project uses the upstream VLA architecture. The exact changes to two OpenPI files are in `patches/openpi-libero10-lora.patch`.

1. `src/openpi/training/config.py`: adds the `episodes` field to `DataConfig` and registers `pi05_libero10_lora`, with the 329 training IDs, `pi05_base` initialization, a 10-step action horizon, and the official LoRA variants.
2. `src/openpi/training/data_loader.py`: passes the selected episodes to `LeRobotDataset` and adapts the episode-boundary lookup for sparse original episode IDs.

In the pinned LeRobot version, rows retain global `episode_index` values while the filtered boundary table is compact. Expanding that table restores the original-ID lookup, so sampled action chunks stay within the correct episode and use endpoint padding.

## Data contract

- Agent-view and wrist-view images are rotated 180 degrees and resized/padded to 224 by the original client.
- State has 8 dimensions: end-effector position (3), axis-angle orientation (3), and gripper state (2).
- Actions have 7 dimensions; training chunks contain 10 actions.
- By default the client executes the first 5 actions and then replans.
- `extra_delta_transform=False` matches this LIBERO adapter.

The historical real-loader check confirmed 329 episode IDs and 87,769 frames, including 658 episode-boundary samples. See `results/verification/loader_verification.json`.

## Normalization

Statistics use only the 329 training episodes. Since 87,769 = 101 × 869, statistics were computed with an in-process batch size of 101 and 4 workers over 869 batches, avoiding a dropped final frame.

Normalization batch size 101 is distinct from training batch size 2. The historical statistics report was produced while the preparation-stage training config still had batch size 1; the final run later used batch size 2.

The adapter uses quantile normalization. Inference must use the matching saved `norm_stats.json`; the upstream full-dataset `meta/stats.json` is provenance, not train-only statistics.

## Configuration and actual run

The source config defaults to 10,000 steps. The actual run overrides it with `--num-train-steps 40000`, while the learning-rate decay remains 10,000 steps.

Training starts as a new run from `pi05_base`, rather than resuming the 20-step smoke checkpoint. Actual arguments and source hashes are in `configs/experiment_record.json`.

The official LoRA variants and `freeze_filter` were used. This project did not enumerate every trainable parameter and does not claim that all non-LoRA parameters were frozen.

## Runtime source preparation

`scripts/bootstrap_openpi.sh` fetches the pinned full OpenPI checkout, initializes its submodules, and applies the patch. By default that checkout lives at `/root/autodl-tmp/pi-libero/openpi`, separately from this reproduction repository.

`scripts/stage_metadata.py` copies the split IDs and saved normalization into the runtime layout. No source-tree merging is required. See [Reproduction](reproduction.md).
