# OpenPI Runtime Dependency

English | [简体中文](README.zh-CN.md)

OpenPI is pinned in [upstream.lock.json](../upstream.lock.json). This directory contains dependency notes, not the training source tree.

From this project's root, run:

```bash
bash scripts/bootstrap_openpi.sh /root/autodl-tmp/pi-libero/openpi
python3 scripts/stage_metadata.py
```

The first script fetches the full OpenPI checkout and submodules, then applies the project patch. The second places split IDs and saved normalization into the runtime layout.

Actual source: `/root/autodl-tmp/pi-libero/openpi/src/openpi/`.
Actual training entry: `/root/autodl-tmp/pi-libero/openpi/scripts/train.py`.

The reproduction repository and runtime checkout remain separate. Environment installation, dataset acquisition, and initialization weights are still required; see [Reproduction](../docs/reproduction.md).
