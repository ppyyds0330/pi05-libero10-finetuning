import collections
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from openpi.training import config, data_loader

root = Path("/root/autodl-tmp/pi-libero")
report_path = root / "data/libero_10_splits/loader_verification.json"
started = time.monotonic()
train_path = root / "data/libero_10_splits/train329/episode_ids.json"
val_path = root / "data/libero_10_splits/validation50/episode_ids.json"
train_ids = json.loads(train_path.read_text())
val_ids = json.loads(val_path.read_text())
assert len(train_ids) == len(set(train_ids)) == 329
assert len(val_ids) == len(set(val_ids)) == 50
assert not (set(train_ids) & set(val_ids))
cfg = config.get_config("pi05_libero10_lora")
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
assert list(dc.episodes) == train_ids
assert cfg.model.action_horizon == 10
print("CONFIG_PASS: 329 unique train IDs; 50 held-out IDs; no overlap", flush=True)
print("Loading the real OpenPI LeRobot data loader (CPU, offline)...", flush=True)
wrapped = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
raw = wrapped._dataset
assert list(raw.episodes) == train_ids
table = raw.hf_dataset.data.table
episode_column = np.asarray(table.column("episode_index").to_numpy())
frame_column = np.asarray(table.column("frame_index").to_numpy())
actual_ids, actual_counts = np.unique(episode_column, return_counts=True)
assert set(actual_ids.tolist()) == set(train_ids)
assert not (set(actual_ids.tolist()) & set(val_ids))
assert len(wrapped) == len(raw) == table.num_rows == 87769
meta_path = root / "data/lerobot/physical-intelligence/libero/meta/episodes.jsonl"
metadata = {row["episode_index"]: row for row in
            (json.loads(line) for line in meta_path.read_text().splitlines())}
expected_frames = sum(metadata[eid]["length"] for eid in train_ids)
assert expected_frames == 87769
per_task = collections.Counter()
checked_samples = 0
for eid, count in zip(actual_ids.tolist(), actual_counts.tolist(), strict=True):
    rows = np.flatnonzero(episode_column == eid)
    lo, hi = int(rows[0]), int(rows[-1]) + 1
    assert count == metadata[eid]["length"] == hi - lo
    assert np.array_equal(frame_column[rows], np.arange(count))
    assert int(raw.episode_data_index["from"][eid]) == lo
    assert int(raw.episode_data_index["to"][eid]) == hi
    for position in {lo, hi - 1}:
        query, padding = raw._get_query_indices(position, eid)
        assert len(query["actions"]) == 10
        assert all(lo <= q < hi for q in query["actions"])
        assert all(int(episode_column[q]) == eid for q in query["actions"])
        item = wrapped[position]
        actions = np.asarray(item["actions"])
        state = np.asarray(item["state"])
        assert int(item["episode_index"]) == eid
        assert actions.shape == (10, 7), actions.shape
        assert state.shape == (8,), state.shape
        assert np.isfinite(actions).all() and np.isfinite(state).all()
        assert isinstance(item["prompt"], str) and item["prompt"]
        expected_actions = np.stack([
            np.asarray(raw.hf_dataset[q]["actions"]) for q in query["actions"]
        ])
        assert np.allclose(actions, expected_actions)
        checked_samples += 1
    per_task[tuple(metadata[eid]["tasks"])] += 1
    if checked_samples % 100 == 0:
        print(f"Verified {checked_samples} episode-boundary samples", flush=True)
assert len(per_task) == 10
source_files = [root / "openpi/src/openpi/training" / name
                for name in ("config.py", "data_loader.py")]
report = {
    "status": "PASS",
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "elapsed_seconds": round(time.monotonic() - started, 2),
    "config_name": cfg.name,
    "interpreter": os.sys.executable,
    "dataset_root": str(raw.root),
    "configured_train_episodes": len(train_ids),
    "actually_loaded_unique_episodes": len(actual_ids),
    "actually_loaded_frames": len(raw),
    "expected_train_frames": expected_frames,
    "held_out_validation_episodes": len(val_ids),
    "train_validation_overlap": [],
    "actual_episode_ids": actual_ids.tolist(),
    "checked_boundary_samples": checked_samples,
    "all_episode_boundaries_and_frame_counts_match": True,
    "sample_action_shape": [10, 7],
    "sample_state_shape": [8],
    "task_episode_counts": [{"tasks": list(task), "episodes": count}
                            for task, count in sorted(per_task.items())],
    "normalization_run": False,
    "training_run": False,
    "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in source_files},
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items()
                  if key not in ("actual_episode_ids", "source_sha256")},
                 ensure_ascii=False, indent=2), flush=True)
print("REPORT:", report_path, flush=True)

