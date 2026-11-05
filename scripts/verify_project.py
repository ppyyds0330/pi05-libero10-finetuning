"""Offline check of split IDs, saved statistics, evaluation and public manifest."""
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import re
ROOT = Path(__file__).resolve().parents[1]
def load(name):
    return json.loads((ROOT / name).read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()
def main():
    train = load("data/splits/train329.json")
    val = load("data/splits/validation50.json")
    source = load("data/splits/source_manifest.json")
    assert len(train) == len(set(train)) == 329
    assert len(val) == len(set(val)) == 50
    assert not set(train) & set(val)
    assert train == source["train_episode_ids"]
    assert val == source["validation_episode_ids"]
    assert set(train) | set(val) == {x["episode_id"] for x in source["files"]}
    assert sum(x["frames"] for x in source["files"] if x["split"] == "train") == 87769
    rng = random.Random(7)
    expected_train, expected_val = [], []
    for task in source["task_counts"]:
        ids = sorted(task["train_episode_ids"] + task["validation_episode_ids"])
        rng.shuffle(ids)
        expected_val.extend(ids[:5])
        expected_train.extend(ids[5:])
    assert sorted(expected_train) == train and sorted(expected_val) == val
    norm = load("results/normalization/norm_stats.json")
    norm_report = load("results/verification/normalization_verification.json")
    assert sha(ROOT / "results/normalization/norm_stats.json") == norm_report["stats_sha256"]
    stats = norm.get("norm_stats", norm)
    for name, dims in (("state", 8), ("actions", 7)):
        for field in ("mean", "std", "q01", "q99"):
            assert len(stats[name][field]) == dims
            assert all(math.isfinite(v) for v in stats[name][field])
    summary = load("results/summary.json")
    with (ROOT / "results/task_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10 and sum(x["success"] == "True" for x in rows) == 8
    assert summary["evaluation"]["success_rate"] == .8
    log = (ROOT / "results/logs/eval_40000.log").read_text()
    observed = re.findall(r"Task: ([^\n]+).*?INFO:root:Success: (True|False)", log, re.S)
    assert [(x["task"], x["success"]) for x in rows] == observed
    assert "Total success rate: 0.8" in log and "Total episodes: 10" in log
    videos = list((ROOT / "media/finetuned_40000").glob("*.mp4"))
    assert len(videos) == 10
    for row in rows:
        suffix = "success" if row["success"] == "True" else "failure"
        expected = "rollout_" + row["task"].replace(" ", "_") + "_" + suffix + ".mp4"
        assert (ROOT / "media/finetuned_40000" / expected).is_file(), expected
    baseline = load("results/pi05_base_summary.json")
    baseline_files = list((ROOT / "media/pi05_base").glob("*.mp4"))
    assert len(baseline_files) == baseline["videos"] == 10
    assert baseline["successes_from_video_labels"] == 0
    for item in baseline["files"]:
        p = ROOT / item["path"]
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"]
        assert p.name.endswith("_failure.mp4")
    manifest = load("PUBLIC_MANIFEST.json")
    for item in manifest["files"]:
        p = ROOT / item["path"]
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"], item["path"]
    report = {"status": "PASS", "manifest_files_verified": len(manifest["files"]),
              "train_episodes": 329, "held_out_episodes": 50, "overlap": 0,
              "split_recomputed_from_seed": True, "train_frames": 87769,
              "evaluation_episodes": 10, "evaluation_successes": 8,
              "videos": 10, "base_videos": 10, "full_local_checkpoint_available": False,
              "gpu_training_rerun": False}
    print(json.dumps(report, indent=2))
    return report
if __name__ == "__main__":
    main()

