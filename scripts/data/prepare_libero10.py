"""Download a pinned OpenPI LIBERO-10 subset and record a reproducible split.

No training is performed. Original Parquet files and metadata remain unchanged.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import threading
import time

import requests

REPO = "physical-intelligence/libero"
REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
TASKS = [
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
]
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = True
    return LOCAL.session


def fetch(url):
    for attempt in range(6):
        try:
            response = session().get(url, timeout=(15, 60))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 15))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(root):
    source = root / REPO
    (source / "meta").mkdir(parents=True, exist_ok=True)
    resolve = "https://huggingface.co/datasets/%s/resolve/%s/" % (REPO, REVISION)
    metadata = {}
    for name in ["info.json", "tasks.jsonl", "episodes.jsonl", "stats.json"]:
        content = fetch(resolve + "meta/" + name).content
        (source / "meta" / name).write_bytes(content)
        metadata["meta/" + name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    tasks = [json.loads(line) for line in (source / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
    task_indices = {entry["task"]: entry["task_index"] for entry in tasks}
    selected = [episode for episode in episodes if set(episode["tasks"]) & set(TASKS)]
    assert len(selected) == 379
    assert sum(episode["length"] for episode in selected) == 101469
    rng = random.Random(7)
    train, validation, task_counts = [], [], []
    for suite_id, task in enumerate(TASKS):
        ids = sorted(episode["episode_index"] for episode in selected if task in episode["tasks"])
        rng.shuffle(ids)
        held_out, training = sorted(ids[:5]), sorted(ids[5:])
        train.extend(training)
        validation.extend(held_out)
        task_counts.append({"suite_task_id": suite_id, "source_task_index": task_indices[task], "task": task,
                            "total": len(ids), "train": len(training), "validation": len(held_out),
                            "train_episode_ids": training, "validation_episode_ids": held_out})
    train, validation = sorted(train), sorted(validation)
    assert len(train) == 329 and len(validation) == 50 and not set(train) & set(validation)
    index = {}
    chunks = sorted({episode["episode_index"] // 1000 for episode in selected})
    for chunk in chunks:
        url = "https://huggingface.co/api/datasets/%s/tree/%s/data/chunk-%03d?limit=1000" % (REPO, REVISION, chunk)
        while url:
            response = fetch(url)
            index.update({entry["path"]: entry for entry in response.json() if entry["type"] == "file"})
            url = response.links.get("next", {}).get("url")
    files = []
    for episode in selected:
        episode_id = episode["episode_index"]
        path = "data/chunk-%03d/episode_%06d.parquet" % (episode_id // 1000, episode_id)
        entry = index[path]
        sha = entry.get("lfs", {}).get("oid")
        if not sha or len(sha) != 64:
            raise RuntimeError("Missing LFS SHA-256 for " + path)
        files.append({"path": path, "episode_id": episode_id, "frames": episode["length"],
                      "tasks": episode["tasks"], "bytes": entry["size"], "sha256": sha,
                      "split": "train" if episode_id in train else "validation"})
    manifest = {"source_repo": REPO, "source_revision": REVISION, "suite": "libero_10", "seed": 7,
                "split_algorithm": "Python random.Random(7); shuffle sorted episode IDs in local suite task order; first 5 per task held out",
                "all_episodes": 379, "all_frames": 101469, "train_episode_ids": train,
                "validation_episode_ids": validation, "task_counts": task_counts, "metadata": metadata, "files": files}
    manifest["total_bytes"] = sum(item["bytes"] for item in files)
    manifest["train_bytes"] = sum(item["bytes"] for item in files if item["split"] == "train")
    manifest["train_frames"] = sum(item["frames"] for item in files if item["split"] == "train")
    save_json(root / "libero_10_split_seed7.json", manifest)
    save_json(root / "train_episode_ids.json", train)
    save_json(root / "validation_episode_ids.json", validation)
    (root / "DATA_LAYOUT.md").write_text(
        "# Cloud LIBERO-10 source and split\n\n"
        "Source: physical-intelligence/libero @ " + REVISION + "\n\n"
        "This is the full LIBERO-10 subset (379 original Parquet episodes), not all 1693 episodes and not the 130-task raw HDF5 release.\n"
        "Original source metadata is preserved unchanged and describes the upstream full dataset.\n"
        "Training selection: 329 source episode IDs in train_episode_ids.json. Validation selection: 50 IDs held out in this cloud directory.\n"
        "Use the episode selection explicitly when training; do not request all upstream episodes.\n"
        "The upstream meta/stats.json is provenance metadata, not normalization fitted to this training split.\n"
        "Compute fresh normalization using training episodes only before training. No training has been started.\n",
        encoding="utf-8")
    return manifest, source, resolve


def download_one(item, source, resolve):
    destination = source / item["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == item["bytes"] and digest(destination) == item["sha256"]:
            return item["episode_id"]
        raise RuntimeError("Existing completed file failed integrity check: " + str(destination))
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(12):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == item["bytes"]:
            if digest(partial) == item["sha256"]:
                partial.replace(destination)
                return item["episode_id"]
            partial.unlink()
            offset = 0
        try:
            response = session().get(resolve + item["path"] + "?download=true",
                                     headers={"Range": "bytes=%d-" % offset} if offset else {},
                                     stream=True, timeout=(20, 90))
            response.raise_for_status()
            if offset and response.status_code != 206:
                offset = 0
            if offset and not response.headers.get("Content-Range", "").startswith("bytes %d-" % offset):
                raise RuntimeError("Invalid resume range")
            with response, partial.open("ab" if offset else "wb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        output.write(block)
            if partial.stat().st_size != item["bytes"]:
                raise RuntimeError("Incomplete file")
            if digest(partial) != item["sha256"]:
                raise RuntimeError("SHA-256 mismatch")
            partial.replace(destination)
            return item["episode_id"]
        except Exception as error:
            print(json.dumps({"retry_episode": item["episode_id"], "attempt": attempt + 1,
                              "error": str(error)[:220]}), flush=True)
            if attempt == 11:
                raise
            time.sleep(min(2 ** min(attempt, 4), 20))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    manifest, source, resolve = prepare(args.root)
    print(json.dumps({"prepared": True, "episodes": 379, "train": 329, "validation": 50,
                      "bytes": manifest["total_bytes"], "train_bytes": manifest["train_bytes"]}), flush=True)
    if args.prepare_only:
        return
    remaining = sum(max(0, item["bytes"] - (source / item["path"]).stat().st_size)
                    if (source / item["path"]).exists() else item["bytes"] for item in manifest["files"])
    physical_free = shutil.disk_usage(args.root).free
    if physical_free < remaining + 4 * 1024 ** 3:
        raise RuntimeError("Insufficient cloud data disk space (including 4 GiB reserve)")
    started = time.monotonic()
    completed, verified_bytes = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, item, source, resolve): item for item in manifest["files"]}
        for future in as_completed(futures):
            episode = future.result()
            item = futures[future]
            completed.append(episode)
            verified_bytes += item["bytes"]
            status = {"status": "DOWNLOADING" if len(completed) < 379 else "ALL_379_SHA256_VERIFIED",
                      "verified_files": len(completed), "verified_bytes": verified_bytes,
                      "total_files": 379, "total_bytes": manifest["total_bytes"],
                      "elapsed_seconds": round(time.monotonic() - started, 1),
                      "completed_episode_ids": sorted(completed)}
            save_json(args.root / "download_status.json", status)
            print(json.dumps({key: value for key, value in status.items() if key != "completed_episode_ids"}), flush=True)
    save_json(args.root / "cloud_verification.json", {
        "status": "ALL_379_SHA256_VERIFIED", "source_revision": REVISION,
        "verified_files": 379, "verified_bytes": verified_bytes,
        "train_files": 329, "validation_files": 50,
        "verified_at_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()


