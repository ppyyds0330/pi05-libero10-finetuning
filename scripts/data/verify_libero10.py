"""Verify the cloud dataset and expose immutable train/validation file views."""
from pathlib import Path
import hashlib,json,os,socket,time
from datetime import datetime,timezone
import numpy as np
import pyarrow.parquet as pq

ROOT=Path("/root/autodl-tmp/pi-libero/data/lerobot")
OUT=Path("/root/autodl-tmp/pi-libero/data/libero_10_splits")
def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
    return h.hexdigest()
def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def main():
    deadline=time.monotonic()+3600
    while not (ROOT/"cloud_verification.json").exists():
        if time.monotonic()>deadline: raise RuntimeError("Download completion timeout")
        time.sleep(10)
    m=json.loads((ROOT/"libero_10_split_seed7.json").read_text())
    train=set(m["train_episode_ids"]); valid=set(m["validation_episode_ids"])
    assert len(train)==329 and len(valid)==50 and not train&valid
    assert train|valid=={x["episode_id"] for x in m["files"]}
    OUT.mkdir(exist_ok=True)
    records={"train":[],"validation":[]}
    frames={"train":0,"validation":0}
    tasks={"train":{},"validation":{}}
    for position,item in enumerate(m["files"],1):
        path=ROOT/m["source_repo"]/item["path"]
        assert path.stat().st_size==item["bytes"],str(path)
        assert sha256(path)==item["sha256"],str(path)
        pf=pq.ParquetFile(path)
        assert pf.metadata.num_rows==item["frames"]
        assert {"image","wrist_image","state","actions","episode_index","task_index"}<=set(pf.schema_arrow.names)
        t=pf.read(columns=["state","actions","episode_index","frame_index","task_index"])
        state=np.asarray(t["state"].to_pylist(),dtype=np.float32)
        actions=np.asarray(t["actions"].to_pylist(),dtype=np.float32)
        assert state.shape==(item["frames"],8) and actions.shape==(item["frames"],7)
        assert np.isfinite(state).all() and np.isfinite(actions).all()
        assert set(t["episode_index"].to_pylist())=={item["episode_id"]}
        assert t["frame_index"].to_pylist()==list(range(item["frames"]))
        split=item["split"]
        assert (item["episode_id"] in train)==(split=="train")
        folder=OUT/("train329" if split=="train" else "validation50")
        destination=folder/item["path"]
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists(): os.link(path,destination)
        assert os.path.samefile(path,destination)
        records[split].append(item)
        frames[split]+=item["frames"]
        task=item["tasks"][0]
        tasks[split][task]=tasks[split].get(task,0)+1
        if position%50==0: print(json.dumps({"audit_files":position,"total":379}),flush=True)
    for split,items in records.items():
        folder=OUT/("train329" if split=="train" else "validation50")
        ids=[x["episode_id"] for x in items]
        save(folder/"episode_ids.json",ids)
        save(folder/"file_manifest.json",{"source_repo":m["source_repo"],"source_revision":m["source_revision"],"split":split,"episodes":len(ids),"frames":frames[split],"files":items})
        (folder/"README.txt").write_text(
            "Original source Parquet files; source episode IDs are preserved.\n"
            "This folder is a file-selection view, not a separately reindexed LeRobot repository.\n"
            "Use source repo physical-intelligence/libero at HF_LEROBOT_HOME="+str(ROOT)+"\n"
            "with the exact episode_ids.json selection. Compute fresh normalization on TRAIN ONLY.\n"
            "The original source meta/stats.json must not be treated as train-only normalization.\n",encoding="utf-8")
    for relative,expected in m["metadata"].items():
        assert sha256(ROOT/m["source_repo"]/relative)==expected["sha256"]
    receipt={"status":"CLOUD_LIBERO10_TRAIN329_VALID50_READY","hostname":socket.gethostname(),
        "source_repo":m["source_repo"],"source_revision":m["source_revision"],
        "download_method":"Direct cloud HTTPS download via AutoDL network turbo; no local dataset copy",
        "all_files":379,"all_frames":sum(frames.values()),"all_bytes":sum(x["bytes"] for x in m["files"]),
        "train_files":len(records["train"]),"train_frames":frames["train"],"train_bytes":sum(x["bytes"] for x in records["train"]),
        "validation_files":len(records["validation"]),"validation_frames":frames["validation"],
        "overlapping_episode_ids":[],"state_dimension":8,"action_dimension":7,"all_numeric_values_finite":True,
        "all_files_sha256_verified":True,"train_dir":str(OUT/"train329"),"validation_dir":str(OUT/"validation50"),
        "source_dir":str(ROOT/m["source_repo"]),"training_started":False,
        "per_task_counts":tasks,"finished_at_utc":datetime.now(timezone.utc).isoformat()}
    save(OUT/"preparation_complete.json",receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
if __name__=="__main__":
    try: main()
    except Exception as e:
        OUT.mkdir(exist_ok=True)
        save(OUT/"preparation_error.json",{"error":str(e),"at":datetime.now(timezone.utc).isoformat()})
        raise

