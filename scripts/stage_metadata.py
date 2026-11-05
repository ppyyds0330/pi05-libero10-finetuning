"""Stage saved train IDs and normalization; downloads no data or weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/pi-libero"))
args = parser.parse_args()
if args.root.resolve() != Path("/root/autodl-tmp/pi-libero"):
    parser.error("The archived patch uses /root/autodl-tmp/pi-libero; change the patch paths first.")
project = Path(__file__).resolve().parents[1]
train = json.loads((project / "data/splits/train329.json").read_text())
valid = json.loads((project / "data/splits/validation50.json").read_text())
assert len(set(train)) == 329 and len(set(valid)) == 50 and not set(train) & set(valid)
pairs = [
 ("data/splits/train329.json", "data/libero_10_splits/train329/episode_ids.json"),
 ("data/splits/validation50.json", "data/libero_10_splits/validation50/episode_ids.json"),
 ("data/splits/train329.json", "data/lerobot/train_episode_ids.json"),
 ("data/splits/validation50.json", "data/lerobot/validation_episode_ids.json"),
 ("data/splits/source_manifest.json", "data/lerobot/libero_10_split_seed7.json"),
 ("results/normalization/norm_stats.json",
  "assets/pi05_libero10_lora/physical-intelligence/libero/norm_stats.json"),
]
for source, target in pairs:
    source = project / source
    target = args.root / target
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise SystemExit("Existing file differs; inspect before replacing: " + str(target))
    shutil.copy2(source, target)
print("Staged 329/50 episode IDs and saved training normalization.")
