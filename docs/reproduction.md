# Reproduction: Cloud Training and Local WSL Simulation

English | [简体中文](reproduction.zh-CN.md)

These commands follow the recorded workflow. Documentation maintenance only checks files and patches; it does not rerun data downloads, training, or simulation. The local artifacts do not contain a complete fine-tuned checkpoint, so serving that model requires a full checkpoint or retraining.

## 1. Prepare the pinned cloud runtime

Copy or clone this project onto the cloud host, then run from its root:

```bash
export PROJECT_DIR="$PWD"
export PI_ROOT=/root/autodl-tmp/pi-libero
bash scripts/bootstrap_openpi.sh "$PI_ROOT/openpi"
python3 scripts/stage_metadata.py
cd "$PI_ROOT/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
source .venv/bin/activate

export OPENPI_DATA_HOME="$PI_ROOT/cache/openpi"
export HF_HOME="$PI_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$PI_ROOT/data/lerobot"
```

Install uv first and use a Python 3.11 / Ubuntu 22.04 / NVIDIA CUDA environment compatible with the pinned dependencies. Recorded cloud Python was 3.11.18; the archived platform report shows approximately 32 GiB GPU memory. Use a real smoke run to validate the chosen device and recipe.

The two scripts connect separate directories:

- `PROJECT_DIR`: this reproduction project.
- `PI_ROOT/openpi`: the full upstream checkout, including `src/openpi/` and `scripts/train.py`.
- `bootstrap_openpi.sh`: fetch the pinned source and submodules, then apply the project patch.
- `stage_metadata.py`: copy split IDs and saved normalization into the runtime layout. It does not download the dataset or weights, or start training.

This setup retains `/root/autodl-tmp/pi-libero`, the path used in the archived patch. Changing the runtime layout also requires updating the patch's configuration paths. Exact dependency records are in `environments/`; rebuild against the upstream `uv.lock` rather than treating a package-version snapshot as a complete installer.

## 2. Download and verify data on the cloud host

On AutoDL, `source /etc/network_turbo` enables its academic-resource acceleration in the current terminal. See the [official instructions](https://www.autodl.com/docs/network_turbo/); availability is not guaranteed.

```bash
python "$PROJECT_DIR/scripts/data/prepare_libero10.py" \
  --root "$HF_LEROBOT_HOME" --workers 6

python "$PROJECT_DIR/scripts/data/verify_libero10.py"
```

This retrieves 379 original-ID Parquet episodes from the pinned LIBERO_10 subset: 329 training and 50 held out. Source metadata still describes the full upstream dataset, so explicit episode selection is required.

Verification checks SHA256, frame counts, finite values, and 8/7-dimensional state/action arrays. It creates hard-link views for the split without duplicating the Parquet bytes. These training Parquet files are not needed by the local simulation client.

Obtain the `pi05_base` initialization with the upstream downloader or the corresponding upstream instructions:

```bash
python - <<'PY'
from openpi.shared import download
print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base"))
PY
```

The archived configuration expects `/root/autodl-tmp/pi-libero/cache/openpi/openpi-assets/checkpoints/pi05_base/params`. Confirm the resulting path. Dependency, model, tokenizer, and dataset acquisition require working network access before offline checks.

## 3. Verify the actual training subset

After data preparation:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
  python "$PROJECT_DIR/scripts/data/verify_train329_loader.py"
```

The real loader should select 329 episodes, 87,769 frames, and 658 checked boundary samples, with no held-out episode leakage. Historical reports are under `results/verification/`.

## 4. Compute training-only normalization

Saved statistics are in `results/normalization/norm_stats.json`. To reproduce their computation, use the original statistics script with an in-process normalization-only batch override:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 python - <<'PY'
import dataclasses
from openpi.training import config
from scripts import compute_norm_stats

original_get_config = config.get_config
def norm_config(name):
    return dataclasses.replace(original_get_config(name), batch_size=101, num_workers=4)
config.get_config = norm_config
compute_norm_stats.main("pi05_libero10_lora")
PY
```

101 × 869 = 87,769, covering every training frame. This override does not rewrite the training recipe. Inference must use statistics matching the trained checkpoint.

## 5. Run a GPU smoke check and train

```bash
export CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

python scripts/train.py pi05_libero10_lora \
  --exp-name "smoke_b2_$(date +%Y%m%d_%H%M%S)" \
  --batch-size 2 --num-train-steps 20
```

Check real optimizer updates, finite loss/gradients, checkpoint saving, and reloadability before a long run. The historical reload receipt is `results/verification/smoke_b2_reload.json`; its synthetic-input interface check is not a task-success measurement.

Recorded training command:

```bash
python -u scripts/train.py pi05_libero10_lora \
  --exp-name libero10_b2_s40000_20261103_234738 \
  --batch-size 2 --num-train-steps 40000
```

Choose a new experiment name for a new run. Learning-rate decay remains 10,000 steps. A checkpoint includes `params`, `assets`, and resumable `train_state`; one recorded checkpoint is about 8.9 GiB. Also allow disk space for data, initialization weights, dependencies, and smoke checkpoints.

## 6. Serve the final checkpoint on the cloud host

A complete checkpoint is required. Step 39999 represents 40,000 optimizer updates with zero-based numbering.

```bash
CKPT="$PI_ROOT/checkpoints/pi05_libero10_lora/libero10_b2_s40000_20261103_234738/39999"
python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_libero10_lora \
  --policy.dir "$CKPT"
```

Keep the service running. `--env LIBERO` alone selects the upstream default policy, not this custom fine-tuned checkpoint.

## 7. Prepare the local Ubuntu 22.04 / WSL client

On the original machine, the working client checkout is `/home/pp/projects/openpi`, with its Python environment under `examples/libero/.venv`.

For a fresh client, obtain the same source revision and submodules and create a separate Python 3.8 environment:

```bash
# PROJECT_DIR points to this project's root
export PROJECT_DIR=/home/pp/projects/pi05-libero10-finetuning
cd /home/pp/projects
git clone --no-checkout https://github.com/Physical-Intelligence/openpi.git openpi
cd openpi
git checkout --detach 215abfb217dbac7d5f1273282331b9b1866c0479
git submodule update --init --recursive
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync "$PROJECT_DIR/environments/client_requirements_cpu.txt" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match
uv pip install -e packages/openpi-client -e third_party/libero
```

The local paths below use the original example username `pp`; adapt them to your machine consistently. WSLg graphics acceleration and a compatible host GPU driver are required. The recorded setup uses:

```bash
export PYTHONPATH="$PWD/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH=/home/pp/projects/pi-libero-client/libero-config
export MUJOCO_GL=glx PYOPENGL_PLATFORM=glx
export MESA_LOADER_DRIVER_OVERRIDE=d3d12 GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
```

For first use, create the local LIBERO path configuration:

```bash
python - <<'PY'
from pathlib import Path
import os, yaml
benchmark = Path.cwd() / "third_party/libero/libero/libero"
state = Path("/home/pp/projects/pi-libero-client")
datasets = state / "datasets"
datasets.mkdir(parents=True, exist_ok=True)
folder = Path(os.environ["LIBERO_CONFIG_PATH"])
folder.mkdir(parents=True, exist_ok=True)
expected = dict(benchmark_root=str(benchmark), bddl_files=str(benchmark/"bddl_files"),
                init_states=str(benchmark/"init_files"), datasets=str(datasets),
                assets=str(benchmark/"assets"))
path = folder / "config.yaml"
if path.exists() and yaml.safe_load(path.read_text()) != expected:
    raise SystemExit("Existing LIBERO config differs; inspect before replacing.")
path.write_text(yaml.safe_dump(expected))
PY
```

The `datasets` directory here is an empty client configuration path, not the training-data download destination.

## 8. Tunnel, evaluate, and save videos

In one local WSL terminal, substitute the current instance's actual SSH host and port:

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:18000:127.0.0.1:8000 \
  -p YOUR_SSH_PORT root@YOUR_SSH_HOST
```

In another WSL terminal, activate the client and set the graphics variables from the previous section, then run:

```bash
cd /home/pp/projects/openpi
source examples/libero/.venv/bin/activate
VIDEO_DIR="$PWD/data/libero/videos/finetuned_40000_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$VIDEO_DIR"
set -o pipefail
python examples/libero/main.py \
  --args.host 127.0.0.1 --args.port 18000 \
  --args.task-suite-name libero_10 \
  --args.num-trials-per-task 1 --args.seed 7 \
  --args.video-out-path "$VIDEO_DIR" 2>&1 | tee "$VIDEO_DIR/eval.log"
```

MP4 files are saved locally and can be opened in VS Code. The first model request includes compilation time. This pinned client iterates through the entire suite and has no `--task-id` option.

For multiple trials, the original task-plus-result filename scheme can overwrite repeated outcomes. Use distinct output directories per round or record an intentional naming change to retain every video.
