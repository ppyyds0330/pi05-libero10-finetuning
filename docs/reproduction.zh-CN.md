# 复现步骤：云端训练与本地 WSL 仿真

[English](reproduction.md) | 简体中文

本文件按本次实际流程整理。项目整理时只做资料与补丁核验，没有重新下载数据、训练或仿真。完整训练后权重未备份到本地，因此启动该微调模型之前需要取得完整检查点或重新训练。

## 1. 云端准备固定版本

把本项目放在云端任意目录，从本项目根目录运行：

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

需要预先安装 uv，并使用可运行当前依赖的 Python 3.11 / Ubuntu 22.04 / NVIDIA CUDA 环境。确切依赖见 environments 和上游 uv.lock。云端记录的解释器为 Python 3.11.18；归档时平台报告 GPU 显存约 32 GiB。GPU 配方以 smoke 的实际结果为准。

两个脚本将独立目录配套使用：

- `PROJECT_DIR`：本复现项目。
- `PI_ROOT/openpi`：完整上游源码，包含 `src/openpi/` 和 `scripts/train.py`。
- `bootstrap_openpi.sh`：取得固定版本源码及子模块并应用补丁。
- `stage_metadata.py`：放置划分 ID 和保存的归一化文件，不下载数据或权重，也不启动训练。

脚本保留原实验的 `/root/autodl-tmp/pi-libero` 布局。更换运行目录时也要修改补丁中的配置路径。重建环境以原项目 `uv.lock` 为依据，依赖版本快照不能独立代替安装流程。

## 2. 云端下载与核验数据

如果使用 AutoDL 学术资源加速，在需要访问 GitHub / Hugging Face 的当前终端运行：
`source /etc/network_turbo`。来源：[AutoDL 官方说明](https://www.autodl.com/docs/network_turbo/)；该服务不承诺稳定性。

```bash
python "$PROJECT_DIR/scripts/data/prepare_libero10.py" \
  --root "$HF_LEROBOT_HOME" --workers 6

python "$PROJECT_DIR/scripts/data/verify_libero10.py"
```

下载固定版本下属于 LIBERO_10 的 379 条原始编号 Parquet，其中训练 329、留出验证 50。源元数据仍描述完整上游数据，训练必须显式传入 ID。核验脚本逐文件检查 SHA256、帧数、数值有限性和 8/7 维 state/action，并创建不重复占用数据空间的硬链接视图。

此步骤在云端执行；本地仿真不需要下载这些训练 Parquet。

获取 pi05_base 初始化权重可使用上游下载工具，或按上游 README 取得同一检查点：

```bash
python - <<'PY'
from openpi.shared import download
print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base"))
PY
```

预期配置读取：/root/autodl-tmp/pi-libero/cache/openpi/openpi-assets/checkpoints/pi05_base/params。取得文件后核对实际目录。依赖、模型、tokenizer 及数据资源需要在离线检查前通过可用网络准备。

## 3. 检查实际只加载训练子集

数据准备完成后：

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
  python "$PROJECT_DIR/scripts/data/verify_train329_loader.py"
```

应检查 329 个 episode、87,769 帧和 658 个回合边界样本，没有验证 ID 混入。公开目录中的历史报告保存于 results/verification。

## 4. 训练集归一化

已有本次统计保存在 results/normalization/norm_stats.json。若复现完整计算，使用原版统计脚本，在进程内单独修改统计 batch，以完整覆盖所有帧：

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

101×869=87,769，统计覆盖全部训练帧。这个进程内的 batch 不会改写训练配方。归一化与模型推理必须使用同一统计。

## 5. GPU smoke 与正式训练

```bash
export CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

python scripts/train.py pi05_libero10_lora \
  --exp-name "smoke_b2_$(date +%Y%m%d_%H%M%S)" \
  --batch-size 2 --num-train-steps 20
```

检查真实更新、有限 loss/梯度、最终检查点保存和重新加载，再开始长训练。历史 batch=2 重载报告在 results/verification/smoke_b2_reload.json；它的合成输入接口检查不能当成任务成功率。

本次正式命令：

```bash
python -u scripts/train.py pi05_libero10_lora \
  --exp-name libero10_b2_s40000_20261103_234738 \
  --batch-size 2 --num-train-steps 40000
```

复现新实验时换一个 exp-name，不覆盖旧结果。学习率 decay_steps 保持 10,000。检查点包含 params、assets 和续训用 train_state；本次完整一份约 8.9 GiB，另需给数据、初始模型、环境及 smoke 留空间。

## 6. 云端加载最终检查点

必须先具有完整检查点。目录编号 39999 对应从 0 开始的 40,000 步训练。

```bash
CKPT="$PI_ROOT/checkpoints/pi05_libero10_lora/libero10_b2_s40000_20261103_234738/39999"
python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_libero10_lora \
  --policy.dir "$CKPT"
```

保持模型服务运行。单独使用 --env LIBERO 会加载官方默认模型，不会加载本次的自定义微调检查点。

## 7. 本地 Ubuntu 22.04 / WSL 客户端

已存在本机可运行环境时，继续使用 /home/pp/projects/openpi 和 examples/libero/.venv。

从头配置客户端时，先在本地克隆同一上游版本及子模块，再使用独立 Python 3.8 环境。客户端与云端 Python 环境分开：

```bash
# PROJECT_DIR 指向本项目根目录
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

以下本地路径使用原实验用户名 pp 作为示例，换机器时需一致地调整。WSLg 图形加速和本机 GPU 驱动需可用。本实验使用以下渲染变量：

```bash
export PYTHONPATH="$PWD/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH=/home/pp/projects/pi-libero-client/libero-config
export MUJOCO_GL=glx PYOPENGL_PLATFORM=glx
export MESA_LOADER_DRIVER_OVERRIDE=d3d12 GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
```

初次使用时生成本地 LIBERO 路径配置：

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

这里的 datasets 只是客户端配置需要的空目录，不是训练数据下载位置。

## 8. SSH 隧道、本地评估与视频

本地 WSL 一个终端建立隧道，填入当前实例实际 SSH 主机与端口：

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:18000:127.0.0.1:8000 \
  -p YOUR_SSH_PORT root@YOUR_SSH_HOST
```

另一个 WSL 终端激活客户端环境并设置上一节的渲染变量后：

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

仿真录制为 MP4，直接在本地 VS Code 中打开。首个请求有模型编译等待。此版本会遍历整个套件，不提供 --task-id 参数。



当前整理没有创建远端仓库或推送内容。完整权重、数据、环境和私有原始归档不属于提交内容。GitHub 普通 Git 仓库会阻止超过 100 MiB 的单个文件，参见 [GitHub 官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

多次评估时，原版视频命名可能覆盖相同任务的同类结果；需要保留每条视频时，分轮使用不同目录，或记录命名方式的修改。
