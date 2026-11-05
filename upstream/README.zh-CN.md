# OpenPI 运行依赖

[English](README.md) | 简体中文

OpenPI 版本固定在 [upstream.lock.json](../upstream.lock.json)。本目录只有依赖说明，不是训练源码目录。

在本项目根目录运行：

```bash
bash scripts/bootstrap_openpi.sh /root/autodl-tmp/pi-libero/openpi
python3 scripts/stage_metadata.py
```

第一个脚本获取完整 OpenPI 仓库及子模块，再应用本项目补丁；第二个脚本把划分 ID 和保存的归一化文件放到运行目录。

实际源码：`/root/autodl-tmp/pi-libero/openpi/src/openpi/`。
实际训练入口：`/root/autodl-tmp/pi-libero/openpi/scripts/train.py`。

复现项目与运行源码保持独立。还需准备环境、数据和初始化权重，见 [完整复现步骤](../docs/reproduction.zh-CN.md)。
