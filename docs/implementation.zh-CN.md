# 修改内容与数据契约

[English](implementation.md) | 简体中文

## 已实现的修改

本项目没有提出新的 VLA 模型结构。代码改动集中在 OpenPI 的两个文件，完整差异见 patches/openpi-libero10-lora.patch。

1. src/openpi/training/config.py：为 DataConfig 增加 episodes 字段，注册 pi05_libero10_lora 配方，使用固定的 329 条训练 ID、pi05_base 初始化权重、action_horizon=10 及官方 LoRA 模型变体。
2. src/openpi/training/data_loader.py：把 episodes 参数传入 LeRobotDataset，并修复稀疏原始 episode ID 与紧凑边界表之间的索引映射。

后者用于本次固定的 LeRobot 版本：数据行仍带原始全局 episode_index，筛选后的 episode_data_index 却按子集紧凑排列。扩展边界表后，每个动作序列能按正确边界取样，并在回合末尾填充，不会误取其他 episode。

## 数据契约

- 仿真图像与腕部图像；输入时按原版客户端旋转 180° 并 resize/pad 到 224。
- state 为 8 维：末端位置 3、轴角姿态 3、夹爪状态 2。
- 真实动作是 7 维，训练动作序列长度为 10。
- 客户端默认每次执行前 5 步动作后重新规划。
- extra_delta_transform=False，保持与该 LIBERO 数据适配器一致。

在历史真实加载器检查中，329 个原始 episode ID、87,769 帧已核对，并检查了 658 个回合首尾样本。报告保存在 results/verification/loader_verification.json。

## 归一化

归一化仅使用 329 条训练演示，不使用 50 条验证演示。87,769=101×869，因此统计时仅在进程内将 batch 改为 101、num_workers 改为 4，完整处理 869 批，避免 drop_last 丢失最后一帧。

归一化的 batch=101 不等于训练 batch=2。历史归一化报告产生于训练配方仍是 batch=1 的准备阶段；后来正式训练改为 batch=2。报告内的时间点和字段应按阶段理解。

保存统计使用 quantile normalization，norm_stats.json 必须与训练检查点一起用于推理；不能用上游全量 meta/stats.json 代替训练集统计。

## 配方与实际命令

源配方 num_train_steps 默认 10,000，正式命令用 --num-train-steps 40000 覆盖。学习率 decay_steps 仍为 10,000，没有随总步数扩大。

运行从 pi05_base 新建实验，而非从 20 步 smoke 检查点继续。完整实际参数与源码 SHA256 在 configs/experiment_record.json。

采用官方 LoRA 变体及 freeze_filter。没有在本项目中逐个统计可训练参数，因此不宣称所有非 LoRA 参数均被冻结。

## 上游源码

`scripts/bootstrap_openpi.sh` 会拉取固定版本的完整 OpenPI 源码及其子模块，再应用补丁。默认目录为 `/root/autodl-tmp/pi-libero/openpi`，与本复现项目分开存放。

`scripts/stage_metadata.py` 把训练/验证 ID 和保存的归一化统计放到配方使用的运行目录，无须合并两个源码目录。详见 [复现步骤](reproduction.zh-CN.md)。
