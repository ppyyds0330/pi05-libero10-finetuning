# 数据清单

原始 LIBERO 官方入口：https://github.com/Lifelong-Robot-Learning/LIBERO#Dataset

本实验实际训练读取的是 physical-intelligence/libero 的 LeRobot 格式数据，而非直接读取原始 HDF5：
https://huggingface.co/datasets/physical-intelligence/libero

固定 revision：a4336d589d589045d1c56423ffdf3b88a0e19b1f。

- LIBERO_10 来源子集共 379 条演示、101,469 帧。
- train329.json：329 条训练 ID，87,769 帧。
- validation50.json：50 条留出 ID，13,700 帧。
- 按任务分别打乱，每任务留出 5 条；seed=7，具体任务顺序与算法写在 source_manifest.json。
- IDs 保留上游原始编号，没有重新索引。
- source_manifest.json 保存源文件 SHA256、大小、任务和划分，可以重新校验下载。
- 50 条验证演示未参与训练或归一化，本次没有另外计算验证损失。

此目录不含实际 Parquet、图像或模型权重。数据下载命令应在云端执行，见 docs/reproduction.md。
