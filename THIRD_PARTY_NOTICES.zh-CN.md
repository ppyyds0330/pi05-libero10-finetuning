# 第三方归属与许可说明

[English](THIRD_PARTY_NOTICES.md) | 简体中文

本复现与工程实践项目基于：

- Physical Intelligence OpenPI：https://github.com/Physical-Intelligence/openpi
- LIBERO：https://github.com/Lifelong-Robot-Learning/LIBERO
- PI 提供的 LeRobot 格式 LIBERO 数据：https://huggingface.co/datasets/physical-intelligence/libero

代码与数据的固定版本见 [upstream.lock.json](upstream.lock.json)。OpenPI 使用 Apache-2.0，本项目保留其 [LICENSE](LICENSE) 原文；补丁包含 OpenPI 的代码上下文，并保留对原作者的归属。

本项目的工作包括数据选择与核验工具、episode 索引适配、训练配置补丁、部署文档和实验记录。基础 VLA 模型、OpenPI 框架、LIBERO 基准和其他依赖不属于本项目原创。

项目文件不包含模型权重或原始数据文件。`pi05_base` 等权重仍须遵循上游模型及组件许可，包括 OpenPI 的 [LICENSE_GEMMA.txt](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE_GEMMA.txt)。

数据和基准资产遵循各自发布方的条款。依赖版本记录用于复现，不改变依赖本身的许可证。
