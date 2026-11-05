# Third-Party Notices

English | [简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

This reproduction and engineering project is built on:

- Physical Intelligence OpenPI: https://github.com/Physical-Intelligence/openpi
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- PI's LeRobot-formatted LIBERO data: https://huggingface.co/datasets/physical-intelligence/libero

Exact code and data revisions are recorded in [upstream.lock.json](upstream.lock.json). OpenPI is licensed under Apache-2.0; its license text is retained in [LICENSE](LICENSE). The patch contains context from OpenPI and preserves attribution to its authors.

Project-specific work includes dataset selection and verification utilities, the episode-index adaptation and training-configuration patch, deployment documentation, and experiment evidence. The project does not claim authorship of the base VLA model, OpenPI framework, LIBERO benchmark, or other dependencies.

Model weights and source dataset files are not included. Use of `pi05_base` and other weights remains subject to the upstream model and component licenses, including OpenPI's [LICENSE_GEMMA.txt](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE_GEMMA.txt).

Dataset and benchmark assets retain their publishers' terms. Package-version records support reproducibility and do not relicense dependencies.
