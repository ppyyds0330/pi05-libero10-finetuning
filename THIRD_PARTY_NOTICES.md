# Third-party attribution

This is a reproduction and engineering project built on:
- Physical Intelligence OpenPI: https://github.com/Physical-Intelligence/openpi
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- PI's LeRobot-formatted LIBERO data: https://huggingface.co/datasets/physical-intelligence/libero

Exact code and data revisions are recorded in upstream.lock.json. OpenPI is licensed under Apache-2.0; the Apache-2.0 text is retained in LICENSE. The patch contains context from OpenPI and preserves attribution to its authors.

Project-specific additions include dataset selection and verification utilities, the episode-index adaptation and training configuration patch, deployment documentation and experiment evidence. This repository does not claim authorship of the underlying VLA model, OpenPI framework, LIBERO benchmark, or third-party dependencies.

No model weights or original dataset files are distributed here. Access to pi05_base and any other model weights remains subject to the upstream model and component licenses, including the upstream LICENSE_GEMMA.txt:
https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE_GEMMA.txt

Dataset and benchmark assets retain the terms set by their respective publishers. Package version records are for reproducibility, not a relicensing of dependencies.
