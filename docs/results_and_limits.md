# Results and Limitations

English | [简体中文](results_and_limits.zh-CN.md)

## Verified outcomes

- Records cover data preparation, real-loader subset checks, train-only normalization, batch-1/2 smoke runs, and a batch-2 checkpoint reload.
- Training completed 40,000 optimizer steps at batch size 2, saved checkpoint 39999, and exited with code 0.
- That checkpoint completed 10 LIBERO_10 closed-loop rollouts through the cloud service and local client: 8 successes and 2 failures.
- The fine-tuned per-task CSV, ten MP4 files, and full evaluation log agree.
- The ten original historical `pi05_base` videos are retained, all labeled failure (0/10), with byte-for-byte archive SHA256 matches.

Failed tasks:
1. put both the cream cheese box and the butter in the basket
2. put the yellow and white mug in the microwave and close it

No frame-by-frame failure analysis has established the cause, so these outcomes are not attributed to a particular training or control factor.

## Interpreting historical comparisons

Fine-tuned evidence: [per-task results](../results/task_results.csv) and [evaluation log](../results/logs/eval_40000.log). Base evidence: [video provenance](../media/pi05_base/README.md) and [historical summary](../results/pi05_base_summary.json).

The base result is inferred from historical video filenames. Its full evaluation log and exact inference settings are not preserved here. It used a LIBERO interface and borrowed LIBERO normalization, so it is not a strict target-data-free zero-shot baseline.

A separate historical official LIBERO-policy run had 10/10 success-labeled videos. Those videos are not included in this project, and that comparison was not rerun.

## Current limits

- One rollout per task is insufficient to claim a multi-initial-state, multi-seed benchmark result.
- The 50 demonstrations were held out, but no dedicated offline validation loss was computed.
- Training and held-out demonstrations belong to the same 10 tasks; unseen-task generalization was not tested.
- No real-robot deployment or full-model inference on the local 8GB GPU was verified.
- The local artifacts do not contain a complete fine-tuned checkpoint; rerunning the model requires a full checkpoint or retraining.

## Further experiments

Compare models under aligned initializations and evaluation budgets, then increase the trials per task. The original client names videos by task and success/failure, so repeated outcomes overwrite files. Use separate directories per round or record a naming change if all videos must be kept.

Training and loader reports are historical evidence. Documentation maintenance only checks static files and evidence consistency; it does not rerun training or simulation.
