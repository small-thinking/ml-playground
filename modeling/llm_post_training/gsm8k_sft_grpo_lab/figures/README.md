# Curated W&B Figures

Store reviewable screenshots here, not raw W&B downloads.

Use these filename patterns:

```text
e0-base__outcome-and-group-difficulty.png
e1-sft-clean__train-and-eval.png
e1-e2-e3__sft-debugging-comparison.png
e4-grpo-clean__learning-dynamics.png
e4-e5-e6-e7__grpo-failure-comparison.png
```

Each screenshot must be linked from `../experiment_log.md` with its W&B run
URL and frozen config. Redact tokens, private URLs, and sensitive raw examples
before committing. Use Git LFS only for figures that exceed the repository's
small-image policy; do not track every PNG by default.
