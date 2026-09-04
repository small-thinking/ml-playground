# Curated figures

`gsm8k-posttraining-formal-results-v2.png` is the current publication-style
summary of the completed E0, E2, and E4–E9 selected-checkpoint results under
the shared GSM8K formal protocol. It keeps the score series visually constant
and labels the comparison scopes: E4 is the controlled SFT→GRPO reference, E7
uses 12.7× E4's optimized input tokens, E8 is the approximate token-matched
control, and E9 is a separate full-corpus Base→KD route. E1 remains excluded as
the teaching failure mode documented in the main README. The older v1 image is
retained as the historical E2-to-E7 view.

Generate the current image from the audited values in `../experiment_log.md`:

```bash
UV_CACHE_DIR=.uv-cache uv run python figures/plot_formal_results.py
```

The script, its generated v2 PNG, and the values in `../README.md` must change
together. The PNG is tracked with Git LFS through `.gitattributes`.

Keep raw W&B exports, prompts, completions, and model artifacts out of this
directory. New figures should be linked from the main README or experiment
ledger and should cite their W&B source runs and frozen protocol.
