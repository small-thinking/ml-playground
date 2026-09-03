# Curated figures

`gsm8k-posttraining-formal-results-v1.png` is the publication-style summary
of the E2-to-E7 ablation path on the shared GSM8K formal protocol. E1 is
intentionally excluded from the graphic and retained as a teaching failure mode
in the main README. The figure is generated from W&B run summaries listed in
`../experiment_log.md`, checked against their source values, and tracked with
Git LFS through the exact path in the repository's `.gitattributes`.

Keep raw W&B exports, prompts, completions, and model artifacts out of this
directory. New figures should be linked from the main README or experiment
ledger and should cite their W&B source runs and frozen protocol.
