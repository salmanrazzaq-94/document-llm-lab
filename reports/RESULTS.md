# Local lease extraction experiments

Experiments completed on October 14, 2026, on an Apple M4 Pro with 48 GB RAM.
All model training and inference ran locally, with no paid APIs or cloud GPU.
The [model card](MODEL_CARD.md) records the exact base revision, training settings, scope, and licenses.

## Synthetic held-out test: 50 PDFs

| Model | Documents | Schema valid | Field accuracy | Exact document | Median generation | MLX peak |
|---|---:|---:|---:|---:|---:|---:|
| Original Qwen3-1.7B | 50 | 64.00% | 63.25% | 58.00% | 2.32 s | 3.84 GB |
| LoRA v1 | 50 | 100.00% | 100.00% | 100.00% | 4.73 s | 4.13 GB |
| LoRA v2 | 50 | 100.00% | 99.50% | 96.00% | 4.88 s | 4.14 GB |

All three models used identical prompts, greedy generation, token budgets, and the same versioned
test examples. Entire template families were excluded from project training and validation.
Invalid JSON/schema outputs count as incorrect for every field. These are deliberately simple,
related synthetic templates; the scores do not establish accuracy on arbitrary leases.
Full per-field/non-null accuracy, gold-disagreement rates, predictions, and run configs are in
[evaluations/](evaluations/).

## Separate format reliability from factual extraction

The original model wrapped **17 of 50** responses in Markdown fences.
A post-hoc diagnostic that removes only one enclosing JSON fence raises its schema-valid rate to
**98.00%**, field score to
**96.75%**, and exact-document score to
**88.00%**.
One remaining response copies an ambiguous date into the date field, violating the schema;
other errors turn ambiguous dates into definite ISO dates instead of null.

The large raw-score improvement therefore includes substantial **format compliance** gains.
It should not be described as a similarly large improvement in factual understanding.
This diagnostic was computed after the frozen experiment and did not alter primary scores,
training, prompts, or checkpoint selection. Normalized predictions are saved so it can be rerun:

```sh
uv run lease-lab score reports/baseline-format-diagnostic/predictions.jsonl
```

In a production comparison, a baseline with output normalization or constrained JSON generation
would be worth evaluating before attributing all gains to training. That additional model
experiment is outside this release.

## Controlled v2 data experiment

Both runs used 200 updates, the same base model, prompt, optimizer, seed, LoRA settings,
and validation/test examples. v1 trained on 200 examples; v2 trained on 304, adding 104
training-only missing-rent and ambiguous-date examples. v2 starts from the original model,
not from v1's adapter. This holds update budget fixed, not number of dataset passes.

Both adapters scored 100% on the 50-document synthetic validation set. v2 showed no improvement
on that saturated validation metric. v1 is the primary learning adapter because it achieved the
same validation accuracy with the smaller dataset and marginally lower validation loss;
v2 is retained as a transparent ablation. Test results were reported after freezing those choices.

On the synthetic test set, v2 regressed from v1's 100% field / 100% exact-document accuracy
to 99.5% field / 96% exact-document accuracy. Additional examples did not improve this run.

Selected checkpoints each have 199 completed updates, because MLX evaluates before the next
update. They were selected by the lowest full-validation completion loss. v1 took about
300 seconds and v2 about 314 seconds, each peaking at 4.43 GB of MLX allocation.
Recorded curves and settings are in [v1/](v1/) and [v2/](v2/).

## Independently sourced real-lease challenge: only two PDFs

| Model | Documents | Schema valid | Field accuracy | Exact document | Median generation | MLX peak |
|---|---:|---:|---:|---:|---:|---:|
| Original Qwen3-1.7B | 2 | 0.00% | 0.00% | 0.00% | 9.68 s | 4.02 GB |
| LoRA v1 | 2 | 100.00% | 50.00% | 0.00% | 5.08 s | 4.66 GB |
| LoRA v2 | 2 | 50.00% | 37.50% | 0.00% | 4.51 s | 4.66 GB |

Both original-model responses on this challenge were wrapped in Markdown. Removing only those
fences yields **100% schema validity and 75% field accuracy**, compared with v1's 50% and v2's
37.5% raw field accuracy. This tiny challenge therefore shows a **regression relative to the
format-normalized baseline**, not a demonstrated real-world extraction improvement.
See [the diagnostic](real-format-diagnostic.json); its predictions can be rescored with
`uv run lease-lab score reports/real-format-diagnostic/predictions.jsonl`.

V1 returned null for all four explicit written-month dates, and both baseline and adapters
confused the parties in one lease. V2 also emitted two separate JSON objects for one document.
A concrete coverage gap is that synthetic training only uses ISO dates or deliberately ambiguous
numeric dates, whereas the real contracts spell out months. This is a plausible contributor to
the date regression, not a proven causal diagnosis. More examples from the same narrow generator
did not supply that missing diversity. A next experiment should add diverse, independently
annotated training documents and establish a new held-out benchmark before tuning further.

These are real lease texts from [ALeaseBert](https://uvaauas.figshare.com/articles/dataset/ALeaseBert/19732993),
converted from HTML into new PDFs, with independently reviewed task labels. They retain the
source's CC BY-NC 4.0 terms and were never used for training or checkpoint selection.
Their original layout is not preserved and their presence in base-model pretraining is unknown.
This is a qualitative challenge set, not a representative real-world accuracy benchmark.

Read [the field-level errors](real-challenge-errors.json) alongside the original PDFs and
annotation notes. Strict matching treats punctuation differences as errors, so inspect the
underlying values rather than equating every disagreement with an invented fact. No broad
real-world improvement claim is justified by two documents.

## Reproduction and interpretation

Follow the [README experiment commands](../README.md#run-the-experiment). Every comparison checks
matching dataset version, document IDs, model/revision, prompt hash, and generation configuration.
The release contains the selected MLX adapters and their training provenance; base weights are
loaded separately at the pinned revision. Both selected adapters were reloaded in fresh processes
for evaluation. The v1 extraction command was also exercised end to end after reloading.

Generation timings exclude model download/loading, PDF parsing, and tokenization. Peak memory
is MLX allocation, not total system RAM. Cross-platform/runtime differences can affect timings
and floating-point results. The source, lockfile, original datasets, predictions, and checkpoint
checksums make the experiments reviewable; they do not promise bit-identical training on every Mac.
