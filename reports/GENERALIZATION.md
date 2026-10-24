# Generalization experiment: data diversity and prompting

Local experiments on October 24, 2026 (UTC), Apple M4 Pro / 48 GB RAM. No paid APIs,
cloud training, Hypha documents, or external contract text in training.
This report extends the [original experiment](RESULTS.md); it preserves the unsuccessful runs.

## What changed in v3

The first adapter dropped explicit written-month dates on both original real leases.
V3 tests a broader **original synthetic dataset**, retaining the original training split for
rehearsal: 600 training examples, 100 validation, 100 test. Coverage includes five date styles,
indirect party roles in both directions, company punctuation, named currencies and ambiguous
symbols, later increases, and monthly rates collected quarterly.
See [coverage counts](v3-coverage.json) and [the spot audit](v3-audit.json).

The pinned base model, original prompt, strict parser/scoring, rank 8, final 8 layers, batch size 1,
learning rate `1e-4`, seed 42, and 200-update budget stay fixed. V3 starts from the base model,
not v1. With 600 training rows, 200 steps are roughly one third of a pass, not one epoch.
Several coverage dimensions change together: this tests the dataset intervention, not the
individual causal contribution of each dimension.

The 20-step check passed. The full run took **684.8 seconds**, with **4.43 GB** peak MLX allocation.
The minimum full-validation completion loss was **0.0035595 at 139 completed updates**;
that checkpoint was frozen before test predictions. The final 199-update validation checkpoint
had a slightly higher loss and was not selected. Curves, configuration, and weight checksum
are in [v3/](v3/).

## Original-prompt comparisons

| Evaluation set | Model | Schema valid | Field accuracy | After common fence cleanup |
|---|---|---:|---:|---:|
| New synthetic test, 100 | Base | 76.0% | 71.25% | 92.25% |
| New synthetic test, 100 | V3 | 100.0% | 99.375% | 99.375% |
| Original real cases, 2 | Base | 0.0% | 0.0% | 75.0% |
| Original real cases, 2 | V1 | 100.0% | 50.0% | 50.0% |
| Original real cases, 2 | V3 | 50.0% | 37.5% | 37.5% |
| Prospective real cases, 3 | Base | 33.3% | 29.17% | 29.17% |
| Prospective real cases, 3 | V1 | 66.7% | 29.17% | 29.17% |
| Prospective real cases, 3 | V3 | 66.7% | 54.17% | 54.17% |

The broader synthetic task improved relative to the base model, including after removing its
Markdown fences. V3 recovered all four explicit written-month dates and the ambiguous currency
on the original real cases. However, it omitted the required tenant key in case 237 and still
swapped parties in case 73. **The original two-lease score regressed**, despite those partial fixes.

On the three prospectively labeled cases, v3 improved the strict field score, but still selected
a per-square-meter rent, missed a conflicting landlord, and selected the wrong initial rent
period/amount. It is not a reliable general-purpose lease extractor.

Invalid schema outputs count as wrong for **all eight fields**, even when some individual
values are correct. Consequently, a schema correction can raise this score without improving
every fact. The fence diagnostic removes only one enclosing Markdown fence, with the same rule
for every model; it does not repair missing keys, rent units, currency values, or party roles.
Read the actual [field errors](v3-errors.json), not just the aggregate numbers.

## Prospective versus development evaluation

The original two real cases were inspected to guide v3 and are development/regression cases.
Three additional leases (84, 166, 201) were independently read and labeled before v3 predictions.
They are a tiny convenience sample from the same ALeaseBert corpus; case 84 shares a tenant
corporate family with original case 237. Source pretraining exposure is unknown. They are not
independent evidence of broad real-world accuracy.
Annotations were authored and reviewed by Codex against source text; they have not been
independently adjudicated by a human. The dataset reviewer field identifies Codex, not a human
annotator. Human review is still needed before treating these as a reliable benchmark.

Case 84 initially exceeded the input budget because of artificial PDF line wrapping: 1,689
prompt tokens at 78 characters and 1,675 at 90. Reflowing all fresh PDFs at 105 characters retained
all source text and labels, stayed within page margins, and produced prompts of 1,656 / 1,274 /
1,322 tokens. The first attempted baseline run rejected the oversized input before producing
any predictions. Reflow and input hashes were recorded before the successful evaluations;
no truncation, token-budget increase, document removal, or label change was used.
See [the frozen hypothesis and preflight record](v3-hypothesis.json).

The synthetic split still shares clause subpatterns and a PDF renderer across families. Its
prompts span only 362–436 tokens, whereas the fresh real cases reach 1,656. Synthetic success
therefore leaves a material context-length and document-complexity gap.

## Controlled prompt ablation

After inspecting these real errors, one fixed `careful` prompt was written to clarify required
keys, transaction roles, whole-property rent, and named currencies. It is **inference-only**;
all adapters were trained with `original`. The same revised prompt is applied to base and v3,
with unchanged generation settings, schema, and scoring. The default remains `original`.
Prompt hashes prevent accidental comparisons between different profiles.

All five real cases are **development/regression examples for this prompt experiment**.
They are not a new held-out test of the revised prompt. Any gains from the revised instructions
must be separated from training gains. The candidate and its protocol are committed under
[v3/](v3/); a ten-example synthetic validation slice is checked before the real comparisons.

| Development/regression set | Model with `careful` | Schema valid | Field accuracy |
|---|---|---:|---:|
| Original real cases, 2 | Base | 100.0% | 75.0% |
| Original real cases, 2 | V1 | 100.0% | 50.0% |
| Original real cases, 2 | V3 | 100.0% | 75.0% |
| Other real cases, 3 | Base | 66.7% | 58.33% |
| Other real cases, 3 | V1 | 66.7% | 25.0% |
| Other real cases, 3 | V3 | 66.7% | 58.33% |

V1 was added as an additional exploratory control after the base/v3 prompt comparison, without
changing the candidate prompt or any weights. With identical revised instructions, v3 improves
over v1 but **ties the base model on both real sets**. Across all five examples, base and v3 each
score 26/40 fields (65%) with four schema-valid documents; v1 scores 14/40 (35%). This aggregate
is descriptive only, not a generalization estimate. No Markdown fences were removed from these
revised-prompt outputs.

The revised prompt has a cost. V3's original synthetic regression test falls from **98.75% fields /
100% schema validity** with `original` to **92.5% fields / 94% schema validity** with `careful`.
The ten-example v3 synthetic validation slice scores 96.25% fields for v3 and 82.5% for base under
`careful`; this slice is a sanity check, not full validation or checkpoint selection.
The profile remains optional. It is not an across-the-board replacement for the original prompt.

Remaining real mistakes include a party reversal, overconfident currency on an ambiguous symbol,
company punctuation, per-area rent instead of an explicit whole-property rate, conflicting
landlord identities, and choosing 1.2 million instead of the explicitly stated monthly 100,000.
A small schema-validity improvement does not resolve these factual failures.

## Reproduce and inspect

Follow the [README v3 commands](../README.md#improve-generalization-with-v3). The selected adapter,
training provenance, model card, and original base-model license are packaged in release v0.2.0.
The packaged adapter was reloaded in a fresh process through PDF parsing and the extraction CLI;
its weight checksum matches the frozen adapter used in evaluation.

Every comparison is checked for identical base revision, dataset version, document IDs, prompt
hash, and generation settings. Full configs, predictions, timings, and memory measurements are
in [evaluations/](evaluations/). Summary metrics are in [v3-summary.json](v3-summary.json).
Primary predictions can be rescored directly; the same applies to the saved fence-only diagnostic:

```sh
uv run lease-lab score reports/evaluations/v3-new-test/predictions.jsonl
uv run lease-lab compare reports/evaluations/careful-real-baseline reports/evaluations/careful-real-v3
uv run lease-lab score reports/v3-format-diagnostic/baseline-new-test/predictions.jsonl
```

The fence diagnostic uses a full-string match of one enclosing three-backtick block with an
optional `json` tag, a newline before/after its body, and optional outside whitespace. Only the
body replaces the output; every field and all other text remain untouched. The saved diagnostic
predictions make those scores inspectable without relying on an automatic repair policy.

External source text, including copies inside prediction records, retains **CC BY-NC 4.0** with
ALeaseBert attribution in the datasets. Code, original synthetic data, and adapters are Apache-2.0.
The external texts never enter training. Generation timings exclude loading, PDF extraction,
and tokenization; MLX allocation is not total system RAM.

## What this establishes and what comes next

The adapter's original two-lease score can reach 75% with the revised instructions, improving
over v1 under the same instructions. It **does not beat the equally prompted base model** on
these real examples. Synthetic gains and restored written-date behavior are useful learning
results, not evidence that fine-tuning is necessary or superior for the present real task.

The next data experiment should address the observed distribution gap: longer original synthetic
leases with realistic distractors, less direct role attribution, conflicting names, and explicit
whole-property rates beside unit rates. Validation must measure factual fields and schema
failures on those cases, not only token loss on short templates. A stronger free local base model
would be a separate capacity experiment. Neither has been run in this release.

Before another generalization claim, collect a larger independently human-reviewed real test set
from different sources and freeze it before further tuning. Preserve these five as development
and regression cases. The right outcome may still be the base model with improved instructions;
this project should measure that possibility rather than assume an adapter must win.
