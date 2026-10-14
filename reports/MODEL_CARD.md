# Lease extraction LoRA model card

## Model and intended use

Educational MLX LM LoRA adapters for `Qwen/Qwen3-1.7B`, base revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. They extract explicit party names,
premises, dates, and initial base rent from short English lease text.
They are intended for learning supervised fine-tuning and evaluating extraction behavior.
They are not validated for operational decisions, legal review, or unattended contract processing.

The released files are **MLX adapters**, not standalone models and not PEFT-format weights.
Use the repository's extraction command with the pinned original model and the selected `best/`
adapter directory. See each release archive's README for a complete loading command.
The original model weights are downloaded separately and retain Qwen attribution.

## Training

- Backend: MLX LM 0.32.0, MLX 0.32.3, Python 3.12; dependency versions in `uv.lock`.
- Hardware: Apple M4 Pro, 48 GB unified memory. All model experiments ran locally.
- Method: supervised LoRA with frozen base weights; 8 final transformer layers, rank 8,
  scale 20, dropout 0, approximately 2.490 million trainable parameters (0.145%).
- Adam, learning rate `1e-4`, batch size 1, 200 optimizer updates, seed 42,
  gradient checkpointing, completion-only loss, 2,048-token maximum training sequence.
- Identical non-thinking chat templates during training and inference; deterministic greedy
  generation with a maximum of 384 output tokens during evaluation.
- v1: 200 original synthetic training examples. v2: 304 original synthetic training examples,
  including 104 additional missing-rent and ambiguous-date examples. Both start from the base model.
- Both use the same 50 synthetic validation examples and select the lowest completion-loss
  checkpoint on the entire validation split. Selected adapters have 199 completed updates;
  MLX evaluates before update 200. The final-update weights are retained locally separately.
- v1 training took about 300 seconds; v2 about 314 seconds. Peak MLX allocation was about 4.43 GB
  for each run. This is not total system RAM usage or a performance guarantee for other hardware.

## Evaluation protocol

The synthetic test set contains 50 examples from two template families excluded from project
training and validation. Full source/label records, seed, family assignments, and SHA256 dataset
versions are committed. Synthetic templates deliberately make the task simpler than real leases.
No test predictions were used to select training settings or construct v2.

A separate challenge set contains two real lease texts from the ALeaseBert corpus, reflowed
from HTML into text-based PDFs. Labels were independently reviewed against the source text,
with evidence and interpretation notes recorded in the dataset. Source layout is not preserved.
These documents were excluded from training and checkpoint selection. Their possible presence
in Qwen's original pretraining corpus is unknown. Two examples cannot support broad accuracy claims.

Metrics are strict schema validity, eight-field accuracy, non-null field accuracy, exact-document
accuracy, gold-disagreement rate for non-null predictions, runtime, and peak MLX memory.
Names and addresses ignore case and repeated whitespace; punctuation differences count as errors.
The gold-disagreement metric is not a semantic hallucination detector. Scores from the independent
challenge and the synthetic test set must be reported separately.
See [the experiment report](RESULTS.md) and committed predictions for measured comparisons.

V1 scored 100% and v2 99.5% on synthetic test fields. On the two real leases, raw field accuracy
was 50% for v1 and 37.5% for v2. The original model, after removing only enclosing Markdown
fences, scored 75% there. Thus these runs improved synthetic/format compliance but did not
demonstrate better real-document factual extraction. V1 dropped explicit written-month dates;
v2 also produced a multi-object response. These are unresolved generalization failures.

## Limitations and failure modes

- Training data is small, synthetic, and heavily templated. Related phrasing remains across splits.
- A valid JSON object can contain incorrect or invented facts. The schema does not verify evidence.
- Ambiguous dates, currency symbols, initial versus later rent, party roles, and deposits can be confused.
- The current schema does not capture full rent schedules, incentives, options, amendments, or conflicts.
- Scanned, blank-page, encrypted/unreadable, and over-budget PDFs require explicit handling.
  The introductory reader rejects blank/unreadable pages and PDFs over 20 pages. Inference additionally
  rejects prompts over 1,664 tokens to reserve 384 output tokens; no silent truncation is used.
- Real layouts, long documents, OCR, multilingual leases, and Excel rent rolls are outside this release.
- Synthetic success does not establish generalization to documents from new sources.

## Data, attribution, and license

Code, original synthetic data, and adapters: Apache-2.0, with attribution to Salman Razzaq
and the Qwen team for the base model. No Hypha or customer documents were used.

External evaluation data retains CC BY-NC 4.0 as specified by the
[ALeaseBert dataset authors](https://uvaauas.figshare.com/articles/dataset/ALeaseBert/19732993):
Spyretta Leivaditi, Julien Rossi, and Evangelos Kanoulas.
DOI: `10.21942/uva.19732993.v1`. Changes: HTML text reflowed into PDFs, typographic punctuation
normalized, and task-specific labels/evidence added. It was never used to train these adapters.
The repository's Apache-2.0 license does not replace the external evaluation data's terms.
