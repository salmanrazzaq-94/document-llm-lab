# Document LLM Lab

Learn to fine-tune a language model by teaching it to extract explicit lease terms from PDFs.
Everything runs locally on Apple Silicon. No paid API, cloud GPU, or hosted service is needed.

This is a learning project: the first dataset contains deliberately simple synthetic leases.
Good scores on these examples do **not** establish accuracy on unfamiliar real contracts.

## Start here

Read [the learning guide](docs/LEARNING.md), then run the original model before training anything.
You need an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/), and space for the roughly
3.5 GB model download, environment, and checkpoints. The tested machine is an M4 Pro with 48 GB RAM.
Python 3.12 is installed by uv if needed. The first model download needs internet; subsequent
model operations can use the cache. There are no billable service calls.

```sh
git clone https://github.com/salmanrazzaq-94/document-llm-lab.git
cd document-llm-lab
uv sync --locked
uv run lease-lab prepare
uv run lease-lab extract data/samples/train-f00-001.pdf --output artifacts/first-extraction.json
```

The model is **Qwen/Qwen3-1.7B**, pinned to revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. It is already pretrained and instruction tuned.
Our training adds a task-specific LoRA adapter. Thinking is disabled in both training and
inference, with an identical chat-template prefix and loss only on the JSON response.

## Run the experiment

Commands below use fresh output directories. Existing runs are never overwritten.
The default evaluation split is validation; use it to investigate errors and select experiments.

```sh
# Measure the original model, then check that training works.
uv run lease-lab evaluate --split valid --output artifacts/my-baseline-valid
uv run lease-lab train --steps 20 --output artifacts/my-smoke

# Train v1 and reload the checkpoint chosen by validation loss.
uv run lease-lab train --output artifacts/my-v1
uv run lease-lab evaluate --split valid --adapter artifacts/my-v1/best --output artifacts/my-v1-valid
uv run lease-lab extract data/samples/train-f00-001.pdf --adapter artifacts/my-v1/best --output artifacts/my-trained-extraction.json

# Controlled second experiment: additional missing-rent and ambiguous-date examples.
# Each run starts from the same original model, keeping the comparison interpretable.
uv run lease-lab prepare --variant v2 --output data/generated/v2
uv run lease-lab train --dataset data/generated/v2 --output artifacts/my-v2
uv run lease-lab evaluate --split valid --adapter artifacts/my-v2/best --output artifacts/my-v2-valid

# After choosing experiments, evaluate the untouched synthetic test set.
uv run lease-lab evaluate --split test --output artifacts/my-baseline-test
uv run lease-lab evaluate --split test --adapter artifacts/my-v1/best --output artifacts/my-v1-test
uv run lease-lab evaluate --split test --adapter artifacts/my-v2/best --output artifacts/my-v2-test
uv run lease-lab compare artifacts/my-baseline-test artifacts/my-v1-test
uv run lease-lab compare artifacts/my-v1-test artifacts/my-v2-test
```

LoRA uses rank 8 in the last 8 layers, batch size 1, Adam at `1e-4`, seed 42,
gradient checkpointing, and a 2,048-token training limit. Runs are capped at 200 steps.
These are starter experiment settings, not a claim of optimal hyperparameters.
Validation covers all 50 validation examples every 20 iterations. `best/` stores the lowest
validation-loss weights; `last/` stores the final weights and periodic checkpoints.
MLX evaluates before each reported training iteration, so the chosen checkpoint's metadata
records the number of **completed updates** (for example, 199 at the evaluation before step 200).
The untrained adapter at step zero can win if training never improves validation loss.

Each evaluation saves `config.json`, incremental `predictions.jsonl`, and `metrics.json`.
Each training run saves `training.json`, `events.jsonl`, `selection.json`, and adapters.
Output limits and prompt lengths are checked explicitly; inputs are never silently truncated.
Inference reserves 384 output tokens within the 2,048-token introductory budget.

## Data and extraction contract

v1 has **200 training, 50 validation, and 50 test** documents. Template-family IDs stay in
one split. v2 keeps validation/test unchanged and adds 104 original training examples.
v3 has **600 training, 100 validation, and 100 test** documents: 400 diverse original training
leases plus the original 200 for rehearsal. Its new validation/test families cover written dates,
indirect party roles, company punctuation, currency ambiguity, and rate versus collection period.
The original v1 test set remains a separate regression check. Clause patterns and the PDF renderer
still overlap across synthetic splits; family isolation does not make these independent real leases.
Dataset content is hashed, and each split is verified against the manifest before use.
The original synthetic text and labels are committed under `data/synthetic/v1/`;
`prepare` regenerates all 300 PDFs under `data/generated/v1/`. Three samples are included.

The required JSON keys are `tenant`, `landlord`, `premises_address`, `commencement_date`,
`expiration_date`, and `base_rent`. Missing or ambiguous fields are `null`.
Dates use `YYYY-MM-DD` and must be explicit, not calculated from a duration.
`base_rent` is null or `{ "amount": 6721.0, "currency": "EUR", "period": "year" }`,
with nullable components. Extract the initial rate in its stated period, not a computed
monthly equivalent, deposit, penalty, or later rent increase. A currency symbol can be ambiguous.
The schema validates shape and types; it cannot establish that a fact is supported by the PDF.

Metrics include valid JSON/schema rate, accuracy for eight flattened fields, non-null field
accuracy, exact-document accuracy, runtime, and peak MLX memory. Invalid outputs count as
incorrect for every field. The `unsupported_value_rate` is a **gold-label disagreement proxy**:
non-null predictions that disagree with annotated values divided by all non-null predictions
in schema-valid outputs. Inspect it alongside schema validity. It is not an entailment classifier.
Names/addresses are compared ignoring case and repeated whitespace, but punctuation remains significant.

## Independent real-document challenge

Two independently sourced lease texts from
[ALeaseBert](https://uvaauas.figshare.com/articles/dataset/ALeaseBert/19732993) are converted
to new text-based PDFs, with independently reviewed labels and evidence notes.
Their original HTML layout is not retained. They are a **tiny qualitative challenge set**,
not a representative real-world accuracy benchmark. They never enter training or checkpoint selection.

```sh
uv run lease-lab prepare-real
uv run lease-lab evaluate --dataset data/real/benchmark --split test --output artifacts/my-real-baseline
uv run lease-lab evaluate --dataset data/real/benchmark --split test --adapter artifacts/my-v1/best --output artifacts/my-real-v1
uv run lease-lab compare artifacts/my-real-baseline artifacts/my-real-v1
```

These external materials retain **CC BY-NC 4.0**, as stated by the dataset authors.
Their attribution, source URLs, changes, checksum, and annotation notes are recorded beside them.
They are separate from the repository's Apache-2.0 code and original synthetic data.
Do not incorporate this noncommercial corpus into a permissively released training dataset.

## Improve generalization with v3

The original two real leases have now been inspected to guide the data-diversity experiment.
Treat them as development/regression cases. Three additional real leases were labeled before
v3 predictions for a prospective check; these remain a small convenience sample from the same
corpus, with related styles and a shared corporate family in one case. All external text stays
out of training. The model, extraction prompt, strict parser, and scoring rules are unchanged.

```sh
uv run lease-lab prepare --variant v3 --output data/generated/v3
uv run lease-lab train --dataset data/generated/v3 --steps 20 --output artifacts/my-v3-smoke
uv run lease-lab train --dataset data/generated/v3 --output artifacts/my-v3

# Freeze the validation-selected adapter before checking either test set.
uv run lease-lab evaluate --dataset data/generated/v3 --split test --output artifacts/my-v3-data-baseline
uv run lease-lab evaluate --dataset data/generated/v3 --split test --adapter artifacts/my-v3/best --output artifacts/my-v3-test
uv run lease-lab compare artifacts/my-v3-data-baseline artifacts/my-v3-test
uv run lease-lab evaluate --dataset data/generated/v1 --split test --adapter artifacts/my-v3/best --output artifacts/my-v3-old-test

# Fresh real check, using the same base and adapter prompts/settings.
uv run lease-lab prepare-real --suite fresh --output data/real/fresh
uv run lease-lab evaluate --dataset data/real/fresh --split test --output artifacts/my-fresh-baseline
uv run lease-lab evaluate --dataset data/real/fresh --split test --adapter artifacts/my-v3/best --output artifacts/my-fresh-v3
uv run lease-lab compare artifacts/my-fresh-baseline artifacts/my-fresh-v3
```

Validation loss covers all 100 v3 validation examples at each checkpoint. No real-lease
predictions select a checkpoint. The [v3 hypothesis](reports/v3-hypothesis.json) records the
protocol and dataset hashes before evaluation. See lesson 7 in the learning guide for why
rehearsal, coverage, and a fresh check matter.

An optional **inference-only** prompt ablation is available as `--prompt-profile careful`.
It clarifies transaction roles, required keys, whole-property rent, and named currencies.
Training and the default inference profile remain `original`. Configs record the selected
profile and its prompt hash; `compare` refuses to compare different prompts.
Apply the same profile to the baseline and adapter when measuring their difference:

```sh
uv run lease-lab evaluate --dataset data/real/benchmark --split test --prompt-profile careful --output artifacts/my-careful-baseline
uv run lease-lab evaluate --dataset data/real/benchmark --split test --prompt-profile careful --adapter artifacts/my-v3/best --output artifacts/my-careful-v3
uv run lease-lab compare artifacts/my-careful-baseline artifacts/my-careful-v3
uv run lease-lab extract data/samples/train-f00-001.pdf --prompt-profile careful --adapter artifacts/my-v3/best --output artifacts/my-careful-extraction.json
```

The revised prompt was written after inspecting real-document errors. Its scores on those
same documents are exploratory development results, not a new held-out test. Prompting gains
must not be attributed to training. A new independent benchmark is needed before a broad claim.

## Tests and releases

```sh
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

Unit tests run on Linux without MLX. Actual model inference/training requires Apple Silicon;
the local experiments exercise that path. CI runs tests and lint, never GPU training.
Results and model-card limitations are published under [reports/](reports/).
The first runs reached 100% (v1) and 99.5% (v2) synthetic test field accuracy, but regressed
against the format-normalized baseline on the two real leases. See [the results](reports/RESULTS.md)
for the distinction between output-format compliance and factual extraction.
Adapters are published as release assets, rather than embedding base-model weights in Git.
Extract an adapter archive at the repository root, then pass its `best/` directory to `--adapter`.
The base model is downloaded separately at its pinned revision.

Code and original synthetic examples: [Apache-2.0](LICENSE).
Base model: [Qwen3-1.7B, Apache-2.0](https://huggingface.co/Qwen/Qwen3-1.7B).
External evaluation material: separate CC BY-NC 4.0 attribution in `data/real/benchmark/LICENSE.txt`.
