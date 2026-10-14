# Learn through experiments

Work through one lesson at a time. Before running each command, predict what will happen.
Afterward, open the saved JSON and explain any difference between your prediction and the result.

MLX lets us use your Mac's GPU locally. Tokens, loss, gradients, optimization, LoRA, and evaluation
transfer to PyTorch and Hugging Face TRL, but the training APIs and saved adapter format differ.
Keep that distinction clear when explaining this project in an interview.

## 1. Inference before training

Run the README's first extraction command. Compare its JSON with the sample PDF.
The model reads **tokens**, which are pieces of text. A tokenizer maps text to token IDs.
The model's learned **weights** turn those IDs into probabilities for the next token.
Generation repeatedly selects a token and feeds it back into the model.
Here we use deterministic greedy generation so baseline and trained runs are comparable.

Find `make_messages` in `src/lease_lab/prompts.py`. Identify the instruction, input document,
and expected assistant answer. The PDF parser extracts text; the LLM does not see PDF pixels.
Compare `train-f00-000.pdf` with `train-f00-001.pdf`: the first deliberately omits rent and uses
an ambiguous commencement date. A sensible answer needs nulls, not confident guesses.

**Explain back:** Which errors would require better PDF extraction, and which require better model behavior?

## 2. The dataset is the specification

Open a few rows of `data/generated/v1/examples.jsonl`, then open their PDFs.
Each row pairs an input with its correct structured answer. Those pairs teach a behavior.
Merely collecting PDFs does not create a supervised training dataset.
The generator renders known facts, extracts the resulting PDF text, and retains the gold labels.
Review is still necessary: generation bugs can produce systematically incorrect labels.

Training examples update weights. Validation examples guide experiment and checkpoint choices.
Test examples measure the final chosen experiment. Looking at test mistakes and repeatedly
tuning against them turns the test set into another validation set.
Template-family splits reduce leakage, but these small synthetic templates are still related.

**Exercise:** Audit one ordinary example, one missing-rent example, and one ambiguous-date example.
Check that the deposit is not labeled as rent. Describe a realistic lease these examples do not cover.

## 3. What LoRA learns

For a weight matrix W, LoRA learns an adjustment represented by two smaller matrices:

```text
effective W = frozen W + scale × B × A
```

The small inner dimension is the **rank**. Increasing rank allows a richer adjustment and
uses more trainable parameters. The original model still participates in every forward pass.
In this project about 2.5 million parameters are trainable in a 1.7-billion-parameter model.

The **loss** measures how surprising the correct output tokens are to the model.
**Backpropagation** computes how each trainable parameter contributed to that loss.
The **optimizer** updates those parameters. The **learning rate** controls update size.
We mask the input prompt: the objective learns the JSON completion, not to reproduce the lease text.

**Exercise:** Run 20 steps. Open `events.jsonl` and identify training versus validation losses.
Explain why lower training loss alone does not establish better extraction.

## 4. Train v1 and evaluate

A **step** is an optimizer update here because gradient accumulation is one.
An **epoch** is a pass through the training dataset; steps and epochs are not interchangeable.
With 200 examples, batch size 1, and 200 steps, this is approximately one pass.
The framework shuffles and length-batches examples, so record actual settings rather than assuming order.

Training saves both last and validation-selected adapters. Evaluate `best/` after reloading it
in a fresh process. Inspect per-field accuracy and invalid outputs, not just a single overall score.
Null fields can inflate average accuracy; the separate non-null score makes this visible.
Peak memory is MLX allocation, not total system RAM or every framework's allocation.

**Explain back:** Could validation loss improve while exact-document accuracy stays unchanged? Why?

## 5. A controlled v2 experiment

Use validation failures to form a hypothesis before generating more training examples.
Our starter v2 hypothesis is that extra ambiguous-date and missing-rent examples reinforce abstention.
Keep model, prompt, rank, learning rate, and evaluation data fixed; change the training data.
Start v2 from the original model. This avoids confusing dataset improvement with additional
training time carried over from v1. Continuing an adapter is a different experiment.

If v1 already succeeds on the validation cases, v2 is an ablation of additional hard-example
data rather than a promised fix. A tie or regression is a useful result. Do not invent a success story.

**Exercise:** Write the hypothesis, controlled variables, results, and one limitation in five sentences.

## 6. Release something another person can reproduce

A useful release includes source code, reusable data, pinned base-model revision, adapter weights,
training settings, evaluation predictions, and a model card explaining intended use and limitations.
Someone else should be able to regenerate the synthetic dataset and load the same adapter.
They may get small floating-point differences on different hardware or library versions.

For an interview, explain one actual failure, the evidence that identified its cause, the
experiment you ran, and the measured result. Distinguish formatting improvements from factual
accuracy and synthetic-template success from generalization to real contracts.
