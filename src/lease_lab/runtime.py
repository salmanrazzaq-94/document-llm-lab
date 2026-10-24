"""One local inference/training path, with explicit provenance and bounded inputs."""

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

from .dataset import validate_dataset, write_json, write_jsonl
from .evaluation import score_predictions
from .prompts import make_messages

MODEL = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
GENERATION = {"max_tokens": 384, "temperature": 0.0, "enable_thinking": False}


def check_length(tokens: list[int], limit: int = 2048) -> None:
    if len(tokens) > limit:
        raise ValueError(
            f"Input has {len(tokens)} tokens; limit is {limit}. No truncation applied."
        )


def select_best_checkpoint(events: list[dict]) -> dict:
    finite = [e for e in events if math.isfinite(e["val_loss"])]
    if not finite:
        raise ValueError("No finite validation losses; no checkpoint can be selected")
    return min(finite, key=lambda e: e["val_loss"])


def comparable_runs(first: dict, second: dict) -> None:
    for key in [
        "model",
        "dataset_version",
        "split",
        "ids",
        "model_revision",
        "prompt_hash",
        "generation",
    ]:
        if first[key] != second[key]:
            raise ValueError(f"Runs differ in {key}; a controlled comparison is not possible")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def inference_config(
    adapter: Path | None = None, input_pdf: Path | None = None, *, prompt_profile: str = "original"
) -> dict:
    config = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "generation": GENERATION,
        "prompt_hash": hashlib.sha256(
            json.dumps(make_messages("", profile=prompt_profile), sort_keys=True).encode()
        ).hexdigest(),
        "prompt_profile": prompt_profile,
        "adapter": str(adapter) if adapter else None,
    }
    if input_pdf is not None:
        config["input_pdf"] = input_pdf.name
        config["input_pdf_sha256"] = hashlib.sha256(input_pdf.read_bytes()).hexdigest()
    if adapter:
        training = json.loads((adapter.parent / "training.json").read_text())
        if training["model"] != MODEL or training["model_revision"] != MODEL_REVISION:
            raise ValueError("Adapter was trained against a different pinned model")
        config["adapter_sha256"] = hashlib.sha256(
            (adapter / "adapters.safetensors").read_bytes()
        ).hexdigest()
    return config


class NonThinkingTokenizer:
    """Force the same template in MLX's dataset processor and inference."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)


def load_local(adapter: Path | None = None):
    from mlx_lm import load

    return load(MODEL, revision=MODEL_REVISION, adapter_path=str(adapter) if adapter else None)


def prediction(model, tokenizer, text: str, *, prompt_profile: str = "original") -> dict:
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    tokens = tokenizer.apply_chat_template(
        make_messages(text, profile=prompt_profile),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    check_length(tokens, 2048 - GENERATION["max_tokens"])
    started = time.perf_counter()
    response = "".join(
        part.text
        for part in stream_generate(
            model,
            tokenizer,
            tokens,
            max_tokens=GENERATION["max_tokens"],
            sampler=make_sampler(temp=0),
        )
    )
    return {
        "output": response,
        "seconds": time.perf_counter() - started,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }


def evaluate_dataset(
    dataset: Path,
    split: str,
    output: Path,
    adapter: Path | None = None,
    limit: int | None = None,
    *,
    prompt_profile: str = "original",
) -> dict:
    import mlx.core as mx

    output.mkdir(parents=True, exist_ok=False)
    manifest = validate_dataset(dataset)
    rows = read_jsonl(dataset / f"{split}.jsonl")
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("Evaluation needs at least one document")
    config = {
        **inference_config(adapter, prompt_profile=prompt_profile),
        "dataset_version": manifest["version"],
        "split": split,
        "ids": [r["id"] for r in rows],
        "limit": limit,
    }
    write_json(output / "config.json", config)
    model, tokenizer = load_local(adapter)
    mx.reset_peak_memory()
    results = []
    for i, row in enumerate(rows, 1):
        result = dict(
            id=row["id"],
            expected=row["expected"],
            text=row["text"],
            **prediction(model, tokenizer, row["text"], prompt_profile=prompt_profile),
        )
        results.append(result)
        write_jsonl(output / "predictions.jsonl", results)
        print(f"Evaluated {i}/{len(rows)}: {row['id']}", flush=True)
    metrics = score_predictions(results)
    write_json(output / "metrics.json", metrics)
    return metrics


def train_adapter(
    dataset: Path, output: Path, steps: int = 200, learning_rate: float = 1e-4
) -> dict:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import ChatDataset
    from mlx_lm.tuner.trainer import TrainingCallback

    manifest = validate_dataset(dataset)
    if not all(manifest["counts"].get(split, 0) for split in ["train", "valid"]):
        raise ValueError("Training requires non-empty train and validation splits")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if not 1 <= steps <= 200:
        raise ValueError("Introductory runs support 1 to 200 steps")
    output.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_local()
    np.random.seed(42)
    mx.random.seed(42)
    template = NonThinkingTokenizer(tokenizer)
    sets = {}
    for split in ["train", "valid"]:
        rows = read_jsonl(dataset / f"{split}.jsonl")
        chats = [{"messages": make_messages(r["text"], r["expected"])} for r in rows]
        processed = ChatDataset(chats, template, mask_prompt=True)
        for row in chats:
            tokens, offset = processed.process(row)
            check_length(tokens)
            prefix = template.apply_chat_template(
                row["messages"][:-1], add_generation_prompt=True, return_dict=False
            )
            if tokens[:offset] != prefix:
                raise ValueError("Training prompt differs from inference prompt")
        sets[split] = processed
    settings = copy.deepcopy(CONFIG_DEFAULTS)
    settings.update(
        model=MODEL,
        data=str(dataset),
        train=True,
        seed=42,
        num_layers=8,
        batch_size=1,
        iters=steps,
        learning_rate=learning_rate,
        mask_prompt=True,
        max_seq_length=2048,
        steps_per_eval=20,
        steps_per_report=10,
        val_batches=-1,
        save_every=20,
        grad_checkpoint=True,
        adapter_path=str(output / "last"),
    )
    provenance = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset_version": manifest["version"],
        "settings": settings,
        "checkpoint_selection": "minimum full-validation completion loss; includes step-zero baseline",
    }
    write_json(output / "training.json", provenance)
    events = []
    best_dir = output / "best"
    best_dir.mkdir()

    class Recorder(TrainingCallback):
        def on_train_loss_report(self, info):
            events.append({"type": "train", **info})
            write_jsonl(output / "events.jsonl", events)

        def on_val_loss_report(self, info):
            events.append({"type": "validation", **info})
            write_jsonl(output / "events.jsonl", events)
            best = select_best_checkpoint([e for e in events if e["type"] == "validation"])
            if best is events[-1]:
                mx.save_safetensors(
                    str(best_dir / "adapters.safetensors"),
                    dict(tree_flatten(model.trainable_parameters())),
                )
                write_json(best_dir / "adapter_config.json", settings)
                write_json(output / "selection.json", best)

    started = time.perf_counter()
    mx.reset_peak_memory()
    train_model(SimpleNamespace(**settings), model, sets["train"], sets["valid"], Recorder())
    provenance.update(
        seconds=time.perf_counter() - started,
        peak_memory_gb=mx.get_peak_memory() / 1e9,
        best_checkpoint=json.loads((output / "selection.json").read_text()),
    )
    write_json(output / "training.json", provenance)
    return provenance
