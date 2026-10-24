import argparse
import json
from pathlib import Path

from .dataset import build_dataset, write_json
from .evaluation import score_predictions
from .pdf import read_pdf
from .real_data import build_real_dataset
from .runtime import (
    comparable_runs,
    evaluate_dataset,
    inference_config,
    load_local,
    prediction,
    read_jsonl,
    train_adapter,
)
from .schema import parse_prediction


def main():
    parser = argparse.ArgumentParser(description="Local lease PDF fine-tuning lab")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Generate reproducible original lease PDFs and labels"
    )
    prepare.add_argument("--output", type=Path, default=Path("data/generated/v1"))
    prepare.add_argument("--variant", choices=["v1", "v2", "v3"], default="v1")
    real = commands.add_parser(
        "prepare-real", help="Prepare the separate CC BY-NC external evaluation PDFs"
    )
    real.add_argument("--output", type=Path, default=Path("data/real/benchmark"))
    real.add_argument("--archive", type=Path, help="Optional already-downloaded pinned archive")
    real.add_argument("--suite", choices=["original", "fresh"], default="original")
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate baseline or adapter against one dataset split"
    )
    evaluate.add_argument("--dataset", type=Path, default=Path("data/generated/v1"))
    evaluate.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    evaluate.add_argument("--adapter", type=Path)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--output", type=Path, required=True)
    train = commands.add_parser(
        "train", help="Train LoRA with completion-only loss and validation selection"
    )
    train.add_argument("--dataset", type=Path, default=Path("data/generated/v1"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=200)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    extract = commands.add_parser("extract", help="Extract one new, short, text-based PDF")
    extract.add_argument("pdf", type=Path)
    extract.add_argument("--adapter", type=Path)
    extract.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score", help="Recompute metrics from saved predictions")
    score.add_argument("predictions", type=Path)
    compare = commands.add_parser(
        "compare", help="Compare two runs after checking identical evaluation conditions"
    )
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = build_dataset(args.output, args.variant)
        elif args.command == "prepare-real":
            result = build_real_dataset(args.output, args.archive, args.suite)
        elif args.command == "evaluate":
            if args.limit is not None and args.limit < 1:
                raise ValueError("--limit must be positive")
            result = evaluate_dataset(
                args.dataset, args.split, args.output, args.adapter, args.limit
            )
        elif args.command == "train":
            result = train_adapter(args.dataset, args.output, args.steps, args.learning_rate)
        elif args.command == "extract":
            config = inference_config(args.adapter, args.pdf)
            text = read_pdf(args.pdf)
            model, tokenizer = load_local(args.adapter)
            result = prediction(model, tokenizer, text)
            result["config"] = config
            write_json(args.output, result)
            result["extraction"] = parse_prediction(result["output"]).model_dump()
            write_json(args.output, result)
        elif args.command == "score":
            result = score_predictions(read_jsonl(args.predictions))
        else:
            configs = [
                json.loads((p / "config.json").read_text()) for p in [args.first, args.second]
            ]
            comparable_runs(*configs)
            result = {
                "first": json.loads((args.first / "metrics.json").read_text()),
                "second": json.loads((args.second / "metrics.json").read_text()),
            }
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
