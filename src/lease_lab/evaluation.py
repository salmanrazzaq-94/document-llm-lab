import statistics

from .schema import Lease, parse_prediction

FIELDS = [
    "tenant",
    "landlord",
    "premises_address",
    "commencement_date",
    "expiration_date",
    "base_rent.amount",
    "base_rent.currency",
    "base_rent.period",
]


def flatten(value: dict) -> dict:
    rent = value["base_rent"] or {"amount": None, "currency": None, "period": None}
    return {
        **{k: v for k, v in value.items() if k != "base_rent"},
        **{f"base_rent.{key}": val for key, val in rent.items()},
    }


def equal(expected, predicted) -> bool:
    if isinstance(expected, str) and isinstance(predicted, str):
        return " ".join(expected.split()).casefold() == " ".join(predicted.split()).casefold()
    return expected == predicted


def score_predictions(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot score an empty prediction set")
    correct = {field: 0 for field in FIELDS}
    nonnull_correct = nonnull_total = unsupported = predicted_nonnull = valid = exact = 0
    for row in rows:
        expected = flatten(Lease.model_validate(row["expected"]).model_dump())
        nonnull_total += sum(value is not None for value in expected.values())
        try:
            predicted = flatten(parse_prediction(row["output"]).model_dump())
        except ValueError:
            continue
        valid += 1
        all_fields_match = True
        for field in FIELDS:
            match = equal(expected[field], predicted[field])
            all_fields_match = all_fields_match and match
            correct[field] += int(match)
            if expected[field] is not None:
                nonnull_correct += int(match)
            if predicted[field] is not None:
                predicted_nonnull += 1
                unsupported += int(not match)
        exact += int(all_fields_match)
    seconds = [r.get("seconds", 0) for r in rows]
    return {
        "documents": len(rows),
        "schema_validity": valid / len(rows),
        "field_accuracy": sum(correct.values()) / (len(rows) * len(FIELDS)),
        "nonnull_field_accuracy": nonnull_correct / nonnull_total if nonnull_total else None,
        "exact_document_accuracy": exact / len(rows),
        "per_field_accuracy": {k: v / len(rows) for k, v in correct.items()},
        "unsupported_value_count": unsupported,
        "unsupported_value_rate": unsupported / predicted_nonnull if predicted_nonnull else 0,
        "unsupported_value_definition": "Non-null predictions disagreeing with reviewed gold labels; not semantic entailment",
        "invalid_prediction_count": len(rows) - valid,
        "total_seconds": sum(seconds),
        "median_seconds": statistics.median(seconds),
        "peak_memory_gb": max((r.get("peak_memory_gb", 0) for r in rows), default=0),
    }
