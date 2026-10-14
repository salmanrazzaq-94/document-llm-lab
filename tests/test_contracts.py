import json

import pytest
from pydantic import ValidationError
from reportlab.pdfgen import canvas

from lease_lab.dataset import build_dataset
from lease_lab.evaluation import score_predictions
from lease_lab.pdf import read_pdf
from lease_lab.prompts import make_messages
from lease_lab.schema import Lease, parse_prediction


def empty_lease():
    return {
        "tenant": None,
        "landlord": None,
        "premises_address": None,
        "commencement_date": None,
        "expiration_date": None,
        "base_rent": None,
    }


def test_schema_requires_fields_and_rejects_invented_keys():
    assert Lease.model_validate(empty_lease()).base_rent is None
    with pytest.raises(ValidationError):
        Lease.model_validate({})
    with pytest.raises(ValidationError):
        Lease.model_validate(empty_lease() | {"deposit": 50})


def test_dates_and_money_are_strict():
    for bad in ["03/04/2026", "2026-02-30"]:
        with pytest.raises(ValidationError):
            Lease.model_validate(empty_lease() | {"commencement_date": bad})
    with pytest.raises(ValidationError):
        Lease.model_validate(
            empty_lease() | {"base_rent": {"amount": -1, "currency": "USD", "period": "month"}}
        )


def test_prediction_must_be_one_json_object():
    assert parse_prediction(json.dumps(empty_lease())).tenant is None
    for bad in ["```json\n{}\n```", "Explanation: {}", "{} {}"]:
        with pytest.raises(ValueError):
            parse_prediction(bad)


def test_pdf_preserves_page_boundaries_and_rejects_blank(tmp_path):
    path = tmp_path / "lease.pdf"
    doc = canvas.Canvas(str(path))
    doc.drawString(50, 700, "Tenant: Example LLC")
    doc.showPage()
    doc.drawString(50, 700, "Rent: USD 100 per month")
    doc.save()
    text = read_pdf(path)
    assert "[Page 1]" in text and "[Page 2]" in text
    assert text.index("Example LLC") < text.index("USD 100")
    blank = tmp_path / "blank.pdf"
    doc = canvas.Canvas(str(blank))
    doc.showPage()
    doc.save()
    with pytest.raises(ValueError, match="selectable text"):
        read_pdf(blank)


def test_deterministic_dataset_has_isolated_families(tmp_path):
    first = build_dataset(tmp_path / "a")
    second = build_dataset(tmp_path / "b")
    assert first["version"] == second["version"]
    assert first["counts"] == {"train": 200, "valid": 50, "test": 50}
    families = first["families"]
    assert not set(families["train"]) & set(families["valid"])
    assert not set(families["train"]) & set(families["test"])
    examples = [json.loads(x) for x in (tmp_path / "a" / "examples.jsonl").read_text().splitlines()]
    missing = [e for e in examples if e["expected"]["base_rent"] is None]
    assert missing and all("Security deposit" in e["text"] for e in missing)
    ambiguous = [e for e in examples if "03/04/2026" in e["text"]]
    assert ambiguous and all(e["expected"]["commencement_date"] is None for e in ambiguous)


def test_prompt_teaches_abstention_and_separates_completion():
    messages = make_messages("LEASE TEXT", empty_lease())
    assert [x["role"] for x in messages] == ["system", "user", "assistant"]
    assert "deposit" in messages[0]["content"]
    assert "null" in messages[0]["content"]
    assert json.loads(messages[-1]["content"]) == empty_lease()


def test_scores_invalid_json_and_unsupported_values():
    expected = empty_lease() | {"tenant": "Example LLC"}
    result = score_predictions(
        [
            {"expected": expected, "output": json.dumps(expected), "seconds": 1},
            {
                "expected": expected,
                "output": json.dumps(expected | {"landlord": "Invented"}),
                "seconds": 2,
            },
            {"expected": expected, "output": "not json", "seconds": 1},
        ]
    )
    assert result["schema_validity"] == pytest.approx(2 / 3)
    assert result["field_accuracy"] < 1
    assert result["unsupported_value_count"] == 1
    assert result["invalid_prediction_count"] == 1
