from lease_lab.dataset import build_dataset, validate_dataset
from lease_lab.real_data import benchmark_cases, fresh_cases
from lease_lab.runtime import read_jsonl
from lease_lab.schema import Lease


def test_v3_covers_generalization_gaps_without_external_training_data(tmp_path):
    manifest = build_dataset(tmp_path, "v3")
    assert validate_dataset(tmp_path) == manifest
    assert manifest["counts"] == {"train": 600, "valid": 100, "test": 100}
    rows = read_jsonl(tmp_path / "examples.jsonl")
    for split in manifest["counts"]:
        diverse = [r for r in rows if r["split"] == split and "features" in r]
        assert {r["features"]["date_style"] for r in diverse} >= {
            "iso",
            "month_first",
            "day_first",
            "ordinal",
            "abbreviated",
        }
        assert any(r["expected"]["tenant"].endswith(".") for r in diverse)
        assert any(r["features"]["indirect_roles"] for r in diverse)
        assert any(r["features"]["currency_kind"] == "ambiguous" for r in diverse)
        assert any(r["features"]["currency_kind"] == "named" for r in diverse)
        assert any(r["features"]["collection_differs"] for r in diverse)
        assert any(r["expected"]["commencement_date"] is None for r in diverse)
        assert all(r["source"] == "original-synthetic" for r in diverse)
    second = build_dataset(tmp_path / "again", "v3")
    assert manifest["version"] == second["version"]


def test_fresh_external_cases_are_disjoint_and_label_conflicting_parties_as_ambiguous():
    cases = fresh_cases()
    assert not {r["id"] for r in cases} & {r["id"] for r in benchmark_cases()}
    for row in cases:
        Lease.model_validate(row["expected"])
    conflict = next(r for r in cases if "landlord_conflict" in r["evidence"])
    assert conflict["expected"]["landlord"] is None
