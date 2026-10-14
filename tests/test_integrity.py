import json

import pytest

from lease_lab.dataset import build_dataset, validate_dataset


def test_edited_labels_cannot_reuse_an_existing_dataset_version(tmp_path):
    build_dataset(tmp_path)
    path = tmp_path / "examples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["expected"]["tenant"] = "Changed after versioning"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="version"):
        validate_dataset(tmp_path)


def test_training_split_cannot_silently_substitute_test_documents(tmp_path):
    build_dataset(tmp_path)
    (tmp_path / "train.jsonl").write_text((tmp_path / "test.jsonl").read_text())
    with pytest.raises(ValueError, match="train"):
        validate_dataset(tmp_path)
