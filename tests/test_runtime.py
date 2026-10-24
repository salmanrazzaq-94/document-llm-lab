import pytest

from lease_lab.runtime import (
    MODEL,
    MODEL_REVISION,
    check_length,
    comparable_runs,
    inference_config,
    select_best_checkpoint,
)


def test_length_limit_rejects_without_truncation():
    check_length([1] * 2048)
    with pytest.raises(ValueError, match="2049"):
        check_length([1] * 2049)


def test_checkpoint_selection_uses_validation_and_rejects_nan(tmp_path):
    events = [
        {"iteration": 0, "val_loss": 3.0},
        {"iteration": 19, "val_loss": 0.5},
        {"iteration": 39, "val_loss": 0.8},
    ]
    assert select_best_checkpoint(events)["iteration"] == 19
    with pytest.raises(ValueError):
        select_best_checkpoint([{"iteration": 1, "val_loss": float("nan")}])


def test_comparison_refuses_different_datasets_or_prompts():
    run = {
        "model": "base-model",
        "dataset_version": "abc",
        "split": "test",
        "ids": ["one"],
        "model_revision": "rev",
        "prompt_hash": "prompt",
        "generation": {"max_tokens": 384},
    }
    comparable_runs(run, dict(run))
    for key in ["model", "dataset_version", "prompt_hash", "model_revision", "ids"]:
        changed = dict(run) | {key: "different"}
        with pytest.raises(ValueError):
            comparable_runs(run, changed)


def test_inference_config_records_pinned_inputs_and_rejects_wrong_adapter_model(tmp_path):
    pdf = tmp_path / "lease.pdf"
    pdf.write_bytes(b"lease")
    adapter = tmp_path / "best"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"weights")
    (tmp_path / "training.json").write_text(
        '{"model": "' + MODEL + '", "model_revision": "' + MODEL_REVISION + '"}'
    )
    config = inference_config(adapter, pdf)
    assert config["model"] == MODEL
    assert config["model_revision"] == MODEL_REVISION
    assert config["input_pdf"] == "lease.pdf"
    assert config["input_pdf_sha256"]
    assert config["adapter_sha256"]

    (tmp_path / "training.json").write_text(
        '{"model": "other", "model_revision": "' + MODEL_REVISION + '"}'
    )
    with pytest.raises(ValueError, match="pinned model"):
        inference_config(adapter, pdf)


def test_prompt_ablation_records_a_distinct_prompt_and_keeps_original_default():
    from lease_lab.prompts import make_messages
    from lease_lab.runtime import inference_config

    original = inference_config()
    careful = inference_config(prompt_profile="careful")
    assert original["prompt_profile"] == "original"
    assert original["prompt_hash"] != careful["prompt_hash"]
    with pytest.raises(ValueError, match="prompt_hash"):
        comparable_runs(
            original | {"dataset_version": "same", "split": "test", "ids": ["same"]},
            careful | {"dataset_version": "same", "split": "test", "ids": ["same"]},
        )
    messages = make_messages("Document", profile="careful")
    assert "whole-property" in messages[0]["content"]
    assert "ALL keys" in messages[0]["content"]
