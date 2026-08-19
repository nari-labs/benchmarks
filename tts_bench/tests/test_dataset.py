from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest

import tts_bench.dataset as dataset_module
from tests.helpers import make_prompts
from tts_bench.artifacts import write_dataset_snapshot
from tts_bench.dataset import PromptPool, load_prompt_pool, prepare_seedtts_dataset
from tts_bench.models import DatasetManifest, Phase


def test_prepare_seedtts_dataset_projects_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        b"sample-1|Reference one|prompt-wavs/one.wav|Synthesize this sentence.\n"
        b"sample-2|Reference two|prompt-wavs/two.wav|And synthesize this one.\n"
    )
    expected_rows = (
        {"id": "sample-1", "text": "Synthesize this sentence."},
        {"id": "sample-2", "text": "And synthesize this one."},
    )
    projection = "".join(
        f"{json.dumps(row, separators=(',', ':'))}\n" for row in expected_rows
    ).encode()
    expected_prompts = tuple(
        dataset_module._prompt(row["id"], row["text"], index)
        for index, row in enumerate(expected_rows)
    )
    monkeypatch.setattr(dataset_module, "SEEDTTS_SOURCE_SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(
        dataset_module, "SEEDTTS_PROJECTION_SHA256", hashlib.sha256(projection).hexdigest()
    )
    monkeypatch.setattr(
        dataset_module, "SEEDTTS_POOL_SHA256", dataset_module._pool_hash(expected_prompts)
    )
    monkeypatch.setattr(dataset_module, "SEEDTTS_ROWS", 2)
    monkeypatch.setattr(dataset_module, "urlopen", lambda *_args, **_kwargs: BytesIO(source))

    output, provenance_path = prepare_seedtts_dataset(tmp_path / "seedtts.jsonl")

    assert tuple(json.loads(line) for line in output.read_text().splitlines()) == expected_rows
    provenance = json.loads(provenance_path.read_text())
    assert provenance["source"]["license"] == "CC-BY-4.0"
    assert provenance["source"]["revision"] == dataset_module.SEEDTTS_REVISION
    assert provenance["projection"]["rows"] == 2

    with pytest.raises(dataset_module.DatasetError, match="refusing to overwrite"):
        prepare_seedtts_dataset(output)


def test_saved_snapshot_is_reusable(tmp_path: Path) -> None:
    prompts = make_prompts(7)
    source_manifest = DatasetManifest(
        kind="jsonl_snapshot",
        source_path="original.jsonl",
        source_files=("original.jsonl",),
        source_sha256="a" * 64,
        prompt_count=len(prompts),
        pool_sha256=dataset_module._pool_hash(prompts),
    )
    write_dataset_snapshot(tmp_path, prompts, source_manifest)

    loaded, manifest = load_prompt_pool(tmp_path / "dataset.jsonl")
    assert loaded == prompts
    assert manifest.kind == "jsonl_snapshot"
    assert manifest.pool_sha256 == source_manifest.pool_sha256


def test_prompt_pool_shuffle_is_full_deterministic_and_phase_independent() -> None:
    prompts = make_prompts(31)
    first = PromptPool(prompts, seed=0)
    second = PromptPool(prompts, seed=0)

    measurement_epoch_0 = tuple(first.at(Phase.MEASUREMENT, index) for index in range(31))
    measurement_epoch_1 = tuple(first.at(Phase.MEASUREMENT, index + 31) for index in range(31))
    warmup_epoch_0 = tuple(first.at(Phase.WARMUP, index) for index in range(31))

    assert set(measurement_epoch_0) == set(prompts)
    assert set(measurement_epoch_1) == set(prompts)
    assert measurement_epoch_0 != measurement_epoch_1
    assert warmup_epoch_0 != measurement_epoch_0
    assert measurement_epoch_0 == tuple(second.at(Phase.MEASUREMENT, index) for index in range(31))
    for warmup_requests in (0, 1, 17, 100):
        for index in range(warmup_requests):
            second.at(Phase.WARMUP, index)
        assert tuple(second.at(Phase.MEASUREMENT, index) for index in range(31)) == (
            measurement_epoch_0
        )
