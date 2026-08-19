from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

import tts_bench.runtime as runtime_module
from tests.helpers import make_manifest, make_prompts
from tts_bench.artifacts import ArtifactStore, write_dataset_snapshot
from tts_bench.models import (
    ArrivalPattern,
    ArrivalSlot,
    AudioSuccessCriteria,
    PcmFormat,
    Phase,
    PlaybackRecord,
    Prompt,
    RequestRecord,
    TargetKind,
)
from tts_bench.targets import TargetAdapter


def test_audio_success_criteria_accept_stall_threshold_boundaries() -> None:
    criteria = AudioSuccessCriteria()
    playback = PlaybackRecord(
        complete=True,
        continuous=False,
        audible_offset_ns=0,
        audio_duration_ns=9_000_000_000,
        total_stalled_ns=1_000_000_000,
        largest_stall_ns=500_000_000,
    )

    assert runtime_module._audio_success_failure(playback, criteria=criteria) is None


@pytest.mark.parametrize(
    ("playback", "expected_kind"),
    (
        (
            PlaybackRecord(
                complete=True,
                continuous=False,
                audible_offset_ns=0,
                audio_duration_ns=1_000_000_000,
                total_stalled_ns=500_000_001,
                largest_stall_ns=500_000_001,
            ),
            "audio_stall",
        ),
        (
            PlaybackRecord(
                complete=True,
                continuous=False,
                audible_offset_ns=0,
                audio_duration_ns=1_000_000_000,
                total_stalled_ns=1_000_000_001,
                largest_stall_ns=400_000_000,
            ),
            "audio_stall",
        ),
    ),
)
def test_audio_success_criteria_reject_quality_failures(
    playback: PlaybackRecord, expected_kind: str
) -> None:
    failure = runtime_module._audio_success_failure(
        playback,
        criteria=AudioSuccessCriteria(),
    )

    assert failure is not None
    assert failure[0] == expected_kind


def test_duration_and_audibility_do_not_change_continuity_success() -> None:
    playback = PlaybackRecord(
        complete=True,
        continuous=True,
        audio_duration_ns=30_000_000_000,
        total_stalled_ns=0,
        largest_stall_ns=0,
    )

    assert runtime_module._audio_success_failure(playback, criteria=AudioSuccessCriteria()) is None


@pytest.mark.asyncio
async def test_scheduler_drops_without_waiting_and_drains_started_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = make_prompts(3)
    write_dataset_snapshot(tmp_path, prompts, make_manifest(prompts))
    slots = tuple(
        ArrivalSlot(
            phase=Phase.MEASUREMENT,
            phase_index=index,
            scheduled_elapsed_ns=index,
            prompt_id=prompt.id,
            prompt_word_count=prompt.word_count,
        )
        for index, prompt in enumerate(prompts)
    )
    gate = asyncio.Event()
    sleep_calls = 0

    async def fake_sleep_until(_deadline_ns: int) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 3:
            gate.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def fake_execute_request(**kwargs: Any) -> RequestRecord:
        slot = cast(ArrivalSlot, kwargs["slot"])
        prompt = cast(Prompt, kwargs["prompt"])
        if slot.phase_index == 0:
            await gate.wait()
        playback = PlaybackRecord(
            complete=True,
            continuous=True,
            first_playable_elapsed_ns=slot.scheduled_elapsed_ns,
            playback_end_elapsed_ns=slot.scheduled_elapsed_ns + 1,
            audio_duration_ns=1,
            total_stalled_ns=0,
            largest_stall_ns=0,
        )
        return RequestRecord(
            run_id="run",
            request_id=f"request-{slot.phase_index}",
            phase=slot.phase,
            phase_index=slot.phase_index,
            prompt_id=prompt.id,
            prompt_word_count=prompt.word_count,
            scheduled_elapsed_ns=slot.scheduled_elapsed_ns,
            task_started_elapsed_ns=slot.scheduled_elapsed_ns,
            dispatch_elapsed_ns=slot.scheduled_elapsed_ns,
            completed_elapsed_ns=slot.scheduled_elapsed_ns + 1,
            status_code=200,
            success=True,
            output_bytes=2,
            measurement_window_audio_ns=1,
            playback=playback,
            request_body={"input": prompt.text},
        )

    monkeypatch.setattr(runtime_module, "_sleep_until", fake_sleep_until)
    monkeypatch.setattr(runtime_module, "_execute_request", fake_execute_request)
    store = ArtifactStore(tmp_path)
    store.start()
    target = TargetAdapter(
        kind=TargetKind.VOXSERVE,
        base_url="http://127.0.0.1:9",
        model="model",
        voice="voice",
        language="English",
        expected_format=PcmFormat(),
        request_params={},
    )

    records, dropped, peak = await runtime_module._run_schedule(
        run_id="run",
        adapter=target,
        slots=slots,
        output=tmp_path,
        store=store,
        warmup_ns=0,
        duration_ns=10,
        timeout_s=1,
        max_in_flight=1,
        audio_success_criteria=AudioSuccessCriteria(),
    )
    await store.close()

    assert dropped == {(Phase.MEASUREMENT, 1)}
    assert sorted(record.phase_index for record in records) == [0, 2]
    assert peak == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["type"] for event in events] == ["arrival.dropped"]


@pytest.mark.asyncio
async def test_internal_failure_preserves_incomplete_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_preflight(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("synthetic preflight failure")

    monkeypatch.setattr(runtime_module, "preflight_target", fail_preflight)
    prompts = make_prompts(1)
    output = tmp_path / "incomplete"
    target = TargetAdapter(
        kind=TargetKind.VOXSERVE,
        base_url="http://127.0.0.1:9",
        model="model",
        voice="voice",
        language="English",
        expected_format=PcmFormat(),
        request_params={},
    )
    with pytest.raises(RuntimeError, match="synthetic preflight failure"):
        await runtime_module.run_rate(
            adapter=target,
            prompts=prompts,
            dataset_manifest=make_manifest(prompts),
            output=output,
            options=runtime_module.RateOptions(
                requested_rps=1,
                arrival=ArrivalPattern.CONSTANT,
                seed=0,
                warmup_s=0,
                duration_s=0.1,
                timeout_s=1,
                max_in_flight=1,
            ),
        )

    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "incomplete"
    assert status["error_type"] == "RuntimeError"
    assert (output / "dataset.jsonl").is_file()
    assert (output / "requests.jsonl").is_file()
