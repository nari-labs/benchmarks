"""Deepgram batch transcription and WER contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from tests.helpers import sine_pcm
from tts_bench.audio import pcm_to_wav
from tts_bench.metrics import derive_summary
from tts_bench.models import (
    PcmFormat,
    Phase,
    PlaybackRecord,
    Prompt,
    RequestRecord,
    SetupMetrics,
    TargetKind,
    TransportStatus,
    WERRecord,
    WERStatus,
)
from tts_bench.wer import DeepgramWEROptions, compute_wer, evaluate_measurement_wer, normalize_text

Handler = Callable[[web.Request], Coroutine[Any, Any, web.StreamResponse]]
REFERENCE = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
MODEL_UUID = "36a58a01-9c9a-4e10-b2a4-7a92b71818b6"


@asynccontextmanager
async def deepgram_server(handler: Handler) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_post("/v1/listen", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        yield str(server.make_url("/v1/listen"))
    finally:
        await server.close()


def deepgram_payload(transcript: str) -> dict[str, Any]:
    return {
        "metadata": {
            "request_id": "deepgram-request",
            "models": [MODEL_UUID],
            "model_info": {
                MODEL_UUID: {
                    "name": "general-nova-3",
                    "version": "2026-07-01.12345",
                    "arch": "nova-3",
                }
            },
        },
        "results": {"channels": [{"alternatives": [{"transcript": transcript}]}]},
    }


def make_prompt(*, prompt_id: str = "prompt") -> Prompt:
    return Prompt(
        id=prompt_id,
        text=REFERENCE,
        language="en",
        word_count=10,
        source_index=0,
    )


def make_request(
    root: Path,
    *,
    request_id: str = "request",
    prompt_id: str = "prompt",
    phase: Phase = Phase.MEASUREMENT,
    success: bool = True,
) -> RequestRecord:
    pcm_format = PcmFormat()
    wav = pcm_to_wav(sine_pcm(50), pcm_format)
    relative = f"audio/wav/{phase.value}/{request_id}.wav"
    absolute = root / relative
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(wav)
    return RequestRecord(
        run_id="run",
        request_id=request_id,
        phase=phase,
        phase_index=0,
        prompt_id=prompt_id,
        prompt_word_count=10,
        scheduled_elapsed_ns=0,
        task_started_elapsed_ns=0,
        dispatch_elapsed_ns=0,
        completed_elapsed_ns=50_000_000,
        status_code=200 if success else 503,
        success=success,
        error_kind=None if success else "http_status",
        error_message=None if success else "HTTP 503",
        output_bytes=len(wav),
        measurement_window_audio_ns=50_000_000,
        wav_sha256=hashlib.sha256(wav).hexdigest(),
        wav_audio_path=relative,
        pcm_format=pcm_format,
        playback=PlaybackRecord(
            complete=success,
            continuous=True,
            first_playable_elapsed_ns=1_000_000,
            first_audible_elapsed_ns=1_000_000,
            audible_offset_ns=0,
            playback_end_elapsed_ns=50_000_000,
            audio_duration_ns=50_000_000,
            total_stalled_ns=0,
            largest_stall_ns=0,
        ),
        request_body={"input": REFERENCE},
        transport_status=(TransportStatus.COMPLETE if success else TransportStatus.INCOMPLETE),
    )


async def evaluate_one(
    root: Path,
    handler: Handler,
    *,
    api_key: str = "test-key",
    timeout_s: float = 60.0,
    max_attempts: int = 3,
) -> tuple[list[RequestRecord], tuple[WERRecord, ...]]:
    request = make_request(root)
    async with deepgram_server(handler) as endpoint:
        return await evaluate_measurement_wer(
            output=root,
            run_id="run",
            records=(request,),
            candidate_request_ids=frozenset({request.request_id}),
            prompts=(make_prompt(),),
            options=DeepgramWEROptions(
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
                max_attempts=max_attempts,
            ),
        )


def test_wer_normalization_and_edit_breakdown() -> None:
    assert normalize_text("NUMBERED THIRTY SIX MEMBERS") == "numbered 36 members"
    assert normalize_text("don't worry") == "do not worry"

    result = compute_wer("alpha bravo charlie", "alpha delta charlie echo")
    assert result.reference_words == 3
    assert result.substitutions == 1
    assert result.deletions == 0
    assert result.insertions == 1
    assert result.wer_percentage == pytest.approx(200 / 3)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transcript", "expected_percentage", "expected_deletions"),
    (
        ("alpha bravo charlie delta echo foxtrot golf hotel india", 10.0, 1),
        ("alpha bravo charlie delta echo foxtrot golf hotel", 20.0, 2),
        ("", 100.0, 10),
    ),
)
async def test_wer_is_recorded_without_changing_request_success(
    tmp_path: Path,
    transcript: str,
    expected_percentage: float,
    expected_deletions: int,
) -> None:
    async def handler(_request: web.Request) -> web.Response:
        return web.json_response(deepgram_payload(transcript))

    updated, evaluations = await evaluate_one(tmp_path, handler)

    evaluation = evaluations[0]
    assert evaluation.status is WERStatus.EVALUATED
    assert evaluation.wer_percentage == pytest.approx(expected_percentage)
    assert evaluation.deletions == expected_deletions
    assert evaluation.resolved_models[0].uuid == MODEL_UUID
    assert evaluation.resolved_models[0].version == "2026-07-01.12345"
    assert evaluation.error_kind is None
    assert updated[0].audio_success is True
    assert updated[0].success is True
    assert updated[0].error_kind is None

@pytest.mark.asyncio
async def test_only_explicit_semantic_measurement_candidates_are_submitted(
    tmp_path: Path,
) -> None:
    calls = 0

    async def handler(_request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        return web.json_response(deepgram_payload(REFERENCE))

    measurement = make_request(tmp_path, request_id="measurement")
    warmup = make_request(tmp_path, request_id="warmup", phase=Phase.WARMUP)
    audio_failure = make_request(tmp_path, request_id="failure", success=False)
    async with deepgram_server(handler) as endpoint:
        updated, evaluations = await evaluate_measurement_wer(
            output=tmp_path,
            run_id="run",
            records=(warmup, audio_failure, measurement),
            candidate_request_ids=frozenset({measurement.request_id}),
            prompts=(make_prompt(),),
            options=DeepgramWEROptions(api_key="test-key", endpoint=endpoint),
        )

    assert calls == 1
    assert len(evaluations) == 1
    by_id = {record.request_id: record for record in updated}
    assert by_id["measurement"].wer_status is WERStatus.EVALUATED
    assert by_id["warmup"].wer_status is WERStatus.NOT_APPLICABLE
    assert by_id["failure"].wer_status is WERStatus.NOT_APPLICABLE
    assert by_id["failure"].success is False


@pytest.mark.asyncio
async def test_auth_request_retry_and_raw_response_evidence(tmp_path: Path) -> None:
    secret = "dg-secret-that-must-never-be-persisted"
    statuses = [429, 503, 200]
    requests: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.read()
        requests.append(
            {
                "authorization": request.headers.get("Authorization"),
                "content_type": request.headers.get("Content-Type"),
                "query": dict(request.query),
                "is_wav": body.startswith(b"RIFF"),
            }
        )
        status = statuses[len(requests) - 1]
        if status != 200:
            return web.json_response(
                {"error": f"temporary-{status}"},
                status=status,
                headers={"Retry-After": "0"},
            )
        return web.json_response(deepgram_payload(REFERENCE))

    options = DeepgramWEROptions(api_key=secret)
    assert secret not in repr(options)
    _updated, evaluations = await evaluate_one(tmp_path, handler, api_key=secret)

    assert len(requests) == 3
    assert all(item["authorization"] == f"Token {secret}" for item in requests)
    assert all(item["content_type"] == "audio/wav" for item in requests)
    assert all(item["is_wav"] is True for item in requests)
    assert requests[0]["query"] == {
        "model": "nova-3",
        "language": "en",
        "punctuate": "true",
        "smart_format": "true",
    }
    evaluation = evaluations[0]
    assert evaluation.attempts == 3
    assert evaluation.response_path is not None
    raw_response = tmp_path / evaluation.response_path
    assert raw_response.is_file()
    assert evaluation.response_sha256 == hashlib.sha256(raw_response.read_bytes()).hexdigest()
    assert json.loads(raw_response.read_bytes()) == deepgram_payload(REFERENCE)
    for artifact in (path for path in tmp_path.rglob("*") if path.is_file()):
        assert secret.encode() not in artifact.read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("response_kind", ("unauthorized", "malformed", "timeout"))
async def test_evaluator_failures_are_missing_and_preserve_audio_success(
    tmp_path: Path,
    response_kind: str,
) -> None:
    calls = 0

    async def handler(_request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        if response_kind == "unauthorized":
            return web.json_response({"error": "unauthorized"}, status=401)
        if response_kind == "malformed":
            return web.json_response({"metadata": {}})
        await asyncio.sleep(0.03)
        return web.json_response(deepgram_payload(REFERENCE))

    updated, evaluations = await evaluate_one(
        tmp_path,
        handler,
        timeout_s=0.01 if response_kind == "timeout" else 1,
        max_attempts=2,
    )

    evaluation = evaluations[0]
    assert evaluation.status is WERStatus.MISSING
    assert evaluation.wer_percentage is None
    assert (
        evaluation.error_kind
        == {
            "unauthorized": "deepgram_http_error",
            "malformed": "deepgram_invalid_response",
            "timeout": "deepgram_timeout",
        }[response_kind]
    )
    assert calls == (2 if response_kind == "timeout" else 1)
    assert updated[0].audio_success is True
    assert updated[0].success is True
    assert updated[0].error_kind is None

    summary = derive_summary(
        run_id="run",
        target=TargetKind.VOXSERVE,
        requested_rps=1,
        warmup_ns=0,
        duration_ns=1_000_000_000,
        scheduled_requests=1,
        dropped_requests=0,
        records=updated,
        peak_in_flight=1,
        setup=SetupMetrics(),
        wer_enabled=True,
    )
    assert summary.success_fraction == 1.0
    assert summary.wer_candidate_requests == 1
    assert summary.wer_evaluated_requests == 0
    assert summary.wer_coverage_fraction == 0.0
    assert summary.wer_percentage is not None
    assert summary.wer_percentage.samples == 0
    assert summary.wer_percentage.missing == 1
    for artifact in (path for path in tmp_path.rglob("*") if path.is_file()):
        assert b"test-key" not in artifact.read_bytes()


@pytest.mark.asyncio
async def test_api_key_is_redacted_even_if_an_error_response_echoes_it(tmp_path: Path) -> None:
    secret = "echoed-deepgram-secret"

    async def handler(_request: web.Request) -> web.Response:
        return web.json_response({"error": secret}, status=401)

    updated, evaluations = await evaluate_one(tmp_path, handler, api_key=secret)

    assert updated[0].success is True
    assert evaluations[0].status is WERStatus.MISSING
    assert evaluations[0].error_kind == "deepgram_response_secret_redacted"
    for artifact in (path for path in tmp_path.rglob("*") if path.is_file()):
        assert secret.encode() not in artifact.read_bytes()
