from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer

from tests.helpers import make_manifest, make_prompts, sine_pcm
from tts_bench.audio import pcm_to_wav
from tts_bench.models import ArrivalPattern, PcmFormat, TargetKind, TransportKind
from tts_bench.reporting import load_and_render_report
from tts_bench.runtime import RateOptions, run_rate
from tts_bench.targets import TargetAdapter
from tts_bench.wer import DeepgramWEROptions

Handler = Callable[[web.Request], Coroutine[Any, Any, web.StreamResponse]]


@asynccontextmanager
async def local_server(
    handler: Handler,
    *,
    websocket_handler: Handler | None = None,
) -> AsyncIterator[str]:
    async def health(_request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/v1/audio/speech", handler)
    if websocket_handler is not None:
        app.router.add_get("/v1/audio/speech/ws", websocket_handler)
    server = TestServer(app)
    await server.start_server()
    try:
        yield str(server.make_url("")).rstrip("/")
    finally:
        await server.close()


@asynccontextmanager
async def local_deepgram_server(handler: Handler) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_post("/v1/listen", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        yield str(server.make_url("/v1/listen"))
    finally:
        await server.close()


def make_adapter(base_url: str) -> TargetAdapter:
    return TargetAdapter(
        kind=TargetKind.VOXSERVE,
        base_url=base_url,
        model="test-model",
        voice="test-voice",
        language="English",
        expected_format=PcmFormat(sample_rate_hz=24_000),
        request_params={"temperature": 0.1},
    )


@pytest.mark.asyncio
async def test_short_single_rate_run_persists_consistent_offline_report(tmp_path: Path) -> None:
    seen_bodies: list[dict[str, Any]] = []
    pcm = sine_pcm(80)

    async def speech(request: web.Request) -> web.StreamResponse:
        seen_bodies.append(await request.json())
        response = web.StreamResponse(status=200, headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await asyncio.sleep(0.002)
        await response.write(pcm[:701])
        await asyncio.sleep(0.002)
        await response.write(pcm[701:])
        await response.write_eof()
        return response

    prompts = make_prompts(5)
    output = tmp_path / "single-rate"
    options = RateOptions(
        requested_rps=5,
        arrival=ArrivalPattern.CONSTANT,
        seed=0,
        warmup_s=0,
        duration_s=0.12,
        timeout_s=1,
        max_in_flight=10,
    )
    async with local_server(speech) as base_url:
        summary = await run_rate(
            adapter=make_adapter(base_url),
            prompts=prompts,
            dataset_manifest=make_manifest(prompts),
            output=output,
            options=options,
        )

    assert load_and_render_report(output) == (output / "report.txt").read_text(encoding="utf-8")
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
    assert summary.schema_version == 1
    assert len(seen_bodies) == 1
    assert all(body["temperature"] == 0.1 for body in seen_bodies)

    expected = {
        "status.json",
        "run.json",
        "dataset.jsonl",
        "dataset-manifest.json",
        "arrivals.jsonl",
        "events.jsonl",
        "requests.jsonl",
        "measurement-prompt-sequence.jsonl",
        "summary.json",
        "report.txt",
        "audio-manifest.jsonl",
    }
    assert expected.issubset(path.name for path in output.iterdir())
    persisted_summary = json.loads((output / "summary.json").read_text())
    assert persisted_summary["successful_requests"] == persisted_summary["started_requests"]
    assert "viability_fraction" not in persisted_summary
    assert persisted_summary["underrun_request_fraction"] == 0
    assert persisted_summary["audible_ttfa_ms"]["missing"] == 0
    requests = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert all(item["schema_version"] == 1 for item in requests)
    assert all(item["transport_status"] == "complete" for item in requests)
    sequence = [
        json.loads(line)
        for line in (output / "measurement-prompt-sequence.jsonl").read_text().splitlines()
    ]
    assert [item["prompt_id"] for item in sequence] == [
        item["prompt_id"] for item in sorted(requests, key=lambda item: item["phase_index"])
    ]
    assert all((output / item["raw_audio_path"]).is_file() for item in requests)
    assert all((output / item["wav_audio_path"]).is_file() for item in requests)
    assert all(item["wav_sha256"] for item in requests)
    run_config = json.loads((output / "run.json").read_text())
    assert run_config["schema_version"] == 1
    assert run_config["package_version"] == "0.1.0"
    assert len(run_config["benchmark_git_revision"]) == 40
    assert isinstance(run_config["benchmark_git_dirty"], bool)
    assert len(run_config["dependency_lock_sha256"]) == 64
    assert run_config["measurement_contract"]["playback_startup_buffer_ms"] == 0
    assert run_config["measurement_contract"]["audible_detector"]["version"] == 1
    assert run_config["measurement_contract"]["request_success"] == {
        "capacity": {
            "require_successful_transport": True,
            "require_protocol_valid_pcm": True,
            "require_complete": True,
            "require_nonempty": True,
            "require_frame_aligned": True,
        },
        "continuity": {
            "largest_stall_ms": 500.0,
            "total_stall_ms": 1_000.0,
        },
        "semantic": {
            "audibility": "inaudible_fails_otherwise_unchecked",
            "audio_duration": "raw_telemetry_only",
        },
    }
    assert run_config["request_body_example"]["input"] != "<prompt>"


@pytest.mark.asyncio
async def test_wav_target_stream_is_normalized_to_pcm(tmp_path: Path) -> None:
    kind = TargetKind.MSTAR
    pcm = sine_pcm(80)
    wav = pcm_to_wav(pcm, PcmFormat(sample_rate_hz=24_000))
    seen_body: dict[str, Any] = {}

    async def speech(request: web.Request) -> web.StreamResponse:
        seen_body.update(await request.json())
        response = web.StreamResponse(status=200, headers={"Content-Type": "audio/wav"})
        await response.prepare(request)
        for boundary in (13, 37, len(wav)):
            start = 0 if boundary == 13 else (13 if boundary == 37 else 37)
            await response.write(wav[start:boundary])
        await response.write_eof()
        return response

    prompts = make_prompts(1)
    output = tmp_path / kind.value
    async with local_server(speech) as base_url:
        summary = await run_rate(
            adapter=TargetAdapter(
                kind=kind,
                base_url=base_url,
                model="test-model",
                voice="Ryan",
                language="English",
                expected_format=PcmFormat(sample_rate_hz=24_000),
                request_params={},
            ),
            prompts=prompts,
            dataset_manifest=make_manifest(prompts),
            output=output,
            options=RateOptions(
                requested_rps=5,
                arrival=ArrivalPattern.CONSTANT,
                seed=0,
                warmup_s=0,
                duration_s=0.12,
                timeout_s=1,
                max_in_flight=10,
            ),
        )

    assert seen_body["response_format"] == "wav"
    assert seen_body["stream"] is True
    assert seen_body["non_streaming_mode"] is False
    assert summary.target is kind
    assert summary.successful_requests == 1
    request = json.loads((output / "requests.jsonl").read_text())
    assert request["pcm_sha256"] is not None
    assert (output / request["wav_audio_path"]).read_bytes() == wav


@pytest.mark.asyncio
async def test_wer_runs_after_tts_and_scores_only_audio_success_measurements(
    tmp_path: Path,
) -> None:
    reference = "alpha bravo charlie delta echo"
    pcm = sine_pcm(50)
    tts_calls = 0
    deepgram_calls: list[dict[str, str]] = []
    secret = "integration-deepgram-secret"
    model_uuid = "36a58a01-9c9a-4e10-b2a4-7a92b71818b6"

    async def speech(request: web.Request) -> web.StreamResponse:
        nonlocal tts_calls
        await request.json()
        call_index = tts_calls
        tts_calls += 1
        if call_index == 3:
            return web.Response(status=503, text="synthetic measurement failure")
        return web.Response(body=pcm, headers={"Content-Type": "audio/pcm"})

    async def transcribe(request: web.Request) -> web.Response:
        assert tts_calls == 5
        deepgram_calls.append(dict(request.query))
        await asyncio.sleep(0.2)
        return web.json_response(
            {
                "metadata": {
                    "request_id": f"dg-{len(deepgram_calls)}",
                    "models": [model_uuid],
                    "model_info": {
                        model_uuid: {
                            "name": "general-nova-3",
                            "version": "2026-07-01.12345",
                            "arch": "nova-3",
                        }
                    },
                },
                "results": {"channels": [{"alternatives": [{"transcript": reference}]}]},
            }
        )

    prompts = tuple(
        prompt.model_copy(update={"text": reference, "word_count": 5}) for prompt in make_prompts(5)
    )
    output = tmp_path / "wer-rate"
    async with local_server(speech) as base_url, local_deepgram_server(transcribe) as endpoint:
        summary = await run_rate(
            adapter=make_adapter(base_url),
            prompts=prompts,
            dataset_manifest=make_manifest(prompts),
            output=output,
            options=RateOptions(
                requested_rps=10,
                arrival=ArrivalPattern.CONSTANT,
                seed=0,
                warmup_s=0.11,
                duration_s=0.21,
                timeout_s=1,
                max_in_flight=10,
                wer_options=DeepgramWEROptions(api_key=secret, endpoint=endpoint),
            ),
        )

    assert tts_calls == 5
    assert len(deepgram_calls) == 2
    assert all(
        query
        == {
            "model": "nova-3",
            "language": "en",
            "punctuate": "true",
            "smart_format": "true",
        }
        for query in deepgram_calls
    )
    assert summary.started_requests == 3
    assert summary.audio_successful_requests == 2
    assert summary.successful_requests == 2
    assert summary.success_fraction == pytest.approx(2 / 3)
    assert summary.wer_candidate_requests == 2
    assert summary.wer_evaluated_requests == 2
    assert summary.wer_coverage_fraction == 1.0
    assert summary.wer_percentage is not None
    assert summary.wer_percentage.mean == 0.0
    assert summary.semantic_quality.value == "unchecked"
    assert summary.end_to_end_ms.p95 is not None
    assert summary.end_to_end_ms.p95 < 200
    assert "WER coverage: 2/2 (100.00%)" in (output / "report.txt").read_text()

    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    warmups = [record for record in records if record["phase"] == "warmup"]
    measurements = [record for record in records if record["phase"] == "measurement"]
    assert len(warmups) == 2
    assert all(record["wer_status"] == "not_applicable" for record in warmups)
    assert sum(record["wer_status"] == "evaluated" for record in measurements) == 2
    assert sum(record["wer_status"] == "not_applicable" for record in measurements) == 1
    assert all(record["audio_success"] == record["success"] for record in measurements)

    wer_rows = [json.loads(line) for line in (output / "wer.jsonl").read_text().splitlines()]
    assert len(wer_rows) == 2
    assert all(row["schema_version"] == 1 for row in wer_rows)
    assert all("threshold_percentage" not in row for row in wer_rows)
    assert all("passed" not in row for row in wer_rows)
    assert all(row["resolved_models"][0]["uuid"] == model_uuid for row in wer_rows)
    assert all((output / row["response_path"]).is_file() for row in wer_rows)
    run = json.loads((output / "run.json").read_text())
    assert run["schema_version"] == 1
    assert run["wer_records_path"] == "wer.jsonl"
    assert run["measurement_contract"]["wer_evaluator"]["requested_model"] == "nova-3"
    assert run["measurement_contract"]["wer_evaluator"]["language"] == "en"
    assert run["measurement_contract"]["wer_evaluator"]["effect"] == "record_only"
    assert "wer" not in run["measurement_contract"]["request_success"]
    for artifact in (path for path in output.rglob("*") if path.is_file()):
        assert secret.encode() not in artifact.read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [TargetKind.VOXSERVE, TargetKind.NARI])
async def test_raw_pcm_target_posts_complete_text_to_openai_speech_endpoint(
    tmp_path: Path,
    kind: TargetKind,
) -> None:
    prompt = make_prompts(1)[0].model_copy(update={"text": "one two three four", "word_count": 4})
    output = tmp_path / f"{kind.value}-complete-text"
    pcm = sine_pcm(80)
    seen_body: dict[str, Any] = {}

    async def speech(request: web.Request) -> web.StreamResponse:
        seen_body.update(await request.json())
        response = web.StreamResponse(status=200, headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await response.write(pcm[:701])
        await response.write(pcm[701:])
        await response.write_eof()
        return response

    async with local_server(speech) as base_url:
        summary = await run_rate(
            adapter=TargetAdapter(
                kind=kind,
                base_url=base_url,
                model="test-model",
                voice="Ryan",
                language="English",
                expected_format=PcmFormat(sample_rate_hz=24_000),
                request_params={},
            ),
            prompts=(prompt,),
            dataset_manifest=make_manifest((prompt,)),
            output=output,
            options=RateOptions(
                requested_rps=5,
                arrival=ArrivalPattern.CONSTANT,
                seed=0,
                warmup_s=0,
                duration_s=0.12,
                timeout_s=1,
                max_in_flight=10,
            ),
        )

    assert seen_body["input"] == prompt.text
    assert seen_body["stream"] is True
    assert seen_body["non_streaming_mode"] is False
    assert seen_body["response_format"] == "pcm"
    request = json.loads((output / "requests.jsonl").read_text().strip())
    assert request["success"] is True
    assert request["first_body_elapsed_ns"] >= request["dispatch_elapsed_ns"]
    assert request["input_timing"] is None
    assert summary.target is kind
    run = json.loads((output / "run.json").read_text())
    assert run["transport"] == "http"
    assert run["measurement_contract"]["request_start"] == "first JSON body bytes sent"


@pytest.mark.asyncio
async def test_nari_websocket_streams_one_word_every_ten_milliseconds(
    tmp_path: Path,
) -> None:
    prompt = make_prompts(1)[0].model_copy(update={"text": "one two three four", "word_count": 4})
    pcm = sine_pcm(80)
    seen_start: dict[str, Any] = {}
    seen_chunks: list[dict[str, Any]] = []

    async def unused_http(_request: web.Request) -> web.Response:
        return web.Response(status=500, text="HTTP must not be used")

    async def speech_websocket(request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse(protocols=("nari.speech.v1",))
        await websocket.prepare(request)
        assert websocket.ws_protocol == "nari.speech.v1"
        await websocket.send_json(
            {
                "type": "session.created",
                "protocol": "nari.speech.v1",
                "audio": {
                    "encoding": "pcm_s16le",
                    "sample_rate": 24_000,
                    "channels": 1,
                },
            }
        )
        start = await websocket.receive_json()
        seen_start.update(start)
        await websocket.send_json({"type": "request.configured"})
        request_id = "ws-test-request"
        audio_chunks = 0
        async for message in websocket:
            assert message.type is WSMsgType.TEXT
            payload = json.loads(message.data)
            if payload["type"] == "input_text.append":
                seen_chunks.append(payload)
                await websocket.send_json(
                    {
                        "type": "input_text.ack",
                        "event": "input_text.append",
                        "sequence": payload["sequence"],
                        "request_id": request_id,
                    }
                )
                if payload["sequence"] == 0:
                    await websocket.send_json(
                        {
                            "type": "response.started",
                            "request_id": request_id,
                            "audio": {
                                "encoding": "pcm_s16le",
                                "sample_rate": 24_000,
                                "channels": 1,
                            },
                        }
                    )
                    await websocket.send_bytes(pcm[: 60 * 24_000 * 2 // 1_000])
                    audio_chunks += 1
            elif payload["type"] == "input_text.end":
                await websocket.send_json(
                    {
                        "type": "input_text.ack",
                        "event": "input_text.end",
                        "sequence": payload["sequence"],
                        "request_id": request_id,
                    }
                )
                await websocket.send_bytes(pcm[60 * 24_000 * 2 // 1_000 :])
                audio_chunks += 1
                await websocket.send_json(
                    {
                        "type": "response.done",
                        "request_id": request_id,
                        "stop_reason": "stop",
                        "audio_chunks": audio_chunks,
                    }
                )
                await websocket.close(code=1000)
                break
        return websocket

    output = tmp_path / "nari-websocket"
    async with local_server(unused_http, websocket_handler=speech_websocket) as base_url:
        summary = await run_rate(
            adapter=TargetAdapter(
                kind=TargetKind.NARI,
                base_url=base_url,
                model="test-model",
                voice="Ryan",
                language="English",
                expected_format=PcmFormat(sample_rate_hz=24_000),
                request_params={"temperature": 0.2},
                transport=TransportKind.WEBSOCKET,
            ),
            prompts=(prompt,),
            dataset_manifest=make_manifest((prompt,)),
            output=output,
            options=RateOptions(
                requested_rps=5,
                arrival=ArrivalPattern.CONSTANT,
                seed=0,
                warmup_s=0,
                duration_s=0.12,
                timeout_s=1,
                max_in_flight=10,
            ),
        )

    assert seen_start == {
        "type": "request.start",
        "model": "test-model",
        "voice": "Ryan",
        "language": "English",
        "response_format": "pcm",
        "temperature": 0.2,
    }
    assert [chunk["sequence"] for chunk in seen_chunks] == [0, 1, 2, 3]
    assert [chunk["text"] for chunk in seen_chunks] == ["one ", "two ", "three ", "four"]
    request = json.loads((output / "requests.jsonl").read_text())
    assert request["success"] is True
    assert request["status_code"] == 101
    assert request["transport_status"] == "complete"
    assert request["request_body"]["input"]["chunk_interval_ms"] == 10
    timing = request["input_timing"]
    assert timing["first_text_sent_elapsed_ns"] == request["dispatch_elapsed_ns"]
    assert timing["last_text_sent_elapsed_ns"] - timing["first_text_sent_elapsed_ns"] >= 20_000_000
    assert request["playback"]["first_playable_elapsed_ns"] < timing["end_sent_elapsed_ns"]
    assert summary.transport is TransportKind.WEBSOCKET
    assert summary.input_ingress_ms is not None
    assert summary.input_ingress_ms.samples == 1
    assert summary.audio_before_input_complete_fraction == 1
    assert summary.final_text_to_first_playable_ms is not None
    assert summary.final_text_to_first_playable_ms.p50 is not None
    assert summary.final_text_to_first_playable_ms.p50 < 0
    run = json.loads((output / "run.json").read_text())
    assert run["transport"] == "websocket"
    assert run["measurement_contract"]["websocket_input"] == {
        "protocol": "nari.speech.v1",
        "chunk_unit": "word",
        "chunk_size": 1,
        "chunk_interval_ms": 10,
        "acknowledgement": "required before the next chunk",
        "cadence": "absolute from the first text chunk",
    }
    report = (output / "report.txt").read_text()
    assert "Transport: websocket" in report
    assert "WebSocket setup" in report


@pytest.mark.asyncio
async def test_nari_websocket_failure_before_text_dispatch_is_counted(
    tmp_path: Path,
) -> None:
    prompt = make_prompts(1)[0]

    async def unused_http(_request: web.Request) -> web.Response:
        return web.Response(status=500, text="HTTP must not be used")

    async def speech_websocket(request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse(protocols=("nari.speech.v1",))
        await websocket.prepare(request)
        await websocket.close(code=1000)
        return websocket

    output = tmp_path / "nari-websocket-predispatch-failure"
    async with local_server(unused_http, websocket_handler=speech_websocket) as base_url:
        summary = await run_rate(
            adapter=TargetAdapter(
                kind=TargetKind.NARI,
                base_url=base_url,
                model="test-model",
                voice="Ryan",
                language="English",
                expected_format=PcmFormat(sample_rate_hz=24_000),
                request_params={},
                transport=TransportKind.WEBSOCKET,
            ),
            prompts=(prompt,),
            dataset_manifest=make_manifest((prompt,)),
            output=output,
            options=RateOptions(
                requested_rps=5,
                arrival=ArrivalPattern.CONSTANT,
                seed=0,
                warmup_s=0,
                duration_s=0.12,
                timeout_s=1,
                max_in_flight=10,
            ),
        )

    assert summary.started_requests == 1
    assert summary.dispatched_requests == 0
    assert summary.predispatch_failed_requests == 1
    assert summary.successful_requests == 0
    assert summary.success_fraction == 0
    assert summary.actual_rps == 0
    request = json.loads((output / "requests.jsonl").read_text())
    assert request["dispatch_elapsed_ns"] is None
    assert request["error_kind"] == "websocket_protocol_error"


@pytest.mark.asyncio
async def test_errors_are_observations_and_do_not_fail_completed_run(tmp_path: Path) -> None:
    pcm = sine_pcm(50)

    async def speech(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        text = str(body["input"])
        if "http error" in text:
            return web.Response(status=503, text="busy")
        response = web.StreamResponse(status=200, headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        if "timeout" in text:
            await asyncio.sleep(0.15)
            with contextlib.suppress(ConnectionResetError):
                await response.write(pcm)
        elif "malformed" in text:
            await response.write(b"\x00")
        elif "silent" in text:
            await response.write(b"\x00\x00" * 1_200)
        else:
            await response.write(pcm)
        with contextlib.suppress(ConnectionResetError):
            await response.write_eof()
        return response

    prompts = list(make_prompts(5))
    labels = ("good speech", "http error", "malformed audio", "timeout response", "silent audio")
    prompts = [
        prompt.model_copy(update={"text": label})
        for prompt, label in zip(prompts, labels, strict=True)
    ]
    output = tmp_path / "observations"
    async with local_server(speech) as base_url:
        summary = await run_rate(
            adapter=make_adapter(base_url),
            prompts=tuple(prompts),
            dataset_manifest=make_manifest(tuple(prompts)),
            output=output,
            options=RateOptions(
                requested_rps=10,
                arrival=ArrivalPattern.CONSTANT,
                seed=0,
                warmup_s=0,
                duration_s=0.5,
                timeout_s=0.05,
                max_in_flight=10,
            ),
        )

    assert summary.started_requests == 5
    assert summary.successful_requests == 2
    assert summary.pcm_complete_requests == 2
    assert summary.audible_requests == 1
    assert summary.semantic_quality.value == "fail"
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert {record["error_kind"] for record in records if record["error_kind"]} == {
        "http_status",
        "audio_protocol_error",
        "timeout",
    }
    silent = next(record for record in records if record["request_body"]["input"] == "silent audio")
    assert silent["transport_status"] == "complete"
    assert silent["success"] is True
    assert silent["error_kind"] is None
    assert all((output / record["raw_audio_path"]).exists() for record in records)
    assert "semantic quality: fail" in (output / "report.txt").read_text()
    assert "never pass/fail gates" in (output / "report.txt").read_text()


@pytest.mark.asyncio
async def test_duration_is_raw_telemetry_not_a_request_failure(tmp_path: Path) -> None:
    pcm = sine_pcm(4_000)

    async def speech(_request: web.Request) -> web.Response:
        return web.Response(body=pcm, headers={"Content-Type": "audio/pcm"})

    prompt = make_prompts(1)[0].model_copy(update={"text": "short phrase", "word_count": 2})
    output = tmp_path / "duration-telemetry"
    async with local_server(speech) as base_url:
        summary = await run_rate(
            adapter=make_adapter(base_url),
            prompts=(prompt,),
            dataset_manifest=make_manifest((prompt,)),
            output=output,
            options=RateOptions(
                requested_rps=5,
                arrival=ArrivalPattern.CONSTANT,
                seed=0,
                warmup_s=0,
                duration_s=0.12,
                timeout_s=1,
                max_in_flight=10,
            ),
        )

    assert summary.successful_requests == 1
    assert summary.audio_duration_s.p95 == pytest.approx(4.0)
    record = json.loads((output / "requests.jsonl").read_text())
    assert record["transport_status"] == "complete"
    assert record["success"] is True
    assert record["error_kind"] is None
