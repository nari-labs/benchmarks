"""Asyncio open-loop execution for one requested arrival rate."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any
from uuid import uuid4

import aiohttp

from tts_bench import __version__
from tts_bench.artifacts import (
    ArtifactStore,
    initialize_result_directory,
    write_dataset_snapshot,
    write_json,
    write_jsonl,
    write_jsonl_models,
    write_text,
)
from tts_bench.audio import (
    AudibleOnsetDetector,
    AudioChunkObservation,
    AudioProtocolError,
    PcmStreamDecoder,
    evaluate_playback,
    pcm_duration_ns,
    pcm_to_wav,
)
from tts_bench.dataset import PROMPT_ORDER_VERSION, PromptPool
from tts_bench.metrics import derive_summary
from tts_bench.models import (
    ArrivalPattern,
    ArrivalSlot,
    AudioSuccessCriteria,
    DatasetManifest,
    EventRecord,
    InputChunkRecord,
    InputTimingRecord,
    PcmFormat,
    Phase,
    PlaybackRecord,
    Prompt,
    RequestRecord,
    RequestSuccessCriteria,
    RunConfiguration,
    RunSummary,
    TransportKind,
    TransportStatus,
    WERStatus,
)
from tts_bench.reporting import render_run_report
from tts_bench.scheduling import (
    ARRIVAL_SCHEDULE_VERSION,
    NANOSECONDS_PER_SECOND,
    iter_phase_schedule,
)
from tts_bench.targets import (
    WEBSOCKET_INPUT_CHUNK_INTERVAL_S,
    WEBSOCKET_PROTOCOL,
    TargetAdapter,
    TargetConfigurationError,
    preflight_target,
)
from tts_bench.wer import WER_RECORDS_PATH, DeepgramWEROptions, evaluate_measurement_wer


@dataclass(frozen=True)
class RateOptions:
    requested_rps: float
    arrival: ArrivalPattern
    seed: int
    warmup_s: float
    duration_s: float
    timeout_s: float
    max_in_flight: int
    audio_success_criteria: AudioSuccessCriteria = field(default_factory=AudioSuccessCriteria)
    wer_options: DeepgramWEROptions | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.requested_rps) or self.requested_rps <= 0:
            raise ValueError("requested_rps must be finite and positive")
        if not math.isfinite(self.warmup_s) or self.warmup_s < 0:
            raise ValueError("warmup_s must be finite and non-negative")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")


@dataclass
class _TraceState:
    anchor_ns: int
    dispatch_elapsed_ns: int | None = None
    connection_reused: bool | None = None

    def elapsed_ns(self) -> int:
        return max(0, perf_counter_ns() - self.anchor_ns)


def _benchmark_provenance() -> tuple[str | None, bool | None, str | None]:
    """Resolve source revision, dirty state, and dependency lock identity when available."""
    project_root = Path(__file__).resolve().parents[2]
    lockfile = project_root / "uv.lock"
    lockfile_sha256 = (
        hashlib.sha256(lockfile.read_bytes()).hexdigest() if lockfile.is_file() else None
    )
    try:
        revision = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                "src",
                "pyproject.toml",
                "uv.lock",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None, lockfile_sha256
    return revision or None, bool(status.strip()), lockfile_sha256


async def run_rate(
    *,
    adapter: TargetAdapter,
    prompts: tuple[Prompt, ...],
    dataset_manifest: DatasetManifest,
    output: Path,
    options: RateOptions,
) -> RunSummary:
    """Execute one independently warm-started rate and persist all evidence."""
    if not prompts:
        raise ValueError("prompt pool cannot be empty")
    if dataset_manifest.prompt_count != len(prompts):
        raise ValueError("dataset manifest prompt count does not match the loaded prompt pool")
    initialize_result_directory(output)
    write_json(output / "status.json", {"schema_version": 1, "state": "initializing"})
    write_dataset_snapshot(output, prompts, dataset_manifest)
    run_id = uuid4().hex
    store = ArtifactStore(output)
    store.start()
    try:
        setup = await preflight_target(adapter, timeout_s=options.timeout_s)
        warmup_ns = round(options.warmup_s * NANOSECONDS_PER_SECOND)
        duration_ns = round(options.duration_s * NANOSECONDS_PER_SECOND)
        pool = PromptPool(prompts, seed=options.seed)
        slots = _build_slots(
            pool=pool,
            warmup_ns=warmup_ns,
            duration_ns=duration_ns,
            options=options,
        )
        write_jsonl_models(output / "arrivals.jsonl", slots)

        request_success_contract = RequestSuccessCriteria(
            continuity=options.audio_success_criteria
        ).model_dump(mode="json")
        wer_options = options.wer_options
        wer_contract = wer_options.contract() if wer_options is not None else None
        benchmark_revision, benchmark_dirty, dependency_lock_sha256 = _benchmark_provenance()

        configuration = RunConfiguration(
            run_id=run_id,
            package_version=__version__,
            benchmark_git_revision=benchmark_revision,
            benchmark_git_dirty=benchmark_dirty,
            dependency_lock_sha256=dependency_lock_sha256,
            target=adapter.kind,
            transport=adapter.transport,
            base_url=adapter.base_url,
            model=adapter.model,
            voice=adapter.voice,
            language=adapter.language,
            requested_rps=options.requested_rps,
            arrival=options.arrival,
            seed=options.seed,
            warmup_s=options.warmup_s,
            duration_s=options.duration_s,
            timeout_s=options.timeout_s,
            max_in_flight=options.max_in_flight,
            sample_rate_hz=adapter.expected_format.sample_rate_hz,
            health_path=adapter.health_path,
            request_params=adapter.request_params,
            request_body_template=adapter.request_template(),
            request_body_example=adapter.build_request(pool.at(Phase.MEASUREMENT, 0).text),
            request_records_path="requests.jsonl",
            wer_records_path=WER_RECORDS_PATH if wer_options is not None else None,
            measurement_contract={
                "clock": "time.perf_counter_ns",
                "request_start": (
                    "first input_text.append frame sent"
                    if adapter.transport is TransportKind.WEBSOCKET
                    else "first JSON body bytes sent"
                ),
                "websocket_input": (
                    {
                        "protocol": WEBSOCKET_PROTOCOL,
                        "chunk_unit": "word",
                        "chunk_size": 1,
                        "chunk_interval_ms": 10,
                        "acknowledgement": "required before the next chunk",
                        "cadence": "absolute from the first text chunk",
                    }
                    if adapter.transport is TransportKind.WEBSOCKET
                    else None
                ),
                "arrival_schedule_version": ARRIVAL_SCHEDULE_VERSION,
                "prompt_order_version": PROMPT_ORDER_VERSION,
                "audible_detector": {
                    "version": AudibleOnsetDetector.version,
                    "frame_ms": AudibleOnsetDetector.frame_ms,
                    "hop_ms": AudibleOnsetDetector.hop_ms,
                    "threshold_dbfs": AudibleOnsetDetector.threshold_dbfs,
                    "consecutive_active_frames": (AudibleOnsetDetector.consecutive_active_frames),
                    "dc_removal": "per-frame mean subtraction",
                },
                "playback_startup_buffer_ms": 0,
                "request_success": request_success_contract,
                "wer_evaluator": wer_contract,
                "percentile_method": "nearest-rank",
            },
            dataset=dataset_manifest,
            started_at_utc=datetime.now(UTC).isoformat(),
            client={
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": sys.version,
            },
            setup=setup,
        )
        write_json(output / "run.json", configuration)
        write_json(output / "status.json", {"schema_version": 1, "state": "running"})

        records, dropped, peak = await _run_schedule(
            run_id=run_id,
            adapter=adapter,
            slots=slots,
            output=output,
            store=store,
            warmup_ns=warmup_ns,
            duration_ns=duration_ns,
            timeout_s=options.timeout_s,
            max_in_flight=options.max_in_flight,
            audio_success_criteria=options.audio_success_criteria,
        )
        await store.close()
        completed_slots = tuple(
            slot.model_copy(
                update={
                    "dropped": (slot.phase, slot.phase_index) in dropped,
                    "drop_reason": (
                        "max_in_flight" if (slot.phase, slot.phase_index) in dropped else None
                    ),
                }
            )
            for slot in slots
        )
        write_jsonl_models(output / "arrivals.jsonl", completed_slots)
        measurement_records = _measurement_records(records)
        if wer_options is not None:
            write_json(
                output / "status.json",
                {"schema_version": 1, "state": "evaluating_wer"},
            )
            records, _ = await evaluate_measurement_wer(
                output=output,
                run_id=run_id,
                records=records,
                candidate_request_ids=frozenset(
                    record.request_id
                    for record in measurement_records
                    if _semantic_evaluation_candidate(record)
                ),
                prompts=prompts,
                options=wer_options,
            )
        else:
            records = [
                record.model_copy(
                    update={
                        "audio_success": record.success,
                        "wer_status": WERStatus.DISABLED,
                    }
                )
                for record in records
            ]
        write_jsonl_models(output / "requests.jsonl", records)
        measurement_records = _measurement_records(records)
        write_jsonl(
            output / "measurement-prompt-sequence.jsonl",
            (
                {
                    "phase_index": record.phase_index,
                    "request_id": record.request_id,
                    "prompt_id": record.prompt_id,
                    "prompt_word_count": record.prompt_word_count,
                    "dispatch_elapsed_ns": record.dispatch_elapsed_ns,
                }
                for record in measurement_records
            ),
        )
        measurement_slots = sum(slot.phase is Phase.MEASUREMENT for slot in completed_slots)
        measurement_drops = sum(
            slot.phase is Phase.MEASUREMENT and slot.dropped for slot in completed_slots
        )
        summary = derive_summary(
            run_id=run_id,
            target=adapter.kind,
            transport=adapter.transport,
            requested_rps=options.requested_rps,
            warmup_ns=warmup_ns,
            duration_ns=duration_ns,
            scheduled_requests=measurement_slots,
            dropped_requests=measurement_drops,
            records=records,
            peak_in_flight=peak,
            setup=setup,
            wer_enabled=wer_options is not None,
        )
        write_json(output / "summary.json", summary)
        write_text(output / "report.txt", render_run_report(summary))
        write_json(
            output / "status.json",
            {
                "schema_version": 1,
                "state": "complete",
                "completed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        return summary
    except BaseException as error:
        with contextlib.suppress(BaseException):
            await store.close()
        write_json(
            output / "status.json",
            {
                "schema_version": 1,
                "state": "incomplete",
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


def _measurement_records(records: list[RequestRecord]) -> list[RequestRecord]:
    return sorted(
        (record for record in records if record.phase is Phase.MEASUREMENT),
        key=lambda record: record.phase_index,
    )


def _semantic_evaluation_candidate(record: RequestRecord) -> bool:
    """Return whether a request has complete audible PCM suitable for ASR."""
    return (
        record.transport_status is TransportStatus.COMPLETE
        and record.wav_audio_path is not None
        and record.wav_sha256 is not None
        and record.playback.audible_offset_ns is not None
    )


def _build_slots(
    *,
    pool: PromptPool,
    warmup_ns: int,
    duration_ns: int,
    options: RateOptions,
) -> tuple[ArrivalSlot, ...]:
    values: list[ArrivalSlot] = []
    phases = (
        (Phase.WARMUP, 0, warmup_ns),
        (Phase.MEASUREMENT, warmup_ns, duration_ns),
    )
    for phase, start_ns, phase_duration_ns in phases:
        if phase_duration_ns == 0:
            continue
        for index, scheduled_ns in iter_phase_schedule(
            phase=phase,
            phase_start_ns=start_ns,
            duration_ns=phase_duration_ns,
            rps=options.requested_rps,
            pattern=options.arrival,
            seed=options.seed,
        ):
            prompt = pool.at(phase, index)
            values.append(
                ArrivalSlot(
                    phase=phase,
                    phase_index=index,
                    scheduled_elapsed_ns=scheduled_ns,
                    prompt_id=prompt.id,
                    prompt_word_count=prompt.word_count,
                )
            )
    return tuple(values)


async def _run_schedule(
    *,
    run_id: str,
    adapter: TargetAdapter,
    slots: tuple[ArrivalSlot, ...],
    output: Path,
    store: ArtifactStore,
    warmup_ns: int,
    duration_ns: int,
    timeout_s: float,
    max_in_flight: int,
    audio_success_criteria: AudioSuccessCriteria,
) -> tuple[list[RequestRecord], set[tuple[Phase, int]], int]:
    prompt_by_id = _load_snapshot_prompts(output / "dataset.jsonl")
    trace_config = _trace_config()
    connector = aiohttp.TCPConnector(limit=max_in_flight, force_close=False)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    records: list[RequestRecord] = []
    # "In flight" counts client request tasks from creation until completion, including
    # pre-dispatch/setup work, and its peak spans both warm-up and measurement phases.
    active: set[asyncio.Task[RequestRecord]] = set()
    task_errors: list[BaseException] = []
    dropped_slots: set[tuple[Phase, int]] = set()
    peak_in_flight = 0
    anchor_ns = perf_counter_ns()
    measurement_end = warmup_ns + duration_ns

    def on_done(task: asyncio.Task[RequestRecord]) -> None:
        active.discard(task)
        try:
            records.append(task.result())
        except BaseException as error:
            task_errors.append(error)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        trace_configs=[trace_config],
        headers={"Accept-Encoding": "identity"},
    ) as session:
        try:
            for slot in slots:
                await _sleep_until(anchor_ns + slot.scheduled_elapsed_ns)
                if task_errors:
                    raise RuntimeError(f"request task failed: {task_errors[0]}") from task_errors[0]
                if len(active) >= max_in_flight:
                    dropped_slots.add((slot.phase, slot.phase_index))
                    elapsed = max(0, perf_counter_ns() - anchor_ns)
                    store.enqueue_events(
                        (
                            EventRecord(
                                run_id=run_id,
                                phase=slot.phase,
                                phase_index=slot.phase_index,
                                type="arrival.dropped",
                                elapsed_ns=elapsed,
                                data={
                                    "scheduled_elapsed_ns": slot.scheduled_elapsed_ns,
                                    "reason": "max_in_flight",
                                },
                            ),
                        )
                    )
                    continue
                prompt = prompt_by_id[slot.prompt_id]
                task = asyncio.create_task(
                    _execute_request(
                        run_id=run_id,
                        adapter=adapter,
                        session=session,
                        prompt=prompt,
                        slot=slot,
                        anchor_ns=anchor_ns,
                        measurement_start_ns=warmup_ns,
                        measurement_end_ns=measurement_end,
                        store=store,
                        audio_success_criteria=audio_success_criteria,
                        timeout_s=timeout_s,
                    ),
                    name=f"request-{slot.phase.value}-{slot.phase_index}",
                )
                active.add(task)
                task.add_done_callback(on_done)
                peak_in_flight = max(peak_in_flight, len(active))
            if active:
                await asyncio.gather(*tuple(active), return_exceptions=True)
            if task_errors:
                raise RuntimeError(f"request task failed: {task_errors[0]}") from task_errors[0]
        except BaseException:
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*tuple(active), return_exceptions=True)
            raise
    return records, dropped_slots, peak_in_flight


async def _sleep_until(deadline_ns: int) -> None:
    while True:
        remaining_ns = deadline_ns - perf_counter_ns()
        if remaining_ns <= 0:
            return
        await asyncio.sleep(remaining_ns / NANOSECONDS_PER_SECOND)


async def _execute_request(
    *,
    run_id: str,
    adapter: TargetAdapter,
    session: aiohttp.ClientSession,
    prompt: Prompt,
    slot: ArrivalSlot,
    anchor_ns: int,
    measurement_start_ns: int,
    measurement_end_ns: int,
    store: ArtifactStore,
    audio_success_criteria: AudioSuccessCriteria,
    timeout_s: float,
) -> RequestRecord:
    request_id = uuid4().hex
    trace = _TraceState(anchor_ns=anchor_ns)
    events: list[EventRecord] = []

    def elapsed_ns() -> int:
        return max(0, perf_counter_ns() - anchor_ns)

    task_started = elapsed_ns()
    events.append(
        EventRecord(
            run_id=run_id,
            request_id=request_id,
            phase=slot.phase,
            phase_index=slot.phase_index,
            type="request.task_started",
            elapsed_ns=task_started,
            data={"scheduled_elapsed_ns": slot.scheduled_elapsed_ns, "prompt_id": prompt.id},
        )
    )
    request_body = adapter.build_request(prompt.text)
    raw = bytearray()
    pcm = bytearray()
    chunks: list[AudioChunkObservation] = []
    decoder = PcmStreamDecoder(
        adapter.stream_kind,
        adapter.expected_format,
    )
    detector = AudibleOnsetDetector(adapter.expected_format)
    measurement_window_audio_ns = 0
    response_headers_at: int | None = None
    first_body_at: int | None = None
    status_code: int | None = None
    content_type: str | None = None
    http_version: str | None = None
    success = False
    error_kind: str | None = None
    error_message: str | None = None
    complete_audio = False
    websocket_protocol_complete = False
    input_timing: InputTimingRecord | None = None
    completed_override: int | None = None

    def consume_audio_chunk(raw_chunk: bytes, arrival: int) -> None:
        nonlocal first_body_at
        nonlocal measurement_window_audio_ns

        if not raw_chunk:
            return
        raw.extend(raw_chunk)
        pcm_chunk = decoder.feed(raw_chunk)
        if first_body_at is None:
            first_body_at = arrival
        media_ns = 0
        if pcm_chunk:
            pcm.extend(pcm_chunk)
            detector.feed(pcm_chunk)
            media_ns = pcm_duration_ns(pcm_chunk, adapter.expected_format)
            chunks.append(AudioChunkObservation(arrival, media_ns))
            if measurement_start_ns <= arrival < measurement_end_ns:
                measurement_window_audio_ns += media_ns
        events.append(
            EventRecord(
                run_id=run_id,
                request_id=request_id,
                phase=slot.phase,
                phase_index=slot.phase_index,
                type="response.body_chunk",
                elapsed_ns=arrival,
                data={
                    "raw_bytes": len(raw_chunk),
                    "pcm_bytes": len(pcm_chunk),
                    "media_duration_ns": media_ns,
                    "sha256": hashlib.sha256(raw_chunk).hexdigest(),
                },
            )
        )

    async def consume_audio_response(
        response: aiohttp.ClientResponse,
    ) -> None:
        nonlocal complete_audio
        nonlocal content_type
        nonlocal first_body_at
        nonlocal http_version
        nonlocal measurement_window_audio_ns
        nonlocal response_headers_at
        nonlocal status_code

        response_headers_at = elapsed_ns()
        status_code = response.status
        content_type = response.headers.get("Content-Type")
        version = response.version
        if version is not None:
            http_version = f"HTTP/{version.major}.{version.minor}"
        events.append(
            EventRecord(
                run_id=run_id,
                request_id=request_id,
                phase=slot.phase,
                phase_index=slot.phase_index,
                type="response.headers",
                elapsed_ns=response_headers_at,
                data={"status": response.status, "content_type": content_type},
            )
        )
        if not 200 <= response.status < 300:
            body = await response.read()
            raw.extend(body)
            raise _ObservedFailure(
                "http_status",
                f"HTTP {response.status}: {body[:4096].decode(errors='replace')}",
            )
        adapter.validate_response(response.headers)
        async for raw_chunk in response.content.iter_any():
            consume_audio_chunk(raw_chunk, elapsed_ns())
        decoder.finalize()
        complete_audio = True

    def websocket_json(message: aiohttp.WSMessage) -> dict[str, Any]:
        if message.type is not aiohttp.WSMsgType.TEXT:
            raise _ObservedFailure(
                "websocket_protocol_error",
                f"expected WebSocket JSON event, received {message.type.name}",
            )
        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError) as error:
            raise _ObservedFailure(
                "websocket_protocol_error", "received invalid WebSocket JSON"
            ) from error
        if not isinstance(value, dict):
            raise _ObservedFailure(
                "websocket_protocol_error", "WebSocket event must be a JSON object"
            )
        if value.get("type") == "error":
            details = value.get("error")
            raise _ObservedFailure("websocket_server_error", str(details or value))
        return value

    async def receive_initial_websocket_event(
        websocket: aiohttp.ClientWebSocketResponse,
        expected_type: str,
    ) -> tuple[dict[str, Any], int]:
        message = await websocket.receive()
        received_at = elapsed_ns()
        if message.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
            aiohttp.WSMsgType.ERROR,
        }:
            raise _ObservedFailure(
                "websocket_protocol_error",
                f"WebSocket closed before {expected_type}",
            )
        payload = websocket_json(message)
        if payload.get("type") != expected_type:
            raise _ObservedFailure(
                "websocket_protocol_error",
                f"expected {expected_type}, received {payload.get('type')!r}",
            )
        events.append(
            EventRecord(
                run_id=run_id,
                request_id=request_id,
                phase=slot.phase,
                phase_index=slot.phase_index,
                type=f"websocket.{expected_type}",
                elapsed_ns=received_at,
                data=payload,
            )
        )
        return payload, received_at

    async def await_expected_ack(
        ack_queue: asyncio.Queue[tuple[str, int, int]],
        receiver: asyncio.Task[None],
        *,
        event: str,
        sequence: int,
    ) -> int:
        ack_task = asyncio.create_task(ack_queue.get())
        done, _pending = await asyncio.wait(
            {ack_task, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        if ack_task in done:
            observed_event, observed_sequence, acknowledged_at = ack_task.result()
        else:
            try:
                observed_event, observed_sequence, acknowledged_at = ack_queue.get_nowait()
            except asyncio.QueueEmpty:
                ack_task.cancel()
                await asyncio.gather(ack_task, return_exceptions=True)
                await receiver
                raise _ObservedFailure(
                    "websocket_protocol_error",
                    f"WebSocket ended before {event} ACK {sequence}",
                ) from None
            ack_task.cancel()
            await asyncio.gather(ack_task, return_exceptions=True)
        if (observed_event, observed_sequence) != (event, sequence):
            raise _ObservedFailure(
                "websocket_protocol_error",
                f"expected {event} ACK {sequence}, received "
                f"{observed_event} ACK {observed_sequence}",
            )
        return acknowledged_at

    async def execute_websocket() -> None:
        nonlocal complete_audio
        nonlocal completed_override
        nonlocal content_type
        nonlocal http_version
        nonlocal input_timing
        nonlocal response_headers_at
        nonlocal status_code
        nonlocal websocket_protocol_complete

        connect_started_at = elapsed_ns()
        receiver: asyncio.Task[None] | None = None
        websocket: aiohttp.ClientWebSocketResponse | None = None
        server_request_id: str | None = None
        try:
            async with asyncio.timeout(timeout_s):
                async with session.ws_connect(
                    adapter.websocket_url,
                    protocols=(WEBSOCKET_PROTOCOL,),
                    compress=0,
                ) as websocket:
                    trace.connection_reused = False
                    response_headers_at = elapsed_ns()
                    status_code = 101
                    content_type = "audio/pcm"
                    events.append(
                        EventRecord(
                            run_id=run_id,
                            request_id=request_id,
                            phase=slot.phase,
                            phase_index=slot.phase_index,
                            type="response.headers",
                            elapsed_ns=response_headers_at,
                            data={"status": status_code, "content_type": content_type},
                        )
                    )
                    if websocket.protocol != WEBSOCKET_PROTOCOL:
                        raise _ObservedFailure(
                            "websocket_protocol_error",
                            f"server did not negotiate {WEBSOCKET_PROTOCOL}",
                        )
                    session_created, session_created_at = await receive_initial_websocket_event(
                        websocket, "session.created"
                    )
                    if session_created.get("protocol") != WEBSOCKET_PROTOCOL:
                        raise _ObservedFailure(
                            "websocket_protocol_error", "session.created has an invalid protocol"
                        )
                    adapter.validate_websocket_audio(session_created.get("audio"))

                    start_sent_at = elapsed_ns()
                    await websocket.send_json(adapter.build_websocket_start())
                    events.append(
                        EventRecord(
                            run_id=run_id,
                            request_id=request_id,
                            phase=slot.phase,
                            phase_index=slot.phase_index,
                            type="input.start.sent",
                            elapsed_ns=start_sent_at,
                            data=adapter.build_websocket_start(),
                        )
                    )
                    _configured, configured_at = await receive_initial_websocket_event(
                        websocket, "request.configured"
                    )

                    ack_queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
                    response_started = False
                    received_audio_chunks = 0

                    async def receive_output() -> None:
                        nonlocal complete_audio
                        nonlocal completed_override
                        nonlocal received_audio_chunks
                        nonlocal response_started
                        nonlocal server_request_id
                        nonlocal websocket_protocol_complete

                        while True:
                            message = await websocket.receive()
                            received_at = elapsed_ns()
                            if message.type is aiohttp.WSMsgType.BINARY:
                                if not response_started:
                                    raise _ObservedFailure(
                                        "websocket_protocol_error",
                                        "audio arrived before response.started",
                                    )
                                received_audio_chunks += 1
                                consume_audio_chunk(bytes(message.data), received_at)
                                continue
                            if message.type is aiohttp.WSMsgType.TEXT:
                                payload = websocket_json(message)
                                event_type = payload.get("type")
                                events.append(
                                    EventRecord(
                                        run_id=run_id,
                                        request_id=request_id,
                                        phase=slot.phase,
                                        phase_index=slot.phase_index,
                                        type=f"websocket.{event_type}",
                                        elapsed_ns=received_at,
                                        data=payload,
                                    )
                                )
                                if event_type == "input_text.ack":
                                    ack_event = payload.get("event")
                                    ack_sequence = payload.get("sequence")
                                    if not isinstance(ack_event, str) or not isinstance(
                                        ack_sequence, int
                                    ):
                                        raise _ObservedFailure(
                                            "websocket_protocol_error", "invalid input text ACK"
                                        )
                                    await ack_queue.put((ack_event, ack_sequence, received_at))
                                elif event_type == "response.started":
                                    if response_started:
                                        raise _ObservedFailure(
                                            "websocket_protocol_error",
                                            "received duplicate response.started",
                                        )
                                    value = payload.get("request_id")
                                    if not isinstance(value, str) or not value:
                                        raise _ObservedFailure(
                                            "websocket_protocol_error",
                                            "response.started has no request_id",
                                        )
                                    adapter.validate_websocket_audio(payload.get("audio"))
                                    server_request_id = value
                                    response_started = True
                                elif event_type == "response.done":
                                    if not response_started:
                                        raise _ObservedFailure(
                                            "websocket_protocol_error",
                                            "response.done arrived before response.started",
                                        )
                                    if payload.get("request_id") != server_request_id:
                                        raise _ObservedFailure(
                                            "websocket_protocol_error",
                                            "response.done request_id does not match",
                                        )
                                    if payload.get("stop_reason") != "stop":
                                        raise _ObservedFailure(
                                            "websocket_protocol_error",
                                            f"unexpected stop reason {payload.get('stop_reason')!r}",
                                        )
                                    if payload.get("audio_chunks") != received_audio_chunks:
                                        raise _ObservedFailure(
                                            "websocket_protocol_error",
                                            "response.done audio chunk count does not match",
                                        )
                                    decoder.finalize()
                                    complete_audio = True
                                    websocket_protocol_complete = True
                                    completed_override = received_at
                                else:
                                    raise _ObservedFailure(
                                        "websocket_protocol_error",
                                        f"unexpected WebSocket event {event_type!r}",
                                    )
                                continue
                            if message.type in {
                                aiohttp.WSMsgType.CLOSE,
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.CLOSING,
                            }:
                                if not websocket_protocol_complete:
                                    raise _ObservedFailure(
                                        "websocket_protocol_error",
                                        "WebSocket closed before response.done",
                                    )
                                if websocket.close_code != 1000:
                                    raise _ObservedFailure(
                                        "websocket_protocol_error",
                                        f"WebSocket closed with code {websocket.close_code}",
                                    )
                                return
                            if message.type is aiohttp.WSMsgType.ERROR:
                                raise _ObservedFailure(
                                    "transport_error",
                                    f"WebSocket receive failed: {websocket.exception()}",
                                )

                    receiver = asyncio.create_task(
                        receive_output(), name=f"websocket-receiver-{request_id}"
                    )
                    chunk_records: list[InputChunkRecord] = []
                    first_text_sent_at: int | None = None
                    interval_ns = round(WEBSOCKET_INPUT_CHUNK_INTERVAL_S * NANOSECONDS_PER_SECOND)
                    for sequence, text_chunk in enumerate(adapter.input_chunks(prompt.text)):
                        if first_text_sent_at is not None:
                            await _sleep_until(
                                anchor_ns + first_text_sent_at + sequence * interval_ns
                            )
                        sent_at = elapsed_ns()
                        if first_text_sent_at is None:
                            first_text_sent_at = sent_at
                            trace.dispatch_elapsed_ns = sent_at
                        await websocket.send_json(
                            {
                                "type": "input_text.append",
                                "sequence": sequence,
                                "text": text_chunk,
                            }
                        )
                        events.append(
                            EventRecord(
                                run_id=run_id,
                                request_id=request_id,
                                phase=slot.phase,
                                phase_index=slot.phase_index,
                                type="input.text_chunk.sent",
                                elapsed_ns=sent_at,
                                data={
                                    "sequence": sequence,
                                    "characters": len(text_chunk),
                                    "sha256": hashlib.sha256(text_chunk.encode()).hexdigest(),
                                },
                            )
                        )
                        acknowledged_at = await await_expected_ack(
                            ack_queue,
                            receiver,
                            event="input_text.append",
                            sequence=sequence,
                        )
                        chunk_records.append(
                            InputChunkRecord(
                                sequence=sequence,
                                text=text_chunk,
                                sent_elapsed_ns=sent_at,
                                acknowledged_elapsed_ns=acknowledged_at,
                            )
                        )

                    assert first_text_sent_at is not None
                    end_sequence = len(chunk_records)
                    end_sent_at = elapsed_ns()
                    await websocket.send_json({"type": "input_text.end", "sequence": end_sequence})
                    events.append(
                        EventRecord(
                            run_id=run_id,
                            request_id=request_id,
                            phase=slot.phase,
                            phase_index=slot.phase_index,
                            type="input.end.sent",
                            elapsed_ns=end_sent_at,
                            data={"sequence": end_sequence},
                        )
                    )
                    end_acknowledged_at = await await_expected_ack(
                        ack_queue,
                        receiver,
                        event="input_text.end",
                        sequence=end_sequence,
                    )
                    input_timing = InputTimingRecord(
                        connect_started_elapsed_ns=connect_started_at,
                        session_created_elapsed_ns=session_created_at,
                        start_sent_elapsed_ns=start_sent_at,
                        configured_elapsed_ns=configured_at,
                        first_text_sent_elapsed_ns=first_text_sent_at,
                        last_text_sent_elapsed_ns=chunk_records[-1].sent_elapsed_ns,
                        end_sent_elapsed_ns=end_sent_at,
                        end_acknowledged_elapsed_ns=end_acknowledged_at,
                        chunks=tuple(chunk_records),
                    )
                    await receiver
                    receiver = None
        except BaseException:
            if websocket is not None and not websocket.closed and server_request_id is not None:
                with contextlib.suppress(BaseException):
                    await websocket.send_json({"type": "response.cancel"})
            raise
        finally:
            if receiver is not None and not receiver.done():
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

    try:
        if adapter.transport is TransportKind.WEBSOCKET:
            await execute_websocket()
        else:
            async with session.post(
                adapter.speech_url,
                json=request_body,
                headers={"Accept": "audio/wav, audio/pcm, application/octet-stream"},
                trace_request_ctx=trace,
            ) as response:
                await consume_audio_response(response)
        success = True
    except _ObservedFailure as error:
        error_kind = error.kind
        error_message = str(error)
    except TimeoutError as error:
        error_kind = "timeout"
        error_message = str(error) or "request timed out"
    except aiohttp.ClientError as error:
        error_kind = "transport_error"
        error_message = str(error)
    except (AudioProtocolError, TargetConfigurationError) as error:
        error_kind = "audio_protocol_error"
        error_message = str(error)
    except Exception as error:
        error_kind = "client_error"
        error_message = f"{type(error).__name__}: {error}"

    completed = completed_override if completed_override is not None else elapsed_ns()
    playback = evaluate_playback(
        chunks,
        complete=complete_audio,
        audible_offset_ns=detector.onset_offset_ns,
    )
    if success:
        quality_failure = _audio_success_failure(
            playback,
            criteria=audio_success_criteria,
        )
        if quality_failure is not None:
            success = False
            error_kind, error_message = quality_failure
    raw_hash = hashlib.sha256(raw).hexdigest()
    pcm_hash = hashlib.sha256(pcm).hexdigest() if pcm else None
    phase_dir = slot.phase.value
    raw_path = f"audio/raw/{phase_dir}/{request_id}.response"
    wav_bytes = pcm_to_wav(bytes(pcm), adapter.expected_format) if pcm else None
    wav_hash = hashlib.sha256(wav_bytes).hexdigest() if wav_bytes is not None else None
    wav_path = f"audio/wav/{phase_dir}/{request_id}.wav" if wav_bytes is not None else None
    event_type = "request.completed" if success else "request.failed"
    events.append(
        EventRecord(
            run_id=run_id,
            request_id=request_id,
            phase=slot.phase,
            phase_index=slot.phase_index,
            type=event_type,
            elapsed_ns=completed,
            data={"error_kind": error_kind, "error_message": error_message},
        )
    )
    if trace.dispatch_elapsed_ns is not None:
        events.append(
            EventRecord(
                run_id=run_id,
                request_id=request_id,
                phase=slot.phase,
                phase_index=slot.phase_index,
                type="request.dispatched",
                elapsed_ns=trace.dispatch_elapsed_ns,
                data={"scheduled_elapsed_ns": slot.scheduled_elapsed_ns},
            )
        )
    events.sort(key=lambda event: (event.elapsed_ns, event.type))
    record = RequestRecord(
        run_id=run_id,
        request_id=request_id,
        phase=slot.phase,
        phase_index=slot.phase_index,
        prompt_id=prompt.id,
        prompt_word_count=prompt.word_count,
        scheduled_elapsed_ns=slot.scheduled_elapsed_ns,
        task_started_elapsed_ns=task_started,
        dispatch_elapsed_ns=trace.dispatch_elapsed_ns,
        response_headers_elapsed_ns=response_headers_at,
        first_body_elapsed_ns=first_body_at,
        completed_elapsed_ns=completed,
        status_code=status_code,
        success=success,
        error_kind=error_kind,
        error_message=error_message,
        content_type=content_type,
        connection_reused=trace.connection_reused,
        response_http_version=http_version,
        output_bytes=len(raw),
        measurement_window_audio_ns=measurement_window_audio_ns,
        raw_sha256=raw_hash,
        pcm_sha256=pcm_hash,
        wav_sha256=wav_hash,
        raw_audio_path=raw_path,
        wav_audio_path=wav_path,
        pcm_format=adapter.expected_format if pcm else None,
        playback=playback,
        request_body=request_body,
        input_timing=input_timing,
        transport_status=(
            TransportStatus.COMPLETE
            if _transport_complete(
                status_code=status_code,
                transport=adapter.transport,
                websocket_protocol_complete=websocket_protocol_complete,
                pcm_format=adapter.expected_format if pcm else None,
                pcm_sha256=pcm_hash,
                playback=playback,
            )
            else TransportStatus.INCOMPLETE
        ),
    )
    store.enqueue_request(record, events, bytes(raw), wav_bytes)
    return record


def _transport_complete(
    *,
    status_code: int | None,
    transport: TransportKind,
    websocket_protocol_complete: bool,
    pcm_format: PcmFormat | None,
    pcm_sha256: str | None,
    playback: PlaybackRecord,
) -> bool:
    """Return whether transport produced complete, nonempty, frame-aligned PCM."""
    return (
        (
            (status_code is not None and 200 <= status_code < 300)
            if transport is TransportKind.HTTP
            else status_code == 101 and websocket_protocol_complete
        )
        and pcm_format is not None
        and pcm_sha256 is not None
        and playback.complete
        and playback.audio_duration_ns > 0
    )


def _audio_success_failure(
    playback: PlaybackRecord,
    *,
    criteria: AudioSuccessCriteria,
) -> tuple[str, str] | None:
    largest_stall_ms = playback.largest_stall_ns / 1_000_000
    total_stall_ms = playback.total_stalled_ns / 1_000_000
    if largest_stall_ms > criteria.largest_stall_ms or total_stall_ms > criteria.total_stall_ms:
        return (
            "audio_stall",
            f"audio stalled {largest_stall_ms:.3f}ms at most and {total_stall_ms:.3f}ms total; "
            f"limits are {criteria.largest_stall_ms:.3f}ms and "
            f"{criteria.total_stall_ms:.3f}ms",
        )
    return None


class _ObservedFailure(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _trace_config() -> aiohttp.TraceConfig:
    config = aiohttp.TraceConfig()

    async def request_chunk_sent(
        _session: aiohttp.ClientSession,
        context: Any,
        _params: aiohttp.TraceRequestChunkSentParams,
    ) -> None:
        state = context.trace_request_ctx
        if isinstance(state, _TraceState) and state.dispatch_elapsed_ns is None:
            state.dispatch_elapsed_ns = state.elapsed_ns()

    async def connection_create_start(
        _session: aiohttp.ClientSession,
        context: Any,
        _params: aiohttp.TraceConnectionCreateStartParams,
    ) -> None:
        state = context.trace_request_ctx
        if isinstance(state, _TraceState):
            state.connection_reused = False

    async def connection_reuse(
        _session: aiohttp.ClientSession,
        context: Any,
        _params: aiohttp.TraceConnectionReuseconnParams,
    ) -> None:
        state = context.trace_request_ctx
        if isinstance(state, _TraceState):
            state.connection_reused = True

    config.on_request_chunk_sent.append(request_chunk_sent)
    config.on_connection_create_start.append(connection_create_start)
    config.on_connection_reuseconn.append(connection_reuse)
    return config


def _load_snapshot_prompts(path: Path) -> dict[str, Prompt]:
    import json

    prompts: dict[str, Prompt] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            prompt = Prompt.model_validate(value)
            prompts[prompt.id] = prompt
    return prompts
