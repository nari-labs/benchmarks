"""Versioned persisted contracts used by the benchmark."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """Strict JSON-friendly base model."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Phase(StrEnum):
    WARMUP = "warmup"
    MEASUREMENT = "measurement"


class ArrivalPattern(StrEnum):
    POISSON = "poisson"
    CONSTANT = "constant"


class TargetKind(StrEnum):
    VLLM_OMNI = "vllm-omni"
    SGLANG_OMNI = "sglang-omni"
    VOXSERVE = "voxserve"
    MSTAR = "mstar"
    NARI = "nari"


class TransportKind(StrEnum):
    HTTP = "http"
    WEBSOCKET = "websocket"


class WERStatus(StrEnum):
    DISABLED = "disabled"
    NOT_APPLICABLE = "not_applicable"
    EVALUATED = "evaluated"
    MISSING = "missing"


class StreamKind(StrEnum):
    RAW_PCM = "raw_pcm"
    WAV = "wav"


class PcmFormat(ContractModel):
    sample_rate_hz: int = Field(default=24_000, gt=0)
    channels: Literal[1] = 1
    sample_width_bytes: Literal[2] = 2

    @property
    def bytes_per_frame(self) -> int:
        return self.channels * self.sample_width_bytes


class AudioSuccessCriteria(ContractModel):
    largest_stall_ms: float = Field(default=500.0, ge=0)
    total_stall_ms: float = Field(default=1_000.0, ge=0)

    @model_validator(mode="after")
    def validate_stall_limits(self) -> Self:
        if self.largest_stall_ms > self.total_stall_ms:
            raise ValueError("largest_stall_ms cannot exceed total_stall_ms")
        return self

    def contract(self) -> dict[str, float]:
        return {
            "largest_stall_ms": self.largest_stall_ms,
            "total_stall_ms": self.total_stall_ms,
        }


class TransportStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class SemanticQualityStatus(StrEnum):
    UNCHECKED = "unchecked"
    PASS = "pass"
    FAIL = "fail"


class RequestSuccessCriteria(ContractModel):
    """Versioned capacity, continuity, and semantic measurement contract."""

    capacity: dict[str, bool] = Field(
        default_factory=lambda: {
            "require_successful_transport": True,
            "require_protocol_valid_pcm": True,
            "require_complete": True,
            "require_nonempty": True,
            "require_frame_aligned": True,
        }
    )
    continuity: AudioSuccessCriteria = Field(default_factory=AudioSuccessCriteria)
    semantic: dict[str, str] = Field(
        default_factory=lambda: {
            "audibility": "inaudible_fails_otherwise_unchecked",
            "audio_duration": "raw_telemetry_only",
        }
    )


class Prompt(ContractModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: str = Field(default="en", min_length=1)
    word_count: int = Field(ge=1)
    source_index: int = Field(ge=0)


class DatasetManifest(ContractModel):
    schema_version: Literal[1] = 1
    kind: Literal["jsonl_snapshot"]
    source_path: str
    source_revision: str | None = None
    source_files: tuple[str, ...] = ()
    source_sha256: str
    prompt_count: int = Field(gt=0)
    pool_sha256: str


class ArrivalSlot(ContractModel):
    schema_version: Literal[1] = 1
    phase: Phase
    phase_index: int = Field(ge=0)
    scheduled_elapsed_ns: int = Field(ge=0)
    prompt_id: str
    prompt_word_count: int = Field(ge=1)
    dropped: bool = False
    drop_reason: str | None = None


class UnderrunRecord(ContractModel):
    started_elapsed_ns: int = Field(ge=0)
    recovered_elapsed_ns: int = Field(ge=0)
    stalled_ns: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.recovered_elapsed_ns - self.started_elapsed_ns != self.stalled_ns:
            raise ValueError("stalled_ns must match the underrun interval")
        return self


class PlaybackRecord(ContractModel):
    complete: bool
    continuous: bool
    first_playable_elapsed_ns: int | None = Field(default=None, ge=0)
    first_audible_elapsed_ns: int | None = Field(default=None, ge=0)
    audible_offset_ns: int | None = Field(default=None, ge=0)
    playback_end_elapsed_ns: int | None = Field(default=None, ge=0)
    audio_duration_ns: int = Field(ge=0)
    underruns: tuple[UnderrunRecord, ...] = ()
    total_stalled_ns: int = Field(ge=0)
    largest_stall_ns: int = Field(ge=0)


class InputChunkRecord(ContractModel):
    sequence: int = Field(ge=0)
    text: str = Field(min_length=1)
    sent_elapsed_ns: int = Field(ge=0)
    acknowledged_elapsed_ns: int = Field(ge=0)


class InputTimingRecord(ContractModel):
    connect_started_elapsed_ns: int = Field(ge=0)
    session_created_elapsed_ns: int = Field(ge=0)
    start_sent_elapsed_ns: int = Field(ge=0)
    configured_elapsed_ns: int = Field(ge=0)
    first_text_sent_elapsed_ns: int = Field(ge=0)
    last_text_sent_elapsed_ns: int = Field(ge=0)
    end_sent_elapsed_ns: int = Field(ge=0)
    end_acknowledged_elapsed_ns: int = Field(ge=0)
    chunks: tuple[InputChunkRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_timeline(self) -> Self:
        if self.first_text_sent_elapsed_ns != self.chunks[0].sent_elapsed_ns:
            raise ValueError("first_text_sent_elapsed_ns must match the first chunk")
        if self.last_text_sent_elapsed_ns != self.chunks[-1].sent_elapsed_ns:
            raise ValueError("last_text_sent_elapsed_ns must match the last chunk")
        if self.end_sent_elapsed_ns < self.last_text_sent_elapsed_ns:
            raise ValueError("input end cannot be sent before the last text chunk")
        return self


class RequestRecord(ContractModel):
    schema_version: Literal[1] = 1
    run_id: str
    request_id: str
    phase: Phase
    phase_index: int = Field(ge=0)
    prompt_id: str
    prompt_word_count: int = Field(ge=1)
    scheduled_elapsed_ns: int = Field(ge=0)
    task_started_elapsed_ns: int = Field(ge=0)
    dispatch_elapsed_ns: int | None = Field(default=None, ge=0)
    response_headers_elapsed_ns: int | None = Field(default=None, ge=0)
    first_body_elapsed_ns: int | None = Field(default=None, ge=0)
    completed_elapsed_ns: int = Field(ge=0)
    status_code: int | None = None
    success: bool
    error_kind: str | None = None
    error_message: str | None = None
    content_type: str | None = None
    connection_reused: bool | None = None
    response_http_version: str | None = None
    output_bytes: int = Field(ge=0)
    measurement_window_audio_ns: int = Field(ge=0)
    raw_sha256: str | None = None
    pcm_sha256: str | None = None
    wav_sha256: str | None = None
    raw_audio_path: str | None = None
    wav_audio_path: str | None = None
    pcm_format: PcmFormat | None = None
    playback: PlaybackRecord
    request_body: dict[str, Any]
    input_timing: InputTimingRecord | None = None
    transport_status: TransportStatus = TransportStatus.INCOMPLETE
    audio_success: bool | None = None
    wer_status: WERStatus = WERStatus.DISABLED
    wer_percentage: float | None = Field(default=None, ge=0)
    wer_record_path: str | None = None


class DeepgramModelInfo(ContractModel):
    uuid: str = Field(min_length=1)
    name: str | None = None
    version: str | None = None
    arch: str | None = None


class WERRecord(ContractModel):
    schema_version: Literal[1] = 1
    run_id: str
    request_id: str
    prompt_id: str
    status: Literal[WERStatus.EVALUATED, WERStatus.MISSING]
    provider: Literal["deepgram"] = "deepgram"
    endpoint: str
    requested_model: Literal["nova-3"] = "nova-3"
    language: Literal["en"] = "en"
    punctuate: Literal[True] = True
    smart_format: Literal[True] = True
    normalization: Literal["whisper_normalizer.EnglishTextNormalizer"] = (
        "whisper_normalizer.EnglishTextNormalizer"
    )
    normalization_version: Literal["2"] = "2"
    audio_path: str
    audio_sha256: str
    attempts: int = Field(ge=0)
    response_http_status: int | None = None
    response_path: str | None = None
    response_sha256: str | None = None
    deepgram_request_id: str | None = None
    resolved_models: tuple[DeepgramModelInfo, ...] = ()
    reference: str
    transcript: str | None = None
    normalized_reference: str | None = None
    normalized_hypothesis: str | None = None
    reference_words: int | None = Field(default=None, ge=0)
    substitutions: int | None = Field(default=None, ge=0)
    deletions: int | None = Field(default=None, ge=0)
    insertions: int | None = Field(default=None, ge=0)
    wer_ratio: float | None = Field(default=None, ge=0)
    wer_percentage: float | None = Field(default=None, ge=0)
    error_kind: str | None = None
    error_message: str | None = None

class EventRecord(ContractModel):
    schema_version: Literal[1] = 1
    run_id: str
    request_id: str | None = None
    phase: Phase | None = None
    phase_index: int | None = Field(default=None, ge=0)
    type: str
    elapsed_ns: int = Field(ge=0)
    data: dict[str, Any] = Field(default_factory=dict)


class Distribution(ContractModel):
    samples: int = Field(ge=0)
    missing: int = Field(ge=0)
    minimum: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    maximum: float | None = None
    mean: float | None = None


class SetupMetrics(ContractModel):
    dns_ms: float | None = Field(default=None, ge=0)
    tcp_ms: float | None = Field(default=None, ge=0)
    tls_ms: float | None = Field(default=None, ge=0)
    healthcheck_ms: float | None = Field(default=None, ge=0)
    healthcheck_status: int | None = None


class RunSummary(ContractModel):
    schema_version: Literal[1] = 1
    run_id: str
    target: TargetKind
    transport: TransportKind = TransportKind.HTTP
    requested_rps: float = Field(gt=0)
    measurement_duration_s: float = Field(gt=0)
    scheduled_requests: int = Field(ge=0)
    started_requests: int = Field(ge=0)
    dispatched_requests: int = Field(ge=0)
    late_dispatched_requests: int = Field(ge=0)
    predispatch_failed_requests: int = Field(ge=0)
    dropped_requests: int = Field(ge=0)
    successful_requests: int = Field(ge=0)
    actual_rps: float = Field(ge=0)
    success_fraction: float | None = Field(default=None, ge=0, le=1)
    underrun_requests: int = Field(ge=0)
    underrun_request_fraction: float | None = Field(default=None, ge=0, le=1)
    underrun_count: int = Field(ge=0)
    total_stalled_ms: float = Field(ge=0)
    largest_stall_ms: float = Field(ge=0)
    scheduling_lag_ms: Distribution
    ttfb_ms: Distribution
    first_playable_ms: Distribution
    audible_ttfa_ms: Distribution
    leading_silence_ms: Distribution
    end_to_end_ms: Distribution
    realtime_factor: Distribution
    prompt_word_count: Distribution
    audio_duration_s: Distribution
    generated_audio_s: float = Field(ge=0)
    received_audio_xrt: float = Field(ge=0)
    peak_in_flight: int = Field(ge=0)
    setup: SetupMetrics
    pcm_complete_requests: int = Field(ge=0)
    audible_requests: int = Field(ge=0)
    semantic_quality: SemanticQualityStatus = SemanticQualityStatus.UNCHECKED
    audio_successful_requests: int = Field(ge=0)
    wer_enabled: bool = False
    wer_candidate_requests: int = Field(default=0, ge=0)
    wer_evaluated_requests: int = Field(default=0, ge=0)
    wer_coverage_fraction: float | None = Field(default=None, ge=0, le=1)
    wer_percentage: Distribution | None = None
    websocket_setup_ms: Distribution | None = None
    input_ingress_ms: Distribution | None = None
    final_text_to_first_playable_ms: Distribution | None = None
    final_text_to_audible_ms: Distribution | None = None
    final_text_to_audio_completion_ms: Distribution | None = None
    audio_before_input_complete_fraction: float | None = Field(default=None, ge=0, le=1)
    reference_ttfa_p95_ms: float = 50.0
    reference_underruns: int = 0

class RunConfiguration(ContractModel):
    schema_version: Literal[1] = 1
    run_id: str
    package_version: str
    benchmark_git_revision: str | None = None
    benchmark_git_dirty: bool | None = None
    dependency_lock_sha256: str | None = None
    target: TargetKind
    transport: TransportKind = TransportKind.HTTP
    base_url: str
    model: str
    voice: str
    language: str
    requested_rps: float
    arrival: ArrivalPattern
    seed: int
    warmup_s: float
    duration_s: float
    timeout_s: float
    max_in_flight: int
    sample_rate_hz: int
    health_path: str
    request_params: dict[str, Any]
    request_body_template: dict[str, Any]
    request_body_example: dict[str, Any]
    request_records_path: str
    wer_records_path: str | None = None
    measurement_contract: dict[str, Any]
    dataset: DatasetManifest
    started_at_utc: str
    client: dict[str, Any]
    setup: SetupMetrics
