"""Human-readable rendering for persisted benchmark summaries."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from tts_bench.models import Distribution, RunSummary


class ReportError(ValueError):
    """A result directory does not contain a supported complete summary."""


def render_run_report(summary: RunSummary) -> str:
    """Render a single-rate report without applying pass/fail policy."""
    wer_lines: list[str] = []
    if summary.wer_enabled:
        assert summary.wer_percentage is not None
        wer_lines = [
            "",
            "Output quality",
            "  WER evaluator: Deepgram nova-3 batch, language=en",
            (
                f"  WER coverage: {summary.wer_evaluated_requests}/"
                f"{summary.wer_candidate_requests} "
                f"({_percent(summary.wer_coverage_fraction)})"
            ),
            _distribution_line("WER", summary.wer_percentage, "%"),
        ]
    websocket_lines: list[str] = []
    if summary.websocket_setup_ms is not None:
        assert summary.input_ingress_ms is not None
        assert summary.final_text_to_first_playable_ms is not None
        assert summary.final_text_to_audible_ms is not None
        assert summary.final_text_to_audio_completion_ms is not None
        websocket_lines = [
            _distribution_line("WebSocket setup", summary.websocket_setup_ms, "ms"),
            _distribution_line("Input ingress", summary.input_ingress_ms, "ms"),
            _distribution_line(
                "Final text to first playable",
                summary.final_text_to_first_playable_ms,
                "ms",
            ),
            _distribution_line("Final text to audible", summary.final_text_to_audible_ms, "ms"),
            _distribution_line(
                "Final text to completion",
                summary.final_text_to_audio_completion_ms,
                "ms",
            ),
            (
                "  audio before input complete: "
                f"{_percent(summary.audio_before_input_complete_fraction)}"
            ),
        ]
    lines = [
        "TTS benchmark result",
        "====================",
        f"Target: {summary.target.value}",
        f"Transport: {summary.transport.value}",
        f"Requested RPS: {_number(summary.requested_rps)}",
        f"Actual RPS: {_number(summary.actual_rps)}",
        f"Measurement: {_number(summary.measurement_duration_s)} s",
        "",
        "Requests",
        f"  scheduled: {summary.scheduled_requests}",
        f"  started: {summary.started_requests}",
        f"  dispatched in measurement window: {summary.dispatched_requests}",
        f"  dispatched after measurement window: {summary.late_dispatched_requests}",
        f"  failed before dispatch: {summary.predispatch_failed_requests}",
        f"  dropped: {summary.dropped_requests}",
        f"  PCM complete: {summary.pcm_complete_requests}",
        f"  audible: {summary.audible_requests}",
        f"  audio successful: {summary.audio_successful_requests}",
        f"  successful: {summary.successful_requests} ({_percent(summary.success_fraction)})",
        (
            f"  underrun requests: {summary.underrun_requests} "
            f"({_percent(summary.underrun_request_fraction)})"
        ),
        f"  underrun events: {summary.underrun_count}",
        f"  peak in-flight: {summary.peak_in_flight}",
        "",
        "Latency and prompt distributions (nearest-rank)",
        _distribution_line("Scheduling lag", summary.scheduling_lag_ms, "ms"),
        _distribution_line("TTFB", summary.ttfb_ms, "ms"),
        _distribution_line("First playable", summary.first_playable_ms, "ms"),
        _distribution_line("Audible TTFA", summary.audible_ttfa_ms, "ms"),
        _distribution_line("Leading silence", summary.leading_silence_ms, "ms"),
        _distribution_line("E2E", summary.end_to_end_ms, "ms"),
        _distribution_line("Per-request RTF", summary.realtime_factor, "x"),
        _distribution_line("Prompt words", summary.prompt_word_count, "words"),
        _distribution_line("Audio duration", summary.audio_duration_s, "s"),
        *websocket_lines,
        *wer_lines,
        f"  semantic quality: {summary.semantic_quality.value}",
        "",
        "Streaming",
        f"  generated audio: {_number(summary.generated_audio_s)} s",
        f"  received audio: {_number(summary.received_audio_xrt)} xRT",
        f"  total stall: {_number(summary.total_stalled_ms)} ms",
        f"  largest stall: {_number(summary.largest_stall_ms)} ms",
        "",
        "Preflight",
        f"  DNS: {_optional_number(summary.setup.dns_ms, 'ms')}",
        f"  TCP: {_optional_number(summary.setup.tcp_ms, 'ms')}",
        f"  TLS: {_optional_number(summary.setup.tls_ms, 'ms')}",
        f"  healthcheck: {_optional_number(summary.setup.healthcheck_ms, 'ms')}",
        "",
        "Informational references only (never pass/fail gates)",
        f"  audible TTFA p95: <= {_number(summary.reference_ttfa_p95_ms)} ms",
        f"  underruns: {summary.reference_underruns}",
    ]
    return "\n".join(lines) + "\n"


def load_and_render_report(result_dir: Path) -> str:
    """Load a single-rate result directory and render it."""
    summary_path = result_dir.expanduser().resolve() / "summary.json"
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReportError(f"summary not found: {summary_path}") from error
    except json.JSONDecodeError as error:
        raise ReportError(f"invalid summary JSON: {summary_path}: {error}") from error
    try:
        return render_run_report(RunSummary.model_validate(value))
    except ValidationError as error:
        raise ReportError(f"unsupported summary contract: {error}") from error


def _distribution_line(label: str, value: Distribution, unit: str) -> str:
    return (
        f"  {label}: min={_metric(value.minimum, unit)}, p50={_metric(value.p50, unit)}, "
        f"p95={_metric(value.p95, unit)}, p99={_metric(value.p99, unit)}, "
        f"max={_metric(value.maximum, unit)}, n={value.samples}, missing={value.missing}"
    )


def _metric(value: float | None, unit: str) -> str:
    return "-" if value is None else f"{_number(value)}{unit}"


def _optional_number(value: float | None, unit: str) -> str:
    return "n/a" if value is None else f"{_number(value)} {unit}"


def _number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"
