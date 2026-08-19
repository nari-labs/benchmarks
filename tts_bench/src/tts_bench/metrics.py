"""Pure nearest-rank aggregation for completed benchmark observations."""

from __future__ import annotations

import math
from collections.abc import Sequence

from tts_bench.models import (
    Distribution,
    Phase,
    RequestRecord,
    RunSummary,
    SemanticQualityStatus,
    SetupMetrics,
    TargetKind,
    TransportKind,
    TransportStatus,
    WERStatus,
)

NANOSECONDS_PER_MILLISECOND = 1_000_000
NANOSECONDS_PER_SECOND = 1_000_000_000


def distribution(values: Sequence[float], *, cohort_size: int) -> Distribution:
    """Return a nearest-rank distribution and explicit missing count."""
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return Distribution(samples=0, missing=cohort_size)

    def percentile(percent: int) -> float:
        index = max(0, math.ceil(percent * len(ordered) / 100) - 1)
        return ordered[index]

    return Distribution(
        samples=len(ordered),
        missing=max(0, cohort_size - len(ordered)),
        minimum=ordered[0],
        p50=percentile(50),
        p95=percentile(95),
        p99=percentile(99),
        maximum=ordered[-1],
        mean=sum(ordered) / len(ordered),
    )


def derive_summary(
    *,
    run_id: str,
    target: TargetKind,
    transport: TransportKind = TransportKind.HTTP,
    requested_rps: float,
    warmup_ns: int,
    duration_ns: int,
    scheduled_requests: int,
    dropped_requests: int,
    records: Sequence[RequestRecord],
    peak_in_flight: int,
    setup: SetupMetrics,
    wer_enabled: bool = False,
) -> RunSummary:
    """Aggregate every non-dropped request scheduled for the measurement phase."""
    measurement_end = warmup_ns + duration_ns
    started = [record for record in records if record.phase is Phase.MEASUREMENT]
    dispatched = [record for record in started if record.dispatch_elapsed_ns is not None]
    dispatched_in_window = [
        record
        for record in dispatched
        if record.dispatch_elapsed_ns is not None
        and warmup_ns <= record.dispatch_elapsed_ns < measurement_end
    ]
    late_dispatched = [
        record
        for record in dispatched
        if record.dispatch_elapsed_ns is not None and record.dispatch_elapsed_ns >= measurement_end
    ]
    predispatch_failed = [
        record for record in started if record.dispatch_elapsed_ns is None and not record.success
    ]
    count = len(started)
    successful = [record for record in started if record.success]
    audio_successful = [
        record
        for record in started
        if (record.success if record.audio_success is None else record.audio_success)
    ]
    pcm_complete = [
        record for record in started if record.transport_status is TransportStatus.COMPLETE
    ]
    audible_records = [
        record for record in pcm_complete if record.playback.audible_offset_ns is not None
    ]
    wer_enabled = wer_enabled or any(
        record.wer_status is not WERStatus.DISABLED for record in records
    )
    wer_candidates = [
        record
        for record in started
        if record.wer_status in {WERStatus.EVALUATED, WERStatus.MISSING}
    ]
    wer_evaluated = [
        record for record in wer_candidates if record.wer_status is WERStatus.EVALUATED
    ]
    wer_values = [
        record.wer_percentage for record in wer_evaluated if record.wer_percentage is not None
    ]
    semantic_quality = SemanticQualityStatus.UNCHECKED
    if len(audible_records) != len(pcm_complete):
        semantic_quality = SemanticQualityStatus.FAIL
    underrun_records = [record for record in started if record.playback.underruns]

    def delta_ms(end: int | None, start: int | None) -> float | None:
        if end is None or start is None:
            return None
        return (end - start) / NANOSECONDS_PER_MILLISECOND

    scheduling = [
        (record.dispatch_elapsed_ns - record.scheduled_elapsed_ns) / NANOSECONDS_PER_MILLISECOND
        for record in started
        if record.dispatch_elapsed_ns is not None
    ]
    ttfb = [
        value
        for record in started
        if (value := delta_ms(record.first_body_elapsed_ns, record.dispatch_elapsed_ns)) is not None
    ]
    first_playable = [
        value
        for record in started
        if (
            value := delta_ms(
                record.playback.first_playable_elapsed_ns,
                record.dispatch_elapsed_ns,
            )
        )
        is not None
    ]
    audible = [
        value
        for record in started
        if (
            value := delta_ms(
                record.playback.first_audible_elapsed_ns,
                record.dispatch_elapsed_ns,
            )
        )
        is not None
    ]
    leading = [
        record.playback.audible_offset_ns / NANOSECONDS_PER_MILLISECOND
        for record in started
        if record.playback.audible_offset_ns is not None
    ]
    end_to_end = [
        value
        for record in started
        if (value := delta_ms(record.completed_elapsed_ns, record.dispatch_elapsed_ns)) is not None
    ]
    realtime_factors = [
        (record.completed_elapsed_ns - record.dispatch_elapsed_ns)
        / record.playback.audio_duration_ns
        for record in pcm_complete
        if record.dispatch_elapsed_ns is not None and record.playback.audio_duration_ns > 0
    ]
    words = [float(record.prompt_word_count) for record in started]
    audio_durations = [
        record.playback.audio_duration_ns / NANOSECONDS_PER_SECOND for record in pcm_complete
    ]
    generated_audio_ns = sum(record.playback.audio_duration_ns for record in started)
    window_audio_ns = sum(record.measurement_window_audio_ns for record in records)
    websocket_setup: list[float] = []
    input_ingress: list[float] = []
    final_text_to_first_playable: list[float] = []
    final_text_to_audible: list[float] = []
    final_text_to_completion: list[float] = []
    audio_before_input_complete: list[bool] = []
    if transport is TransportKind.WEBSOCKET:
        for record in started:
            timing = record.input_timing
            if timing is None:
                continue
            websocket_setup.append(
                (timing.configured_elapsed_ns - timing.connect_started_elapsed_ns)
                / NANOSECONDS_PER_MILLISECOND
            )
            input_ingress.append(
                (timing.last_text_sent_elapsed_ns - timing.first_text_sent_elapsed_ns)
                / NANOSECONDS_PER_MILLISECOND
            )
            if record.playback.first_playable_elapsed_ns is not None:
                final_text_to_first_playable.append(
                    (record.playback.first_playable_elapsed_ns - timing.last_text_sent_elapsed_ns)
                    / NANOSECONDS_PER_MILLISECOND
                )
                audio_before_input_complete.append(
                    record.playback.first_playable_elapsed_ns < timing.end_sent_elapsed_ns
                )
            if record.playback.first_audible_elapsed_ns is not None:
                final_text_to_audible.append(
                    (record.playback.first_audible_elapsed_ns - timing.last_text_sent_elapsed_ns)
                    / NANOSECONDS_PER_MILLISECOND
                )
            final_text_to_completion.append(
                (record.completed_elapsed_ns - timing.last_text_sent_elapsed_ns)
                / NANOSECONDS_PER_MILLISECOND
            )
    return RunSummary(
        run_id=run_id,
        target=target,
        transport=transport,
        requested_rps=requested_rps,
        measurement_duration_s=duration_ns / NANOSECONDS_PER_SECOND,
        scheduled_requests=scheduled_requests,
        started_requests=count,
        dispatched_requests=len(dispatched_in_window),
        late_dispatched_requests=len(late_dispatched),
        predispatch_failed_requests=len(predispatch_failed),
        dropped_requests=dropped_requests,
        successful_requests=len(successful),
        actual_rps=len(dispatched_in_window) * NANOSECONDS_PER_SECOND / duration_ns,
        success_fraction=len(successful) / count if count else None,
        underrun_requests=len(underrun_records),
        underrun_request_fraction=len(underrun_records) / count if count else None,
        underrun_count=sum(len(record.playback.underruns) for record in started),
        total_stalled_ms=sum(record.playback.total_stalled_ns for record in started)
        / NANOSECONDS_PER_MILLISECOND,
        largest_stall_ms=max((record.playback.largest_stall_ns for record in started), default=0)
        / NANOSECONDS_PER_MILLISECOND,
        scheduling_lag_ms=distribution(scheduling, cohort_size=count),
        ttfb_ms=distribution(ttfb, cohort_size=count),
        first_playable_ms=distribution(first_playable, cohort_size=count),
        audible_ttfa_ms=distribution(audible, cohort_size=count),
        leading_silence_ms=distribution(leading, cohort_size=count),
        end_to_end_ms=distribution(end_to_end, cohort_size=count),
        realtime_factor=distribution(realtime_factors, cohort_size=count),
        prompt_word_count=distribution(words, cohort_size=count),
        audio_duration_s=distribution(audio_durations, cohort_size=count),
        generated_audio_s=generated_audio_ns / NANOSECONDS_PER_SECOND,
        received_audio_xrt=window_audio_ns / duration_ns,
        peak_in_flight=peak_in_flight,
        setup=setup,
        pcm_complete_requests=len(pcm_complete),
        audible_requests=len(audible_records),
        semantic_quality=semantic_quality,
        audio_successful_requests=len(audio_successful),
        wer_enabled=wer_enabled,
        wer_candidate_requests=len(wer_candidates),
        wer_evaluated_requests=len(wer_evaluated),
        wer_coverage_fraction=(
            len(wer_evaluated) / len(wer_candidates) if wer_candidates else None
        ),
        wer_percentage=(
            distribution(wer_values, cohort_size=len(wer_candidates)) if wer_enabled else None
        ),
        websocket_setup_ms=(
            distribution(websocket_setup, cohort_size=count)
            if transport is TransportKind.WEBSOCKET
            else None
        ),
        input_ingress_ms=(
            distribution(input_ingress, cohort_size=count)
            if transport is TransportKind.WEBSOCKET
            else None
        ),
        final_text_to_first_playable_ms=(
            distribution(final_text_to_first_playable, cohort_size=count)
            if transport is TransportKind.WEBSOCKET
            else None
        ),
        final_text_to_audible_ms=(
            distribution(final_text_to_audible, cohort_size=count)
            if transport is TransportKind.WEBSOCKET
            else None
        ),
        final_text_to_audio_completion_ms=(
            distribution(final_text_to_completion, cohort_size=count)
            if transport is TransportKind.WEBSOCKET
            else None
        ),
        audio_before_input_complete_fraction=(
            sum(audio_before_input_complete) / len(audio_before_input_complete)
            if audio_before_input_complete
            else None
        ),
    )
