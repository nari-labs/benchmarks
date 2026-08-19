from __future__ import annotations

from tts_bench.metrics import derive_summary, distribution
from tts_bench.models import (
    PcmFormat,
    Phase,
    PlaybackRecord,
    RequestRecord,
    SemanticQualityStatus,
    SetupMetrics,
    TargetKind,
    TransportStatus,
    WERStatus,
)


def test_distribution_uses_nearest_rank_and_counts_missing_values() -> None:
    result = distribution([float(value) for value in range(1, 21)], cohort_size=25)
    assert result.minimum == 1
    assert result.p50 == 10
    assert result.p95 == 19
    assert result.p99 == 20
    assert result.maximum == 20
    assert result.samples == 20
    assert result.missing == 5


def test_empty_distribution_preserves_cohort_size() -> None:
    result = distribution([], cohort_size=7)
    assert result.samples == 0
    assert result.missing == 7
    assert result.p95 is None


def test_summary_separates_transport_audibility_and_duration() -> None:
    records = (
        RequestRecord(
            run_id="run",
            request_id="audible",
            phase=Phase.MEASUREMENT,
            phase_index=0,
            prompt_id="prompt-0",
            prompt_word_count=2,
            scheduled_elapsed_ns=0,
            task_started_elapsed_ns=0,
            dispatch_elapsed_ns=0,
            completed_elapsed_ns=1_000_000_000,
            status_code=200,
            success=True,
            output_bytes=48_000,
            measurement_window_audio_ns=1_000_000_000,
            pcm_sha256="a" * 64,
            pcm_format=PcmFormat(),
            playback=PlaybackRecord(
                complete=True,
                continuous=True,
                audible_offset_ns=0,
                audio_duration_ns=1_000_000_000,
                total_stalled_ns=0,
                largest_stall_ns=0,
            ),
            request_body={"input": "hello"},
            transport_status=TransportStatus.COMPLETE,
        ),
        RequestRecord(
            run_id="run",
            request_id="silent",
            phase=Phase.MEASUREMENT,
            phase_index=1,
            prompt_id="prompt-1",
            prompt_word_count=2,
            scheduled_elapsed_ns=1,
            task_started_elapsed_ns=1,
            dispatch_elapsed_ns=1,
            completed_elapsed_ns=30_000_000_001,
            status_code=200,
            success=True,
            output_bytes=1_440_000,
            measurement_window_audio_ns=0,
            pcm_sha256="b" * 64,
            pcm_format=PcmFormat(),
            playback=PlaybackRecord(
                complete=True,
                continuous=True,
                audio_duration_ns=30_000_000_000,
                total_stalled_ns=0,
                largest_stall_ns=0,
            ),
            request_body={"input": "world"},
            transport_status=TransportStatus.COMPLETE,
        ),
    )

    summary = derive_summary(
        run_id="run",
        target=TargetKind.NARI,
        requested_rps=2,
        warmup_ns=0,
        duration_ns=60_000_000_000,
        scheduled_requests=2,
        dropped_requests=0,
        records=records,
        peak_in_flight=2,
        setup=SetupMetrics(),
    )

    assert summary.pcm_complete_requests == 2
    assert summary.audible_requests == 1
    assert summary.audio_duration_s.maximum == 30.0
    assert summary.semantic_quality is SemanticQualityStatus.FAIL


def test_received_audio_xrt_includes_warmup_carryover() -> None:
    warmup = RequestRecord(
        run_id="run",
        request_id="warmup",
        phase=Phase.WARMUP,
        phase_index=0,
        prompt_id="prompt-0",
        prompt_word_count=1,
        scheduled_elapsed_ns=0,
        task_started_elapsed_ns=0,
        dispatch_elapsed_ns=0,
        completed_elapsed_ns=2_000_000_000,
        status_code=200,
        success=True,
        output_bytes=48_000,
        measurement_window_audio_ns=750_000_000,
        playback=PlaybackRecord(
            complete=True,
            continuous=True,
            audio_duration_ns=1_000_000_000,
            total_stalled_ns=0,
            largest_stall_ns=0,
        ),
        request_body={"input": "warmup"},
        transport_status=TransportStatus.COMPLETE,
    )
    measurement = warmup.model_copy(
        update={
            "request_id": "measurement",
            "phase": Phase.MEASUREMENT,
            "dispatch_elapsed_ns": 1_000_000_000,
            "measurement_window_audio_ns": 1_250_000_000,
        }
    )

    summary = derive_summary(
        run_id="run",
        target=TargetKind.NARI,
        requested_rps=1,
        warmup_ns=1_000_000_000,
        duration_ns=2_000_000_000,
        scheduled_requests=1,
        dropped_requests=0,
        records=(warmup, measurement),
        peak_in_flight=2,
        setup=SetupMetrics(),
    )

    assert summary.started_requests == 1
    assert summary.received_audio_xrt == 1.0


def test_wer_value_does_not_change_success_or_semantic_quality() -> None:
    record = RequestRecord(
        run_id="run",
        request_id="measurement",
        phase=Phase.MEASUREMENT,
        phase_index=0,
        prompt_id="prompt-0",
        prompt_word_count=1,
        scheduled_elapsed_ns=0,
        task_started_elapsed_ns=0,
        dispatch_elapsed_ns=0,
        completed_elapsed_ns=1_000_000_000,
        status_code=200,
        success=True,
        output_bytes=48_000,
        measurement_window_audio_ns=1_000_000_000,
        playback=PlaybackRecord(
            complete=True,
            continuous=True,
            audible_offset_ns=0,
            audio_duration_ns=1_000_000_000,
            total_stalled_ns=0,
            largest_stall_ns=0,
        ),
        request_body={"input": "measurement"},
        transport_status=TransportStatus.COMPLETE,
        audio_success=True,
        wer_status=WERStatus.EVALUATED,
        wer_percentage=100.0,
    )

    summary = derive_summary(
        run_id="run",
        target=TargetKind.NARI,
        requested_rps=1,
        warmup_ns=0,
        duration_ns=1_000_000_000,
        scheduled_requests=1,
        dropped_requests=0,
        records=(record,),
        peak_in_flight=1,
        setup=SetupMetrics(),
        wer_enabled=True,
    )

    assert summary.successful_requests == 1
    assert summary.semantic_quality is SemanticQualityStatus.UNCHECKED
    assert summary.wer_percentage is not None
    assert summary.wer_percentage.mean == 100.0


def test_summary_counts_predispatch_failures_and_late_dispatches() -> None:
    failed_before_dispatch = RequestRecord(
        run_id="run",
        request_id="predispatch-failure",
        phase=Phase.MEASUREMENT,
        phase_index=0,
        prompt_id="prompt-0",
        prompt_word_count=1,
        scheduled_elapsed_ns=100_000_000,
        task_started_elapsed_ns=100_000_000,
        completed_elapsed_ns=200_000_000,
        success=False,
        error_kind="transport_error",
        output_bytes=0,
        measurement_window_audio_ns=0,
        playback=PlaybackRecord(
            complete=False,
            continuous=False,
            audio_duration_ns=0,
            total_stalled_ns=0,
            largest_stall_ns=0,
        ),
        request_body={"input": "failed"},
    )
    late_dispatch = failed_before_dispatch.model_copy(
        update={
            "request_id": "late-dispatch",
            "phase_index": 1,
            "scheduled_elapsed_ns": 900_000_000,
            "task_started_elapsed_ns": 900_000_000,
            "dispatch_elapsed_ns": 1_100_000_000,
            "completed_elapsed_ns": 1_200_000_000,
            "success": True,
            "error_kind": None,
            "output_bytes": 48_000,
            "playback": PlaybackRecord(
                complete=True,
                continuous=True,
                audible_offset_ns=0,
                audio_duration_ns=1_000_000_000,
                total_stalled_ns=0,
                largest_stall_ns=0,
            ),
            "transport_status": TransportStatus.COMPLETE,
        }
    )

    summary = derive_summary(
        run_id="run",
        target=TargetKind.NARI,
        requested_rps=2,
        warmup_ns=0,
        duration_ns=1_000_000_000,
        scheduled_requests=2,
        dropped_requests=0,
        records=(failed_before_dispatch, late_dispatch),
        peak_in_flight=2,
        setup=SetupMetrics(),
    )

    assert summary.started_requests == 2
    assert summary.dispatched_requests == 0
    assert summary.late_dispatched_requests == 1
    assert summary.predispatch_failed_requests == 1
    assert summary.successful_requests == 1
    assert summary.success_fraction == 0.5
    assert summary.actual_rps == 0
    assert summary.scheduling_lag_ms.samples == 1
    assert summary.scheduling_lag_ms.missing == 1


def test_realtime_factor_uses_only_complete_pcm_requests() -> None:
    complete = RequestRecord(
        run_id="run",
        request_id="complete",
        phase=Phase.MEASUREMENT,
        phase_index=0,
        prompt_id="prompt-0",
        prompt_word_count=1,
        scheduled_elapsed_ns=0,
        task_started_elapsed_ns=0,
        dispatch_elapsed_ns=0,
        completed_elapsed_ns=1_000_000_000,
        success=True,
        output_bytes=48_000,
        measurement_window_audio_ns=1_000_000_000,
        playback=PlaybackRecord(
            complete=True,
            continuous=True,
            audible_offset_ns=0,
            audio_duration_ns=1_000_000_000,
            total_stalled_ns=0,
            largest_stall_ns=0,
        ),
        request_body={"input": "complete"},
        transport_status=TransportStatus.COMPLETE,
    )
    truncated = complete.model_copy(
        update={
            "request_id": "truncated",
            "phase_index": 1,
            "dispatch_elapsed_ns": 100_000_000,
            "completed_elapsed_ns": 600_000_000,
            "success": False,
            "error_kind": "timeout",
            "playback": PlaybackRecord(
                complete=False,
                continuous=False,
                audio_duration_ns=250_000_000,
                total_stalled_ns=0,
                largest_stall_ns=0,
            ),
            "transport_status": TransportStatus.INCOMPLETE,
        }
    )

    summary = derive_summary(
        run_id="run",
        target=TargetKind.NARI,
        requested_rps=2,
        warmup_ns=0,
        duration_ns=1_000_000_000,
        scheduled_requests=2,
        dropped_requests=0,
        records=(complete, truncated),
        peak_in_flight=2,
        setup=SetupMetrics(),
    )

    assert summary.realtime_factor.samples == 1
    assert summary.realtime_factor.missing == 1
    assert summary.realtime_factor.p50 == 1.0
