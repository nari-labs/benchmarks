from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from tts_bench.cli import main
from tts_bench.models import (
    Distribution,
    RunSummary,
    SetupMetrics,
    TargetKind,
)


def test_version_and_single_rate_option() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "bench, version 0.1.0" in result.output

    help_result = runner.invoke(main, ["run", "--help"])
    assert help_result.exit_code == 0
    assert "One request rate" in help_result.output
    assert "nari" in help_result.output

    list_result = runner.invoke(main, ["run", "--rps", "1,2"])
    assert list_result.exit_code == 2
    assert "not a valid float" in list_result.output


def test_offline_report_command(tmp_path: Path) -> None:
    empty = Distribution(samples=0, missing=1)
    summary = RunSummary(
        run_id="run",
        target=TargetKind.VOXSERVE,
        requested_rps=1,
        measurement_duration_s=60,
        scheduled_requests=1,
        started_requests=0,
        dispatched_requests=0,
        late_dispatched_requests=0,
        predispatch_failed_requests=0,
        dropped_requests=1,
        successful_requests=0,
        actual_rps=0,
        underrun_requests=0,
        underrun_count=0,
        total_stalled_ms=0,
        largest_stall_ms=0,
        scheduling_lag_ms=empty,
        ttfb_ms=empty,
        first_playable_ms=empty,
        audible_ttfa_ms=empty,
        leading_silence_ms=empty,
        end_to_end_ms=empty,
        realtime_factor=empty,
        prompt_word_count=empty,
        audio_duration_s=empty,
        generated_audio_s=0,
        received_audio_xrt=0,
        peak_in_flight=0,
        setup=SetupMetrics(),
        pcm_complete_requests=0,
        audible_requests=0,
        audio_successful_requests=0,
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(summary.model_dump(mode="json")),
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["report", str(tmp_path)])
    assert result.exit_code == 0
    assert "Requested RPS: 1" in result.output
    assert "Transport: http" in result.output
    assert "Informational references only" in result.output


def test_transport_is_exposed_without_input_chunk_options() -> None:
    runner = CliRunner()
    help_result = runner.invoke(main, ["run", "--help"])
    assert help_result.exit_code == 0
    assert "--transport [http|websocket]" in help_result.output
    assert "--input-mode" not in help_result.output
    assert "--input-chunk" not in help_result.output


def test_wer_requires_api_key_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist"
    result = CliRunner().invoke(
        main,
        [
            "run",
            "--target",
            "voxserve",
            "--base-url",
            "http://127.0.0.1:9",
            "--rps",
            "1",
            "--output",
            str(output),
            "--wer",
        ],
        env={"DEEPGRAM_API_KEY": ""},
    )

    assert result.exit_code == 1
    assert "--wer requires the DEEPGRAM_API_KEY environment variable" in result.output
    assert not output.exists()
