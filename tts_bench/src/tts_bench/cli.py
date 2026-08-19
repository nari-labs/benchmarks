"""Command-line interface for tts-bench."""

from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any

import click

from tts_bench import __version__
from tts_bench.dataset import DatasetError, load_prompt_pool, prepare_seedtts_dataset
from tts_bench.models import ArrivalPattern, PcmFormat, TargetKind, TransportKind
from tts_bench.reporting import ReportError, load_and_render_report, render_run_report
from tts_bench.runtime import RateOptions, run_rate
from tts_bench.targets import PreflightError, TargetAdapter, TargetConfigurationError
from tts_bench.wer import DeepgramWEROptions

_DURATION_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(ms|s|m)?$")


class DurationType(click.ParamType[float]):
    name = "duration"

    def __init__(self, *, allow_zero: bool = False) -> None:
        self.allow_zero = allow_zero

    def convert(
        self,
        value: object,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> float:
        if isinstance(value, (int, float)):
            seconds = float(value)
        else:
            match = _DURATION_RE.fullmatch(str(value).strip().lower())
            if match is None:
                self.fail("expected a number with an optional ms, s, or m suffix", param, ctx)
            magnitude = float(match.group(1))
            unit = match.group(2) or "s"
            seconds = magnitude * {"ms": 0.001, "s": 1.0, "m": 60.0}[unit]
        if not math.isfinite(seconds) or seconds < 0 or (seconds == 0 and not self.allow_zero):
            requirement = "non-negative" if self.allow_zero else "positive"
            self.fail(f"duration must be {requirement}", param, ctx)
        return seconds


class RequestParamType(click.ParamType[tuple[str, Any]]):
    name = "KEY=JSON"

    def convert(
        self,
        value: object,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> tuple[str, Any]:
        raw = str(value)
        key, separator, encoded = raw.partition("=")
        if not separator or not key.strip():
            self.fail("expected KEY=JSON", param, ctx)
        try:
            decoded = json.loads(
                encoded,
                parse_constant=lambda constant: _invalid_json_constant(constant),
            )
        except (json.JSONDecodeError, ValueError) as error:
            self.fail(f"invalid JSON value: {error}", param, ctx)
        return key.strip(), decoded


def _invalid_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant {value}")


@click.group(no_args_is_help=True)
@click.version_option(__version__, prog_name="bench")
def main() -> None:
    """Benchmark OpenAI-compatible streaming TTS servers."""


@main.group("dataset")
def dataset_group() -> None:
    """Prepare benchmark prompt datasets."""


@dataset_group.command("prepare")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("datasets/seed-tts-eval.jsonl"),
    show_default=True,
)
@click.option("--overwrite", is_flag=True, help="Replace an existing projection and provenance.")
def prepare_dataset_command(output: Path, overwrite: bool) -> None:
    """Prepare the pinned English Seed-TTS prompt projection."""
    try:
        dataset_path, provenance_path = prepare_seedtts_dataset(output, overwrite=overwrite)
    except DatasetError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Wrote {dataset_path}")
    click.echo(f"Wrote {provenance_path}")


@main.command("run")
@click.option(
    "--target",
    type=click.Choice([target.value for target in TargetKind], case_sensitive=False),
    required=True,
)
@click.option(
    "--transport",
    type=click.Choice([transport.value for transport in TransportKind]),
    default=TransportKind.HTTP.value,
    show_default=True,
    help="HTTP for every target; WebSocket is supported only by nari.",
)
@click.option("--base-url", required=True, help="Server base URL, before the speech endpoint.")
@click.option(
    "--model",
    default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    show_default=True,
)
@click.option("--voice", default="Ryan", show_default=True)
@click.option("--language", default="English", show_default=True)
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    default=Path("datasets/seed-tts-eval.jsonl"),
    show_default=True,
)
@click.option(
    "--rps",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
    show_default=True,
    help="One request rate. Run each RPS point as a separate command.",
)
@click.option("--warmup", type=DurationType(allow_zero=True), default="10s", show_default=True)
@click.option("--duration", type=DurationType(), default="30s", show_default=True)
@click.option(
    "--arrival",
    type=click.Choice([pattern.value for pattern in ArrivalPattern]),
    default=ArrivalPattern.POISSON.value,
    show_default=True,
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--timeout", type=DurationType(), default="120s", show_default=True)
@click.option("--max-in-flight", type=click.IntRange(min=1), default=4096, show_default=True)
@click.option("--health-path", default="/health", show_default=True)
@click.option("--request-param", type=RequestParamType(), multiple=True)
@click.option(
    "--wer",
    is_flag=True,
    help="Score complete audible measurement WAVs with Deepgram Nova-3 (requires DEEPGRAM_API_KEY).",
)
def run_command(
    *,
    target: str,
    transport: str,
    base_url: str,
    model: str,
    voice: str,
    language: str,
    dataset: Path,
    rps: float,
    warmup: float,
    duration: float,
    arrival: str,
    seed: int,
    output: Path,
    timeout: float,
    max_in_flight: int,
    health_path: str,
    request_param: tuple[tuple[str, Any], ...],
    wer: bool,
) -> None:
    """Run one independently managed RPS point."""
    try:
        request_params = _parse_request_params(request_param)
        wer_options = DeepgramWEROptions.from_environment() if wer else None
        prompts, manifest = load_prompt_pool(dataset)
        adapter = TargetAdapter(
            kind=TargetKind(target),
            base_url=base_url,
            model=model,
            voice=voice,
            language=language,
            expected_format=PcmFormat(sample_rate_hz=24_000),
            request_params=request_params,
            health_path=health_path,
            transport=TransportKind(transport),
        )
        options = RateOptions(
            requested_rps=rps,
            arrival=ArrivalPattern(arrival),
            seed=seed,
            warmup_s=warmup,
            duration_s=duration,
            timeout_s=timeout,
            max_in_flight=max_in_flight,
            wer_options=wer_options,
        )
        summary = asyncio.run(
            run_rate(
                adapter=adapter,
                prompts=prompts,
                dataset_manifest=manifest,
                output=output.expanduser().resolve(),
                options=options,
            )
        )
        report = render_run_report(summary)
    except (
        DatasetError,
        FileExistsError,
        PreflightError,
        ReportError,
        TargetConfigurationError,
        ValueError,
    ) as error:
        raise click.ClickException(str(error)) from error
    click.echo(report, nl=False)


@main.command("report")
@click.argument(
    "result_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, resolve_path=True),
)
def report_command(result_dir: Path) -> None:
    """Re-render a saved single-rate result."""
    try:
        report = load_and_render_report(result_dir)
    except ReportError as error:
        raise click.ClickException(str(error)) from error
    click.echo(report, nl=False)


def _parse_request_params(values: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"duplicate request-param key: {key}")
        result[key] = value
    return result
