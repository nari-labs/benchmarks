"""Typed fixtures shared by unit and integration tests."""

from __future__ import annotations

import math
import struct

from tts_bench.models import DatasetManifest, Prompt


def make_prompts(count: int) -> tuple[Prompt, ...]:
    return tuple(
        Prompt(
            id=f"prompt-{index:04d}",
            text=f"Synthetic benchmark sentence number {index}.",
            language="en",
            word_count=5,
            source_index=index,
        )
        for index in range(count)
    )


def make_manifest(prompts: tuple[Prompt, ...]) -> DatasetManifest:
    return DatasetManifest(
        kind="jsonl_snapshot",
        source_path="test-dataset.jsonl",
        source_files=("test-dataset.jsonl",),
        source_sha256="1" * 64,
        prompt_count=len(prompts),
        pool_sha256="2" * 64,
    )


def sine_pcm(
    duration_ms: int,
    *,
    sample_rate_hz: int = 24_000,
    amplitude: int = 2_000,
    frequency_hz: float = 440.0,
) -> bytes:
    sample_count = duration_ms * sample_rate_hz // 1_000
    samples = [
        round(amplitude * math.sin(2 * math.pi * frequency_hz * index / sample_rate_hz))
        for index in range(sample_count)
    ]
    return struct.pack(f"<{sample_count}h", *samples)


def constant_pcm(
    duration_ms: int,
    value: int,
    *,
    sample_rate_hz: int = 24_000,
) -> bytes:
    sample_count = duration_ms * sample_rate_hz // 1_000
    return struct.pack(f"<{sample_count}h", *([value] * sample_count))
