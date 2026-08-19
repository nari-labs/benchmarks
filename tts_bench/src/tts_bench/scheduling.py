"""Versioned open-loop arrival scheduling."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from fractions import Fraction

from tts_bench.models import ArrivalPattern, Phase

NANOSECONDS_PER_SECOND = 1_000_000_000
ARRIVAL_SCHEDULE_VERSION = 1


def iter_phase_schedule(
    *,
    phase: Phase,
    phase_start_ns: int,
    duration_ns: int,
    rps: float,
    pattern: ArrivalPattern,
    seed: int,
) -> Iterator[tuple[int, int]]:
    """Yield ``(phase_index, absolute_scheduled_elapsed_ns)`` slots."""
    if phase_start_ns < 0 or duration_ns < 0:
        raise ValueError("phase times cannot be negative")
    if not math.isfinite(rps) or rps <= 0:
        raise ValueError("rps must be finite and positive")
    if pattern is ArrivalPattern.CONSTANT:
        yield from _constant_slots(phase_start_ns, duration_ns, rps)
    else:
        yield from _poisson_slots(phase, phase_start_ns, duration_ns, rps, seed)


def _constant_slots(phase_start_ns: int, duration_ns: int, rps: float) -> Iterator[tuple[int, int]]:
    rate = Fraction(str(rps))
    index = 0
    while True:
        local_ns = (index * NANOSECONDS_PER_SECOND * rate.denominator) // rate.numerator
        if local_ns >= duration_ns:
            return
        yield index, phase_start_ns + local_ns
        index += 1


def _poisson_slots(
    phase: Phase,
    phase_start_ns: int,
    duration_ns: int,
    rps: float,
    seed: int,
) -> Iterator[tuple[int, int]]:
    cumulative_s = 0.0
    previous_ns = -1
    index = 0
    while True:
        material = (
            f"tts-bench/poisson/v{ARRIVAL_SCHEDULE_VERSION}\0{seed}\0{phase.value}\0{index}"
        ).encode()
        integer = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        uniform = (integer + 0.5) / 2**64
        cumulative_s += -math.log1p(-uniform) / rps
        local_ns = max(previous_ns + 1, math.ceil(cumulative_s * NANOSECONDS_PER_SECOND))
        if local_ns >= duration_ns:
            return
        yield index, phase_start_ns + local_ns
        previous_ns = local_ns
        index += 1
