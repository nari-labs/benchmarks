from __future__ import annotations

from itertools import pairwise

from tests.helpers import make_prompts
from tts_bench.dataset import PromptPool
from tts_bench.models import ArrivalPattern, Phase
from tts_bench.runtime import RateOptions, _build_slots
from tts_bench.scheduling import iter_phase_schedule


def test_constant_schedule_uses_exact_half_open_window() -> None:
    slots = tuple(
        iter_phase_schedule(
            phase=Phase.MEASUREMENT,
            phase_start_ns=10,
            duration_ns=1_000_000_000,
            rps=2.5,
            pattern=ArrivalPattern.CONSTANT,
            seed=0,
        )
    )
    assert slots == (
        (0, 10),
        (1, 400_000_010),
        (2, 800_000_010),
    )


def test_poisson_schedule_is_version_stable_and_phase_separated() -> None:
    def schedule(phase: Phase) -> tuple[tuple[int, int], ...]:
        return tuple(
            iter_phase_schedule(
                phase=phase,
                phase_start_ns=0,
                duration_ns=1_000_000_000,
                rps=8.0,
                pattern=ArrivalPattern.POISSON,
                seed=0,
            )
        )

    measurement = schedule(Phase.MEASUREMENT)
    repeated = schedule(Phase.MEASUREMENT)
    warmup = schedule(Phase.WARMUP)

    assert measurement == repeated
    assert measurement != warmup
    assert all(left[1] < right[1] for left, right in pairwise(measurement))
    assert all(0 <= deadline < 1_000_000_000 for _, deadline in measurement)


def test_warmup_and_measurement_have_separate_phase_boundaries() -> None:
    options = RateOptions(
        requested_rps=2,
        arrival=ArrivalPattern.CONSTANT,
        seed=0,
        warmup_s=1,
        duration_s=1,
        timeout_s=1,
        max_in_flight=10,
    )
    slots = _build_slots(
        pool=PromptPool(make_prompts(8), seed=0),
        warmup_ns=1_000_000_000,
        duration_ns=1_000_000_000,
        options=options,
    )
    warmup = [slot for slot in slots if slot.phase is Phase.WARMUP]
    measurement = [slot for slot in slots if slot.phase is Phase.MEASUREMENT]
    assert [slot.phase_index for slot in warmup] == [0, 1]
    assert [slot.scheduled_elapsed_ns for slot in warmup] == [0, 500_000_000]
    assert [slot.phase_index for slot in measurement] == [0, 1]
    assert [slot.scheduled_elapsed_ns for slot in measurement] == [
        1_000_000_000,
        1_500_000_000,
    ]
    assert [slot.prompt_id for slot in warmup] != [slot.prompt_id for slot in measurement]
