from __future__ import annotations

import struct

import pytest

from tests.helpers import constant_pcm, sine_pcm
from tts_bench.audio import (
    AudibleOnsetDetector,
    AudioChunkObservation,
    AudioProtocolError,
    PcmStreamDecoder,
    evaluate_playback,
    pcm_to_wav,
)
from tts_bench.models import PcmFormat, StreamKind

PCM = PcmFormat(sample_rate_hz=24_000)


def test_raw_pcm_decoder_aligns_split_samples() -> None:
    source = sine_pcm(40)
    decoder = PcmStreamDecoder(StreamKind.RAW_PCM, PCM)
    chunks = [decoder.feed(source[:1]), decoder.feed(source[1:7]), decoder.feed(source[7:])]
    decoder.finalize()
    assert b"".join(chunks) == source
    assert chunks[0] == b""


def test_raw_pcm_decoder_rejects_partial_and_empty_streams() -> None:
    partial = PcmStreamDecoder(StreamKind.RAW_PCM, PCM)
    partial.feed(b"\x00")
    with pytest.raises(AudioProtocolError, match="partial"):
        partial.finalize()
    with pytest.raises(AudioProtocolError, match="without playable"):
        PcmStreamDecoder(StreamKind.RAW_PCM, PCM).finalize()


def test_wav_decoder_handles_every_byte_boundary() -> None:
    source = sine_pcm(50)
    wav = pcm_to_wav(source, PCM)
    decoder = PcmStreamDecoder(StreamKind.WAV, PCM)
    decoded = b"".join(decoder.feed(bytes((byte,))) for byte in wav)
    decoder.finalize()
    assert decoded == source


def test_wav_decoder_handles_open_ended_stream_header() -> None:
    source = sine_pcm(30)
    byte_rate = PCM.sample_rate_hz * PCM.bytes_per_frame
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        PCM.sample_rate_hz,
        byte_rate,
        PCM.bytes_per_frame,
        16,
        b"data",
        0xFFFFFFFF,
    )
    decoder = PcmStreamDecoder(StreamKind.WAV, PCM)
    assert decoder.feed(header[:19]) == b""
    assert decoder.feed(header[19:] + source) == source
    decoder.finalize()


def test_wav_decoder_rejects_wrong_sample_rate_and_truncation() -> None:
    wrong = pcm_to_wav(sine_pcm(20, sample_rate_hz=16_000), PcmFormat(sample_rate_hz=16_000))
    with pytest.raises(AudioProtocolError, match="received 16000 Hz"):
        PcmStreamDecoder(StreamKind.WAV, PCM).feed(wrong)

    source = pcm_to_wav(sine_pcm(20), PCM)
    decoder = PcmStreamDecoder(StreamKind.WAV, PCM)
    decoder.feed(source[:-2])
    with pytest.raises(AudioProtocolError, match="declared size"):
        decoder.finalize()


def test_audible_detector_removes_dc_and_requires_sustained_frames() -> None:
    detector = AudibleOnsetDetector(PCM)
    assert detector.feed(constant_pcm(50, 2_000)) is None
    assert detector.feed(b"\x00\x00" * (PCM.sample_rate_hz * 20 // 1_000)) is None
    audible = sine_pcm(40)
    first_ten_ms = len(audible) // 4
    first_ten_ms -= first_ten_ms % 2
    assert detector.feed(audible[:first_ten_ms]) is None
    onset = detector.feed(audible[first_ten_ms:])
    assert onset == 60_000_000


def test_playback_detects_continuous_and_multiple_underruns() -> None:
    continuous = evaluate_playback(
        [
            AudioChunkObservation(10_000_000, 100_000_000),
            AudioChunkObservation(70_000_000, 100_000_000),
        ],
        complete=True,
        audible_offset_ns=20_000_000,
    )
    assert continuous.complete
    assert continuous.continuous
    assert continuous.first_audible_elapsed_ns == 30_000_000
    assert continuous.underruns == ()

    stalled = evaluate_playback(
        [
            AudioChunkObservation(0, 100_000_000),
            AudioChunkObservation(200_000_000, 100_000_000),
            AudioChunkObservation(350_000_000, 50_000_000),
        ],
        complete=True,
        audible_offset_ns=150_000_000,
    )
    assert not stalled.continuous
    assert len(stalled.underruns) == 2
    assert stalled.total_stalled_ns == 150_000_000
    assert stalled.largest_stall_ns == 100_000_000
    assert stalled.first_audible_elapsed_ns == 250_000_000


def test_playback_empty_or_incomplete_preserves_completion_state() -> None:
    empty = evaluate_playback([], complete=True, audible_offset_ns=None)
    assert empty.complete
    assert not empty.continuous
    assert empty.audio_duration_ns == 0
    incomplete = evaluate_playback(
        [AudioChunkObservation(0, 10_000_000)], complete=False, audible_offset_ns=None
    )
    assert incomplete.continuous
    assert not incomplete.complete
