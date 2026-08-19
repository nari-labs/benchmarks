"""Streaming PCM/WAV normalization, audible onset, and playback simulation."""

from __future__ import annotations

import io
import math
import struct
import wave
from dataclasses import dataclass

import numpy as np

from tts_bench.models import PcmFormat, PlaybackRecord, StreamKind, UnderrunRecord

NANOSECONDS_PER_SECOND = 1_000_000_000


class AudioProtocolError(ValueError):
    """The response is not a complete supported audio stream."""


@dataclass(frozen=True)
class AudioChunkObservation:
    """Playable media delivered at one client-observed arrival time."""

    arrival_elapsed_ns: int
    duration_ns: int


class PcmStreamDecoder:
    """Normalize raw PCM or a fragmented RIFF/WAVE stream to aligned PCM16."""

    def __init__(
        self,
        kind: StreamKind,
        expected_format: PcmFormat,
    ) -> None:
        self.kind = kind
        self.expected_format = expected_format
        self._wav = _WavParser(expected_format) if kind is StreamKind.WAV else None
        self._remainder = bytearray()
        self._saw_pcm = False

    @property
    def pcm_format(self) -> PcmFormat:
        if self._wav is not None and self._wav.observed_format is not None:
            return self._wav.observed_format
        return self.expected_format

    def feed(self, raw: bytes) -> bytes:
        """Consume one transport chunk and return whole PCM frames."""
        unaligned = self._wav.feed(raw) if self._wav is not None else raw
        if not unaligned:
            return b""
        self._remainder.extend(unaligned)
        frame_bytes = self.expected_format.bytes_per_frame
        complete = len(self._remainder) - (len(self._remainder) % frame_bytes)
        if complete == 0:
            return b""
        pcm = bytes(self._remainder[:complete])
        del self._remainder[:complete]
        self._saw_pcm = True
        return pcm

    def finalize(self) -> None:
        """Require a complete, nonempty, frame-aligned audio stream."""
        if self._wav is not None:
            self._wav.finalize()
        if self._remainder:
            raise AudioProtocolError("audio ended with a partial PCM sample frame")
        if not self._saw_pcm:
            raise AudioProtocolError("response completed without playable PCM audio")


class _WavParser:
    """Incremental parser for PCM16 WAVE, including open-ended streaming headers."""

    _OPEN_ENDED = 0xFFFFFFFF

    def __init__(self, expected_format: PcmFormat) -> None:
        self._expected = expected_format
        self._buffer = bytearray()
        self._riff_seen = False
        self._fmt_seen = False
        self._data_seen = False
        self._data_remaining: int | None = None
        self._skip_remaining = 0
        self.observed_format: PcmFormat | None = None

    def feed(self, raw: bytes) -> bytes:
        self._buffer.extend(raw)
        output = bytearray()
        while True:
            if not self._riff_seen:
                if len(self._buffer) < 12:
                    break
                riff, _size, wave_id = struct.unpack("<4sI4s", self._buffer[:12])
                if riff != b"RIFF" or wave_id != b"WAVE":
                    raise AudioProtocolError("response is not a RIFF/WAVE stream")
                del self._buffer[:12]
                self._riff_seen = True
                continue

            if self._data_seen:
                if self._data_remaining == 0:
                    break
                if not self._buffer:
                    break
                take = len(self._buffer)
                if self._data_remaining is not None:
                    take = min(take, self._data_remaining)
                output.extend(self._buffer[:take])
                del self._buffer[:take]
                if self._data_remaining is not None:
                    self._data_remaining -= take
                continue

            if self._skip_remaining:
                if len(self._buffer) < self._skip_remaining:
                    self._skip_remaining -= len(self._buffer)
                    self._buffer.clear()
                    break
                del self._buffer[: self._skip_remaining]
                self._skip_remaining = 0
                continue

            if len(self._buffer) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", self._buffer[:8])
            padded_size = chunk_size + (chunk_size % 2)
            if chunk_id == b"data":
                del self._buffer[:8]
                self._data_seen = True
                self._data_remaining = None if chunk_size == self._OPEN_ENDED else chunk_size
                continue
            if len(self._buffer) < 8 + padded_size:
                break
            payload = bytes(self._buffer[8 : 8 + chunk_size])
            del self._buffer[: 8 + padded_size]
            if chunk_id == b"fmt ":
                self._parse_format(payload)
            continue
        return bytes(output)

    def _parse_format(self, payload: bytes) -> None:
        if len(payload) < 16:
            raise AudioProtocolError("WAVE fmt chunk is truncated")
        codec, channels, sample_rate, byte_rate, block_align, bits = struct.unpack(
            "<HHIIHH", payload[:16]
        )
        if codec != 1:
            raise AudioProtocolError(f"WAVE codec {codec} is not integer PCM")
        if channels != 1 or bits != 16:
            raise AudioProtocolError(
                f"expected mono PCM16 WAVE, received channels={channels}, bits={bits}"
            )
        if block_align != channels * bits // 8:
            raise AudioProtocolError("WAVE block alignment is inconsistent")
        if byte_rate != sample_rate * block_align:
            raise AudioProtocolError("WAVE byte rate is inconsistent")
        observed = PcmFormat(sample_rate_hz=sample_rate, channels=1, sample_width_bytes=2)
        if observed != self._expected:
            raise AudioProtocolError(
                f"expected {self._expected.sample_rate_hz} Hz PCM, received {sample_rate} Hz"
            )
        self.observed_format = observed
        self._fmt_seen = True

    def finalize(self) -> None:
        if not self._riff_seen:
            raise AudioProtocolError("WAVE stream ended before its RIFF header")
        if not self._fmt_seen:
            raise AudioProtocolError("WAVE stream has no fmt chunk")
        if not self._data_seen:
            raise AudioProtocolError("WAVE stream has no data chunk")
        if self._data_remaining not in (None, 0):
            raise AudioProtocolError("WAVE data chunk ended before its declared size")


class AudibleOnsetDetector:
    """Find sustained speech onset without transport chunk-boundary effects."""

    version = 1
    frame_ms = 20
    hop_ms = 10
    threshold_dbfs = -45.0
    consecutive_active_frames = 2

    def __init__(self, pcm_format: PcmFormat) -> None:
        self._format = pcm_format
        self._pcm = bytearray()
        self._next_window_start = 0
        self._active_run_start: int | None = None
        self._active_run_length = 0
        self.onset_offset_ns: int | None = None

    def feed(self, pcm: bytes) -> int | None:
        if self.onset_offset_ns is not None:
            return self.onset_offset_ns
        if len(pcm) % self._format.bytes_per_frame:
            raise ValueError("audible detector requires whole PCM frames")
        self._pcm.extend(pcm)
        sample_rate = self._format.sample_rate_hz
        frame_samples = max(1, round(sample_rate * self.frame_ms / 1000))
        hop_samples = max(1, round(sample_rate * self.hop_ms / 1000))
        total_samples = len(self._pcm) // 2
        threshold = 10 ** (self.threshold_dbfs / 20)

        while self._next_window_start + frame_samples <= total_samples:
            start = self._next_window_start
            start_byte = start * 2
            end_byte = (start + frame_samples) * 2
            samples = np.frombuffer(bytes(self._pcm[start_byte:end_byte]), dtype="<i2").astype(
                np.float32
            )
            samples /= 32768.0
            samples -= float(np.mean(samples))
            rms = float(np.sqrt(np.mean(samples * samples)))
            if rms >= threshold:
                if self._active_run_length == 0:
                    self._active_run_start = start
                self._active_run_length += 1
                if self._active_run_length >= self.consecutive_active_frames:
                    onset = self._active_run_start
                    assert onset is not None
                    self.onset_offset_ns = onset * NANOSECONDS_PER_SECOND // sample_rate
                    return self.onset_offset_ns
            else:
                self._active_run_start = None
                self._active_run_length = 0
            self._next_window_start += hop_samples
        return None


def pcm_duration_ns(pcm: bytes, pcm_format: PcmFormat) -> int:
    if len(pcm) % pcm_format.bytes_per_frame:
        raise ValueError("PCM bytes must contain complete frames")
    frames = len(pcm) // pcm_format.bytes_per_frame
    return frames * NANOSECONDS_PER_SECOND // pcm_format.sample_rate_hz


def evaluate_playback(
    chunks: list[AudioChunkObservation],
    *,
    complete: bool,
    audible_offset_ns: int | None,
) -> PlaybackRecord:
    """Map media time onto an immediate-playback client wall-clock timeline."""
    positive = [chunk for chunk in chunks if chunk.duration_ns > 0]
    if not positive:
        return PlaybackRecord(
            complete=complete,
            continuous=False,
            audio_duration_ns=0,
            total_stalled_ns=0,
            largest_stall_ns=0,
        )

    first_playable = positive[0].arrival_elapsed_ns
    playback_deadline = first_playable
    media_cursor = 0
    first_audible: int | None = None
    underruns: list[UnderrunRecord] = []

    for chunk in positive:
        if chunk.arrival_elapsed_ns > playback_deadline:
            stalled = chunk.arrival_elapsed_ns - playback_deadline
            underruns.append(
                UnderrunRecord(
                    started_elapsed_ns=playback_deadline,
                    recovered_elapsed_ns=chunk.arrival_elapsed_ns,
                    stalled_ns=stalled,
                )
            )
            playback_deadline = chunk.arrival_elapsed_ns
        if (
            audible_offset_ns is not None
            and first_audible is None
            and media_cursor <= audible_offset_ns < media_cursor + chunk.duration_ns
        ):
            first_audible = playback_deadline + audible_offset_ns - media_cursor
        playback_deadline += chunk.duration_ns
        media_cursor += chunk.duration_ns

    if audible_offset_ns is not None and first_audible is None:
        raise AudioProtocolError("audible onset falls outside the observed PCM stream")
    total_stalled = sum(item.stalled_ns for item in underruns)
    largest_stall = max((item.stalled_ns for item in underruns), default=0)
    continuous = not underruns
    return PlaybackRecord(
        complete=complete,
        continuous=continuous,
        first_playable_elapsed_ns=first_playable,
        first_audible_elapsed_ns=first_audible,
        audible_offset_ns=audible_offset_ns,
        playback_end_elapsed_ns=playback_deadline,
        audio_duration_ns=media_cursor,
        underruns=tuple(underruns),
        total_stalled_ns=total_stalled,
        largest_stall_ns=largest_stall,
    )


def pcm_to_wav(pcm: bytes, pcm_format: PcmFormat) -> bytes:
    """Wrap canonical PCM in a finite WAVE container."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(pcm_format.channels)
        wav.setsampwidth(pcm_format.sample_width_bytes)
        wav.setframerate(pcm_format.sample_rate_hz)
        wav.writeframes(pcm)
    return output.getvalue()


def finite_or_none(value: float) -> float | None:
    """Small helper for safe serialization of calculated floats."""
    return value if math.isfinite(value) else None
