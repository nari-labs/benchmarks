"""Explicit adapters for the supported OpenAI-compatible speech endpoints."""

from __future__ import annotations

import asyncio
import re
import socket
import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter_ns
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from tts_bench.models import PcmFormat, SetupMetrics, StreamKind, TargetKind, TransportKind


class TargetConfigurationError(ValueError):
    """The selected target cannot honor the canonical request contract."""


class PreflightError(RuntimeError):
    """Target setup or health validation failed before load generation."""


_CANONICAL_FIELDS = frozenset(
    {
        "input",
        "model",
        "voice",
        "language",
        "response_format",
        "stream",
        "stream_format",
        "non_streaming_mode",
        "type",
    }
)

_PCM_CONTENT_TYPES = frozenset({"audio/pcm", "audio/l16", "application/octet-stream", "audio/raw"})
_WAV_CONTENT_TYPES = frozenset({"audio/wav", "audio/x-wav", "application/octet-stream"})
WEBSOCKET_PROTOCOL = "nari.speech.v1"
WEBSOCKET_INPUT_CHUNK_INTERVAL_S = 0.010


@dataclass(frozen=True)
class TargetContract:
    """Wire-level differences between supported speech servers."""

    response_format: str
    stream_kind: StreamKind
    content_types: frozenset[str]
    stream_format: str | None = None
    sample_rate_header: str | None = None


TARGET_CONTRACTS: dict[TargetKind, TargetContract] = {
    TargetKind.VLLM_OMNI: TargetContract(
        response_format="pcm",
        stream_kind=StreamKind.RAW_PCM,
        stream_format="audio",
        content_types=_PCM_CONTENT_TYPES,
    ),
    TargetKind.SGLANG_OMNI: TargetContract(
        response_format="pcm",
        stream_kind=StreamKind.RAW_PCM,
        sample_rate_header="x-sample-rate",
        content_types=_PCM_CONTENT_TYPES,
    ),
    TargetKind.VOXSERVE: TargetContract(
        response_format="pcm",
        stream_kind=StreamKind.RAW_PCM,
        content_types=_PCM_CONTENT_TYPES,
    ),
    TargetKind.MSTAR: TargetContract(
        response_format="wav",
        stream_kind=StreamKind.WAV,
        content_types=_WAV_CONTENT_TYPES,
    ),
    TargetKind.NARI: TargetContract(
        response_format="pcm",
        stream_kind=StreamKind.RAW_PCM,
        content_types=_PCM_CONTENT_TYPES,
    ),
}


@dataclass(frozen=True)
class TargetAdapter:
    """Target-specific payload and streaming response policy."""

    kind: TargetKind
    base_url: str
    model: str
    voice: str
    language: str
    expected_format: PcmFormat
    request_params: dict[str, Any]
    health_path: str = "/health"
    transport: TransportKind = TransportKind.HTTP

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise TargetConfigurationError("base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise TargetConfigurationError(
                "base_url cannot include credentials, query parameters, or a fragment"
            )
        overlap = _CANONICAL_FIELDS.intersection(self.request_params)
        if overlap:
            raise TargetConfigurationError(
                f"request-param cannot override canonical fields: {sorted(overlap)}"
            )
        if self.transport is TransportKind.WEBSOCKET and self.kind is not TargetKind.NARI:
            raise TargetConfigurationError("WebSocket transport is supported only by nari")

    @property
    def speech_url(self) -> str:
        return _join_url(self.base_url, "/v1/audio/speech")

    @property
    def websocket_url(self) -> str:
        parsed = urlsplit(_join_url(self.base_url, "/v1/audio/speech/ws"))
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunsplit((scheme, parsed.netloc, parsed.path, "", ""))

    @property
    def health_url(self) -> str:
        return _join_url(self.base_url, self.health_path)

    @property
    def contract(self) -> TargetContract:
        return TARGET_CONTRACTS[self.kind]

    @property
    def stream_kind(self) -> StreamKind:
        return self.contract.stream_kind

    def build_request(self, text: str) -> dict[str, Any]:
        if self.transport is TransportKind.WEBSOCKET:
            chunks = self.input_chunks(text)
            return {
                "protocol": WEBSOCKET_PROTOCOL,
                "path": "/v1/audio/speech/ws",
                "start": self.build_websocket_start(),
                "input": {
                    "chunk_unit": "word",
                    "chunk_size": 1,
                    "chunk_interval_ms": 10,
                    "chunks": list(chunks),
                    "end_sequence": len(chunks),
                },
            }
        common: dict[str, Any] = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "language": self.language,
            "stream": True,
            "non_streaming_mode": False,
        }
        common["response_format"] = self.contract.response_format
        if self.contract.stream_format is not None:
            common["stream_format"] = self.contract.stream_format
        return {**common, **self.request_params}

    def request_template(self) -> dict[str, Any]:
        return self.build_request("<prompt>")

    def build_websocket_start(self) -> dict[str, Any]:
        if self.transport is not TransportKind.WEBSOCKET:
            raise TargetConfigurationError("WebSocket start requires WebSocket transport")
        return {
            "type": "request.start",
            "model": self.model,
            "voice": self.voice,
            "language": self.language,
            "response_format": "pcm",
            **self.request_params,
        }

    def input_chunks(self, text: str) -> tuple[str, ...]:
        if self.transport is not TransportKind.WEBSOCKET:
            raise TargetConfigurationError("input chunks require WebSocket transport")
        words = tuple(re.finditer(r"\S+", text))
        if not words:
            raise TargetConfigurationError("WebSocket input must contain a word")
        boundaries = [0, *(word.start() for word in words[1:]), len(text)]
        chunks = tuple(text[start:end] for start, end in pairwise(boundaries) if start < end)
        if not chunks or "".join(chunks) != text:
            raise TargetConfigurationError("WebSocket word chunks must preserve the exact prompt")
        return chunks

    def validate_websocket_audio(self, value: object) -> PcmFormat:
        expected = {
            "encoding": "pcm_s16le",
            "sample_rate": self.expected_format.sample_rate_hz,
            "channels": self.expected_format.channels,
        }
        if value != expected:
            raise TargetConfigurationError(
                f"expected WebSocket audio metadata {expected}, received {value!r}"
            )
        return self.expected_format

    def validate_response(self, headers: Mapping[str, str]) -> PcmFormat:
        """Validate content type and resolve target-provided sample-rate metadata."""
        lowered = {str(key).lower(): str(value) for key, value in headers.items()}
        content_type = lowered.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in self.contract.content_types:
            raise TargetConfigurationError(
                f"expected {sorted(self.contract.content_types)}, "
                f"received {content_type or 'no content type'}"
            )

        sample_rate_header = self.contract.sample_rate_header
        if sample_rate_header is not None:
            raw_rate = lowered.get(sample_rate_header)
            if raw_rate is None:
                raise TargetConfigurationError(
                    f"{self.kind.value} response has no {sample_rate_header} header"
                )
            try:
                observed_rate = int(raw_rate)
            except ValueError as error:
                raise TargetConfigurationError(f"invalid {sample_rate_header} header") from error
            if observed_rate != self.expected_format.sample_rate_hz:
                raise TargetConfigurationError(
                    f"expected {self.expected_format.sample_rate_hz} Hz, received {observed_rate} Hz"
                )
        return self.expected_format


def _join_url(base_url: str, path: str) -> str:
    parsed = urlsplit(base_url)
    base_path = parsed.path.rstrip("/")
    suffix = "/" + path.lstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, base_path + suffix, "", ""))


async def preflight_target(
    adapter: TargetAdapter,
    *,
    timeout_s: float,
) -> SetupMetrics:
    """Measure DNS/TCP/TLS separately, then require a healthy HTTP endpoint."""
    parsed = urlsplit(adapter.base_url)
    host = parsed.hostname
    assert host is not None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    loop = asyncio.get_running_loop()

    dns_started = perf_counter_ns()
    try:
        addresses = await asyncio.wait_for(
            loop.getaddrinfo(host, port, type=socket.SOCK_STREAM), timeout=timeout_s
        )
    except (OSError, TimeoutError) as error:
        raise PreflightError(f"DNS resolution failed for {host}: {error}") from error
    dns_ms = (perf_counter_ns() - dns_started) / 1_000_000
    if not addresses:
        raise PreflightError(f"DNS resolution returned no addresses for {host}")

    family, sock_type, proto, _canonical, sockaddr = addresses[0]
    sock = socket.socket(family, sock_type, proto)
    sock.setblocking(False)
    writer: asyncio.StreamWriter | None = None
    try:
        tcp_started = perf_counter_ns()
        await asyncio.wait_for(loop.sock_connect(sock, sockaddr), timeout=timeout_s)
        tcp_ms = (perf_counter_ns() - tcp_started) / 1_000_000
        tls_ms: float | None = None
        if parsed.scheme == "https":
            tls_started = perf_counter_ns()
            _reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    sock=sock,
                    ssl=ssl.create_default_context(),
                    server_hostname=host,
                ),
                timeout=timeout_s,
            )
            tls_ms = (perf_counter_ns() - tls_started) / 1_000_000
        else:
            sock.close()
    except (OSError, TimeoutError, ssl.SSLError) as error:
        sock.close()
        raise PreflightError(f"connection setup failed for {host}:{port}: {error}") from error
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()

    health_started = perf_counter_ns()
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(adapter.health_url) as response,
        ):
            await response.read()
            health_status = response.status
    except (aiohttp.ClientError, TimeoutError) as error:
        raise PreflightError(f"healthcheck failed: {error}") from error
    health_ms = (perf_counter_ns() - health_started) / 1_000_000
    if not 200 <= health_status < 300:
        raise PreflightError(f"healthcheck returned HTTP {health_status}")
    return SetupMetrics(
        dns_ms=dns_ms,
        tcp_ms=tcp_ms,
        tls_ms=tls_ms,
        healthcheck_ms=health_ms,
        healthcheck_status=health_status,
    )
