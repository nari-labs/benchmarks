from __future__ import annotations

import pytest

from tts_bench.models import PcmFormat, TargetKind, TransportKind
from tts_bench.targets import TARGET_CONTRACTS, TargetAdapter, TargetConfigurationError


def adapter(kind: TargetKind, **params: object) -> TargetAdapter:
    return TargetAdapter(
        kind=kind,
        base_url="http://localhost:8000/api/",
        model="model",
        voice="voice",
        language="English",
        expected_format=PcmFormat(sample_rate_hz=24_000),
        request_params=dict(params),
    )


def test_every_target_has_an_explicit_contract() -> None:
    assert set(TARGET_CONTRACTS) == set(TargetKind)


@pytest.mark.parametrize("kind", tuple(TargetKind))
def test_target_payload_contracts(kind: TargetKind) -> None:
    target = adapter(kind, temperature=0.2)
    body = target.build_request("hello")
    assert body["model"] == "model"
    assert body["input"] == "hello"
    assert body["stream"] is True
    assert body["non_streaming_mode"] is False
    contract = TARGET_CONTRACTS[kind]
    assert body["response_format"] == contract.response_format
    assert body.get("stream_format") == contract.stream_format
    assert target.stream_kind is contract.stream_kind
    assert target.speech_url == "http://localhost:8000/api/v1/audio/speech"


def test_canonical_request_fields_cannot_be_overridden() -> None:
    with pytest.raises(TargetConfigurationError, match="canonical fields"):
        adapter(TargetKind.VOXSERVE, stream=False)
    with pytest.raises(TargetConfigurationError, match="canonical fields"):
        adapter(TargetKind.MSTAR, non_streaming_mode=True)


def test_websocket_transport_is_nari_only_and_has_a_fixed_input_contract() -> None:
    target = TargetAdapter(
        kind=TargetKind.NARI,
        base_url="https://localhost:8000/api/",
        model="model",
        voice="voice",
        language="English",
        expected_format=PcmFormat(sample_rate_hz=24_000),
        request_params={"temperature": 0.2},
        transport=TransportKind.WEBSOCKET,
    )
    text = "  hello, world!  again"
    chunks = target.input_chunks(text)

    assert target.websocket_url == "wss://localhost:8000/api/v1/audio/speech/ws"
    assert chunks == ("  hello, ", "world!  ", "again")
    assert "".join(chunks) == text
    assert target.build_websocket_start() == {
        "type": "request.start",
        "model": "model",
        "voice": "voice",
        "language": "English",
        "response_format": "pcm",
        "temperature": 0.2,
    }
    request = target.build_request(text)
    assert request["input"] == {
        "chunk_unit": "word",
        "chunk_size": 1,
        "chunk_interval_ms": 10,
        "chunks": list(chunks),
        "end_sequence": 3,
    }

    with pytest.raises(TargetConfigurationError, match="only by nari"):
        TargetAdapter(
            kind=TargetKind.VOXSERVE,
            base_url="http://localhost:8000",
            model="model",
            voice="voice",
            language="English",
            expected_format=PcmFormat(sample_rate_hz=24_000),
            request_params={},
            transport=TransportKind.WEBSOCKET,
        )


def test_websocket_audio_metadata_is_strict() -> None:
    target = TargetAdapter(
        kind=TargetKind.NARI,
        base_url="http://localhost:8000",
        model="model",
        voice="voice",
        language="English",
        expected_format=PcmFormat(sample_rate_hz=24_000),
        request_params={},
        transport=TransportKind.WEBSOCKET,
    )
    assert target.validate_websocket_audio(
        {"encoding": "pcm_s16le", "sample_rate": 24_000, "channels": 1}
    ) == PcmFormat(sample_rate_hz=24_000)
    with pytest.raises(TargetConfigurationError, match="WebSocket audio metadata"):
        target.validate_websocket_audio(
            {"encoding": "pcm_s16le", "sample_rate": 16_000, "channels": 1}
        )


def test_sglang_requires_matching_sample_rate_header() -> None:
    target = adapter(TargetKind.SGLANG_OMNI)
    assert target.validate_response(
        {"Content-Type": "audio/pcm", "X-Sample-Rate": "24000"}
    ) == PcmFormat(sample_rate_hz=24_000)
    with pytest.raises(TargetConfigurationError, match="no x-sample-rate"):
        target.validate_response({"Content-Type": "audio/pcm"})
    with pytest.raises(TargetConfigurationError, match="received 16000"):
        target.validate_response({"Content-Type": "audio/pcm", "X-Sample-Rate": "16000"})


def test_content_type_is_validated_per_stream_kind() -> None:
    adapter(TargetKind.MSTAR).validate_response({"content-type": "audio/wav; rate=24000"})
    adapter(TargetKind.NARI).validate_response({"content-type": "audio/pcm"})
    adapter(TargetKind.VOXSERVE).validate_response({"content-type": "audio/pcm"})
    with pytest.raises(TargetConfigurationError, match="expected"):
        adapter(TargetKind.VOXSERVE).validate_response({"content-type": "audio/mpeg"})
