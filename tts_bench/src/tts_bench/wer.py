"""Post-rate Deepgram transcription and WER scoring."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import aiohttp
import jiwer
from whisper_normalizer.english import (  # type: ignore[import-untyped]
    EnglishTextNormalizer,
)

from tts_bench.artifacts import write_bytes, write_jsonl_models
from tts_bench.models import (
    DeepgramModelInfo,
    Prompt,
    RequestRecord,
    WERRecord,
    WERStatus,
)

DEEPGRAM_ENDPOINT = "https://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_LANGUAGE = "en"
NORMALIZATION_NAME = "whisper_normalizer.EnglishTextNormalizer"
NORMALIZATION_VERSION = "2"
WER_RECORDS_PATH = "wer.jsonl"

_normalizer = EnglishTextNormalizer()


@dataclass(frozen=True)
class DeepgramWEROptions:
    api_key: str = field(repr=False)
    endpoint: str = DEEPGRAM_ENDPOINT
    model: str = DEEPGRAM_MODEL
    language: str = DEEPGRAM_LANGUAGE
    concurrency: int = 8
    timeout_s: float = 60.0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("Deepgram API key cannot be empty")
        if self.model != DEEPGRAM_MODEL:
            raise ValueError(f"WER model must be {DEEPGRAM_MODEL!r}")
        if self.language != DEEPGRAM_LANGUAGE:
            raise ValueError(f"WER language must be {DEEPGRAM_LANGUAGE!r}")
        if self.concurrency < 1:
            raise ValueError("WER concurrency must be positive")
        if self.timeout_s <= 0:
            raise ValueError("WER timeout must be positive")
        if self.max_attempts < 1:
            raise ValueError("WER max_attempts must be positive")

    @classmethod
    def from_environment(cls) -> DeepgramWEROptions:
        api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
        if not api_key:
            raise ValueError("--wer requires the DEEPGRAM_API_KEY environment variable")
        return cls(api_key=api_key)

    def contract(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "provider": "deepgram",
            "endpoint": self.endpoint,
            "requested_model": self.model,
            "language": self.language,
            "punctuate": True,
            "smart_format": True,
            "normalization": NORMALIZATION_NAME,
            "normalization_version": NORMALIZATION_VERSION,
            "effect": "record_only",
            "cohort": "complete audible measurement requests",
            "concurrency": self.concurrency,
            "timeout_s": self.timeout_s,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True)
class WERComputation:
    normalized_reference: str
    normalized_hypothesis: str
    reference_words: int
    substitutions: int
    deletions: int
    insertions: int
    wer_ratio: float
    wer_percentage: float


@dataclass(frozen=True)
class _DeepgramResult:
    attempts: int
    http_status: int | None
    body: bytes | None
    payload: dict[str, Any] | None
    error_kind: str | None
    error_message: str | None


@dataclass(frozen=True)
class _EvaluationContext:
    run_id: str
    record: RequestRecord
    reference: str
    options: DeepgramWEROptions
    audio_path: str
    audio_sha256: str

    def missing(
        self,
        error_kind: str,
        error_message: str,
        *,
        result: _DeepgramResult | None = None,
        response_path: str | None = None,
        response_sha256: str | None = None,
    ) -> WERRecord:
        return WERRecord(
            run_id=self.run_id,
            request_id=self.record.request_id,
            prompt_id=self.record.prompt_id,
            status=WERStatus.MISSING,
            endpoint=self.options.endpoint,
            audio_path=self.audio_path,
            audio_sha256=self.audio_sha256,
            attempts=result.attempts if result is not None else 0,
            response_http_status=result.http_status if result is not None else None,
            response_path=response_path,
            response_sha256=response_sha256,
            reference=self.reference,
            error_kind=error_kind,
            error_message=_redact_secret(error_message, self.options.api_key),
        )


def normalize_text(text: str) -> str:
    """Normalize text with the English WER pipeline."""
    return cast(str, _normalizer(text))


def compute_wer(reference: str, hypothesis: str) -> WERComputation:
    """Compute normalized word errors with finite empty-reference behavior."""
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()

    if not reference_words:
        insertions = len(hypothesis_words)
        ratio = 0.0 if not hypothesis_words else float(insertions)
        return WERComputation(
            normalized_reference=normalized_reference,
            normalized_hypothesis=normalized_hypothesis,
            reference_words=0,
            substitutions=0,
            deletions=0,
            insertions=insertions,
            wer_ratio=ratio,
            wer_percentage=ratio * 100,
        )

    result = jiwer.process_words(normalized_reference, normalized_hypothesis)
    ratio = float(result.wer)
    return WERComputation(
        normalized_reference=normalized_reference,
        normalized_hypothesis=normalized_hypothesis,
        reference_words=len(reference_words),
        substitutions=int(result.substitutions),
        deletions=int(result.deletions),
        insertions=int(result.insertions),
        wer_ratio=ratio,
        wer_percentage=ratio * 100,
    )


async def evaluate_measurement_wer(
    *,
    output: Path,
    run_id: str,
    records: Sequence[RequestRecord],
    candidate_request_ids: frozenset[str],
    prompts: Sequence[Prompt],
    options: DeepgramWEROptions,
) -> tuple[list[RequestRecord], tuple[WERRecord, ...]]:
    """Score eligible measurement WAVs without extending TTS request timings."""
    prompt_by_id = {prompt.id: prompt for prompt in prompts}
    candidates = [record for record in records if record.request_id in candidate_request_ids]
    semaphore = asyncio.Semaphore(options.concurrency)
    timeout = aiohttp.ClientTimeout(total=options.timeout_s)
    connector = aiohttp.TCPConnector(limit=options.concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        evaluations = await asyncio.gather(
            *(
                _evaluate_request(
                    output=output,
                    run_id=run_id,
                    record=record,
                    reference=prompt_by_id[record.prompt_id].text,
                    session=session,
                    semaphore=semaphore,
                    options=options,
                )
                for record in candidates
            )
        )

    evaluation_by_request = {evaluation.request_id: evaluation for evaluation in evaluations}
    updated: list[RequestRecord] = []
    for record in records:
        audio_success = record.success if record.audio_success is None else record.audio_success
        evaluation = evaluation_by_request.get(record.request_id)
        if evaluation is None:
            updated.append(
                record.model_copy(
                    update={
                        "audio_success": audio_success,
                        "wer_status": WERStatus.NOT_APPLICABLE,
                        "wer_percentage": None,
                        "wer_record_path": None,
                    }
                )
            )
            continue

        updated.append(
            record.model_copy(
                update={
                    "audio_success": audio_success,
                    "wer_status": evaluation.status,
                    "wer_percentage": evaluation.wer_percentage,
                    "wer_record_path": WER_RECORDS_PATH,
                }
            )
        )

    write_jsonl_models(output / WER_RECORDS_PATH, evaluations)
    return updated, tuple(evaluations)


async def _evaluate_request(
    *,
    output: Path,
    run_id: str,
    record: RequestRecord,
    reference: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    options: DeepgramWEROptions,
) -> WERRecord:
    audio_path = record.wav_audio_path or ""
    audio_sha256 = record.wav_sha256 or ""
    context = _EvaluationContext(
        run_id=run_id,
        record=record,
        reference=reference,
        options=options,
        audio_path=audio_path,
        audio_sha256=audio_sha256,
    )
    absolute_audio = output / audio_path
    if not audio_path or not audio_sha256 or not absolute_audio.is_file():
        return context.missing(
            "missing_audio_artifact",
            "semantic-evaluation candidate has no persisted WAV evidence",
        )

    audio = await asyncio.to_thread(absolute_audio.read_bytes)
    if hashlib.sha256(audio).hexdigest() != audio_sha256:
        return context.missing(
            "audio_sha256_mismatch",
            "persisted WAV SHA-256 does not match requests.jsonl",
        )

    async with semaphore:
        result = await _transcribe(session=session, audio=audio, options=options)

    response_path: str | None = None
    response_sha256: str | None = None
    response_exposed_api_key = False
    if result.body is not None:
        persisted_body = result.body
        encoded_api_key = options.api_key.encode()
        if encoded_api_key in persisted_body:
            response_exposed_api_key = True
            persisted_body = persisted_body.replace(encoded_api_key, b"[REDACTED]")
        response_path = f"wer/deepgram-responses/{record.request_id}.response"
        response_sha256 = hashlib.sha256(persisted_body).hexdigest()
        await asyncio.to_thread(write_bytes, output / response_path, persisted_body)

    if response_exposed_api_key:
        return context.missing(
            "deepgram_response_secret_redacted",
            "Deepgram response contained authorization material and was redacted",
            result=result,
            response_path=response_path,
            response_sha256=response_sha256,
        )

    if result.payload is None:
        return context.missing(
            result.error_kind or "deepgram_invalid_response",
            result.error_message or "Deepgram returned no usable response",
            result=result,
            response_path=response_path,
            response_sha256=response_sha256,
        )

    try:
        transcript = _transcript(result.payload)
        computation = compute_wer(reference, transcript)
    except (KeyError, TypeError, ValueError) as error:
        return context.missing(
            "deepgram_invalid_response",
            str(error),
            result=result,
            response_path=response_path,
            response_sha256=response_sha256,
        )

    metadata = result.payload.get("metadata")
    metadata_mapping = metadata if isinstance(metadata, Mapping) else {}
    request_id = metadata_mapping.get("request_id")
    return WERRecord(
        run_id=run_id,
        request_id=record.request_id,
        prompt_id=record.prompt_id,
        status=WERStatus.EVALUATED,
        endpoint=options.endpoint,
        audio_path=audio_path,
        audio_sha256=audio_sha256,
        attempts=result.attempts,
        response_http_status=result.http_status,
        response_path=response_path,
        response_sha256=response_sha256,
        deepgram_request_id=str(request_id) if request_id is not None else None,
        resolved_models=_resolved_models(metadata_mapping),
        reference=reference,
        transcript=transcript,
        normalized_reference=computation.normalized_reference,
        normalized_hypothesis=computation.normalized_hypothesis,
        reference_words=computation.reference_words,
        substitutions=computation.substitutions,
        deletions=computation.deletions,
        insertions=computation.insertions,
        wer_ratio=computation.wer_ratio,
        wer_percentage=computation.wer_percentage,
    )


async def _transcribe(
    *,
    session: aiohttp.ClientSession,
    audio: bytes,
    options: DeepgramWEROptions,
) -> _DeepgramResult:
    params = {
        "model": options.model,
        "language": options.language,
        "punctuate": "true",
        "smart_format": "true",
    }
    headers = {
        "Authorization": f"Token {options.api_key}",
        "Content-Type": "audio/wav",
        "Accept-Encoding": "identity",
    }
    for attempt in range(1, options.max_attempts + 1):
        try:
            async with session.post(
                options.endpoint,
                params=params,
                data=audio,
                headers=headers,
            ) as response:
                body = await response.read()
                if 200 <= response.status < 300:
                    try:
                        payload = json.loads(body)
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        return _DeepgramResult(
                            attempts=attempt,
                            http_status=response.status,
                            body=body,
                            payload=None,
                            error_kind="deepgram_invalid_json",
                            error_message=str(error),
                        )
                    if not isinstance(payload, dict):
                        return _DeepgramResult(
                            attempts=attempt,
                            http_status=response.status,
                            body=body,
                            payload=None,
                            error_kind="deepgram_invalid_json",
                            error_message="Deepgram response must be a JSON object",
                        )
                    return _DeepgramResult(attempt, response.status, body, payload, None, None)

                retryable = response.status == 429 or response.status >= 500
                if retryable and attempt < options.max_attempts:
                    await asyncio.sleep(_retry_delay(response.headers, attempt))
                    continue
                return _DeepgramResult(
                    attempts=attempt,
                    http_status=response.status,
                    body=body,
                    payload=None,
                    error_kind="deepgram_http_error",
                    error_message=f"Deepgram HTTP {response.status}",
                )
        except (TimeoutError, aiohttp.ClientError) as error:
            if attempt < options.max_attempts:
                await asyncio.sleep(2 ** (attempt - 1))
                continue
            timed_out = isinstance(error, TimeoutError)
            return _DeepgramResult(
                attempts=attempt,
                http_status=None,
                body=None,
                payload=None,
                error_kind="deepgram_timeout" if timed_out else "deepgram_transport_error",
                error_message=(
                    _redact_secret(str(error), options.api_key)
                    or ("request timed out" if timed_out else type(error).__name__)
                ),
            )
    raise AssertionError("unreachable Deepgram retry state")


def _retry_delay(headers: Mapping[str, str], attempt: int) -> float:
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, min(30.0, float(retry_after)))
        except ValueError:
            pass
    return float(2 ** (attempt - 1))


def _transcript(payload: Mapping[str, Any]) -> str:
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("Deepgram response has no results object")
    channels = results.get("channels")
    if not isinstance(channels, list) or not channels or not isinstance(channels[0], Mapping):
        raise ValueError("Deepgram response has no channel result")
    alternatives = channels[0].get("alternatives")
    if (
        not isinstance(alternatives, list)
        or not alternatives
        or not isinstance(alternatives[0], Mapping)
    ):
        raise ValueError("Deepgram response has no transcript alternative")
    transcript = alternatives[0].get("transcript")
    if not isinstance(transcript, str):
        raise ValueError("Deepgram transcript is not a string")
    return transcript


def _resolved_models(metadata: Mapping[str, Any]) -> tuple[DeepgramModelInfo, ...]:
    model_info = metadata.get("model_info")
    raw_models = metadata.get("models")
    model_uuids = (
        [value for value in raw_models if isinstance(value, str) and value]
        if isinstance(raw_models, list)
        else []
    )
    if not isinstance(model_info, Mapping):
        return tuple(DeepgramModelInfo(uuid=uuid) for uuid in model_uuids)
    values: list[DeepgramModelInfo] = []
    for uuid, raw in model_info.items():
        if not isinstance(uuid, str) or not uuid or not isinstance(raw, Mapping):
            continue
        values.append(
            DeepgramModelInfo(
                uuid=uuid,
                name=_optional_string(raw.get("name")),
                version=_optional_string(raw.get("version")),
                arch=_optional_string(raw.get("arch")),
            )
        )
    known = {value.uuid for value in values}
    values.extend(DeepgramModelInfo(uuid=uuid) for uuid in model_uuids if uuid not in known)
    return tuple(values)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _redact_secret(value: str, secret: str) -> str:
    return value.replace(secret, "[REDACTED]") if secret else value
