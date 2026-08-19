"""Durable benchmark artifacts and background audio persistence."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tts_bench.models import DatasetManifest, EventRecord, Prompt, RequestRecord


@dataclass(frozen=True)
class _RequestJob:
    record: RequestRecord
    events: tuple[EventRecord, ...]
    raw: bytes
    wav: bytes | None


@dataclass(frozen=True)
class _EventsJob:
    events: tuple[EventRecord, ...]


_Job = _RequestJob | _EventsJob


class ArtifactStore:
    """Serialize request evidence without blocking the load scheduler on disk I/O."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._queue: asyncio.Queue[_Job | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        for phase in ("warmup", "measurement"):
            (self.root / "audio" / "raw" / phase).mkdir(parents=True, exist_ok=True)
            (self.root / "audio" / "wav" / phase).mkdir(parents=True, exist_ok=True)
        for name in ("events.jsonl", "requests.jsonl", "audio-manifest.jsonl"):
            _atomic_write_text(self.root / name, "")
        self._worker = asyncio.create_task(self._run(), name="artifact-writer")

    def enqueue_request(
        self,
        record: RequestRecord,
        events: Sequence[EventRecord],
        raw: bytes,
        wav: bytes | None,
    ) -> None:
        self._raise_if_failed()
        self._queue.put_nowait(_RequestJob(record, tuple(events), raw, wav))

    def enqueue_events(self, events: Sequence[EventRecord]) -> None:
        self._raise_if_failed()
        self._queue.put_nowait(_EventsJob(tuple(events)))

    async def close(self) -> None:
        if self._worker is None:
            return
        await self._queue.put(None)
        await self._worker
        self._worker = None
        self._raise_if_failed()

    async def _run(self) -> None:
        try:
            while (job := await self._queue.get()) is not None:
                await asyncio.to_thread(self._write_job, job)
        except BaseException as error:
            self._error = error

    def _write_job(self, job: _Job) -> None:
        if isinstance(job, _EventsJob):
            _append_models(self.root / "events.jsonl", job.events)
            return
        _append_models(self.root / "events.jsonl", job.events)
        _append_models(self.root / "requests.jsonl", (job.record,))
        if job.record.raw_audio_path is not None:
            raw_path = self.root / job.record.raw_audio_path
            raw_path.write_bytes(job.raw)
        if job.wav is not None and job.record.wav_audio_path is not None:
            wav_path = self.root / job.record.wav_audio_path
            wav_path.write_bytes(job.wav)
        manifest = {
            "request_id": job.record.request_id,
            "phase": job.record.phase.value,
            "raw_path": job.record.raw_audio_path,
            "wav_path": job.record.wav_audio_path,
            "raw_sha256": job.record.raw_sha256,
            "pcm_sha256": job.record.pcm_sha256,
            "wav_sha256": job.record.wav_sha256,
            "output_bytes": job.record.output_bytes,
            "audio_duration_ns": job.record.playback.audio_duration_ns,
        }
        _append_json(self.root / "audio-manifest.jsonl", (manifest,))

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"artifact writer failed: {self._error}") from self._error


def initialize_result_directory(root: Path) -> None:
    """Create a new result directory without ever reusing old evidence."""
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"output path already exists: {root}")
    root.mkdir(parents=True)


def write_dataset_snapshot(
    root: Path,
    prompts: Sequence[Prompt],
    manifest: DatasetManifest,
) -> None:
    lines = (
        json.dumps(
            {
                "id": prompt.id,
                "text": prompt.text,
                "language": prompt.language,
                "word_count": prompt.word_count,
                "source_index": prompt.source_index,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for prompt in prompts
    )
    _atomic_write_text(root / "dataset.jsonl", "".join(f"{line}\n" for line in lines))
    write_json(root / "dataset-manifest.json", manifest)


def write_json(path: Path, value: BaseModel | dict[str, Any]) -> None:
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    _atomic_write_text(path, text)


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl_models(path: Path, values: Sequence[BaseModel]) -> None:
    """Atomically persist a finite model sequence as JSON Lines."""
    text = "".join(
        json.dumps(value.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        for value in values
    )
    _atomic_write_text(path, text)


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    """Atomically persist a finite stream of JSON objects as JSON Lines."""
    text = "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values)
    _atomic_write_text(path, text)


def _append_models(path: Path, values: Sequence[BaseModel]) -> None:
    serialized = (value.model_dump(mode="json") for value in values)
    _append_json(path, serialized)


def _append_json(path: Path, values: Sequence[dict[str, Any]] | Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
