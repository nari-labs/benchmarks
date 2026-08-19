"""Persisted JSONL dataset loading and deterministic prompt selection."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterator, Sequence
from pathlib import Path
from urllib.request import Request, urlopen

from tts_bench.models import DatasetManifest, Phase, Prompt

PROMPT_ORDER_VERSION = 1
SEEDTTS_REPOSITORY = "zhaochenyang20/seed-tts-eval"
SEEDTTS_REVISION = "8f5e1aa2a35d42f42e940074c1983358b9491f89"
SEEDTTS_SOURCE_FILE = "en/meta.lst"
SEEDTTS_SOURCE_URL = (
    f"https://huggingface.co/datasets/{SEEDTTS_REPOSITORY}/resolve/"
    f"{SEEDTTS_REVISION}/{SEEDTTS_SOURCE_FILE}"
)
SEEDTTS_SOURCE_SHA256 = "7cd29f701743e6c7033edbca764ddc0b979de8c4f4907eff5e5454682114661f"
SEEDTTS_PROJECTION_SHA256 = "c95cb482f71117cbc46ac4e3aa5eab5c199bb0386d9e5600d912e157da8d2866"
SEEDTTS_POOL_SHA256 = "661013d15bb15db122a947513d9e718fefd6c0d9f3dbd32ec1efebf13f6fecb1"
SEEDTTS_ROWS = 1_088
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][A-Za-z0-9]+)*")


class DatasetError(ValueError):
    """The configured prompt source is missing or invalid."""


def count_words(text: str) -> int:
    """Count normalized English word-like tokens."""
    normalized = unicodedata.normalize("NFKC", text)
    return len(_WORD_RE.findall(normalized))


def prepare_seedtts_dataset(output: Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Download and project the pinned Seed-TTS English manifest to JSONL."""
    resolved = output.expanduser().resolve()
    provenance_path = resolved.with_suffix(".provenance.json")
    existing = [path for path in (resolved, provenance_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise DatasetError(f"refusing to overwrite existing dataset files: {names}")

    request = Request(SEEDTTS_SOURCE_URL, headers={"User-Agent": "tts-bench/0.1"})
    try:
        with urlopen(request, timeout=60) as response:
            source = response.read()
    except OSError as error:
        raise DatasetError(f"failed to download Seed-TTS dataset: {error}") from error

    source_hash = hashlib.sha256(source).hexdigest()
    if source_hash != SEEDTTS_SOURCE_SHA256:
        raise DatasetError(
            "Seed-TTS source hash mismatch: "
            f"expected {SEEDTTS_SOURCE_SHA256}, received {source_hash}"
        )

    prompts = _parse_seedtts_manifest(source)
    projection = "".join(
        f"{json.dumps({'id': prompt.id, 'text': prompt.text}, ensure_ascii=False, separators=(',', ':'))}\n"
        for prompt in prompts
    ).encode()
    projection_hash = hashlib.sha256(projection).hexdigest()
    pool_hash = _pool_hash(prompts)
    if (
        len(prompts) != SEEDTTS_ROWS
        or projection_hash != SEEDTTS_PROJECTION_SHA256
        or pool_hash != SEEDTTS_POOL_SHA256
    ):
        raise DatasetError("Seed-TTS projection does not match the pinned dataset contract")

    provenance = {
        "schema_version": 1,
        "source": {
            "repository": SEEDTTS_REPOSITORY,
            "revision": SEEDTTS_REVISION,
            "file": SEEDTTS_SOURCE_FILE,
            "url": SEEDTTS_SOURCE_URL,
            "license": "CC-BY-4.0",
            "sha256": source_hash,
        },
        "projection": {
            "file": resolved.name,
            "fields": {"id": "utterance_id", "text": "target_text"},
            "rows": len(prompts),
            "sha256": projection_hash,
            "pool_sha256": pool_hash,
        },
    }
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(projection)
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return resolved, provenance_path


def _parse_seedtts_manifest(source: bytes) -> tuple[Prompt, ...]:
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DatasetError("Seed-TTS manifest is not valid UTF-8") from error
    prompts: list[Prompt] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 4:
            raise DatasetError(
                f"Seed-TTS manifest row {line_number} has {len(fields)} fields; expected 4"
            )
        prompts.append(_prompt(fields[0], fields[3], len(prompts)))
    _validate_unique_ids(prompts)
    return tuple(prompts)


def load_prompt_pool(path: Path) -> tuple[tuple[Prompt, ...], DatasetManifest]:
    """Load a checked-in or persisted JSONL prompt snapshot."""
    resolved = path.expanduser().resolve()
    if resolved.is_file() and resolved.suffix == ".jsonl":
        return _load_jsonl(resolved)
    raise DatasetError("dataset must be a .jsonl snapshot")


def _load_jsonl(path: Path) -> tuple[tuple[Prompt, ...], DatasetManifest]:
    prompts: list[Prompt] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetError(f"invalid JSON on {path}:{line_number}: {error}") from error
            sample_id = value.get("id", value.get("sample_id"))
            text = value.get("text", value.get("target_text"))
            prompts.append(_prompt(sample_id, text, len(prompts)))
    if not prompts:
        raise DatasetError(f"dataset snapshot is empty: {path}")
    _validate_unique_ids(prompts)
    source_hash = _sha256_file(path)
    manifest = DatasetManifest(
        kind="jsonl_snapshot",
        source_path=str(path),
        source_files=(path.name,),
        source_sha256=source_hash,
        prompt_count=len(prompts),
        pool_sha256=_pool_hash(prompts),
    )
    return tuple(prompts), manifest


def _prompt(sample_id: object, text: object, source_index: int) -> Prompt:
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise DatasetError(f"row {source_index} has no nonempty sample ID")
    if not isinstance(text, str) or not text.strip():
        raise DatasetError(f"row {source_index} has no nonempty target text")
    words = count_words(text)
    if words == 0:
        raise DatasetError(f"row {source_index} has no countable English words")
    return Prompt(
        id=sample_id,
        text=text,
        language="en",
        word_count=words,
        source_index=source_index,
    )


def _validate_unique_ids(prompts: Sequence[Prompt]) -> None:
    ids = [prompt.id for prompt in prompts]
    if len(ids) != len(set(ids)):
        raise DatasetError("dataset sample IDs must be unique")


def _pool_hash(prompts: Sequence[Prompt]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        value = {"id": prompt.id, "language": prompt.language, "text": prompt.text}
        digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class PromptPool:
    """Deterministic epoch-shuffled access to the complete source pool."""

    def __init__(self, prompts: Sequence[Prompt], *, seed: int) -> None:
        if not prompts:
            raise ValueError("prompt pool cannot be empty")
        self._prompts = tuple(prompts)
        self._seed = seed
        self._cache: dict[tuple[Phase, int], tuple[Prompt, ...]] = {}

    def at(self, phase: Phase, phase_index: int) -> Prompt:
        if phase_index < 0:
            raise ValueError("phase_index cannot be negative")
        epoch, offset = divmod(phase_index, len(self._prompts))
        return self._order(phase, epoch)[offset]

    def _order(self, phase: Phase, epoch: int) -> tuple[Prompt, ...]:
        key = (phase, epoch)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        prefix = (
            f"tts-bench/prompt-order/v{PROMPT_ORDER_VERSION}\0"
            f"{self._seed}\0{phase.value}\0{epoch}\0"
        ).encode()
        order = tuple(
            sorted(
                self._prompts,
                key=lambda prompt: hashlib.sha256(prefix + prompt.id.encode()).digest(),
            )
        )
        self._cache[key] = order
        return order

    def iter_phase(self, phase: Phase) -> Iterator[Prompt]:
        index = 0
        while True:
            yield self.at(phase, index)
            index += 1
