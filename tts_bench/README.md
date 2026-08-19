# tts-bench

`tts-bench` is an asyncio open-loop load generator for streaming text-to-speech servers. It uses
`POST /v1/audio/speech` for every target and can additionally use the Nari stack's
`WS /v1/audio/speech/ws` live-text protocol.

The benchmark measures load, latency, immediate-playback behavior, and optional transcript quality.
With `--wer`, it records WER for completed measurement audio through Deepgram Nova-3 after all TTS
timing has finished. WER is record-only telemetry and does not change request success, semantic
status, or process exit status. The benchmark does not collect GPU telemetry. Audible TTFA p95 of
50 ms and zero underruns are printed as informational references only; they never select a passing
RPS or change the process exit status.

See [`../docs/tts_bench/methodology.md`](../docs/tts_bench/methodology.md) for the complete
measurement and reproducibility contract.

## Install

Python 3.12 or newer is required. With [uv](https://docs.astral.sh/uv/):

```bash
cd tts_bench
uv sync --all-groups
uv run bench --version
```

The installed console executable is `bench`.

## Dataset

Prepare the complete 1,088-row English Seed-TTS prompt set before the first run:

```bash
uv run bench dataset prepare
```

This downloads the pinned `en/meta.lst` from
[`zhaochenyang20/seed-tts-eval`](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval)
and writes:

```text
datasets/seed-tts-eval.jsonl
datasets/seed-tts-eval.provenance.json
```

Each line contains only `id` and `text`; embedded source audio is omitted. The upstream revision,
source hash, license identifier, and projection hash are recorded in the provenance file. The
preparation command verifies the downloaded source, projected JSONL, row count, and prompt-pool
hash against pinned values. The canonical dataset contains all 1,088 English source rows and
performs no text deduplication or length filtering.

The source data is not distributed in this repository or the Python package. Its Hugging Face
dataset card identifies it as CC BY 4.0 and credits the original
[`BytedanceSpeech/seed-tts-eval`](https://github.com/BytedanceSpeech/seed-tts-eval) evaluation set.
To replace an existing local projection intentionally, pass `--overwrite`.

Each rate uses a SHA-256-based, seed-versioned shuffle of the complete pool. Warm-up and measurement
have independent streams. A new hash order is generated with the epoch number when more than 1,088
prompts are needed. Consequently, the same seed and RPS produce the same measurement prompt sequence
for every target, regardless of how many warm-up requests ran.

Every result includes `dataset.jsonl`. That snapshot can be supplied directly in a later run:

```bash
uv run bench run \
  --target voxserve \
  --base-url http://server:8000 \
  --dataset results/previous-rate/dataset.jsonl \
  --rps 8 \
  --output results/replay
```

## Run

One rate:

```bash
uv run bench run \
  --target vllm-omni \
  --base-url http://server:8000 \
  --dataset datasets/seed-tts-eval.jsonl \
  --rps 8 \
  --output results/vllm-rps-8
```

Run each RPS point separately. For example:

```bash
uv run bench run \
  --target sglang-omni \
  --base-url http://server:8000 \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --voice Ryan \
  --language English \
  --dataset datasets/seed-tts-eval.jsonl \
  --rps 8 \
  --warmup 10s \
  --duration 30s \
  --arrival poisson \
  --seed 0 \
  --timeout 120s \
  --max-in-flight 4096 \
  --wer \
  --output results/sglang-omni-rps-8
```

The Nari target can instead stream text over WebSocket:

```bash
uv run bench run \
  --target nari \
  --transport websocket \
  --base-url http://server:8000 \
  --dataset datasets/seed-tts-eval.jsonl \
  --rps 8 \
  --output results/nari-websocket-rps-8
```

WebSocket text ingress is fixed by the benchmark: one whitespace-delimited word per append, an
absolute 10 ms cadence from the first append, and an acknowledgement before the next append. There
are no client options for changing the chunk unit, size, or interval. Other targets reject
`--transport websocket`.

`--rps` accepts exactly one positive number. Apply any cooldown, server restart, or target-specific
state reset outside `bench`, then invoke a new command with a new output directory for the next RPS.
This keeps rate points independent without embedding deployment operations in the benchmark client.

`--wer` requires `DEEPGRAM_API_KEY` in the process environment. The key is sent only in the
Deepgram Authorization header and is never persisted in benchmark artifacts, reports, or NOTE
commands. `tts-bench` does not load `.env` files, so set the variable in the shell or service that
runs `bench`:

```bash
DEEPGRAM_API_KEY=... uv run bench run ... --wer
```

The canonical evaluator is Deepgram's prerecorded
`POST https://api.deepgram.com/v1/listen` API with `model=nova-3`, `language=en`,
`punctuate=true`, and `smart_format=true`. Flux supports English but not prerecorded audio and is
therefore outside this evaluator contract. Requests time out after 60 seconds per attempt and retry
timeouts, transport failures, HTTP 429, and HTTP 5xx up to three attempts with bounded backoff.

`--arrival constant` is available as a diagnostic. Duration values accept `ms`, `s`, or `m`.
Additional top-level JSON body values may be repeated:

```bash
--request-param temperature=0.2 \
--request-param 'some_target_option={"enabled":true}'
```

Canonical fields (`input`, `model`, `voice`, `language`, `stream`, `non_streaming_mode`, `type`,
`response_format`, and `stream_format`) cannot be overridden. An output path is never reused, even if
a previous run is incomplete.

The adapters send these target-specific contracts:

| Target | Request | Response |
|---|---|---|
| vLLM-Omni | `stream=true`, `non_streaming_mode=false`, `stream_format=audio`, `response_format=pcm` | raw mono PCM16/24 kHz |
| SGLang-Omni | `stream=true`, `non_streaming_mode=false`, `response_format=pcm` | raw PCM; `x-sample-rate: 24000` required |
| VoxServe | `stream=true`, `non_streaming_mode=false`, `response_format=pcm` | raw mono PCM16 at configured 24 kHz |
| M* | `stream=true`, `non_streaming_mode=false`, `response_format=wav` | incrementally parsed streaming WAV |
| Nari (HTTP) | `stream=true`, `non_streaming_mode=false`, `response_format=pcm` | raw mono PCM16 at configured 24 kHz |
| Nari (WebSocket) | `nari.speech.v1`; word/1/10 ms `input_text.append` followed by `input_text.end` | binary mono PCM16/24 kHz plus JSON control events |

The health endpoint defaults to `/health` and can be changed with `--health-path`.

## Text input contract

HTTP transport sends the complete prompt in one `POST /v1/audio/speech` JSON request and sets
`non_streaming_mode=false`. Nari WebSocket transport negotiates `nari.speech.v1`, configures one
request, streams exact prompt-preserving word chunks with monotonically increasing sequence numbers,
and ends input with the next sequence number. Binary response audio may arrive before input ends.

## Load and timing contract

The benchmark uses deterministic open-loop arrivals, client-observed streaming latency, audible
onset detection, zero-buffer playback simulation, and optional post-rate WER. Results keep capacity,
playback continuity, audibility, and WER telemetry separate. Audio duration is reported as telemetry
and is never classified by a fixed length heuristic. Deepgram failures are reported as missing WER,
with coverage tracked separately.
See [`../docs/tts_bench/methodology.md`](../docs/tts_bench/methodology.md) for timing boundaries,
measurement semantics, aggregation rules, artifacts, and reproducibility guidance.

## Results

`--output` is the result directory for exactly one RPS point and is never reused.
Each completed rate contains:

```text
status.json
run.json
dataset-manifest.json
dataset.jsonl
arrivals.jsonl
measurement-prompt-sequence.jsonl
events.jsonl
requests.jsonl
summary.json
report.txt
audio-manifest.jsonl
wer.jsonl                         # with --wer
wer/deepgram-responses/*          # with --wer
audio/raw/{warmup,measurement}/*
audio/wav/{warmup,measurement}/*
```

`run.json` contains the package version, benchmark Git revision and dirty state, dependency lockfile
SHA-256, resolved configuration, dataset provenance and hashes, an example finalized body, the body
template, and versioned measurement policies. Every exact request body is in `requests.jsonl`.
Persisted artifact records use schema v1. Records keep transport completion, continuity success, WER
status/score, and final request success separate; no WER value is a pass/fail gate.
`arrivals.jsonl` records the ideal schedule and dropped arrivals; the measurement sequence file
records every non-dropped measurement-phase request, including pre-dispatch failures and late
dispatches. The audio manifest includes
SHA-256 values for raw responses, decoded PCM, and canonical WAV artifacts. `wer.jsonl` retains
reference and transcript text, normalized text, edit counts, requested and resolved Deepgram model
metadata, retry/error state, and SHA-256-addressed raw Deepgram responses. Summaries report PCM
completion, audibility, semantic quality, audio-duration distribution, WER candidate/evaluated
counts, coverage, and the request-level macro-mean distribution. WebSocket summaries also report
setup and input-ingress latency, latency relative to the final text chunk, and the fraction of audio
streams that began before input completion.

Re-render any saved result without contacting a server:

```bash
uv run bench report results/sglang-omni
```

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```
