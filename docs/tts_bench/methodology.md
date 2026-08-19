# Methodology

A `tts-bench` result is identified by its benchmark revision, dataset snapshot, resolved run
configuration, and target deployment.

## 1. Dataset and prompt order

The canonical prompt pool is the 1,088-row English standard split of the
[Seed-TTS evaluation set](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval):

```bash
cd tts_bench
uv run bench dataset prepare
```

This downloads pinned revision `8f5e1aa2a35d42f42e940074c1983358b9491f89`, verifies it, and
projects sample IDs and target text without deduplication or length filtering. Each result stores its
exact dataset snapshot and manifest. Prompt order is deterministic by dataset, seed, phase, and phase
index, with independent warm-up and measurement orders.

## 2. Target and audio contract

The benchmark streams TTS output without transforming it. All HTTP targets receive the complete
prompt with `stream=true` and `non_streaming_mode=false`.

| Target | Additional request settings | Response |
|---|---|---|
| vLLM-Omni | `stream_format=audio`, `response_format=pcm` | raw PCM16/24 kHz |
| SGLang-Omni | `response_format=pcm` | raw PCM, `x-sample-rate: 24000` |
| VoxServe | `response_format=pcm` | raw PCM16/24 kHz |
| M* | `response_format=wav` | streaming WAV, PCM16/24 kHz |
| Nari HTTP | `response_format=pcm` | raw PCM16/24 kHz |
| Nari WebSocket | `nari.speech.v1`, one word every 10 ms | binary PCM16/24 kHz and JSON events |

Raw PCM is signed little-endian mono PCM16 at 24 kHz. Available format metadata is validated, and
resolved settings and requests are stored in `run.json` and `requests.jsonl`.

## 3. Open-loop load model

The benchmark uses an open-loop load model: request arrivals are scheduled independently of request
completion. Arrivals follow a Poisson process at the requested RPS.

## 4. Timing boundaries

DNS, TCP, TLS, and health-check timings are reported separately from request latency. Request time
zero is the first JSON body byte sent for HTTP, or immediately before the first `input_text.append`
for WebSocket. WebSocket setup is reported separately and excluded from TTFB.

| Metric | Definition |
|---|---|
| Scheduling lag | request dispatch − ideal arrival deadline |
| TTFB | first non-empty response body byte − dispatch |
| First playable | first complete decoded PCM sample frame − dispatch |
| Audible TTFA | first detected audible sample − dispatch |
| Leading silence | audible sample offset within decoded audio |
| E2E | response EOF or request completion − dispatch |
| Per-request RTF | complete-PCM request E2E duration ÷ generated audio duration |
| Received audio xRT | media received during the measurement window ÷ window duration |
| WebSocket setup | connection start to `request.configured` |
| Input ingress | last text append send − first text append send |
| Final-text latency | first playable, audible, or completion time − last text append send |

## Reproducing a result

1. Check out `benchmark_git_revision` from `run.json` and preserve the diff if
   `benchmark_git_dirty=true`.
2. Verify `dependency_lock_sha256`, then run `uv sync --all-groups` in `tts_bench/`.
3. Restore the saved dataset, target deployment, hardware state, and resolved `run.json` options.
4. For WER, configure `DEEPGRAM_API_KEY` and compare evaluator metadata and coverage.

Hardware, runtime state, network path, and background load can still affect latency.

## Caveats

- Measurements are client-observed and do not isolate model inference or server-side stages.
- The dataset and evaluator are English-specific, and results may not generalize to production.
