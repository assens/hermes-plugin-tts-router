# Hermes Plugin: TTS Router

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) TTS plugin that
routes text-to-speech calls based on language detection:

- **Bulgarian text** (Cyrillic, pure OR mixed with English/Latin) →
  **bg-tts-st** (Ani-Voice-API two-stage: Supertonic → BgTTS, M5 voice) with
  automatic fallback to Edge TTS
- **Everything else** → OpenAI TTS (or any OpenAI-compatible endpoint — Kokoro, OpenAI, DeepInfra, etc.)

This lets you keep a single `tts.provider: tts-router` in `config.yaml` and have
mixed-language output routed automatically — no manual switching.

> The `bg-tts-v5-mlx` and `transformers` (bg-tts-v7) backends are **not** routed
> to — both pure and mixed Bulgarian go to bg-tts-st. The MLX/transformers code
> and config remain in the plugin for testing / later re-enable.

## How It Works

The plugin analyses each TTS request using two signals (either triggers the
Bulgarian route):

1. **Cyrillic script** (primary) — any character in the Unicode Cyrillic block
   (U+0400–U+04FF and extensions). This is the most reliable signal.
2. **Bulgarian word forms** (secondary) — a curated set of ~88 common Bulgarian
   words (здравей, благодаря, моля, това, etc.) matched against normalized
   lowercase text. Catches edge cases where transliterated text appears without
   Cyrillic.

When Bulgarian is detected, the whole text (pure OR mixed with Latin) routes to
**bg-tts-st**:

- **Bulgarian (pure or mixed)** → `bg-tts-st` (Ani-Voice-API two-stage,
  M5 voice), falling back to Edge TTS on any failure.
- **Non-Bulgarian** → OpenAI TTS using `tts.openai.*`.

Each backend reads its own config block independently — all routes stay fully
configurable.

### Bulgarian backend

The router sends all Bulgarian text to a single backend (`transformers`/bg-tts-v7
and `bg-tts-v5-mlx` are kept in the codebase for testing but **not** routed to):

| Route | Backend | Description |
|-------|---------|-------------|
| Bulgarian (pure or mixed) | `bg-tts-st` | **Ani-Voice-API** two-stage (Supertonic → BgTTS, M5 voice). Handles pure Bulgarian and mixed BG/EN text well. Edge TTS is the automatic fallback on any failure. |

## Installation

```bash
# From GitHub shorthand
hermes plugins install assen-sharlandjiev/hermes-plugin-tts-router --enable

# Or full URL
hermes plugins install https://github.com/assens/hermes-plugin-tts-router --enable
```

## Prerequisites (beyond `hermes plugins install`)

The plugin install only copies the Python code — it does **not** set up the
Python venv, model weights, or server scripts that the bg-tts-st backend needs.
Follow these steps on a fresh Hermes instance:

### 1. Create the dedicated Python ≥ 3.13 venv

Hermes's own venv is Python 3.11, too old for `supertonic` and `miocodec`
(both require ≥ 3.12). Create a separate venv:

```bash
python3.13 -m venv ~/.hermes/venvs/ani-voice

env -u PYTHONPATH ~/.hermes/venvs/ani-voice/bin/pip install \
  "supertonic==1.3.1" \
  "bg-text-normalizer==1.1.0" \
  "num2cyrillic==1.0.0" \
  soundfile \
  "onnxruntime>=1.20.0" \
  "huggingface-hub" \
  torch torchaudio \
  "miocodec @ git+https://github.com/Aratako/MioCodec@main"
```

| Package | Purpose |
|---------|---------|
| `supertonic` | Stage 1 — voice-style reference generation (ONNX, CPU) |
| `bg-text-normalizer` | Bulgarian text normalization (numbers, dates) |
| `num2cyrillic` | Number → Cyrillic word conversion |
| `torch` / `torchaudio` | Stage 2 — BgTTS synthesis (MPS on Apple Silicon) |
| `miocodec` | Codec for BgTTS audio encoding/decoding (from git) |
| `onnxruntime` | Supertonic ONNX inference |
| `soundfile` | Audio I/O |

### 2. Download the Ani-Voice-API model (~148 MB)

```bash
~/.hermes/venvs/ani-voice/bin/hf download \
  beleata74/Ani-Voice-API \
  --local-dir ~/.hermes/models/Ani-Voice-API
```

This includes the BgTTS checkpoint (`BgTTS/checkpoint_inference.pt`, 146 MB,
step 258680), the inference source code, and demo WAVs.

### 3. Auto-downloaded models (no manual step)

These are fetched automatically on the **first synthesis request** (so the
first request will be slower):

- **MioCodec** (`Aratako/MioCodec-25Hz-24kHz`) — cached at
  `~/.cache/huggingface/hub/` by `miocodec`
- **Supertonic voice assets** — voice-style references for F1–F5, M1–M5,
  fetched by `supertonic` on first use

### 4. Install the server scripts

If you used `hermes plugins install`, the scripts are in the plugin's
`scripts/` directory. Copy them to `~/.hermes/scripts/`:

```bash
cp ~/.hermes/plugins/tts/tts-router/scripts/bg_tts_st_server.py ~/.hermes/scripts/
cp ~/.hermes/plugins/tts/tts-router/scripts/bg_tts_st_server.sh ~/.hermes/scripts/
chmod +x ~/.hermes/scripts/bg_tts_st_server.{py,sh}
```

The plugin auto-starts/stops the server on `port 8002` — you don't normally
need to run it manually, but the scripts must be at `~/.hermes/scripts/`.

### 5. OpenAI-compatible TTS endpoint (for English)

The English route needs an OpenAI-compatible TTS server on **port 8000**. On
this setup that's [oMLX](https://github.com/NousResearch/omlx) serving
`Kokoro-82M-bf16`. Alternatively, point `tts.openai.base_url` at any
OpenAI-compatible endpoint (real OpenAI API, DeepInfra, etc.).

### 6. `ffplay` (for test CLIs only)

The standalone evaluation CLIs (`ani_voice_cli.py`, `bg_tts_38m_cli.py`,
`bg_tts_v7_cli.py`) play audio via `ffplay`. Not needed for the router itself:

```bash
brew install ffmpeg
```

### Quick checklist

| Step | Required? | Size | Auto? |
|------|-----------|------|-------|
| Install plugin (`hermes plugins install`) | ✅ | small | — |
| Create `~/.hermes/venvs/ani-voice` (Python 3.13) | ✅ | ~3 GB (torch) | Manual |
| Download Ani-Voice-API model | ✅ | 148 MB | Manual |
| MioCodec model | ✅ | ~200 MB | Auto (first run) |
| Supertonic voice assets | ✅ | ~50 MB | Auto (first run) |
| Copy server scripts to `~/.hermes/scripts/` | ✅ | small | Manual |
| Configure `config.yaml` (see below) | ✅ | — | Manual |
| OpenAI/Kokoro endpoint (port 8000) | For English | varies | Separate install |
| `ffplay` (`brew install ffmpeg`) | Test CLIs only | ~100 MB | Manual |
| Edge TTS | Fallback | 0 | Built-in |

## Configuration

In `~/.hermes/config.yaml`:

```yaml
tts:
  provider: tts-router
  edge:
    voice: bg-BG-KalinaNeural        # Bulgarian voice (or any Edge voice)
  openai:
    model: Kokoro-82M-bf16           # or tts-1, tts-1-hd, etc.
    voice: af_sky
    streaming: true
    base_url: http://127.0.0.1:8000/v1   # OpenAI-compatible endpoint
    api_key: omlx-local              # or your real OpenAI key
    speed: 1
    response_format: opus
```

### Optional: Fine-tune detection

```yaml
tts:
  provider: tts-router
  tts_router:
    # Minimum Cyrillic characters before routing to BG (default: 1)
    min_cyrillic_chars: 1
    # Override the default Bulgarian word set (optional)
    bulgarian_words:
      - "здравей"
      - "благодаря"
      - "моля"
    # bg-tts-st (Ani-Voice-API) backend options (used for ALL Bulgarian text —
    # pure or mixed). This is now the single Bulgarian backend.
    bg_tts_st:
      voice_style: M5        # Supertonic voice: F1-F5 female, M1-M5 male
      speed: 1.6
      python: ~/.hermes/venvs/ani-voice/bin/python  # dedicated Python>=3.13 venv
      temperature: 0.7
      top_k: 250
      top_p: 0.95
      rep_penalty: 1.1
      max_new_tokens: 512
      server_url: http://127.0.0.1:8002
    # The bg-tts-v5-mlx and transformers (bg-tts-v7) backends are kept in the
    # codebase for testing / later re-enable but are NOT routed to.
```

> Routing is automatic: Bulgarian (pure OR mixed with Latin) → `bg-tts-st`;
> everything else → OpenAI TTS. Edge TTS is the automatic fallback for the
> Bulgarian route on any failure. The `transformers`/bg-tts-v7 and `mlx`
> backends are kept in the codebase for testing only and are **not** routed to.

After configuring, start a new session (`/new` or `/reset`) for the plugin to
load.

## Logging

The plugin writes every routing decision to its own log file (independent of
the Hermes desktop console, which does not reliably surface plugin logs):

```
~/.hermes/logs/tts-router.log
```

The file is a rotating log (5 MB, 3 backups, UTF-8) and doubles as a
per-request audit trail of which backend handled each piece of text:

```
2026-08-03 19:26:03 INFO hermes_plugins.tts__tts_router: TTS Router: ROUTED → Edge TTS (voice=bg-BG-KalinaNeural)  | text='English: The children are playing...' (86 chars)
2026-08-03 19:26:03 INFO hermes_plugins.tts__tts_router: TTS Router: generating Bulgarian via Edge TTS...
2026-08-03 19:20:54 INFO hermes_plugins.tts__tts_router: TTS Router: ROUTED → bg-tts-st (voice=M5)  | text='Здравей, как сте?' (16 chars)
```

The log line for the bg-tts-st route includes the voice actually used (e.g.
`voice=M5`), and warnings are emitted whenever the bg-tts-st backend fails
and falls back to Edge. To watch live:

```bash
tail -f ~/.hermes/logs/tts-router.log
```

## Setting Up the bg-tts-v5-mlx Backend (for reference — not routed)

> The router no longer sends Bulgarian text to bg-tts-v5-mlx (both pure and
> mixed Bulgarian now go to bg-tts-st). This section is kept for reference and
> in case you re-enable the MLX route later.

The bg-tts-v5-mlx backend is a **native Apple Silicon MLX** TTS model that runs
fully on-device. It needs its own dedicated venv (the model requires MLX, which
is separate from the Hermes runtime).

```bash
# Create a dedicated venv
python3.13 -m venv ~/.hermes/venvs/bg-tts-v5

# Install the MLX stack
env -u PYTHONPATH ~/.hermes/venvs/bg-tts-v5/bin/pip install \
  mlx soundfile numpy \
  "nanocodec-mlx @ git+https://github.com/nineninesix-ai/nanocodec-mlx.git"

# Download the model (~965MB)
env -u PYTHONPATH ~/.hermes/venvs/bg-tts-v5/bin/huggingface-cli download \
  raditotev/bg-tts-v5-mlx --local-dir ~/.hermes/models/bg-tts-v5-mlx
```

The model's codec (`nineninesix/nemo-nano-codec-22khz...`) is downloaded
automatically on first synthesis.

### Persistent server for bg-tts-v5-mlx (recommended)

By default the plugin reloads the ~965MB model on *every* Bulgarian prompt
(spawning a subprocess per request). To avoid that overhead, a **persistent
server** keeps the model resident across requests:

```bash
# Start the server (loads model on first request, unloads after 5 min idle)
~/.hermes/scripts/bg_tts_server.sh start

# Other commands
~/.hermes/scripts/bg_tts_server.sh status
~/.hermes/scripts/bg_tts_server.sh restart
~/.hermes/scripts/bg_tts_server.sh stop
```

The server:
- Runs on **`http://127.0.0.1:8001`** (configurable via
  `tts.tts_router.mlx.server_url`)
- Loads `bg-tts-v5-mlx` + codec **once** on the first request
- Keeps them resident for subsequent requests (near-instant)
- **Auto-unloads after 5 minutes idle** (configurable via
  `--idle-timeout`), reloading lazily on the next request

The tts-router **tries the server first**; if it's not running it transparently
falls back to the subprocess-per-request path. So the server is an
optimization, not a requirement — Bulgarian TTS works either way.

### Auto lifecycle management

The plugin manages the server for you automatically:

- **On load** (`register()`) the plugin starts the server if it isn't already
  running (idempotent — no-op if healthy).
- **On exit** (Hermes process shutdown) the plugin stops the server via an
  `atexit` hook, releasing the model's memory.

So you normally don't need to run the launcher manually — Hermes starts and
stops the server with itself. The launcher remains useful for manual control
(e.g. keeping the server hot across Hermes restarts, or checking status).

> **Note:** This backend runs entirely on Apple Silicon via MLX. It is not
> served through oMLX — oMLX's built-in TTS engine doesn't support the custom
> `bg-tts-v5-mlx` architecture, so the plugin calls the model's own inference
> module directly.

**If the backend can't load or fails for any reason, the router transparently
falls back to Edge TTS** for Bulgarian text, so the plugin keeps working even
before the model stack is fully set up.

### Persistent server for bg-tts-st (Ani-Voice-API)

The mixed-Bulgarian backend (`bg-tts-st`) also runs as a **persistent server**
that keeps Supertonic + BgTTS resident across requests (avoids reloading both
models on every mixed BG/EN prompt):

```bash
# Start the server (voice=M5, loads models on first request, unloads after 5 min idle)
~/.hermes/scripts/bg_tts_st_server.sh start

# Other commands
~/.hermes/scripts/bg_tts_st_server.sh status
~/.hermes/scripts/bg_tts_st_server.sh restart
~/.hermes/scripts/bg_tts_st_server.sh stop
```

The server:
- Runs on **`http://127.0.0.1:8002`** (configurable via
  `tts.tts_router.bg_tts_st.server_url`)
- Loads Supertonic + BgTTS **once** on the first request
- Two-stage pipeline: Supertonic renders a reference clip in the configured
  voice style (default **M5**), then BgTTS uses it as the speaker embedding
- **Auto-unloads after 5 minutes idle**, reloading lazily on the next request
- Applies Bulgarian text normalization (numbers/dates) automatically

The tts-router **tries the bg-tts-st server first**, falling back to the
subprocess-per-request path if it's not running, and to Edge TTS if both fail.

#### Streaming (`/stream`)

The server also exposes a **streaming endpoint** for "hear-as-generated"
playback. It returns per-sentence audio chunks as NDJSON (HTTP chunked), one
line per sentence as each is generated:

```bash
curl -sN -X POST http://127.0.0.1:8002/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Здравейте! Това е първото изречение. Ето второто.", "voice_style": "M5"}'
```

Each NDJSON line is `{"chunk_index": i, "pcm_base64": "...", "duration": 0.0}`
where `pcm_base64` is **raw little-endian int16 mono 24 kHz PCM** — ready for
streaming playback without any conversion.

The tts-router plugin **registers a streaming provider** (under the name
`tts-router`) with Hermes's streaming-TTS machinery. The provider routes by
language (same `classify_text` logic as `synthesize`): Bulgarian/mixed →
bg-tts-st `/stream`, English/other → delegates to the OpenAI streaming
provider (Kokoro).

#### Where streaming TTS is available

Streaming playback (hear audio while the LLM is still generating) depends on
the Hermes surface — not all adapters implement the streaming-TTS contract:

| Surface | Streaming TTS | How it works |
|---------|:-------------:|--------------|
| **TUI** (`hermes tui`) | ✅ Yes | `tui_gateway/server.py` → `stream_tts_to_speaker` → `resolve_streaming_provider` → our `BgTtsStStreamer`. Plays each sentence via tempfile→afplay as it's generated. Activate with `HERMES_VOICE_TTS=1` env var + `/voice on`. |
| **CLI** (`hermes` voice mode) | ✅ Yes | `hermes_cli/voice.py` → `speak_text()` → `stream_tts_to_speaker` → our streamer. Same sentence-overlap pipeline as TUI. |
| **Telegram / messaging gateway** | ✅ Yes | `StreamingTTSConsumer` feeds PCM chunks to the adapter's `write_streaming_tts` while the LLM generates. Requires the adapter to override `supports_streaming_tts` (Telegram does). |
| **Desktop (Electron app)** | ❌ No | The `APIServerAdapter` inherits `supports_streaming_tts() → False` and never overrides it. The streaming consumer is created but silently exits; the whole-file `synthesize()` path runs instead (audio plays after the LLM finishes). **No config option or env var activates streaming on the desktop** — it would require the adapter to implement the streaming-TTS methods (a core Hermes change). |

> **Desktop users:** TTS works correctly on the desktop — it just plays the
> complete audio after the response is generated, not incrementally during
> generation. For "hear-as-generated" playback, use `hermes tui` in voice mode.

#### Streaming sample-rate detail

BgTTS synthesizes at 24 kHz (mono, int16). The streaming provider advertises
`sample_rate=24000, channels=1, sample_width=2` so Hermes's playback layer
opens the correct sounddevice stream (or writes an accurate temp WAV).

## Setting Up the bg-tts-v7 Backend (Alternative)

The bg-tts-v7 transformers backend needs `torch`, `transformers`, and
`miocodec`. Because `miocodec` requires **Python >= 3.12** (the Hermes runtime
venv is Python 3.11), these run in a separate dedicated venv that the plugin
shells out to.

```bash
# Create a Python 3.12+ venv (adjust version to what's available)
python3.13 -m venv ~/.hermes/venvs/bg-tts

# Install the stack (run with PYTHONPATH cleared so it stays self-contained)
env -u PYTHONPATH ~/.hermes/venvs/bg-tts/bin/pip install \
  "transformers<5" soundfile \
  "miocodec @ git+https://github.com/Aratako/MioCodec@main" \
  accelerate

# Download a speaker preset (voice identity) — no Bulgarian one exists, so
# we use the English female reference from MioTTS-Inference
mkdir -p ~/.hermes/venvs/bg-tts/presets
curl -L -o ~/.hermes/venvs/bg-tts/presets/en_female.pt \
  https://raw.githubusercontent.com/Aratako/MioTTS-Inference/main/presets/en_female.pt
```

On the first TTS call the model (~360 MB) and codec are downloaded from the
Hugging Face Hub automatically. Subsequent calls use the local cache.

**If the backend can't load or fails for any reason, the router transparently
falls back to Edge TTS** for Bulgarian text, so the plugin keeps working even
before the model stack is fully set up.

### Testing the bg-tts-v7 backend with the CLI

To experiment with the transformers model in isolation (without going through
the router), use the bundled test CLI. It synthesizes the text and **plays it
back with ffplay** (no file is saved):

```bash
~/.hermes/scripts/bg_tts_v7_cli.py "Здравейте, как сте днес?"
```

It shells out to the same `bg-tts` venv and synthesis logic the plugin's
(disabled) transformers backend uses, but **does not touch routing** — it's a
pure test harness. Options mirror the transformers config block:

```bash
~/.hermes/scripts/bg_tts_v7_cli.py "Текст" --temperature 0.9      # sampling
~/.hermes/scripts/bg_tts_v7_cli.py "Текст" --max-new-tokens 800   # length
~/.hermes/scripts/bg_tts_v7_cli.py --help                          # all options
```

Run `--help` for the full option list (model/codec/preset overrides, dtype,
top-p, repetition penalty, speech offset).

### Testing BgTTS-38M with the CLI

[`beleata74/BgTTS-38M`](https://huggingface.co/beleata74/BgTTS-38M) is a
lightweight **38M-param** Bulgarian/English encoder-decoder TTS with zero-shot
voice cloning via MioCodec. Unlike bg-tts-v7 it has a proper end-of-speech
token, so it terminates cleanly. Use the bundled test CLI to evaluate it
(does **not** touch routing):

```bash
# First download the model (once)
~/.hermes/venvs/bg-tts/bin/huggingface-cli download \
  beleata74/BgTTS-38M --local-dir ~/.hermes/models/BgTTS-38M

# Synthesize & play (uses the bundled Bulgarian sample as the speaker)
~/.hermes/scripts/bg_tts_38m_cli.py "Това е тест на българския синтез на реч."
```

The model needs a **reference speaker** (voice cloning). By default it uses
the model's bundled `sample_mixed.wav`; pass your own for a different voice:

```bash
# Clone from your own reference WAV (3-10s of clean speech)
~/.hermes/scripts/bg_tts_38m_cli.py "Здравейте!" --speaker-wav my_voice.wav

# English text (works too, model trained on BG + EN)
~/.hermes/scripts/bg_tts_38m_cli.py "Hello there." --speaker-wav en_ref.wav

# Use a pre-saved embedding (.pt or .npy) instead of a WAV
~/.hermes/scripts/bg_tts_38m_cli.py "Текст" --speaker-emb my_voice_emb.pt

# Tune sampling
~/.hermes/scripts/bg_tts_38m_cli.py "Текст" --temperature 0.5   # stable
~/.hermes/scripts/bg_tts_38m_cli.py "Текст" --temperature 0.8   # expressive
```

Run `--help` for the full option list (device, top-k/top-p, repetition
penalty, max tokens). The model runs on Apple Silicon via **MPS** by default
(`--device mps`; also `cpu`/`cuda`). It works best with short sentences (up
to ~8s / 200 tokens) — split longer text.

### Testing Ani-Voice-API with the CLI

[`beleata74/Ani-Voice-API`](https://huggingface.co/beleata74/Ani-Voice-API) is
a **two-stage** Bulgarian TTS: Supertonic generates a reference clip in a
chosen voice style, then BgTTS uses it as the speaker embedding. It also runs
Bulgarian text normalization so numbers/dates are read correctly. Use the
bundled test CLI to evaluate it (does **not** touch routing):

```bash
# First download the model (once)
~/.hermes/venvs/ani-voice/bin/hf download \
  beleata74/Ani-Voice-API --local-dir ~/.hermes/models/Ani-Voice-API

# Synthesize & play (uses the F5 female voice by default)
~/.hermes/scripts/ani_voice_cli.py "Здравейте! Това е тест на гласа."
```

Pick a voice style (**F1-F5** female, **M1-M5** male):

```bash
~/.hermes/scripts/ani_voice_cli.py "Текст" --voice-style F2  # female
~/.hermes/scripts/ani_voice_cli.py "Текст" --voice-style M1  # male
```

Numbers/dates are normalized automatically (e.g. "15 май 2026" is read in
full). Run `--help` for the full option list (speed, device, temperature,
top-k/top-p, repetition penalty, max tokens). The model runs on Apple Silicon
via **MPS** by default.

## Requirements

- `edge-tts` Python package (for the Bulgarian route): `pip install edge-tts`
- `openai` Python package (for the default route): `pip install openai`
- An OpenAI-compatible TTS endpoint (local Kokoro, OpenAI API, DeepInfra, etc.)
- No API key needed for Edge TTS (free, uses Microsoft's public endpoint)

## Adapting for Other Languages

The router pattern works for any language pair. To adapt it:

1. Change the Cyrillic detection regex to match your target script (e.g.,
   Greek U+0370–U+03FF, Arabic U+0600–U+06FF, CJK U+4E00–U+9FFF)
2. Replace the `bulgarian_words` set with your target language's common words
3. Configure the appropriate Edge TTS voice for that language
4. Rebuild and reinstall

The `is_bulgarian()` function and `_DEFAULT_BULGARIAN_WORDS` set in
`__init__.py` are the only language-specific parts — everything else is
generic routing logic.

## How It Hooks Into Hermes

The plugin registers a `TTSProvider` (via the
[`agent.tts_provider.TTSProvider`](https://hermes-agent.nousresearch.com/docs) ABC)
under the name `tts-router`. When `tts.provider: tts-router` is set in
`config.yaml`, Hermes's TTS dispatcher:

1. Checks if `tts-router` is a built-in → no
2. Checks for a command-type provider → no
3. Checks the plugin TTS registry → **found** → dispatches to
   `TTSRouterProvider.synthesize()`

The plugin then:
- **Routes** non-Bulgarian text to the built-in `_generate_openai_tts`
- **Implements** the Bulgarian backend itself:
  - `bg-tts-st` (Ani-Voice-API two-stage, M5 voice) — the single routed
    Bulgarian backend (`_generate_st_bg`, via the persistent server on 8002
    or subprocess fallback)
  - `bg-tts-v5-mlx` via a dedicated MLX venv (`_generate_mlx_bg`, kept for
    reference, not routed to)
  - `bg-tts-v7` via transformers (`_generate_transformers_bg`, kept for
    testing, not routed to)
  - Edge TTS via the built-in `_generate_edge_tts`, passing the full tts
    config so the configured Bulgarian voice (`bg-BG-KalinaNeural`) is used

## License

MIT — see [LICENSE](LICENSE).
