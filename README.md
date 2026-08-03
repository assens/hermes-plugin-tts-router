# Hermes Plugin: TTS Router

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) TTS plugin that
routes text-to-speech calls based on language detection:

- **Pure Bulgarian text** (Cyrillic, no Latin mixed in) → **bg-tts-v5-mlx**
  (native Apple Silicon MLX) with automatic fallback to Edge TTS
- **Mixed Bulgarian** (Cyrillic + Latin letters, e.g. code/brand names) →
  Edge TTS (Microsoft neural voices, free, no API key)
- **Everything else** → OpenAI TTS (or any OpenAI-compatible endpoint — Kokoro, OpenAI, DeepInfra, etc.)

This lets you keep a single `tts.provider: tts-router` in `config.yaml` and have
mixed-language output routed automatically — no manual switching.

## How It Works

The plugin analyses each TTS request using two signals (either triggers the
Bulgarian route):

1. **Cyrillic script** (primary) — any character in the Unicode Cyrillic block
   (U+0400–U+04FF and extensions). This is the most reliable signal.
2. **Bulgarian word forms** (secondary) — a curated set of ~88 common Bulgarian
   words (здравей, благодаря, моля, това, etc.) matched against normalized
   lowercase text. Catches edge cases where transliterated text appears without
   Cyrillic.

When Bulgarian is detected, it's further split into **pure** (Cyrillic only,
no Latin letters) vs **mixed** (Cyrillic + Latin):

- **Pure Bulgarian** → `bg-tts-v5-mlx` (native Apple Silicon MLX), falling
  back to Edge TTS on any failure.
- **Mixed Bulgarian** (contains Latin letters — e.g. code, brand names, URLs)
  → Edge TTS.
- **Non-Bulgarian** → OpenAI TTS using `tts.openai.*`.

Each backend reads its own config block independently — all routes stay fully
configurable.

### Bulgarian backends

The router routes Bulgarian to two backends (the `transformers`/bg-tts-v7
backend is kept in the codebase for testing but **not** routed to):

| Route | Backend | Description |
|-------|---------|-------------|
| Pure Bulgarian | `mlx` | **bg-tts-v5-mlx** (native Apple Silicon MLX, ~965MB model). Best quality for Bulgarian. Edge TTS is the automatic fallback on any failure. |
| Mixed Bulgarian | `edge` | Microsoft Edge neural voices (free, no heavy deps). Robust for text containing Latin letters. |

## Installation

```bash
# From GitHub shorthand
hermes plugins install assen-sharlandjiev/hermes-plugin-tts-router --enable

# Or full URL
hermes plugins install https://github.com/assen-sharlandjiev/hermes-plugin-tts-router --enable
```

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
    # bg-tts-v5-mlx backend options (used for pure Bulgarian text)
    mlx:
      checkpoint: ~/.hermes/models/bg-tts-v5-mlx   # model directory
      python: ~/.hermes/venvs/bg-tts-v5/bin/python  # dedicated Python>=3.12 venv
      speaker_id: 0                                 # 0=AI voice, 1=audiobook narrator
      temperature: 0.25
      top_k: 50
      top_p: 0.8
      rep_penalty: 1.1
      max_tokens: 2000
```

> Routing is automatic: pure Bulgarian (Cyrillic, no Latin) → `mlx`;
> mixed Bulgarian (contains Latin letters) → Edge TTS; everything else →
> OpenAI TTS. The `transformers`/bg-tts-v7 backend is kept in the codebase
> for testing only and is **not** part of the routing.

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
2026-08-03 19:20:54 INFO hermes_plugins.tts__tts_router: TTS Router: ROUTED → bg-tts-v5-mlx (MLX)  | text='Здравей, как сте?' (16 chars)
```

The log line for Edge routing includes the voice actually used (e.g.
`voice=bg-BG-KalinaNeural`), and warnings are emitted whenever the MLX
backend fails and falls back to Edge. To watch live:

```bash
tail -f ~/.hermes/logs/tts-router.log
```

## Setting Up the bg-tts-v5-mlx Backend

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

### Persistent server (recommended)

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
the model's bundled `sample_bg.wav`; pass your own for a different voice:

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
- **Implements** the Bulgarian backends itself:
  - `bg-tts-v5-mlx` via a dedicated MLX venv (`_generate_mlx_bg`, shells out
    to the model's own `tts_mlx` inference module)
  - `bg-tts-v7` via transformers (`_generate_transformers_bg`, kept for
    testing, not routed to)
  - Edge TTS via the built-in `_generate_edge_tts`, passing the full tts
    config so the configured Bulgarian voice (`bg-BG-KalinaNeural`) is used

## License

MIT — see [LICENSE](LICENSE).
