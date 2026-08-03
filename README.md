# Hermes Plugin: TTS Router

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) TTS plugin that
routes text-to-speech calls between two backends based on language detection:

- **Bulgarian text** → Edge TTS (Microsoft neural voices, free, no API key)
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

When Bulgarian is detected, the text is sent to the configured Bulgarian
backend. By default this is **bg-tts-v7** (a fine-tuned Qwen3 TTS model) via
transformers, with **Edge TTS** as an automatic fallback. Otherwise, it's sent
to **OpenAI TTS** using the endpoint configured in `tts.openai.*`.

Each backend reads its own config block independently — all routes stay fully
configurable.

### Bulgarian backends

The router supports two backends for Bulgarian text, selected via
`tts.tts_router.bg_backend`:

| Backend | Description |
|---------|-------------|
| `transformers` (default) | **bg-tts-v7** (Qwen3-0.6B fine-tune + MioCodec) run locally via transformers. Edge TTS is the automatic fallback on any failure. |
| `edge` | Always Microsoft Edge neural voices (free, no heavy deps). |

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
    # Bulgarian backend: "transformers" (bg-tts-v7, default) or "edge"
    bg_backend: transformers
    # bg-tts-v7 backend options (only read when bg_backend: transformers)
    transformers:
      model_name: beleata74/bg-tts-v7
      codec_model: Aratako/MioCodec-25Hz-24kHz
      python: ~/.hermes/venvs/bg-tts/bin/python   # dedicated Python>=3.12 venv
      preset_path: ~/.hermes/venvs/bg-tts/presets/en_female.pt
      max_new_tokens: 500
      temperature: 0.7
      top_p: 0.9
      repetition_penalty: 1.1
```

After configuring, start a new session (`/new` or `/reset`) for the plugin to
load.

## Setting Up the bg-tts-v7 Backend

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

The plugin then delegates to the built-in `_generate_edge_tts` and
`_generate_openai_tts` functions — it only decides *which* to call.

## License

MIT — see [LICENSE](LICENSE).
