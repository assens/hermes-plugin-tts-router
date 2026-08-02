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

When Bulgarian is detected, the text is sent to **Edge TTS** using the voice
configured in `tts.edge.voice`. Otherwise, it's sent to **OpenAI TTS** using
the endpoint configured in `tts.openai.*`.

Each backend reads its own config block independently — both stay fully
configurable.

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
    base_url: http://127.0.0.1:8000/v1   # OpenAI-compatible endpoint
    api_key: omlx-local              # or your real OpenAI key
    response_format: opus
    streaming: true
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
```

After configuring, start a new session (`/new` or `/reset`) for the plugin to
load.

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
