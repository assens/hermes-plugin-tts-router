"""
TTS Router Plugin
=================

Routes text-to-speech calls between two backends based on language detection:

- **Bulgarian text** → a Bulgarian TTS backend, defaulting to
  `bg-tts-v7` (transformers direct) with automatic fallback to Edge TTS.
- **Everything else** → OpenAI TTS (local Kokoro or configured endpoint)

Bulgarian detection uses two signals (either triggers the BG route):

1. **Cyrillic script** — any character in the Unicode Cyrillic block
   (U+0400–U+04FF) or Cyrillic Supplement (U+0500–U+052F).
2. **Bulgarian word forms** — a curated set of common Bulgarian words
   and lemmas (lowercase, accent-stripped) that don't require Cyrillic
   to be recognised (e.g. Latin transliterations are NOT covered — this
   is strictly Cyrillic + Bulgarian-specific patterns).

This lets the user keep a single ``tts.provider: tts-router`` in
config.yaml and have mixed-language output routed automatically.

Configuration (all optional — sensible defaults from existing tts.edge
and tts.openai config blocks):

.. code-block:: yaml

    tts:
      provider: tts-router
      edge:
        voice: bg-BG-KalinaNeural
      openai:
        model: Kokoro-82M-bf16
        voice: af_sky
        base_url: http://127.0.0.1:8000/v1
        api_key: omlx-local
        response_format: opus
      tts_router:
        # Which backend handles Bulgarian text: "transformers" (bg-tts-v7,
        # recommended) or "edge" (Microsoft neural voices). Edge is always
        # used as the fallback if transformers fails or isn't installed.
        bg_backend: transformers
        # Transformers backend options (only read when bg_backend: transformers)
        transformers:
          model_name: beleata74/bg-tts-v7
          codec_model: Aratako/MioCodec-25Hz-24kHz
          device: auto              # "auto", "cuda", "cpu", or "mps"
          dtype: bfloat16           # torch dtype name
          max_new_tokens: 500
          temperature: 0.7
          top_p: 0.9
          repetition_penalty: 1.1
        # Override the default Cyrillic word list (optional)
        bulgarian_words:
          - "здравей"
          - "благодаря"
        # Minimum number of Cyrillic chars before routing to BG (default: 1)
        min_cyrillic_chars: 1

The plugin delegates the actual synthesis to the built-in ``_generate_edge_tts``
and ``_generate_openai_tts`` functions — it only decides *which* to call. For
the ``bg-tts-v7`` transformers backend it implements the "Direct with
transformers" pipeline from the model card directly (ChatML prompt → generate
speech tokens → decode via MioCodec → wav).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import unicodedata
from typing import Any, Dict, Optional

from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)

# Speech-token vocab offset for Qwen3-based bg-tts-v7 (base vocab 151669 +
# 12800 speech tokens). See model card "Direct with transformers".
_SPEECH_OFFSET = 151669
_SPEECH_VOCAB = 12800
_CODEC_SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# Bulgarian detection
# ---------------------------------------------------------------------------

# Cyrillic Unicode ranges: basic + supplement + extended
_CYRILLIC_RE = re.compile(
    r"[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F]"
)

# Curated set of common Bulgarian words (lowercase, no accents).
# These are uniquely or predominantly Bulgarian (as opposed to Russian,
# Serbian, etc.) when they appear together with Cyrillic text.
# This list is a secondary signal — Cyrillic detection is primary.
_DEFAULT_BULGARIAN_WORDS: frozenset = frozenset({
    # Greetings / politeness
    "здравей", "здрасти", "здравейте",
    "благодаря", "благодарам",
    "моля", "извинете", "извинявай",
    "привет",
    # Common function words unique/distinctive to Bulgarian
    "съм", "си", "е", "сме", "сте", "са",  # copula "to be" (BG-specific short forms)
    "това", "онова", "тук", "там",
    "как", "къде", "кога", "защо", "колко",
    "да", "не",
    "и", "но", "или",
    # Question words / fillers
    "нали", "та", "де", "хайде",
    # Days / time
    "днес", "утре", "вчера",
    "неделя", "понеделник", "вторник", "сряда", "четвъртък", "петък", "събота",
    # Common verbs (BG aorist/imperfect stems are distinctive)
    "правя", "правиш", "прави",
    "искам", "искаш", "иска",
    "мога", "можеш", "може",
    "виждам", "виждаш", "вижда",
    "зная", "знаеш", "знае",
    # Common nouns
    "работа", "време", "ден", "човек", "хора", "дом", "град",
    "вода", "храна", "пари",
    # Family
    "майка", "баща", "син", "дъщеря", "брат", "сестра",
    # Filler / discourse
    "добре", "ясно", "разбира", "съжалявам",
    # Numbers (BG-specific)
    "едно", "две", "три", "четири", "пет", "шест", "седем", "осем", "девет", "десет",
})


def _normalize_text(text: str) -> str:
    """Lowercase + strip accents for word matching."""
    # Decompose accents (NFD) and drop combining marks
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.lower()


def _count_cyrillic(text: str) -> int:
    """Count Cyrillic characters in text."""
    return len(_CYRILLIC_RE.findall(text))


def _contains_bulgarian_words(
    text: str, word_set: frozenset, min_chars: int = 2
) -> bool:
    """Check if text contains any known Bulgarian word (>= min_chars long).

    Words shorter than ``min_chars`` are skipped to avoid false positives
    on short particles like ``е``, ``и``, ``да``, ``не`` when they appear
    in isolation in non-Bulgarian context. When Cyrillic is already
    detected, this function is not called — it's a secondary signal only.
    """
    normalized = _normalize_text(text)
    # Split on non-letter characters (works for Cyrillic too)
    words = re.findall(r"[^\W\d_]+", normalized, re.UNICODE)
    for word in words:
        if len(word) >= min_chars and word in word_set:
            return True
    return False


def is_bulgarian(
    text: str,
    bulgarian_words: Optional[frozenset] = None,
    min_cyrillic_chars: int = 1,
) -> bool:
    """Determine whether text should be routed to the Bulgarian TTS voice.

    Detection logic (first hit wins):

    1. **Cyrillic script** — if the text contains at least
       ``min_cyrillic_chars`` Cyrillic characters, route to Bulgarian.
       This is the primary and most reliable signal.
    2. **Bulgarian word forms** — if no Cyrillic is found but the text
       contains known Bulgarian words from the curated set, route to
       Bulgarian. This catches edge cases where only Latin transliteration
       is used (rare, but possible).

    Args:
        text: The input text to analyse.
        bulgarian_words: Optional override for the default word set.
        min_cyrillic_chars: Minimum Cyrillic char count to trigger BG route.

    Returns:
        True if the text should use the Bulgarian TTS voice.
    """
    if not text or not text.strip():
        return False

    word_set = bulgarian_words or _DEFAULT_BULGARIAN_WORDS

    # Signal 1: Cyrillic characters (primary)
    cyrillic_count = _count_cyrillic(text)
    if cyrillic_count >= min_cyrillic_chars:
        return True

    # Signal 2: Bulgarian word forms (secondary, no Cyrillic found)
    if _contains_bulgarian_words(text, word_set):
        return True

    return False


# ---------------------------------------------------------------------------
# bg-tts-v7 (transformers direct) synthesis
# ---------------------------------------------------------------------------

# The transformers/miocodec stack requires Python >= 3.12 (miocodec hard
# constraint), which the Hermes runtime venv (3.11) can't satisfy. So the
# actual model runs in a SEPARATE dedicated venv (created at install time),
# and this plugin shells out to a small worker script in that venv. The
# worker reads its arguments/options as JSON on argv and writes the audio
# file to the requested path; stdout carries a JSON status line. Any non-zero
# exit or JSON error causes the caller to fall back to Edge TTS.

_WORKER_SCRIPT = r'''
import json, sys

def main():
    try:
        opts = json.loads(sys.argv[1])
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"bad opts: {exc}"}))
        return 1

    text = opts["text"]
    output_path = opts["output_path"]
    cfg = opts.get("cfg", {})
    out_format = opts.get("format", "mp3")

    import os
    import torch
    from miocodec.model import MioCodecModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import soundfile as sf

    model_name = cfg.get("model_name", "beleata74/bg-tts-v7")
    codec_model = cfg.get("codec_model", "Aratako/MioCodec-25Hz-24kHz")
    dtype = getattr(torch, str(cfg.get("dtype", "bfloat16")), torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map="auto"
    )
    codec = MioCodecModel.from_pretrained(codec_model).eval()
    codec_device = next(codec.parameters()).device

    # Load the speaker preset (global embedding) that drives voice identity.
    # bg-tts-v7 has no Bulgarian reference, so we use an English preset and
    # accept the resulting accent. Path can be overridden via cfg["preset_path"].
    preset_path = cfg.get("preset_path") or os.path.expanduser(
        "~/.hermes/venvs/bg-tts/presets/en_female.pt"
    )
    if not os.path.exists(preset_path):
        raise RuntimeError(f"speaker preset not found: {preset_path}")
    global_embedding = torch.load(preset_path, map_location=codec_device, weights_only=True)
    if global_embedding.dim() == 2:
        global_embedding = global_embedding.squeeze(0)
    global_embedding = global_embedding.to(codec_device)

    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(cfg.get("max_new_tokens", 500)),
            temperature=float(cfg.get("temperature", 0.7)),
            top_p=float(cfg.get("top_p", 0.9)),
            repetition_penalty=float(cfg.get("repetition_penalty", 1.1)),
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    speech_offset = int(cfg.get("speech_offset", 151669))
    speech_vocab = 12800
    audio_codes = [
        int(t.item()) - speech_offset
        for t in generated
        if speech_offset <= int(t.item()) < speech_offset + speech_vocab
    ]
    if not audio_codes:
        print(json.dumps({"ok": False, "error": "no speech tokens generated"}))
        return 1

    tokens = torch.tensor(audio_codes, dtype=torch.long, device=codec_device)
    with torch.inference_mode():
        wav = codec.decode(
            global_embedding=global_embedding,
            content_token_indices=tokens,
        )
    wav_np = wav.cpu().numpy()
    sample_rate = int(codec.config.sample_rate)

    if out_format == "wav" or output_path.lower().endswith(".wav"):
        sf.write(output_path, wav_np, sample_rate)
    else:
        import subprocess, tempfile
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp, wav_np, sample_rate)
            subprocess.run(["ffmpeg", "-y", "-i", tmp, output_path],
                           check=True, capture_output=True)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    print(json.dumps({"ok": True, "output_path": output_path}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''


def _generate_transformers_bg(
    text: str,
    output_path: str,
    cfg: Dict[str, Any],
    format: str = "mp3",
) -> str:
    """Synthesize Bulgarian speech with bg-tts-v7 in a dedicated venv.

    The heavy stack (torch, transformers, miocodec) lives in a separate
    Python >= 3.12 venv — see ``~/.hermes/venvs/bg-tts``. This function shells
    out to ``_WORKER_SCRIPT`` there, passing text/options as JSON. Raises on
    any failure so the caller falls back to Edge TTS.

    The venv python path can be overridden via the ``python`` option in
    ``tts.tts_router.transformers`` (default:
    ``~/.hermes/venvs/bg-tts/bin/python``).
    """
    import json
    import os
    import subprocess

    python_bin = cfg.get("python") or os.path.expanduser(
        "~/.hermes/venvs/bg-tts/bin/python"
    )
    if not os.path.exists(python_bin):
        raise RuntimeError(
            f"bg-tts venv python not found at {python_bin}. Create it with: "
            "python3.13 -m venv ~/.hermes/venvs/bg-tts && "
            "~/.hermes/venvs/bg-tts/bin/pip install transformers<5 soundfile "
            "'miocodec @ git+https://github.com/Aratako/MioCodec@main'"
        )

    opts = {
        "text": text,
        "output_path": output_path,
        "cfg": cfg,
        "format": format,
    }
    # The Hermes runtime sets PYTHONPATH to include its OWN venv
    # (lib/python3.11/site-packages). If that leaks into this subprocess,
    # the bg-tts venv's python will import numpy/torch built for 3.11 and
    # crash. Strip PYTHONPATH (and PYTHONHOME) so the dedicated venv is
    # fully self-contained.
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("PYTHONPATH", "PYTHONHOME")}
    proc = subprocess.run(
        [python_bin, "-c", _WORKER_SCRIPT, json.dumps(opts)],
        capture_output=True,
        text=True,
        timeout=600,
        env=clean_env,
    )
    # Parse the last stdout line as JSON status.
    result = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                result = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if proc.returncode != 0 or not result or not result.get("ok"):
        detail = (result or {}).get("error") or proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"bg-tts-v7 synthesis failed: {detail}")
    return result.get("output_path") or output_path


# ---------------------------------------------------------------------------
# TTS Provider
# ---------------------------------------------------------------------------


class TTSRouterProvider(TTSProvider):
    """Routes TTS calls between a BG backend and OpenAI (default).

    Reads configuration from the standard ``tts.edge`` and ``tts.openai``
    config blocks, plus a ``tts.tts_router`` block for routing options.

    Bulgarian backend selection (``tts_router.bg_backend``):
    - ``transformers`` (default) — bg-tts-v7 via transformers, falling back
      to Edge TTS on any error or missing dependency.
    - ``edge`` — always Microsoft Edge neural voices.
    """

    def __init__(self) -> None:
        self._bg_words_override: Optional[frozenset] = None
        self._min_cyrillic: int = 1
        self._bg_backend: str = "transformers"
        self._transformers_cfg: Dict[str, Any] = {}

    def _load_router_config(self, tts_config: Dict[str, Any]) -> None:
        """Load router-specific overrides from ``tts.tts_router`` block."""
        router_cfg = (
            tts_config.get("tts_router") if isinstance(tts_config, dict) else None
        ) or {}
        if not isinstance(router_cfg, dict):
            return

        words = router_cfg.get("bulgarian_words")
        if isinstance(words, list) and words:
            self._bg_words_override = frozenset(
                _normalize_text(w) for w in words if isinstance(w, str)
            )

        min_c = router_cfg.get("min_cyrillic_chars")
        if isinstance(min_c, (int, float)) and min_c >= 1:
            self._min_cyrillic = int(min_c)

        backend = router_cfg.get("bg_backend", "transformers")
        if isinstance(backend, str):
            self._bg_backend = backend.strip().lower()
        else:
            self._bg_backend = "transformers"

        t_cfg = router_cfg.get("transformers")
        if isinstance(t_cfg, dict):
            self._transformers_cfg = dict(t_cfg)
        else:
            self._transformers_cfg = {}

    @property
    def name(self) -> str:
        return "tts-router"

    @property
    def display_name(self) -> str:
        return "TTS Router (BG→bg-tts-v7/Edge, Other→OpenAI)"

    def is_available(self) -> bool:
        """Check that the fallback (edge) and default (openai) are importable.

        The transformers backend is optional — if it isn't installed the
        router transparently falls back to Edge TTS for Bulgarian text, so
        it isn't required for the plugin to be "available".
        """
        try:
            from tools.tts_tool import _import_edge_tts, _import_openai_client
            _import_edge_tts()
            _import_openai_client()
            return True
        except ImportError:
            return False

    @property
    def voice_compatible(self) -> bool:
        """Output is suitable for voice-bubble delivery (Opus via pipeline)."""
        return True

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "free",
            "tag": "Routes Bulgarian text to bg-tts-v7/Edge, everything else to OpenAI TTS",
            "env_vars": [],
        }

    def _generate_bg(
        self,
        text: str,
        output_path: str,
        format: str,
    ) -> str:
        """Run the configured Bulgarian backend, falling back to Edge TTS.

        If ``bg_backend: transformers`` is set, try bg-tts-v7 first; on any
        exception (missing deps, model download failure, empty output, …)
        log a warning and fall back to Edge TTS. If ``bg_backend: edge`` or
        transformers isn't configured, go straight to Edge.
        """
        from tools.tts_tool import _generate_edge_tts

        if self._bg_backend == "transformers":
            try:
                logger.info(
                    "TTS Router: generating Bulgarian via bg-tts-v7 (transformers)..."
                )
                return _generate_transformers_bg(
                    text, output_path, self._transformers_cfg, format
                )
            except Exception as exc:  # noqa: BLE001 — any failure falls back
                logger.warning(
                    "TTS Router: bg-tts-v7 failed (%s); falling back to Edge TTS",
                    exc,
                )

        logger.info("TTS Router: generating Bulgarian via Edge TTS...")
        import asyncio
        import concurrent.futures

        edge_config = {}  # _generate_edge_tts reads its own tts.edge block
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(
                    lambda: asyncio.run(
                        _generate_edge_tts(text, output_path, edge_config)
                    )
                ).result(timeout=60)
        except RuntimeError:
            asyncio.run(_generate_edge_tts(text, output_path, edge_config))
        return output_path

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = "mp3",
        **extra: Any,
    ) -> str:
        """Route text to the appropriate TTS backend and synthesize.

        Delegates Bulgarian text to :meth:`_generate_bg` (bg-tts-v7 with
        Edge fallback) and non-Bulgarian text to the built-in
        ``_generate_openai_tts``. The ``voice`` and ``model`` params from the
        dispatcher are ignored — each backend reads its own config block so
        the routes stay independently configurable.
        """
        from tools.tts_tool import _generate_openai_tts, _load_tts_config

        tts_config = _load_tts_config()
        self._load_router_config(tts_config)

        is_bg = is_bulgarian(
            text,
            bulgarian_words=self._bg_words_override,
            min_cyrillic_chars=self._min_cyrillic,
        )

        if is_bg:
            logger.info(
                "TTS Router: Bulgarian text detected (%d chars)", len(text),
            )
            return self._generate_bg(text, output_path, format)

        logger.info(
            "TTS Router: non-Bulgarian text (%d chars), routing to OpenAI TTS",
            len(text),
        )
        _generate_openai_tts(text, output_path, tts_config)
        return output_path


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire TTSRouterProvider into the TTS registry."""
    ctx.register_tts_provider(TTSRouterProvider())
    logger.info("TTS Router plugin registered: BG→bg-tts-v7/Edge, Other→OpenAI")