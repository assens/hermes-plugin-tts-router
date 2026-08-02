"""
TTS Router Plugin
=================

Routes text-to-speech calls between two backends based on language detection:

- **Bulgarian text** → Edge TTS (``bg-BG-KalinaNeural`` or configured voice)
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
        # Override the default Cyrillic word list (optional)
        bulgarian_words:
          - "здравей"
          - "благодаря"
        # Minimum number of Cyrillic chars before routing to BG (default: 1)
        min_cyrillic_chars: 1

The plugin delegates the actual synthesis to the built-in ``_generate_edge_tts``
and ``_generate_openai_tts`` functions — it only decides *which* to call.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)


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
# TTS Provider
# ---------------------------------------------------------------------------


class TTSRouterProvider(TTSProvider):
    """Routes TTS calls between Edge (Bulgarian) and OpenAI (default).

    Reads configuration from the standard ``tts.edge`` and ``tts.openai``
    config blocks. A ``tts.tts_router`` block can override detection
    parameters (``bulgarian_words``, ``min_cyrillic_chars``).
    """

    def __init__(self) -> None:
        self._bg_words_override: Optional[frozenset] = None
        self._min_cyrillic: int = 1

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

    @property
    def name(self) -> str:
        return "tts-router"

    @property
    def display_name(self) -> str:
        return "TTS Router (BG→Edge, Other→OpenAI)"

    def is_available(self) -> bool:
        """Check that both edge-tts and openai packages are importable."""
        try:
            from tools.tts_tool import _import_edge_tts, _import_openai_client
            _import_edge_tts()
            _import_openai_client()
            return True
        except ImportError:
            return False

    @property
    def voice_compatible(self) -> bool:
        """Output is suitable for voice-bubble delivery (Opus via OpenAI)."""
        return True

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "free",
            "tag": "Routes Bulgarian text to Edge TTS, everything else to OpenAI TTS",
            "env_vars": [],
        }

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

        Delegates to the built-in ``_generate_edge_tts`` (Bulgarian) or
        ``_generate_openai_tts`` (default) functions. The ``voice`` and
        ``model`` params from the dispatcher are ignored — each backend
        reads its own config block (``tts.edge.voice``, ``tts.openai.voice``,
        etc.) so the two routes stay independently configurable.
        """
        from tools.tts_tool import (
            _generate_edge_tts,
            _generate_openai_tts,
            _load_tts_config,
        )
        import asyncio

        tts_config = _load_tts_config()
        self._load_router_config(tts_config)

        is_bg = is_bulgarian(
            text,
            bulgarian_words=self._bg_words_override,
            min_cyrillic_chars=self._min_cyrillic,
        )

        if is_bg:
            logger.info(
                "TTS Router: Bulgarian text detected (%d chars), routing to Edge TTS",
                len(text),
            )
            # Edge TTS is async — run it in a thread pool like the built-in
            # dispatcher does (concurrent.futures with a fresh event loop).
            import concurrent.futures

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(
                        lambda: asyncio.run(
                            _generate_edge_tts(text, output_path, tts_config)
                        )
                    ).result(timeout=60)
            except RuntimeError:
                # No running event loop — safe to call asyncio.run directly
                asyncio.run(_generate_edge_tts(text, output_path, tts_config))

            return output_path
        else:
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
    logger.info("TTS Router plugin registered: BG→Edge, Other→OpenAI")
