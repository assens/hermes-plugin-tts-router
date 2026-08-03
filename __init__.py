"""
TTS Router Plugin
=================

Routes text-to-speech calls based on language detection:

- **Pure Bulgarian text** (Cyrillic, no Latin mixed in) → `bg-tts-v5-mlx`
  (native Apple Silicon MLX) with automatic fallback to Edge TTS.
- **Mixed Bulgarian** (Cyrillic + Latin letters, e.g. code or brand names)
  → Microsoft Edge neural voices.
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

import atexit
import json
import logging
import logging.handlers
import os
import re
import subprocess
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
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


# Latin letters (A-Z a-z). Used to distinguish "pure" Bulgarian text from
# mixed (Cyrillic + Latin) text.
_LATIN_RE = re.compile(r"[A-Za-z]")


def _count_latin(text: str) -> int:
    """Count Latin (A-Z a-z) characters in text."""
    return len(_LATIN_RE.findall(text))


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


def classify_text(
    text: str,
    bulgarian_words: Optional[frozenset] = None,
    min_cyrillic_chars: int = 1,
) -> str:
    """Classify text into a routing category.

    Returns one of:
    - ``"mlx"`` — pure Bulgarian (Cyrillic, no Latin letters mixed in).
    - ``"edge"`` — Bulgarian with mixed Latin letters (e.g. code, brand
      names, URLs) OR text detected as Bulgarian only via word forms with
      Latin letters present.
    - ``"other"`` — non-Bulgarian (route to OpenAI TTS).

    Distinguishes "pure" from "mixed" Bulgarian by checking for Latin
    letters. A text with Cyrillic plus any Latin letter(s) is treated as
    mixed and sent to the more robust Edge voice rather than the MLX model.
    """
    if not text or not text.strip():
        return "other"

    cyrillic_count = _count_cyrillic(text)
    latin_count = _count_latin(text)
    word_set = bulgarian_words or _DEFAULT_BULGARIAN_WORDS

    # Pure Bulgarian: Cyrillic present, no Latin letters.
    if cyrillic_count >= min_cyrillic_chars and latin_count == 0:
        return "mlx"

    # Bulgarian present but mixed with Latin → Edge (robust).
    if cyrillic_count >= min_cyrillic_chars:
        return "edge"

    # No Cyrillic, but Bulgarian word forms found.
    if _contains_bulgarian_words(text, word_set):
        # If there are also Latin letters (e.g. transliteration) → edge.
        return "edge" if latin_count > 0 else "mlx"

    return "other"


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
    preset_path = os.path.expanduser(preset_path)  # expand ~ if config used it
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
    python_bin = os.path.expanduser(python_bin)  # expand ~ if config used it
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
# bg-tts-v5-mlx (native MLX) synthesis
# ---------------------------------------------------------------------------

# bg-tts-v5-mlx is a self-contained MLX (Apple Silicon) TTS model. It runs
# entirely via its bundled ``tts_mlx`` package — no server needed. Because
# oMLX's TTS engine doesn't support the custom ``bg-tts-v5-mlx`` architecture,
# we call the model's own ``synthesize()`` function directly in a dedicated
# venv. The worker mirrors the model card's usage and writes the wav, then
# (optionally) converts to the requested container format.

_WORKER_V5_SCRIPT = r'''
import json, os, sys

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

    checkpoint = cfg.get("checkpoint") or os.path.expanduser(
        "~/.hermes/models/bg-tts-v5-mlx"
    )
    checkpoint = os.path.expanduser(checkpoint)  # expand ~ if config used it
    sys.path.insert(0, checkpoint)

    from tts_mlx.inference import synthesize

    wav_path = output_path if str(out_format).lower() == "wav" \
        or output_path.lower().endswith(".wav") else output_path + ".wav"

    synthesize(
        checkpoint=checkpoint,
        text=text,
        output=wav_path,
        speaker_id=int(cfg.get("speaker_id", 0)),
        temperature=float(cfg.get("temperature", 0.25)),
        top_k=int(cfg.get("top_k", 50)),
        top_p=float(cfg.get("top_p", 0.8)),
        rep_penalty=float(cfg.get("rep_penalty", 1.1)),
        max_tokens=int(cfg.get("max_tokens", 2000)),
    )

    # Convert to requested container format if it isn't wav.
    if wav_path != output_path:
        import subprocess, tempfile
        subprocess.run(["ffmpeg", "-y", "-i", wav_path, output_path],
                       check=True, capture_output=True)
        os.unlink(wav_path)

    print(json.dumps({"ok": True, "output_path": output_path}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''


_DEFAULT_MLX_SERVER = "http://127.0.0.1:8001"

# Dedicated venv + server script paths for the bg-tts-v5-mlx persistent server.
_MLX_SERVER_SCRIPT = os.path.expanduser("~/.hermes/scripts/bg_tts_server.py")
_MLX_SERVER_LAUNCHER = os.path.expanduser("~/.hermes/scripts/bg_tts_server.sh")
_MLX_VENV_PYTHON = os.path.expanduser("~/.hermes/venvs/bg-tts-v5/bin/python")
_MLX_SERVER_PIDFILE = os.path.expanduser("~/.hermes/logs/bg-tts-server.pid")


def _server_healthy(server_url: str = _DEFAULT_MLX_SERVER) -> bool:
    """Return True if the bg-tts server is reachable and healthy."""
    try:
        req = urllib.request.Request(f"{server_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status != 200:
                return False
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("status") == "ok"
    except Exception:  # noqa: BLE001 — any failure means not healthy
        return False


def _ensure_server_running(
    cfg: Dict[str, Any] | None = None,
) -> bool:
    """Start the persistent bg-tts-v5-mlx server if it isn't already running.

    Idempotent: checks ``/health`` first; only starts the subprocess if the
    server is unreachable. Returns True if a server is running (started now or
    already up). Never raises — a failure just means the router falls back to
    the subprocess-per-request path.
    """
    server_url = (cfg or {}).get("server_url") or _DEFAULT_MLX_SERVER
    if _server_healthy(server_url):
        return True

    script = _MLX_SERVER_SCRIPT
    python_bin = (cfg or {}).get("python") or _MLX_VENV_PYTHON
    python_bin = os.path.expanduser(python_bin)
    if not os.path.exists(script) or not os.path.exists(python_bin):
        logger.warning(
            "TTS Router: cannot auto-start bg-tts server (missing %s or %s); "
            "will use subprocess-per-request",
            script, python_bin,
        )
        return False

    # Start the server detached, writing logs + pidfile like the launcher.
    log_dir = os.path.expanduser("~/.hermes/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "bg-tts-server.log")
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("PYTHONPATH", "PYTHONHOME")}
    # Pre-load the model on startup so the first real request is fast?
    # No — let the model load lazily (idle unload timer starts on load).
    try:
        logf = open(log_path, "ab")
        proc = subprocess.Popen(
            [python_bin, script, "--port", "8001", "--idle-timeout", "300"],
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=clean_env,
            start_new_session=True,  # detach from Hermes process group
            close_fds=True,
        )
        logf.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("TTS Router: failed to start bg-tts server: %s", exc)
        return False

    # Write pidfile (mirror the launcher's convention).
    try:
        with open(_MLX_SERVER_PIDFILE, "w") as f:
            f.write(str(proc.pid))
    except Exception:  # noqa: BLE001 — pidfile is best-effort
        pass

    logger.info(
        "TTS Router: started bg-tts-v5-mlx server (pid %d) on %s",
        proc.pid, server_url,
    )
    # Give it a moment to bind the socket.
    for _ in range(20):
        if _server_healthy(server_url):
            return True
        time.sleep(0.5)
    logger.warning("TTS Router: bg-tts server started but not yet healthy")
    return True


def _stop_server(server_url: str = _DEFAULT_MLX_SERVER) -> None:
    """Stop the persistent bg-tts-v5-mlx server if it's running.

    Terminates the server subprocess and removes the pidfile. Safe to call
    multiple times.
    """
    # 1) Kill via pidfile if present.
    pid = None
    try:
        if os.path.exists(_MLX_SERVER_PIDFILE):
            with open(_MLX_SERVER_PIDFILE) as f:
                pid = int(f.read().strip())
    except Exception:  # noqa: BLE001
        pid = None
    if pid:
        try:
            os.kill(pid, 15)  # SIGTERM — server handles it and unloads
        except ProcessLookupError:
            pass
        except Exception:  # noqa: BLE001
            logger.warning("TTS Router: could not kill server pid %s", pid)
        try:
            os.unlink(_MLX_SERVER_PIDFILE)
        except OSError:
            pass

    # 2) If still reachable, ask it to shut down (last resort).
    if _server_healthy(server_url):
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{server_url}/shutdown", method="POST"),
                timeout=2,
            )
        except Exception:  # noqa: BLE001
            pass

    logger.info("TTS Router: stopped bg-tts-v5-mlx server")


def _synthesize_via_server(
    text: str,
    output_path: str,
    cfg: Dict[str, Any],
    format: str = "mp3",
) -> bool:
    """Try to synthesize via the persistent bg-tts-v5-mlx HTTP server.

    Returns True if the server handled the request (audio written), False if
    the server is not available (so the caller falls back to subprocess).
    Raises RuntimeError if the server is up but synthesis fails.

    The server keeps the model resident across requests, avoiding the
    per-request ~965MB model reload. ``cfg`` may include ``server_url``
    (default ``http://127.0.0.1:8001``) and any synthesis params.
    """
    import json
    import urllib.request
    import urllib.error

    server_url = (cfg.get("server_url") or _DEFAULT_MLX_SERVER).rstrip("/")
    payload = {
        "text": text,
        "speaker_id": int(cfg.get("speaker_id", 0)),
        "temperature": float(cfg.get("temperature", 0.25)),
        "top_k": int(cfg.get("top_k", 50)),
        "top_p": float(cfg.get("top_p", 0.8)),
        "rep_penalty": float(cfg.get("rep_penalty", 1.1)),
        "max_tokens": int(cfg.get("max_tokens", 2000)),
    }
    req = urllib.request.Request(
        f"{server_url}/tts",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            audio = resp.read()
    except urllib.error.HTTPError as exc:
        # Server responded with an error (4xx/5xx) -> server is up but failed.
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"bg-tts server error {exc.code}: {body}") from exc
    except urllib.error.URLError:
        # Server not reachable -> caller falls back to subprocess path.
        return False

    if not audio:
        raise RuntimeError("bg-tts server returned empty audio")

    # Server returns wav (22050 Hz). Write it, then convert to requested format.
    if str(format).lower() == "wav" or output_path.lower().endswith(".wav"):
        with open(output_path, "wb") as f:
            f.write(audio)
        return True

    import subprocess as _sp
    tmp_wav = output_path + ".server.wav"
    with open(tmp_wav, "wb") as f:
        f.write(audio)
    try:
        _sp.run(
            ["ffmpeg", "-y", "-i", tmp_wav, output_path],
            check=True, capture_output=True,
        )
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)
    return True


def _generate_mlx_bg(
    text: str,
    output_path: str,
    cfg: Dict[str, Any],
    format: str = "mp3",
) -> str:
    """Synthesize Bulgarian speech with bg-tts-v5-mlx.

    Prefers the persistent HTTP server (``_synthesize_via_server``) which
    keeps the model resident across requests. If the server isn't running,
    falls back to the subprocess-per-request path (``_WORKER_V5_SCRIPT``),
    which reloads the ~965MB model each time. Raises on any failure so the
    caller falls back to Edge TTS. The venv python can be overridden via the
    ``python`` option, and the server URL via ``server_url``, in
    ``tts.tts_router.mlx``.
    """
    import json
    import os
    import subprocess

    # 1) Try the persistent server first.
    try:
        if _synthesize_via_server(text, output_path, cfg, format):
            logger.info("TTS Router: bg-tts-v5-mlx served by persistent server")
            return output_path
    except Exception as exc:  # noqa: BLE001 — server failed, fall back
        logger.warning(
            "TTS Router: bg-tts-v5-mlx server failed (%s); using subprocess",
            exc,
        )

    # 2) Fall back to subprocess-per-request.
    python_bin = cfg.get("python") or os.path.expanduser(
        "~/.hermes/venvs/bg-tts-v5/bin/python"
    )
    python_bin = os.path.expanduser(python_bin)  # expand ~ if config used it
    if not os.path.exists(python_bin):
        raise RuntimeError(
            f"bg-tts-v5 venv python not found at {python_bin}. Create it with: "
            "python3.13 -m venv ~/.hermes/venvs/bg-tts-v5 && "
            "~/.hermes/venvs/bg-tts-v5/bin/pip install mlx soundfile numpy "
            "'nanocodec-mlx @ git+https://github.com/nineninesix-ai/nanocodec-mlx.git'"
        )

    opts = {
        "text": text,
        "output_path": output_path,
        "cfg": cfg,
        "format": format,
    }
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("PYTHONPATH", "PYTHONHOME")}
    proc = subprocess.run(
        [python_bin, "-c", _WORKER_V5_SCRIPT, json.dumps(opts)],
        capture_output=True,
        text=True,
        timeout=600,
        env=clean_env,
    )
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
        raise RuntimeError(f"bg-tts-v5 synthesis failed: {detail}")
    return result.get("output_path") or output_path


# ---------------------------------------------------------------------------
# TTS Provider
# ---------------------------------------------------------------------------


class TTSRouterProvider(TTSProvider):
    """Routes TTS calls between a BG backend and OpenAI (default).

    Reads configuration from the standard ``tts.edge`` and ``tts.openai``
    config blocks, plus a ``tts.tts_router`` block for routing options.

    Bulgarian routing:
    - ``mlx`` — pure Bulgarian text (Cyrillic, no Latin mixed in) →
      bg-tts-v5-mlx (native Apple Silicon MLX), falling back to Edge TTS
      on any error or missing dependency.
    - ``edge`` — Bulgarian text mixed with Latin letters (e.g. code, brand
      names, URLs) → Microsoft Edge neural voices.
    - ``other`` — non-Bulgarian → OpenAI TTS.
    """

    def __init__(self) -> None:
        self._bg_words_override: Optional[frozenset] = None
        self._min_cyrillic: int = 1
        self._bg_backend: str = "transformers"
        self._transformers_cfg: Dict[str, Any] = {}
        self._mlx_cfg: Dict[str, Any] = {}

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

        m_cfg = router_cfg.get("mlx")
        if isinstance(m_cfg, dict):
            self._mlx_cfg = dict(m_cfg)
        else:
            self._mlx_cfg = {}

    @property
    def name(self) -> str:
        return "tts-router"

    @property
    def display_name(self) -> str:
        return "TTS Router (BG→MLX/Edge, Other→OpenAI)"

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
            "tag": "Pure Bulgarian→bg-tts-v5-mlx, mixed Bulgarian→Edge, other→OpenAI TTS",
            "env_vars": [],
        }

    def _generate_edge(
        self,
        text: str,
        output_path: str,
        tts_config: Dict[str, Any],
    ) -> str:
        """Synthesize via Microsoft Edge neural voices.

        ``tts_config`` is the **full** ``tts:`` config dict (which contains
        the ``edge`` block). ``_generate_edge_tts`` internally reads
        ``tts_config.get("edge").voice`` to pick the voice, so the whole
        config must be passed through — not just the ``edge`` sub-block,
        otherwise it falls back to the default English voice.
        """
        from tools.tts_tool import _generate_edge_tts

        logger.info("TTS Router: generating Bulgarian via Edge TTS...")
        import asyncio
        import concurrent.futures

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(
                    lambda: asyncio.run(
                        _generate_edge_tts(text, output_path, tts_config)
                    )
                ).result(timeout=60)
        except RuntimeError:
            asyncio.run(_generate_edge_tts(text, output_path, tts_config))
        return output_path

    def _generate_bg(
        self,
        text: str,
        output_path: str,
        format: str,
        category: str,
        tts_config: Dict[str, Any],
    ) -> str:
        """Route Bulgarian text to the MLX model or Edge TTS.

        ``category`` comes from :func:`classify_text`:
        - ``"mlx"`` — pure Bulgarian → bg-tts-v5-mlx (native Apple Silicon
          MLX). Any failure falls back to Edge TTS.
        - ``"edge"`` — mixed Bulgarian (Cyrillic + Latin) → Edge TTS.

        ``tts_config`` is the **full** ``tts:`` config dict, passed through
        to ``_generate_edge_tts`` so the configured Bulgarian voice is
        honoured.

        The ``transformers`` (bg-tts-v7) backend is intentionally **not**
        routed to — it is kept in the codebase for testing only.
        """
        if category == "mlx":
            try:
                logger.info(
                    "TTS Router: generating Bulgarian via bg-tts-v5-mlx (MLX)..."
                )
                return _generate_mlx_bg(text, output_path, self._mlx_cfg, format)
            except Exception as exc:  # noqa: BLE001 — any failure falls back
                logger.warning(
                    "TTS Router: bg-tts-v5-mlx FAILED (%s); FALLING BACK → Edge TTS",
                    exc,
                )

        return self._generate_edge(text, output_path, tts_config)

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

        Routing (via :func:`classify_text`):
        - ``"mlx"`` — pure Bulgarian → bg-tts-v5-mlx (MLX), Edge on failure.
        - ``"edge"`` — mixed Bulgarian (Cyrillic + Latin) → Edge TTS.
        - ``"other"`` — non-Bulgarian → OpenAI TTS (Kokoro).

        The ``voice`` and ``model`` params from the dispatcher are ignored —
        each backend reads its own config block so the routes stay
        independently configurable.
        """
        from tools.tts_tool import _generate_openai_tts, _load_tts_config

        tts_config = _load_tts_config()
        self._load_router_config(tts_config)
        edge_block = (
            tts_config.get("edge") if isinstance(tts_config, dict) else None
        ) or {}
        edge_voice = edge_block.get("voice", "DEFAULT")

        category = classify_text(
            text,
            bulgarian_words=self._bg_words_override,
            min_cyrillic_chars=self._min_cyrillic,
        )

        if category in ("mlx", "edge"):
            if category == "mlx":
                logger.info(
                    "TTS Router: ROUTED → bg-tts-v5-mlx (MLX)  | text='%s' (%d chars)",
                    text, len(text),
                )
            else:
                logger.info(
                    "TTS Router: ROUTED → Edge TTS (voice=%s)  | text='%s' (%d chars)",
                    edge_voice, text, len(text),
                )
            return self._generate_bg(
                text, output_path, format, category, tts_config
            )

        logger.info(
            "TTS Router: ROUTED → OpenAI TTS (Kokoro)  | text='%s' (%d chars)",
            text, len(text),
        )
        _generate_openai_tts(text, output_path, tts_config)
        return output_path


def _ensure_console_handler() -> None:
    """Attach a rotating file handler so routing decisions are always logged
    to ``~/.hermes/logs/tts-router.log``.

    The Hermes desktop console does not reliably surface INFO records from
    plugin loggers (the root logger level/handlers vary by launch mode). Logging
    to our own file is deterministic and survives even when the console is
    quiet. Each routing decision is written here, so the file doubles as a
    per-request audit trail of which backend handled each piece of text.
    """
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    log_dir = os.path.expanduser("~/.hermes/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "tts-router.log")
    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    # Stop propagating to the root logger — our own file is the authoritative
    # destination for routing messages, so don't double-write to the console.
    logger.propagate = False


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire TTSRouterProvider into the TTS registry.

    Also manages the persistent bg-tts-v5-mlx server lifecycle:
    - starts the server on plugin load (idempotent — no-op if already running)
    - stops it when the Hermes process exits (via ``atexit``)
    """
    _ensure_console_handler()
    ctx.register_tts_provider(TTSRouterProvider())

    # Manage the persistent bg-tts-v5-mlx server.
    try:
        _ensure_server_running()
        # Stop the server when the Hermes process exits.
        if not getattr(_stop_server, "_registered", False):
            atexit.register(_stop_server)
            _stop_server._registered = True
    except Exception as exc:  # noqa: BLE001 — never block plugin registration
        logger.warning("TTS Router: server lifecycle setup failed: %s", exc)

    logger.info("TTS Router plugin registered: BG→MLX/Edge, Other→OpenAI")