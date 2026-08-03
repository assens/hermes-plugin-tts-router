#!/usr/bin/env python3
"""
bg-tts-v5-mlx persistent TTS server.

Loads the bg-tts-v5-mlx model + nanocodec ONCE and keeps them resident,
serving synthesis requests over HTTP. This eliminates the per-request
model-load overhead (the plugin's subprocess-per-request path reloads the
~965MB model every time).

Endpoints
---------
- GET  /health  -> {"status": "ok", "loaded": bool, "uptime": ...}
- POST /tts     -> body JSON {"text": "...", "speaker_id": 0, ...}
                   returns audio bytes (wav, 22050 Hz) with Content-Type
                   application/octet-stream.

Idle unload: the model is freed after ``idle_timeout`` seconds (default 300
= 5 min) with no requests, and reloaded lazily on the next request.

Run in the dedicated bg-tts-v5 venv (Python>=3.12, mlx, nanocodec-mlx).
"""

import argparse
import json
import os
import sys
import threading
import time

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Make the model's tts_mlx package importable.
_CHECKPOINT = os.path.expanduser("~/.hermes/models/bg-tts-v5-mlx")
if os.path.isdir(_CHECKPOINT):
    sys.path.insert(0, _CHECKPOINT)


class TTSState:
    """Holds the loaded model + codec, with idle-based unload."""

    def __init__(self, checkpoint, idle_timeout):
        self.checkpoint = checkpoint
        self.idle_timeout = idle_timeout
        self.lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.codec = None
        self.last_used = 0.0
        self._loaded = False

    # -- load / unload -------------------------------------------------
    def _do_load(self):
        from tts_mlx.inference import load_from_safetensors
        from tts_mlx.tokenizer import TTSTokenizer
        from tts_mlx.config import NANOCODEC_MODEL_NAME, CODEC_NUM_CODEBOOKS
        from nanocodec_mlx.models.audio_codec import AudioCodecModel

        print("📦 Loading bg-tts-v5-mlx model + codec...", flush=True)
        t0 = time.time()
        model = load_from_safetensors(self.checkpoint)
        model.eval()
        tokenizer = TTSTokenizer()
        codec = AudioCodecModel.from_pretrained(NANOCODEC_MODEL_NAME)
        print(f"✅ Model loaded in {time.time()-t0:.1f}s", flush=True)
        self.model = model
        self.tokenizer = tokenizer
        self.codec = codec
        self._codec_nbooks = CODEC_NUM_CODEBOOKS
        self._loaded = True

    def _do_unload(self):
        print("🛑 Idle timeout reached — unloading model.", flush=True)
        self.model = None
        self.tokenizer = None
        self.codec = None
        self._loaded = False

    def ensure_loaded(self):
        """Return True if loaded; load if not. Call under lock."""
        if self._loaded:
            self.last_used = time.time()
            return True
        try:
            self._do_load()
            self.last_used = time.time()
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"❌ Model load failed: {exc}", flush=True)
            return False

    def is_loaded(self):
        return self._loaded

    def synthesize(self, text, speaker_id=0, temperature=0.25,
                   top_k=50, top_p=0.8, rep_penalty=1.1, max_tokens=2000):
        """Synthesize text -> wav numpy array. Model must be loaded."""
        import numpy as np
        import mlx.core as mx
        from tts_mlx.inference import generate
        from tts_mlx.config import AUDIO_OFFSET, NUM_AUDIO_TOKENS, CODEC_NUM_CODEBOOKS

        tokens = generate(
            self.model, self.tokenizer, text, speaker_id, max_tokens,
            temperature, top_k, top_p, rep_penalty,
        )
        if tokens is None or len(tokens) == 0:
            raise RuntimeError("No audio generated")

        tokens = tokens[: len(tokens) - len(tokens) % CODEC_NUM_CODEBOOKS]
        num_frames = len(tokens) // CODEC_NUM_CODEBOOKS
        codes = tokens.reshape(num_frames, CODEC_NUM_CODEBOOKS).T  # [4, T]
        codes_mx = mx.array(codes.astype(np.int32))[None, :, :]     # [1,4,T]
        tokens_len = mx.array([num_frames], dtype=mx.int32)
        wav_mx, _ = self.codec.decode(codes_mx, tokens_len)
        mx.eval(wav_mx)
        wav_np = np.array(wav_mx[0, 0, :])
        return wav_np

    # -- idle unload loop ---------------------------------------------
    def idle_loop(self, stop_event):
        while not stop_event.is_set():
            time.sleep(1.0)
            with self.lock:
                if self._loaded and (time.time() - self.last_used) > self.idle_timeout:
                    self._do_unload()


def build_wav(numpy_audio, sample_rate=22050):
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, numpy_audio, sample_rate, format="WAV")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--checkpoint", default=_CHECKPOINT)
    parser.add_argument("--idle-timeout", type=int, default=300,
                        help="Unload model after Ns idle (default 300 = 5min)")
    args = parser.parse_args()

    state = TTSState(args.checkpoint, args.idle_timeout)
    stop_event = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *fargs):
            sys.stderr.write("  [%s] %s\n" % (self.log_date_time_string(), fmt % fargs))

        def _send_json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == "/health":
                with state.lock:
                    loaded = state.is_loaded()
                    uptime = time.time() - start_time
                self._send_json(200, {
                    "status": "ok",
                    "loaded": loaded,
                    "uptime": round(uptime, 1),
                    "idle_timeout": args.idle_timeout,
                })
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.split("?")[0] != "/tts":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                text = body.get("text", "")
                if not text or not text.strip():
                    self._send_json(400, {"error": "empty text"})
                    return
                opts = {
                    "speaker_id": int(body.get("speaker_id", 0)),
                    "temperature": float(body.get("temperature", 0.25)),
                    "top_k": int(body.get("top_k", 50)),
                    "top_p": float(body.get("top_p", 0.8)),
                    "rep_penalty": float(body.get("rep_penalty", 1.1)),
                    "max_tokens": int(body.get("max_tokens", 2000)),
                }
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {"error": f"bad request: {exc}"})
                return

            with state.lock:
                if not state.ensure_loaded():
                    self._send_json(500, {"error": "model not loaded"})
                    return
                t0 = time.time()
                try:
                    wav_np = state.synthesize(text, **opts)
                except Exception as exc:  # noqa: BLE001
                    self._send_json(500, {"error": f"synthesis failed: {exc}"})
                    return
                dur = time.time() - t0

            try:
                wav_bytes = build_wav(wav_np)
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": f"wav encode failed: {exc}"})
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(wav_bytes)))
            self.send_header("X-Synth-Ms", str(round(dur * 1000)))
            self.end_headers()
            self.wfile.write(wav_bytes)
            print(f"✅ {text[:40]!r} -> {len(wav_bytes)} bytes wav ({dur:.2f}s)", flush=True)

    global start_time
    start_time = time.time()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    t = threading.Thread(target=state.idle_loop, args=(stop_event,), daemon=True)
    t.start()
    print(f"🚀 bg-tts-v5-mlx server on http://{args.host}:{args.port} "
          f"(idle unload {args.idle_timeout}s)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()