#!/usr/bin/env python3
"""
bg-tts-st persistent TTS server (Ani-Voice-API two-stage pipeline).

Loads the Supertonic reference-generator and the BgTTS synthesizer ONCE and
keeps them resident, serving synthesis requests over HTTP. This avoids the
per-request model-load overhead of the subprocess-per-request path.

Pipeline (per request):
  1. **Supertonic** renders the text in a chosen voice style (F1-F5 female,
     M1-M5 male) -> a reference clip (the voice identity).
  2. **BgTTS** uses that reference as the speaker embedding and synthesizes
     the final audio.
  Optionally, Bulgarian text normalization (bg-text-normalizer + num2cyrillic)
  is applied first so numbers/dates are read correctly.

Endpoints
---------
- GET  /health  -> {"status": "ok", "loaded": bool, "uptime": ...}
- POST /tts     -> body JSON {"text": "...", "voice_style": "M5", "speed": 1.6,
                              "temperature": 0.7, "top_k": 250, "top_p": 0.95,
                              "rep_penalty": 1.1, "max_new_tokens": 512}
                   returns audio bytes (wav, 24000 Hz).
- POST /stream  -> same body; returns NDJSON (chunked), one line per sentence
                   as it's generated: {"chunk_index": i, "pcm_base64": "...",
                   "duration": 0.0} where pcm_base64 is raw little-endian
                   int16 mono 24 kHz PCM (for streaming playback).
- POST /shutdown -> graceful stop.

Idle unload: the model is freed after ``idle_timeout`` seconds (default 300 =
5 min) with no requests, and reloaded lazily on the next request.

Run in the dedicated ani-voice venv (Python>=3.13, supertonic, bg-text-
normalizer, num2cyrillic, torch, torchaudio, miocodec).
"""

import argparse
import json
import os
import sys
import threading
import time

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_MODEL_DIR = os.path.expanduser("~/.hermes/models/Ani-Voice-API")


class TTSState:
    """Holds the loaded Supertonic + BgTTS models, with idle-based unload."""

    def __init__(self, model_dir, idle_timeout, voice_style="M5", speed=1.6):
        self.model_dir = model_dir
        self.idle_timeout = idle_timeout
        self.default_voice_style = voice_style
        self.default_speed = speed
        self.lock = threading.Lock()
        self.supertonic = None
        self.bgtts_model = None
        self.bgtts_tokenizer = None
        self.codec = None
        self.last_used = 0.0
        self._loaded = False
        self._bgtts_checkpoint = os.path.join(model_dir, "BgTTS", "checkpoint_inference.pt")

    # -- load / unload -------------------------------------------------
    def _do_load(self):
        # BgTTS sources (inference.py etc.) live in the model repo dir.
        sys.path.insert(0, self.model_dir)
        sys.path.insert(0, os.path.join(self.model_dir, "BgTTS"))

        from supertonic import TTS

        from model import load_for_inference
        from tokenizer import TTSTokenizer
        from codec import CodecV6

        print("📦 Loading Supertonic + BgTTS...", flush=True)
        t0 = time.time()

        st = TTS(auto_download=True)
        bgtts_model = load_for_inference(self._bgtts_checkpoint, device="mps")
        bgtts_tokenizer = TTSTokenizer()
        codec = CodecV6(device="mps")

        print(f"✅ Loaded in {time.time()-t0:.1f}s", flush=True)
        self.supertonic = st
        self.bgtts_model = bgtts_model
        self.bgtts_tokenizer = bgtts_tokenizer
        self.codec = codec
        self._loaded = True

    def _do_unload(self):
        print("🛑 Idle timeout reached — unloading model.", flush=True)
        self.supertonic = None
        self.bgtts_model = None
        self.bgtts_tokenizer = None
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

    def synthesize(self, text, voice_style=None, speed=None,
                   temperature=0.7, top_k=250, top_p=0.95,
                   rep_penalty=1.1, max_new_tokens=512):
        """Synthesize text -> wav bytes. Model must be loaded.

        Returns (wav_bytes, duration_seconds).
        """
        import io
        import tempfile
        import wave
        import shutil

        import numpy as np

        from inference import synthesize as bgtts_synthesize
        from normalizer import normalize_text

        voice_style = voice_style or self.default_voice_style
        speed = speed if speed is not None else self.default_speed

        # 1. Supertonic reference in the requested voice style.
        v_style = self.supertonic.get_voice_style(voice_name=voice_style)
        clean_text = text.replace('"', '').replace('„', '').replace('“', '') \
                         .replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-") \
                         .replace("*", "")
        if not clean_text.strip():
            raise RuntimeError("empty text after cleaning")

        wav_array, _ = self.supertonic.synthesize(
            clean_text, voice_style=v_style, lang="bg", speed=speed
        )
        wav_data = np.asarray(wav_array).flatten()
        wav_max = np.max(np.abs(wav_data))
        if wav_max > 0:
            wav_data = wav_data / wav_max
        pcm = (wav_data * 32767).astype(np.int16)

        fd, ref_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(ref_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(pcm.tobytes())

            # 2. BgTTS synthesis using the reference as speaker.
            normalized = normalize_text(clean_text)
            fd2, final_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd2)
            os.unlink(final_path)
            try:
                bgtts_synthesize(
                    checkpoint=self._bgtts_checkpoint,
                    text=normalized,
                    output=final_path,
                    speaker_wav=ref_path,
                    device="mps",
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    rep_penalty=float(rep_penalty),
                    max_tokens=int(max_new_tokens),
                )
                if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
                    raise RuntimeError("BgTTS produced no audio output")
                with open(final_path, "rb") as f:
                    wav_bytes = f.read()
                with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                    dur = wf.getnframes() / wf.getframerate()
            finally:
                try:
                    os.remove(final_path)
                except OSError:
                    pass
        finally:
            try:
                os.remove(ref_path)
            except OSError:
                pass

        return wav_bytes, float(dur)

    def synthesize_stream(self, text, voice_style=None, speed=None,
                          temperature=0.7, top_k=250, top_p=0.95,
                          rep_penalty=1.1, max_new_tokens=512):
        """Synthesize text -> raw PCM chunks (int16, 24 kHz mono), sentence by
        sentence.

        Splits ``text`` into sentences and yields ``(pcm_bytes, duration_s)``
        for each as it's generated. This is what enables "hear-as-generated"
        playback — the client gets each sentence the moment it's ready instead
        of waiting for the whole response.

        Each yielded chunk is raw little-endian int16 mono PCM at 24 kHz
        (the BgTTS output format), ready for the streaming-TTS provider
        contract.
        """
        import io
        import re
        import tempfile
        import wave

        import numpy as np

        from inference import synthesize as bgtts_synthesize
        from normalizer import normalize_text

        voice_style = voice_style or self.default_voice_style
        speed = speed if speed is not None else self.default_speed

        # Split into sentence chunks (mirror tts_engine.split_text_for_tts).
        def _split(text):
            text = text.strip()
            if not text:
                return []
            raw = re.split(r"(?<=[.!?…])\s+|\n+", text)
            chunks, buf = [], ""
            for part in raw:
                part = part.strip()
                if not part:
                    continue
                if not buf or len(buf) < 80 or len(buf) + len(part) + 1 <= 200:
                    buf = (buf + " " + part).strip()
                else:
                    chunks.append(buf)
                    buf = part
            if buf:
                chunks.append(buf)
            return chunks

        # Supertonic reference (one per request; the voice identity).
        v_style = self.supertonic.get_voice_style(voice_name=voice_style)
        clean_text = text.replace('"', '').replace('„', '').replace('“', '') \
                         .replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-") \
                         .replace("*", "")
        if not clean_text.strip():
            return

        wav_array, _ = self.supertonic.synthesize(
            clean_text, voice_style=v_style, lang="bg", speed=speed
        )
        wav_data = np.asarray(wav_array).flatten()
        wav_max = np.max(np.abs(wav_data))
        if wav_max > 0:
            wav_data = wav_data / wav_max
        pcm = (wav_data * 32767).astype(np.int16)

        fd, ref_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(ref_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(pcm.tobytes())

            normalized = normalize_text(clean_text)
            for chunk in _split(normalized):
                if not chunk.strip():
                    continue
                fd2, final_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd2)
                os.unlink(final_path)
                try:
                    bgtts_synthesize(
                        checkpoint=self._bgtts_checkpoint,
                        text=chunk,
                        output=final_path,
                        speaker_wav=ref_path,
                        device="mps",
                        temperature=float(temperature),
                        top_k=int(top_k),
                        top_p=float(top_p),
                        rep_penalty=float(rep_penalty),
                        max_tokens=int(max_new_tokens),
                    )
                    if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
                        raise RuntimeError("BgTTS produced no audio output")
                    # Convert wav -> raw PCM (int16 mono 24kHz).
                    with wave.open(final_path, "rb") as wf:
                        nframes = wf.getnframes()
                        sr = wf.getframerate()
                        raw_pcm = wf.readframes(nframes)
                    yield raw_pcm, nframes / sr
                finally:
                    try:
                        os.remove(final_path)
                    except OSError:
                        pass
        finally:
            try:
                os.remove(ref_path)
            except OSError:
                pass

    # -- idle unload loop ---------------------------------------------
    def idle_loop(self, stop_event):
        while not stop_event.is_set():
            time.sleep(1.0)
            with self.lock:
                if self._loaded and (time.time() - self.last_used) > self.idle_timeout:
                    self._do_unload()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model-dir", default=_MODEL_DIR)
    parser.add_argument("--voice-style", default="M5")
    parser.add_argument("--speed", type=float, default=1.6)
    parser.add_argument("--idle-timeout", type=int, default=300,
                        help="Unload model after Ns idle (default 300 = 5min)")
    args = parser.parse_args()

    state = TTSState(args.model_dir, args.idle_timeout,
                     voice_style=args.voice_style, speed=args.speed)
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
            if self.path.split("?")[0] == "/shutdown":
                threading.Thread(target=lambda: (
                    time.sleep(0.1),
                    self.server.shutdown(),
                ), daemon=True).start()
                self._send_json(200, {"status": "shutting down"})
                return

            path = self.path.split("?")[0]
            if path not in ("/tts", "/stream"):
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
                    "voice_style": body.get("voice_style", args.voice_style),
                    "speed": float(body.get("speed", args.speed)),
                    "temperature": float(body.get("temperature", 0.7)),
                    "top_k": int(body.get("top_k", 250)),
                    "top_p": float(body.get("top_p", 0.95)),
                    "rep_penalty": float(body.get("rep_penalty", 1.1)),
                    "max_new_tokens": int(body.get("max_new_tokens", 512)),
                }
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {"error": f"bad request: {exc}"})
                return

            with state.lock:
                if not state.ensure_loaded():
                    self._send_json(500, {"error": "model not loaded"})
                    return
                t0 = time.time()

                if path == "/stream":
                    # Streaming: yield one NDJSON line per sentence chunk as
                    # it's generated (base64-encoded raw PCM, 24 kHz mono).
                    # Each line: {"chunk_index": i, "pcm_base64": "...",
                    #             "duration": 0.0}
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    import base64 as _b64
                    idx = 0
                    try:
                        for pcm_bytes, dur in state.synthesize_stream(text, **opts):
                            line = json.dumps({
                                "chunk_index": idx,
                                "pcm_base64": _b64.b64encode(pcm_bytes).decode("utf-8"),
                                "duration": round(dur, 3),
                            }) + "\n"
                            payload = line.encode("utf-8")
                            self.wfile.write(f"{len(payload):X}\r\n".encode("utf-8"))
                            self.wfile.write(payload)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                            idx += 1
                            print(f"  ➤ chunk {idx-1}: {dur:.2f}s PCM "
                                  f"({len(pcm_bytes)} bytes)", flush=True)
                        self.wfile.write(b"0\r\n\r\n")  # chunked terminator
                        self.wfile.flush()
                    except Exception as exc:  # noqa: BLE001
                        # Terminate the chunked stream on error.
                        try:
                            err_line = (json.dumps({"error": str(exc)}) + "\n").encode("utf-8")
                            self.wfile.write(f"{len(err_line):X}\r\n".encode("utf-8"))
                            self.wfile.write(err_line)
                            self.wfile.write(b"\r\n")
                            self.wfile.write(b"0\r\n\r\n")
                            self.wfile.flush()
                        except Exception:  # noqa: BLE001
                            pass
                    print(f"✅ {text[:40]!r} -> {idx} streamed chunks "
                          f"({time.time()-t0:.2f}s)", flush=True)
                    return

                # Whole-file /tts path.
                try:
                    wav_bytes, dur = state.synthesize(text, **opts)
                except Exception as exc:  # noqa: BLE001
                    self._send_json(500, {"error": f"synthesis failed: {exc}"})
                    return
                synth_time = time.time() - t0

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(wav_bytes)))
            self.send_header("X-Synth-Ms", str(round(synth_time * 1000)))
            self.send_header("X-Duration-S", str(round(dur, 3)))
            self.end_headers()
            self.wfile.write(wav_bytes)
            print(f"✅ {text[:40]!r} -> {len(wav_bytes)} bytes wav ({dur:.2f}s, "
                  f"synth {synth_time:.2f}s)", flush=True)

    global start_time
    start_time = time.time()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    t = threading.Thread(target=state.idle_loop, args=(stop_event,), daemon=True)
    t.start()
    print(f"🚀 bg-tts-st server on http://{args.host}:{args.port} "
          f"(voice={args.voice_style}, idle unload {args.idle_timeout}s)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()