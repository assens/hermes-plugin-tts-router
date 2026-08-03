#!/usr/bin/env python3
"""
Ani-Voice-API test CLI.

A small command-line tool to test the beleata74/Ani-Voice-API Bulgarian TTS
in isolation (does NOT touch tts-router routing).

The model is a two-stage pipeline:
  1. **Supertonic** generates a reference audio clip in a chosen voice style
     (F1-F5 female, M1-M5 male) for the given text.
  2. **BgTTS** uses that reference as the speaker embedding and synthesizes
     the final audio.

It also applies Bulgarian text normalization (bg-text-normalizer +
num2cyrillic) so numbers/dates are read correctly.

Runs in a dedicated venv (~/.hermes/venvs/ani-voice), synthesizes ``text`` to
a temp wav, plays it with ffplay, and deletes the temp file.

Usage
-----
    ~/.hermes/scripts/ani_voice_cli.py "Текст за синтез" [--voice-style F5]
"""

import argparse
import json
import os
import subprocess
import sys

DEFAULT_VENV = os.path.expanduser("~/.hermes/venvs/ani-voice/bin/python")
DEFAULT_MODEL_DIR = os.path.expanduser("~/.hermes/models/Ani-Voice-API")

# Worker synthesis logic — mirrors tts_engine.py's two-stage pipeline.
_WORKER = r'''
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

    import os
    import io
    import wave
    import numpy as np
    import torch
    import tempfile

    model_dir = os.path.expanduser(cfg.get("model_dir", "~/.hermes/models/Ani-Voice-API"))
    os.chdir(model_dir)
    sys.path.insert(0, model_dir)
    sys.path.insert(0, os.path.join(model_dir, "BgTTS"))

    from normalizer import normalize_text
    from inference import synthesize

    device = cfg.get("device", "mps")
    voice_style = cfg.get("voice_style", "F5")
    speed = float(cfg.get("speed", 1.6))

    # ---- 1. Supertonic: generate reference audio in the voice style ----
    import supertonic
    from supertonic import TTS
    st = TTS(auto_download=True)
    v_style = st.get_voice_style(voice_name=voice_style)

    clean_text = text.replace('"', '').replace('„', '').replace('“', '') \
                     .replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-") \
                     .replace("*", "")
    if not clean_text.strip():
        raise RuntimeError("empty text after cleaning")

    wav_array, _ = st.synthesize(clean_text, voice_style=v_style, lang="bg", speed=speed)
    wav_data = np.asarray(wav_array).flatten()
    wav_max = np.max(np.abs(wav_data))
    if wav_max > 0:
        wav_data = wav_data / wav_max
    pcm_data = (wav_data * 32767).astype(np.int16)

    fd, ref_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(ref_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(pcm_data.tobytes())

    # ---- 2. BgTTS: synthesize final audio using the reference as speaker ----
    fd2, final_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd2)
    os.unlink(final_path)  # inference.synthesize writes it
    try:
        normalized = normalize_text(clean_text)
        synthesize(
            checkpoint=os.path.join(model_dir, "BgTTS", "checkpoint_inference.pt"),
            text=normalized,
            output=final_path,
            speaker_wav=ref_path,
            device=device,
            temperature=float(cfg.get("temperature", 0.7)),
            top_k=int(cfg.get("top_k", 250)),
            top_p=float(cfg.get("top_p", 0.95)),
            rep_penalty=float(cfg.get("rep_penalty", 1.1)),
            max_tokens=int(cfg.get("max_new_tokens", 512)),
        )
        if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            raise RuntimeError("BgTTS produced no audio output")

        # Copy the final wav to the requested output path.
        import shutil
        shutil.copyfile(final_path, output_path)

        # Determine duration via the codec sample rate (24000).
        dur = 0.0
        try:
            with wave.open(output_path, "rb") as wf:
                dur = wf.getnframes() / wf.getframerate()
        except Exception:
            pass
        print(json.dumps({"ok": True, "output_path": output_path,
                          "voice_style": voice_style, "duration": round(dur, 3)}))
        return 0
    finally:
        for p in (ref_path, final_path):
            try:
                os.remove(p)
            except OSError:
                pass

if __name__ == "__main__":
    sys.exit(main())
'''


def build_cfg(args) -> dict:
    cfg = {}
    if args.model_dir:
        cfg["model_dir"] = os.path.expanduser(args.model_dir)
    if args.voice_style:
        cfg["voice_style"] = args.voice_style
    if args.speed is not None:
        cfg["speed"] = args.speed
    if args.device:
        cfg["device"] = args.device
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.top_k is not None:
        cfg["top_k"] = args.top_k
    if args.top_p is not None:
        cfg["top_p"] = args.top_p
    if args.rep_penalty is not None:
        cfg["rep_penalty"] = args.rep_penalty
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens
    return cfg


def main():
    parser = argparse.ArgumentParser(
        prog="ani_voice_cli",
        description="Test the beleata74/Ani-Voice-API Bulgarian TTS "
                    "(Supertonic reference + BgTTS) in isolation "
                    "(does not touch tts-router routing).",
    )
    parser.add_argument("text", nargs="?", help="Bulgarian text to synthesize")
    parser.add_argument("--python", default=DEFAULT_VENV,
                        help="Path to the ani-voice venv python (default: the "
                             "dedicated ~/.hermes/venvs/ani-voice/bin/python)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="Path to the Ani-Voice-API model directory "
                             "(default: ~/.hermes/models/Ani-Voice-API)")
    parser.add_argument("--voice-style", default="F5",
                        help="Supertonic voice style (default: F5). "
                             "Female: F1-F5, Male: M1-M5")
    parser.add_argument("--speed", type=float, default=None,
                        help="Reference speech speed (default: 1.6)")
    parser.add_argument("--device", default="mps",
                        help="Inference device: mps (Apple Silicon), cpu, cuda "
                             "(default: mps)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="BgTTS sampling temperature (default: 0.7)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="BgTTS top-k (default: 250)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="BgTTS nucleus threshold (default: 0.95)")
    parser.add_argument("--rep-penalty", type=float, default=None,
                        help="BgTTS repetition penalty (default: 1.1)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="BgTTS max decoder steps (default: 512)")
    args = parser.parse_args()

    if not args.text:
        parser.error("text argument is required (e.g. ani_voice_cli.py \"Здравей\")")

    # Generate to a temp wav, play with ffplay, then delete.
    import tempfile
    fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="ani-voice-")
    os.close(fd)
    os.unlink(output_path)  # worker writes it

    cfg = build_cfg(args)
    opts = {
        "text": args.text,
        "output_path": output_path,
        "cfg": cfg,
    }

    python_bin = os.path.expanduser(args.python)
    if not os.path.exists(python_bin):
        print(f"ERROR: ani-voice venv python not found at {python_bin}")
        print("Create it with: python3.13 -m venv ~/.hermes/venvs/ani-voice && "
              "~/.hermes/venvs/ani-voice/bin/pip install supertonic "
              "bg-text-normalizer num2cyrillic soundfile numpy torch torchaudio "
              "'miocodec @ git+https://github.com/Aratako/MioCodec@main'")
        sys.exit(1)

    model_dir = cfg.get("model_dir", DEFAULT_MODEL_DIR)
    if not os.path.exists(os.path.join(model_dir, "BgTTS", "checkpoint_inference.pt")):
        print(f"ERROR: Ani-Voice-API BgTTS checkpoint not found in {model_dir}")
        print("Download it with: "
              "~/.hermes/venvs/ani-voice/bin/hf download beleata74/Ani-Voice-API "
              "--local-dir ~/.hermes/models/Ani-Voice-API")
        sys.exit(1)

    # Strip PYTHONPATH/HOME so the dedicated venv stays self-contained.
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("PYTHONPATH", "PYTHONHOME")}

    print(f"Synthesizing via Ani-Voice-API (voice={cfg.get('voice_style', 'F5')}, "
          f"device={cfg.get('device', 'mps')})...")
    print(f"  text: {args.text}")
    proc = subprocess.run(
        [python_bin, "-c", _WORKER, json.dumps(opts)],
        capture_output=True, text=True, timeout=600, env=clean_env,
    )

    # Parse the last JSON status line from stdout.
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
        print(f"ERROR: synthesis failed: {detail}")
        if proc.stderr.strip():
            print("\n--- stderr (last 30 lines) ---")
            print("\n".join(proc.stderr.strip().splitlines()[-30:]))
        sys.exit(1)

    out = result.get("output_path") or output_path
    if result.get("duration"):
        print(f"Voice style: {result['voice_style']}, duration: {result['duration']}s")

    # Play the audio with ffplay, then clean up the temp file.
    print("Playing audio...")
    try:
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", out],
                       check=True)
    except FileNotFoundError:
        print("ERROR: ffplay not found (install ffmpeg with the ffplay binary). "
              f"Generated audio left at: {out}")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: ffplay exited with code {exc.returncode}")
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass
    print("Done.")


if __name__ == "__main__":
    main()