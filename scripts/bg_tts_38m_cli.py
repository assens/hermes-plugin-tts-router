#!/usr/bin/env python3
"""
BgTTS-38M test CLI.

A small command-line tool to test the beleata74/BgTTS-38M encoder-decoder TTS
model in isolation (does NOT touch tts-router routing).

The model is a lightweight 38M-param Bulgarian/English TTS with zero-shot
voice cloning via MioCodec. Unlike bg-tts-v7 it has a proper end-of-speech
token, so it terminates cleanly. It requires a reference speaker (a WAV clip
or a pre-saved embedding).

It runs in the dedicated bg-tts venv (Python >= 3.12, torch + miocodec),
synthesizes ``text`` to a temp wav, plays it with ffplay, and deletes the
temp file.

Usage
-----
    ~/.hermes/scripts/bg_tts_38m_cli.py "Текст за синтез" [--speaker-wav ref.wav]

If no speaker is given, the model's bundled Bulgarian/English mixed sample
(samples/sample_mixed.wav) is used as the reference voice.
"""

import argparse
import json
import os
import subprocess
import sys

DEFAULT_VENV = os.path.expanduser("~/.hermes/venvs/bg-tts/bin/python")
DEFAULT_MODEL_DIR = os.path.expanduser("~/.hermes/models/BgTTS-38M")
DEFAULT_SPEAKER = os.path.join(DEFAULT_MODEL_DIR, "samples", "sample_mixed.wav")

# Worker synthesis logic — mirrors the model card's inference path.
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
    import torch

    model_dir = os.path.expanduser(cfg.get("model_dir", "~/.hermes/models/BgTTS-38M"))
    os.chdir(model_dir)
    sys.path.insert(0, model_dir)

    from model import load_for_inference
    from tokenizer import TTSTokenizer
    from codec import CodecV6
    from inference import generate

    device = cfg.get("device", "mps")

    print(f"Loading BgTTS-38M on {device}...")
    model = load_for_inference(model_dir, device=device)
    tokenizer = TTSTokenizer()
    codec = CodecV6(device=device)

    # Resolve the speaker embedding: a WAV clip (voice cloning) or a saved
    # .pt/.npy embedding.
    speaker_wav = cfg.get("speaker_wav")
    speaker_emb_path = cfg.get("speaker_emb_path")
    if speaker_emb_path:
        emb_path = os.path.expanduser(speaker_emb_path)
        if emb_path.endswith(".npy"):
            import numpy as np
            speaker_emb = torch.from_numpy(np.load(emb_path)).to(device)
        else:
            speaker_emb = torch.load(emb_path, map_location=device, weights_only=False)
        if isinstance(speaker_emb, dict):
            speaker_emb = speaker_emb.get("global_embedding", speaker_emb.get("embedding"))
        if speaker_emb.dim() > 1:
            speaker_emb = speaker_emb.squeeze()
        speaker_emb = speaker_emb.to(device)
        print(f"Speaker from preset: {tuple(speaker_emb.shape)}")
    elif speaker_wav:
        wav_path = os.path.expanduser(speaker_wav)
        if not os.path.exists(wav_path):
            raise RuntimeError(f"speaker wav not found: {wav_path}")
        ref = codec.encode(wav_path)
        speaker_emb = ref["global_embedding"].to(device)
        print(f"Speaker from wav: {wav_path}")
    else:
        raise RuntimeError("No speaker provided — pass --speaker-wav or --speaker-emb")

    codes = generate(
        model, tokenizer, text, speaker_emb,
        max_new_tokens=int(cfg.get("max_new_tokens", 512)),
        temperature=float(cfg.get("temperature", 0.7)),
        top_k=int(cfg.get("top_k", 250)),
        top_p=float(cfg.get("top_p", 0.95)),
        rep_penalty=float(cfg.get("rep_penalty", 1.1)),
        device=device,
    )

    if codes is None or len(codes) == 0:
        print(json.dumps({"ok": False, "error": "no audio generated"}))
        return 1

    # Decode to wav (CodecV6.tokens_to_wav saves the file and returns samples).
    wav = codec.tokens_to_wav(codes, speaker_emb, output_path)
    print(json.dumps({"ok": True, "output_path": output_path,
                      "num_codes": int(len(codes)),
                      "duration": round(len(wav) / 24000.0, 3)}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''


def build_cfg(args) -> dict:
    cfg = {}
    if args.model_dir:
        cfg["model_dir"] = os.path.expanduser(args.model_dir)
    # Default the speaker to the bundled Bulgarian sample unless the user
    # explicitly provided a speaker (wav or embedding).
    if args.speaker_emb:
        cfg["speaker_emb_path"] = os.path.expanduser(args.speaker_emb)
    elif args.speaker_wav:
        cfg["speaker_wav"] = os.path.expanduser(args.speaker_wav)
    else:
        cfg["speaker_wav"] = DEFAULT_SPEAKER
    if args.device:
        cfg["device"] = args.device
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.top_k is not None:
        cfg["top_k"] = args.top_k
    if args.top_p is not None:
        cfg["top_p"] = args.top_p
    if args.rep_penalty is not None:
        cfg["rep_penalty"] = args.rep_penalty
    return cfg


def main():
    parser = argparse.ArgumentParser(
        prog="bg_tts_38m_cli",
        description="Test the beleata74/BgTTS-38M encoder-decoder TTS model "
                    "in isolation (does not touch tts-router routing).",
    )
    parser.add_argument("text", nargs="?", help="Bulgarian/English text to synthesize")
    parser.add_argument("--python", default=DEFAULT_VENV,
                        help="Path to the bg-tts venv python (default: the "
                             "dedicated ~/.hermes/venvs/bg-tts/bin/python)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="Path to the BgTTS-38M model directory "
                             "(default: ~/.hermes/models/BgTTS-38M)")
    parser.add_argument("--speaker-wav", default=None,
                        help="Reference audio WAV for voice cloning (default: "
                             "the model's bundled sample_mixed.wav). Required if "
                             "--speaker-emb is not given.")
    parser.add_argument("--speaker-emb", default=None,
                        help="Path to a saved speaker embedding (.pt or .npy). "
                             "Overrides --speaker-wav if given.")
    parser.add_argument("--device", default="mps",
                        help="Inference device: mps (Apple Silicon), cpu, cuda "
                             "(default: mps)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default: 0.7; 0.5 stable, "
                             "0.7-0.8 expressive)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k filtering (default: 250)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling threshold (default: 0.95)")
    parser.add_argument("--rep-penalty", type=float, default=None,
                        help="Repetition penalty (default: 1.1)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Max decoder steps (default: 512; model degrades "
                             "past ~475 / 19s)")
    args = parser.parse_args()

    if not args.text:
        parser.error("text argument is required (e.g. bg_tts_38m_cli.py \"Здравей\")")

    # Generate to a temp wav, play with ffplay, then delete.
    import tempfile
    fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="bg-tts-38m-")
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
        print(f"ERROR: bg-tts venv python not found at {python_bin}")
        print("Create it with: python3.13 -m venv ~/.hermes/venvs/bg-tts && "
              "~/.hermes/venvs/bg-tts/bin/pip install 'transformers<5' soundfile "
              "torchaudio 'miocodec @ git+https://github.com/Aratako/MioCodec@main' accelerate")
        sys.exit(1)

    model_dir = cfg.get("model_dir", DEFAULT_MODEL_DIR)
    if not os.path.exists(os.path.join(model_dir, "checkpoint.pt")):
        print(f"ERROR: BgTTS-38M checkpoint not found in {model_dir}")
        print("Download it with: "
              "~/.hermes/venvs/bg-tts/bin/huggingface-cli download beleata74/BgTTS-38M "
              "--local-dir ~/.hermes/models/BgTTS-38M")
        sys.exit(1)

    # Strip PYTHONPATH/HOME so the dedicated venv stays self-contained.
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("PYTHONPATH", "PYTHONHOME")}

    print(f"Synthesizing via BgTTS-38M (device={cfg.get('device', 'mps')})...")
    print(f"  text: {args.text}")
    if cfg.get("speaker_wav"):
        print(f"  speaker: {cfg['speaker_wav']}")
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
        print(f"Generated {result['num_codes']} codes, duration: {result['duration']}s")

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