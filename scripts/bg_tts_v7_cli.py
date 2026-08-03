#!/usr/bin/env python3
"""
bg-tts-v7 (transformers) test CLI.

A small command-line tool to test the beleata74/bg-tts-v7 transformers TTS
backend in isolation — useful for experimenting with the model without going
through the tts-router (this tool does NOT touch routing).

It shells out to the dedicated bg-tts venv (Python >= 3.12, torch +
transformers + miocodec), synthesizes ``text`` to a temp wav, plays it with
ffplay, and deletes the temp file.

Usage
-----
    ~/.hermes/scripts/bg_tts_v7_cli.py "Текст за синтез" [options]

Options mirror the transformers config block from the tts-router plugin.
"""

import argparse
import json
import os
import subprocess
import sys

DEFAULT_VENV = os.path.expanduser("~/.hermes/venvs/bg-tts/bin/python")
DEFAULT_PRESET = os.path.expanduser("~/.hermes/venvs/bg-tts/presets/en_female.pt")

# The worker synthesis logic (mirrors the plugin's _WORKER_SCRIPT).
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
    out_format = opts.get("format", "wav")

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

    preset_path = cfg.get("preset_path") or os.path.expanduser(
        "~/.hermes/venvs/bg-tts/presets/en_female.pt"
    )
    preset_path = os.path.expanduser(preset_path)
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

    # Truncate at the first non-speech control token. The bg-tts-v7 model
    # does NOT emit an EOS token — it keeps generating audio tokens past the
    # end of the spoken text (producing gibberish) until it hits
    # max_new_tokens. The response is terminated by a non-speech special
    # token (<|im_end|> = 151645, or <|endoftext|> = 151643). We stop the
    # first time we see a token below the speech offset (i.e. not an audio
    # token), which cleanly cuts the real speech from the hallucinated tail.
    generated_list = [int(t.item()) for t in generated]
    cut = len(generated_list)
    for i, tid in enumerate(generated_list):
        if tid < speech_offset:
            cut = i
            break
    generated_list = generated_list[:cut]

    audio_codes = [
        t - speech_offset
        for t in generated_list
        if speech_offset <= t < speech_offset + speech_vocab
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
        import subprocess as sp, tempfile
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp, wav_np, sample_rate)
            sp.run(["ffmpeg", "-y", "-i", tmp, output_path],
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


def build_cfg(args) -> dict:
    cfg = {}
    if args.model_name:
        cfg["model_name"] = args.model_name
    if args.codec_model:
        cfg["codec_model"] = args.codec_model
    if args.preset_path:
        cfg["preset_path"] = os.path.expanduser(args.preset_path)
    if args.dtype:
        cfg["dtype"] = args.dtype
    if args.max_new_tokens:
        cfg["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.top_p is not None:
        cfg["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        cfg["repetition_penalty"] = args.repetition_penalty
    if args.speech_offset is not None:
        cfg["speech_offset"] = args.speech_offset
    return cfg


def main():
    parser = argparse.ArgumentParser(
        prog="bg_tts_v7_cli",
        description="Test the beleata74/bg-tts-v7 transformers TTS backend "
                    "in isolation (does not touch tts-router routing).",
    )
    parser.add_argument("text", nargs="?", help="Bulgarian text to synthesize")
    parser.add_argument("--python", default=DEFAULT_VENV,
                        help="Path to the bg-tts venv python (default: the "
                             "dedicated ~/.hermes/venvs/bg-tts/bin/python)")
    parser.add_argument("--model-name", default=None,
                        help="Override model name (default: beleata74/bg-tts-v7)")
    parser.add_argument("--codec-model", default=None,
                        help="Override codec model (default: Aratako/MioCodec-25Hz-24kHz)")
    parser.add_argument("--preset-path", default=None,
                        help="Path to speaker preset .pt (default: "
                             "~/.hermes/venvs/bg-tts/presets/en_female.pt)")
    parser.add_argument("--dtype", default=None,
                        help="Model dtype (default: bfloat16)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Max new tokens for generation (default: 500)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling top-p (default: 0.9)")
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="Repetition penalty (default: 1.1)")
    parser.add_argument("--speech-offset", type=int, default=None,
                        help="Speech token offset (default: 151669)")
    args = parser.parse_args()

    if not args.text:
        parser.error("text argument is required (e.g. bg_tts_v7_cli.py \"Здравей\")")

    # Generate to a temp wav file, play it with ffplay, then delete it.
    import tempfile
    fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="bg-tts-v7-")
    os.close(fd)
    os.unlink(output_path)  # worker writes it

    cfg = build_cfg(args)
    opts = {
        "text": args.text,
        "output_path": output_path,
        "cfg": cfg,
        "format": "wav",
    }

    python_bin = os.path.expanduser(args.python)
    if not os.path.exists(python_bin):
        print(f"ERROR: bg-tts venv python not found at {python_bin}")
        print("Create it with: python3.13 -m venv ~/.hermes/venvs/bg-tts && "
              "~/.hermes/venvs/bg-tts/bin/pip install 'transformers<5' soundfile "
              "'miocodec @ git+https://github.com/Aratako/MioCodec@main' accelerate")
        sys.exit(1)

    # Strip PYTHONPATH/HOME so the dedicated venv stays self-contained.
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("PYTHONPATH", "PYTHONHOME")}

    print(f"Synthesizing via bg-tts-v7 (model={cfg.get('model_name', 'beleata74/bg-tts-v7')})...")
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
        # Show the tail of stderr for debugging.
        if proc.stderr.strip():
            print("\n--- stderr (last 30 lines) ---")
            print("\n".join(proc.stderr.strip().splitlines()[-30:]))
        sys.exit(1)

    out = result.get("output_path") or output_path

    # Report duration.
    try:
        dur = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", out],
            capture_output=True, text=True,
        ).stdout.strip()
        if dur:
            print(f"Duration: {dur}s")
    except Exception:
        pass

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