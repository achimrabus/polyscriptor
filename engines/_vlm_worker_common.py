"""Shared command line for the subprocess OCR-VLM workers (GLM-OCR, dots.mocr).

Standard library only: each worker runs in its own venv and imports this module from
its own directory.

Two call forms:

    worker.py '<config_json>' <image_path>
        one image, prints ONE JSON object (the form the engine plugins use)

    worker.py '<config_json>' --list <file>
        <file> holds one image path per line; the model is loaded once and every image
        is answered with one JSON line {"image": ..., "text": ..., "seconds": ...}
        (or {"image": ..., "error": ...}). For evaluation and batch runs, where a
        model load per image would dominate the run time.
"""

import json
import sys
import time
import traceback


def _emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def run(name, load, transcribe, meta=None):
    """Parse argv, load the model once, transcribe one image or a list.

    load(config) -> state; transcribe(state, image_path) -> text;
    meta(config) -> dict of extra fields for the single-image result.
    """
    usage = f"Usage: {name} <config_json> <image_path> | <config_json> --list <file>"
    args = sys.argv[1:]
    if len(args) not in (2, 3) or (len(args) == 3 and args[1] != "--list"):
        _emit({"error": usage})
        sys.exit(1)
    try:
        config = json.loads(args[0])
    except json.JSONDecodeError as e:
        _emit({"error": f"Bad config JSON: {e}"})
        sys.exit(1)

    if len(args) == 3:
        try:
            with open(args[2], encoding="utf-8") as f:
                images = [line.strip() for line in f if line.strip()]
        except OSError as e:
            _emit({"error": f"Cannot read list file: {e}"})
            sys.exit(1)
    else:
        images = [args[1]]

    try:
        state = load(config)
    except Exception as e:
        _emit({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
        sys.exit(1)

    if len(args) == 2:
        try:
            text = transcribe(state, images[0])
        except Exception as e:
            _emit({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
            sys.exit(1)
        _emit({"text": text, **(meta(config) if meta else {})})
        return

    for path in images:
        t0 = time.time()
        try:
            text = transcribe(state, path)
            _emit({"image": path, "text": text, "seconds": round(time.time() - t0, 3)})
        except Exception as e:
            _emit({"image": path, "error": f"{type(e).__name__}: {e}"})
