"""Kraken 7 line recognition worker (PP-OCRv6 and other kraken 7 recognition models).

Runs in venv_kraken7 (kraken 7.1.x, which the main venv cannot host: blla,
the Kraken engine and Party depend on kraken 6). Three call forms:

    venv_kraken7/bin/python engines/kraken7_worker.py '<config_json>' <line_image>
    venv_kraken7/bin/python engines/kraken7_worker.py '<config_json>' --list <file>
        standard command line of the subprocess workers, see _vlm_worker_common.py

    venv_kraken7/bin/python engines/kraken7_worker.py '<config_json>' --serve
        persistent mode for the Polyscriptor engine: loads the model once, prints
        {"ready": true, ...} and then answers one JSON request per stdin line,
        {"image": "<path>"}, with one JSON line {"text", "confidence", "seconds"}
        (or {"error": ...}). Ends on EOF or {"cmd": "quit"}.

Each input is ONE pre-cut line image; it is recognised as a single bbox line covering
the whole image (what `kraken ocr --no-segmentation` does).

Config keys:
    model_path   .safetensors / .mlmodel file (required)
    device       "cuda:0" (default) or "cpu"
    batch_size   default 1 (one line per call)
    precision    default "32-true"; bf16/16-true are flagged unstable by kraken
"""

import json
import sys
import time
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _vlm_worker_common import run  # noqa: E402


def load(config):
    from kraken.configs import RecognitionInferenceConfig
    from kraken.ketos.util import to_ptl_device
    from kraken.tasks import RecognitionTaskModel

    model_path = config.get("model_path")
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"model_path not found: {model_path!r}")
    model = RecognitionTaskModel.load_model(model_path)
    # kraken 7 configs take Lightning's (accelerator, devices) pair, not "cuda:0"
    accelerator, devices = to_ptl_device(config.get("device", "cuda:0"))
    infer_cfg = RecognitionInferenceConfig(
        accelerator=accelerator,
        device=devices,
        batch_size=int(config.get("batch_size", 1)),
        precision=config.get("precision", "32-true"),
    )
    return {"model": model, "config": infer_cfg}


def recognise(state, image_path):
    """Return (text, mean character confidence or None)."""
    from kraken.containers import BBoxLine, Segmentation
    from PIL import Image

    im = Image.open(image_path)
    im.load()
    seg = Segmentation(
        type="bbox",
        text_direction="horizontal-lr",
        imagename=str(image_path),
        script_detection=False,
        lines=[BBoxLine(id=f"_{uuid.uuid4()}", bbox=(0, 0, im.width, im.height))],
    )
    preds = list(state["model"].predict(im=im, segmentation=seg, config=state["config"]))
    text = "".join(p.prediction for p in preds)
    confs = [c for p in preds for c in (getattr(p, "confidences", None) or [])]
    return text, (sum(confs) / len(confs) if confs else None)


def transcribe(state, image_path):
    return recognise(state, image_path)[0]


def meta(config):
    return {"engine": "kraken7", "model_path": config.get("model_path")}


def _emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve(config):
    # kraken/lightning print banners to stdout; the protocol channel must stay clean
    proto, sys.stdout = sys.stdout, sys.stderr
    try:
        state = load(config)
    except Exception as e:
        sys.stdout = proto
        _emit({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
        sys.exit(1)

    def reply(obj):
        proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proto.flush()

    reply({"ready": True, "model_path": config.get("model_path")})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            reply({"error": f"bad request: {e}"})
            continue
        if req.get("cmd") == "quit":
            break
        t0 = time.time()
        try:
            text, conf = recognise(state, req["image"])
            reply({"text": text, "confidence": conf, "seconds": round(time.time() - t0, 3)})
        except Exception as e:
            reply({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2] == "--serve":
        try:
            cfg = json.loads(sys.argv[1])
        except json.JSONDecodeError as e:
            _emit({"error": f"Bad config JSON: {e}"})
            sys.exit(1)
        serve(cfg)
    else:
        run("kraken7_worker.py", load, transcribe, meta)
