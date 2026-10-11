#!/usr/bin/env python3
"""
PaddleOCR worker — runs inside venv_paddle, called as subprocess by paddle_engine.py.

Usage:
    python paddle_worker.py '<config_json>' '<image_path>'

Writes a single JSON object to stdout:
    {"lines": [...], "confidences": [...], "lang": "...", "use_gpu": false}

On error writes:
    {"error": "<message>"}
"""

import os
import sys
import json
import traceback

# Disable slow model-source connectivity check and suppress verbose output
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_DISABLE_RESOURCE_CHECK", "1")
# Note: FLAGS_use_mkldnn env var is NOT honoured by paddle.inference.Config in Paddle 3.x.
# MKL-DNN is disabled via enable_mkldnn=False passed to PaddleOCR() constructor instead.


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: paddle_worker.py <config_json> <image_path>"}))
        sys.exit(1)

    try:
        config = json.loads(sys.argv[1])
        image_path = sys.argv[2]
    except (json.JSONDecodeError, IndexError) as e:
        print(json.dumps({"error": f"Bad arguments: {e}"}))
        sys.exit(1)

    try:
        import numpy as np
        from PIL import Image
        from paddleocr import PaddleOCR
    except ImportError as e:
        print(json.dumps({"error": f"Import failed: {e}"}))
        sys.exit(1)

    lang = config.get("lang", "en")
    use_gpu = config.get("use_gpu", False)
    angle_cls = config.get("use_angle_cls", True)
    det_thresh = config.get("det_db_thresh", 0.30)

    # PaddleOCR 3.x renamed several parameters
    try:
        ocr = PaddleOCR(
            use_textline_orientation=angle_cls,  # was: use_angle_cls
            lang=lang,
            device="gpu" if use_gpu else "cpu",  # was: use_gpu=bool
            text_det_thresh=det_thresh,           # was: det_db_thresh
            enable_mkldnn=False,                  # avoid ConvertPirAttribute crash on CPU
        )

        img = np.array(Image.open(image_path).convert("RGB"))
        result = ocr.predict(img)  # ocr() is deprecated in PaddleOCR 3.x

        lines = []
        confidences = []
        if result:
            page = result[0]  # OCRResult object (dict-like) for the first (only) image
            try:
                rec_texts = page["rec_texts"]
                rec_scores = page["rec_scores"]
                for text, conf in zip(rec_texts, rec_scores):
                    text = str(text)
                    if text:
                        lines.append(text)
                        confidences.append(float(conf))
            except (KeyError, TypeError):
                pass

        print(json.dumps({
            "lines": lines,
            "confidences": confidences,
            "lang": lang,
            "use_gpu": use_gpu,
        }, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}))
        sys.exit(1)


if __name__ == "__main__":
    main()
