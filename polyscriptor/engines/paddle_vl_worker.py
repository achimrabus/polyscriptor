#!/usr/bin/env python3
"""
PaddleOCR-VL worker — runs inside the isolated PaddleOCR-VL venv, called as a
subprocess by paddle_vl_engine.py.

Uses the `paddleocr.PaddleOCRVL` pipeline (native PaddlePaddle backend), which
loads and runs reliably. NOTE: the default eager backend is fast per text line
(~1.4s on GPU) but slow on dense full pages; for high page-level throughput the
vendor path is a vLLM server (vl_rec_backend="vllm-server"), not wired here.

Usage:
    python paddle_vl_worker.py '<config_json>' '<image_path>'

config_json keys:
    use_gpu (bool), gpu_index (int), pipeline_version (str), prompt (str, unused
    by the pipeline path), cache_root (str)

Writes a single JSON object to stdout:
    {"text": "...", "markdown": "...", "lines": [...], "use_gpu": bool,
     "prompt": "...", "pipeline_version": "..."}
On error:
    {"error": "<message>", "traceback": "..."}
"""

import os
import sys
import json
import traceback

# Disable slow model-source connectivity check.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _markdown_text(res) -> str:
    """Return the reflowed markdown string for a result (for the 'markdown' field)."""
    md = getattr(res, "markdown", None)
    if isinstance(md, dict):
        return str(md.get("markdown_texts") or md.get("text") or "")
    if md is not None:
        return str(md)
    return ""


def _extract_blocks(res) -> list:
    """
    Return an ordered list of text blocks for a result.

    Prefers res.json['parsing_res_list'] (one entry per detected layout block,
    e.g. paragraph/heading) sorted by reading order — this preserves structure
    instead of collapsing the whole page into one reflowed blob. Falls back to
    the markdown string split on blank lines.
    """
    try:
        j = res.json if hasattr(res, "json") else None
        blocks = (j or {}).get("parsing_res_list")
        if blocks:
            ordered = sorted(
                blocks,
                key=lambda b: (b.get("block_order") if b.get("block_order") is not None else 1e9),
            )
            texts = []
            for b in ordered:
                content = str(b.get("block_content", "")).strip()
                if content:
                    texts.append(content)
            if texts:
                return texts
    except Exception:
        pass
    # Fallback: split markdown on blank lines
    md = _markdown_text(res)
    return [p.strip() for p in md.split("\n\n") if p.strip()]


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: paddle_vl_worker.py <config_json> <image_path>"}))
        sys.exit(1)

    try:
        config = json.loads(sys.argv[1])
        image_path = sys.argv[2]
    except (json.JSONDecodeError, IndexError) as e:
        print(json.dumps({"error": f"Bad arguments: {e}"}))
        sys.exit(1)

    use_gpu = bool(config.get("use_gpu", True))
    gpu_index = int(config.get("gpu_index", 0))
    pipeline_version = config.get("pipeline_version", "v1.5")
    prompt = config.get("prompt", "OCR:")

    # Optional predict() knobs — only forwarded when set (None = pipeline default).
    predict_kwargs = {}
    prompt_label = config.get("prompt_label") or None
    if prompt_label:
        predict_kwargs["prompt_label"] = prompt_label
    if config.get("max_new_tokens"):
        predict_kwargs["max_new_tokens"] = int(config["max_new_tokens"])
    if config.get("repetition_penalty"):
        predict_kwargs["repetition_penalty"] = float(config["repetition_penalty"])

    try:
        import paddle
        # The engine pins the physical GPU via CUDA_VISIBLE_DEVICES, so the
        # device is always addressed as index 0 here.
        paddle.set_device("gpu:0" if use_gpu else "cpu")

        from paddleocr import PaddleOCRVL
        pipeline = PaddleOCRVL(pipeline_version=pipeline_version)

        output = list(pipeline.predict(image_path, **predict_kwargs))

        block_texts = []
        md_parts = []
        for res in output:
            block_texts.extend(_extract_blocks(res))
            md = _markdown_text(res)
            if md:
                md_parts.append(md)

        # One block (paragraph/region) per line — preserves layout structure
        # instead of merging everything into a single blob.
        full_text = "\n".join(block_texts).strip()
        full_md = "\n\n".join(md_parts).strip()
        lines = [ln for ln in full_text.splitlines() if ln.strip()]

        print(json.dumps({
            "text": full_text,
            "markdown": full_md,
            "lines": lines,
            "use_gpu": use_gpu,
            "prompt": prompt,
            "pipeline_version": pipeline_version,
        }, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}))
        sys.exit(1)


if __name__ == "__main__":
    main()
