#!/usr/bin/env python3
"""
DeepSeek-OCR-2 worker — runs inside venv_deepseek, called as subprocess by deepseek_ocr_engine.py.

Setup (venv_deepseek):
    python -m venv /path/to/dhlab-slavistik/venv_deepseek
    source venv_deepseek/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install "transformers==4.46.3" einops addict easydict pillow
    # Optional (faster inference, requires CUDA toolkit headers):
    pip install flash-attn --no-build-isolation
    deactivate

Usage:
    python deepseek_worker.py '<config_json>' '<image_path>'

Config keys (JSON):
    model_id           – HuggingFace model ID or local path (default: deepseek-ai/DeepSeek-OCR-2)
    device             – "cuda:0", "cuda:1", "cpu" (default: "cuda:0")
    ocr_mode           – "document" (markdown with layout) or "free" (plain text) (default: "document")
    strip_markdown     – bool, remove markdown symbols from output (default: false)
    base_size          – int, patch resolution base (512–2048, default: 1024)
    image_size         – int, output resolution (512–1024, default: 640)
    crop_mode          – bool, enable crop mode (default: true)
    repetition_penalty – float, penalise repeated tokens (1.0 = disabled, default: 1.3)

Writes a single JSON object to stdout:
    {"text": "...", "model_id": "deepseek-ai/DeepSeek-OCR-2", "ocr_mode": "document"}

On error writes:
    {"error": "<message>", "traceback": "..."}
"""

import json
import re
import shutil
import sys
import tempfile
import traceback


# ---------------------------------------------------------------------------
# Markdown stripping (kept in worker so it runs in the correct venv)
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: deepseek_worker.py <config_json> <image_path>"}))
        sys.exit(1)

    try:
        config = json.loads(sys.argv[1])
        image_path = sys.argv[2]
    except (json.JSONDecodeError, IndexError) as e:
        print(json.dumps({"error": f"Bad arguments: {e}"}))
        sys.exit(1)

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        print(json.dumps({
            "error": (
                f"Import failed: {e}. "
                "Activate venv_deepseek and install: "
                "pip install \"transformers==4.46.3\" einops addict easydict"
            )
        }))
        sys.exit(1)

    model_id = config.get("model_id") or "deepseek-ai/DeepSeek-OCR-2"
    device = config.get("device", "cuda:0")
    ocr_mode = config.get("ocr_mode", "document")
    do_strip_md = config.get("strip_markdown", False)
    base_size = config.get("base_size", 1024)
    image_size = config.get("image_size", 640)
    crop_mode = config.get("crop_mode", True)
    repetition_penalty = float(config.get("repetition_penalty", 1.3))

    try:
        # Detect flash-attn availability
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "eager"

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        model = model.to(device)
        model.eval()

        # Suppress repetition loops — common in autoregressive OCR on easy images
        if repetition_penalty != 1.0:
            model.generation_config.repetition_penalty = repetition_penalty

        if ocr_mode == "document":
            prompt = "<image>\nOCR the document with markdown format."
        else:
            prompt = "<image>\nOCR this image."

        # infer() only returns the text when eval_mode=True.
        # Other modes write to files or return None.
        # output_path is still required (makedirs called unconditionally).
        out_tmp = tempfile.mkdtemp(prefix="deepseek_ocr_")
        try:
            with torch.no_grad():
                result = model.infer(
                    tokenizer,
                    prompt,
                    image_path,
                    output_path=out_tmp,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    eval_mode=True,
                )
        finally:
            shutil.rmtree(out_tmp, ignore_errors=True)

        if do_strip_md:
            result = _strip_markdown(result)

        print(json.dumps({
            "text": result,
            "model_id": model_id,
            "ocr_mode": ocr_mode,
        }, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
