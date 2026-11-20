#!/usr/bin/env python3
"""
GLM-OCR worker — runs inside venv_glm, called as subprocess by glm_ocr_engine.py.

GLM-OCR (zai-org/GLM-OCR, ~1.3B) is built on the GLM-V encoder-decoder with a
0.5B language decoder.  It requires transformers >= 5.3.0 (AutoModelForMultimodalLM),
which conflicts with the main venv — hence the isolated venv + worker pattern.

Setup (venv_glm):
    python -m venv /path/to/dhlab-slavistik/venv_glm
    source venv_glm/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # transformers 5.x needs torch >= 2.6
    pip install "transformers>=5.3.0" accelerate pillow
    deactivate

Usage:
    python glm_ocr_worker.py '<config_json>' '<image_path>'
    python glm_ocr_worker.py '<config_json>' --list <file>   # one path per line, JSON lines out

Config keys (JSON):
    model_id           – HuggingFace model ID or local path (default: zai-org/GLM-OCR)
    device             – "cuda:0", "cuda:1", "cpu" (default: "cuda:0")
    prompt_mode        – "text" | "formula" | "table" | "free" (default: "text")
    strip_markdown     – bool, remove markdown symbols from output (default: false)
    max_new_tokens     – int, generation limit (default: 8192)
    repetition_penalty – float (1.0 = disabled, default: 1.1)

Whole-page usage (no layout pipeline): the benchmark convention is the single
prompt "Text Recognition:" on the full page; the official SDK additionally runs
PP-DocLayoutV3 + per-region recognition.

Writes a single JSON object to stdout:
    {"text": "...", "model_id": "zai-org/GLM-OCR", "prompt_mode": "text"}

On error writes:
    {"error": "<message>", "traceback": "..."}
"""

import re

from _vlm_worker_common import run


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


_PROMPTS = {
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
    "free": "Extract all readable content from this document page.",
}


def load(config):
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM
    except ImportError as e:
        raise RuntimeError(
            f"Import failed: {e}. Activate venv_glm and install: "
            'pip install "transformers>=5.3.0" accelerate pillow'
        )
    model_id = config.get("model_id") or "zai-org/GLM-OCR"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=config.get("device", "cuda:0"),
    )
    model.eval()
    generate_kwargs = dict(max_new_tokens=int(config.get("max_new_tokens", 8192)), do_sample=False)
    repetition_penalty = float(config.get("repetition_penalty", 1.1))
    if repetition_penalty != 1.0:
        generate_kwargs["repetition_penalty"] = repetition_penalty
    return {
        "torch": torch, "processor": processor, "model": model,
        "prompt": _PROMPTS.get(config.get("prompt_mode", "text"), _PROMPTS["text"]),
        "strip_md": config.get("strip_markdown", False),
        "generate_kwargs": generate_kwargs,
    }


def transcribe(st, image_path):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": st["prompt"]},
    ]}]
    # The image must go through AutoProcessor; AutoTokenizer.apply_chat_template
    # silently drops it (13 input ids, empty Markdown block as output).
    inputs = st["processor"].apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(st["model"].device)
    with st["torch"].no_grad():
        output_ids = st["model"].generate(**inputs, **st["generate_kwargs"])
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = st["processor"].decode(new_tokens, skip_special_tokens=True)
    return _strip_markdown(text) if st["strip_md"] else text


def _meta(config):
    return {"model_id": config.get("model_id") or "zai-org/GLM-OCR",
            "prompt_mode": config.get("prompt_mode", "text")}


if __name__ == "__main__":
    run("glm_ocr_worker.py", load, transcribe, _meta)
