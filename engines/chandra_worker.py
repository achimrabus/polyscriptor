#!/usr/bin/env python3
"""
Chandra worker — runs inside the isolated Chandra venv (chandra-ocr[hf] +
transformers >= 5.x), called as a subprocess by chandra_engine.py.

Loads datalab-to/chandra-ocr-2 (5B VLM) via plain transformers
(AutoModelForImageTextToText) and OCRs one full page image. The model is
trained to emit HTML; the worker converts that to plain text for the engine
while also returning the raw HTML.

Usage:
    python chandra_worker.py '<config_json>' '<image_path>'

config_json keys:
    model_id (str, default "datalab-to/chandra-ocr-2")
    use_gpu (bool), prompt_mode ("ocr" | "ocr_layout" | "custom"),
    custom_prompt (str), max_new_tokens (int, default 4096)

Writes a single JSON object to stdout:
    {"text": "...", "html": "...", "lines": [...], "use_gpu": bool}
On error:
    {"error": "<message>", "traceback": "..."}
"""

import html as html_lib
import json
import re
import sys
import traceback

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t]+")
THINK_RE = re.compile(r"<think>.*?</think>", flags=re.S)
BLOCK_RE = re.compile(r"</(?:p|div|tr|li|h[1-6])>|<br\s*/?>", flags=re.I)

FALLBACK_OCR_PROMPT = (
    "OCR this image to HTML. Preserve the text exactly as written, "
    "including line breaks (use <br> or <p> per line)."
)


def html_to_text(s: str) -> str:
    """Convert Chandra's HTML output to plain text, one line per block element."""
    # If stop tokens failed, keep only the first assistant answer
    s = s.split("\nassistant\n")[0].split("assistant\n<think>")[0]
    s = THINK_RE.sub(" ", s)
    s = BLOCK_RE.sub("\n", s)
    s = TAG_RE.sub(" ", s)
    s = html_lib.unescape(s)
    lines = [WS_RE.sub(" ", ln).strip() for ln in s.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def main():
    config = json.loads(sys.argv[1])
    image_path = sys.argv[2]

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    Image.MAX_IMAGE_PIXELS = None

    model_id = config.get("model_id") or "datalab-to/chandra-ocr-2"
    use_gpu = bool(config.get("use_gpu", True)) and torch.cuda.is_available()
    # Engine pins the physical GPU via CUDA_VISIBLE_DEVICES; address it as cuda:0
    device = "cuda:0" if use_gpu else "cpu"
    dtype = torch.bfloat16 if use_gpu else torch.float32
    max_new_tokens = int(config.get("max_new_tokens") or 4096)

    prompt_mode = config.get("prompt_mode") or "ocr"
    if prompt_mode == "custom" and config.get("custom_prompt"):
        prompt_text = config["custom_prompt"]
    else:
        try:
            from chandra.prompts import PROMPT_MAPPING
            prompt_text = PROMPT_MAPPING.get(prompt_mode) or PROMPT_MAPPING["ocr"]
        except Exception:
            prompt_text = FALLBACK_OCR_PROMPT

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
    model = model.to(device)
    model.eval()

    # Stop tokens as in chandra/model/hf.py: <|endoftext|> AND <|im_end|>,
    # otherwise the model keeps generating additional "assistant" turns.
    tok = processor.tokenizer
    stop_ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for t in ("<|endoftext|>", "<|im_end|>"):
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid >= 0:
            stop_ids.add(tid)

    pil = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": pil},
                    {"type": "text", "text": prompt_text}],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, eos_token_id=sorted(stop_ids))
    input_len = inputs["input_ids"].shape[1]
    raw = processor.decode(out[0, input_len:], skip_special_tokens=True)

    text = html_to_text(raw)
    print(json.dumps({
        "text": text,
        "html": raw.strip(),
        "lines": text.split("\n") if text else [],
        "use_gpu": use_gpu,
        "model_id": model_id,
        "prompt_mode": prompt_mode,
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
        sys.exit(1)
