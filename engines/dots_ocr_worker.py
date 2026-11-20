#!/usr/bin/env python3
"""
dots.mocr worker — runs inside venv_dots, called as subprocess by dots_ocr_engine.py.

dots.mocr (rednote-hilab/dots.mocr, ~3B, MIT) is a multilingual document-parsing
VLM based on a Qwen2.5-VL-style architecture.  It needs transformers==4.56.1 +
qwen-vl-utils + trust_remote_code, which conflicts with the main venv — hence
the isolated venv + worker pattern.

IMPORTANT: download the weights to a local directory WITHOUT periods in the name
(temporary upstream workaround), e.g. models/DotsMOCR:
    python tools/download_model.py   # from the dots.ocr repo
    # or: huggingface-cli download rednote-hilab/dots.mocr --local-dir models/DotsMOCR

Setup (venv_dots):
    python -m venv /path/to/dhlab-slavistik/venv_dots
    source venv_dots/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install "transformers==4.56.1" accelerate "qwen-vl-utils" pillow
    # flash-attn is REQUIRED by the dots.mocr model code (prebuilt wheel, no compile):
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
    deactivate

Usage:
    python dots_ocr_worker.py '<config_json>' '<image_path>'
    python dots_ocr_worker.py '<config_json>' --list <file>   # one path per line, JSON lines out

Config keys (JSON):
    model_path      – local model dir (no periods!) or HF id (default: models/DotsMOCR)
    device          – "cuda:0", "cuda:1", "cpu" (default: "cuda:0")
    prompt_mode     – "ocr" | "layout" | "scene" (default: "ocr")
    max_new_tokens  – int, generation limit (default: 8192)
    do_sample       – bool (default: true = model-card setting, temperature 0.1,
                      top_p 0.9).  Greedy (false) degenerates on manuscript pages: the
                      model then spells the page letter by letter with spaces in between
                      (smoke test 2026-11-16), so it is only an option.
    seed            – int, torch seed set before every image (default: 42), so sampled
                      runs repeat on the same hardware
    max_pixels      – int, upper bound for the processor's image resize
                      (default: 11289600 = upstream value with flash-attn; capped at
                      2359296 = 1536x1536 under eager attention, where the upstream
                      value needs ~71 GiB for the attention matrix)

Writes a single JSON object to stdout:
    {"text": "...", "model_path": "...", "prompt_mode": "ocr"}

On error writes:
    {"error": "<message>", "traceback": "..."}
"""

from _vlm_worker_common import run

_PROMPTS = {
    # prompt_ocr: plain text, page headers/footers excluded by design
    "ocr": "Extract the text content from this image.",
    # prompt_layout_all_en: full layout info as a single JSON object
    "layout": (
        "Please output the layout information from the PDF image, including each "
        "layout element's bbox, its category, and the corresponding text content "
        "within the bbox.\n\n"
        "1. Bbox format: [x1, y1, x2, y2]\n\n"
        "2. Layout Categories: The possible categories are ['Caption', 'Footnote', "
        "'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', "
        "'Section-header', 'Table', 'Text', 'Title'].\n\n"
        "3. Text Extraction & Formatting Rules:\n"
        "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
        "    - Formula: Format its text as LaTeX.\n"
        "    - Table: Format its text as HTML.\n"
        "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
        "4. Constraints:\n"
        "    - The output text must be the original text from the image, with no translation.\n"
        "    - All layout elements must be sorted according to human reading order.\n\n"
        "5. Final Output: The entire output must be a single JSON object.\n"
    ),
    # prompt_scene_spotting
    "scene": "Detect and recognize the text in the image.",
}


_MAX_PIXELS_FLASH = 11289600
_MAX_PIXELS_EAGER = 1536 * 1536


def load(config):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise RuntimeError(
            f"Import failed: {e}. Activate venv_dots and install: "
            'pip install "transformers==4.56.1" accelerate qwen-vl-utils pillow'
        )
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    model_path = config.get("model_path") or "models/DotsMOCR"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation=attn_impl, device_map=config.get("device", "cuda:0"),
    )
    model.eval()

    max_pixels = int(config.get("max_pixels", _MAX_PIXELS_FLASH))
    if attn_impl == "eager":
        max_pixels = min(max_pixels, _MAX_PIXELS_EAGER)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, min_pixels=3136, max_pixels=max_pixels,
    )

    generate_kwargs = dict(max_new_tokens=int(config.get("max_new_tokens", 8192)))
    if config.get("do_sample", True):
        generate_kwargs.update(do_sample=True, temperature=0.1, top_p=0.9)
    else:
        generate_kwargs.update(do_sample=False)
    return {
        "torch": torch, "model": model, "processor": processor,
        "process_vision_info": process_vision_info,
        "prompt": _PROMPTS.get(config.get("prompt_mode", "ocr"), _PROMPTS["ocr"]),
        "generate_kwargs": generate_kwargs,
        "seed": int(config.get("seed", 42)),
    }


def transcribe(st, image_path):
    processor = st["processor"]
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": st["prompt"]},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = st["process_vision_info"](messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(st["model"].device)
    st["torch"].manual_seed(st["seed"])
    with st["torch"].no_grad():
        generated_ids = st["model"].generate(**inputs, **st["generate_kwargs"])
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()


def _meta(config):
    return {"model_path": config.get("model_path") or "models/DotsMOCR",
            "prompt_mode": config.get("prompt_mode", "ocr")}


if __name__ == "__main__":
    run("dots_ocr_worker.py", load, transcribe, _meta)
