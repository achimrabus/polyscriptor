"""
Qwen3-VL Prompt Presets for HTR

Collection of optimized prompts for different document types and languages.
These prompts help Qwen3-VL focus on specific characteristics of the input.
"""

QWEN3_PROMPT_PRESETS = {
    "default": {
        "name": "Default",
        "prompt": "Transcribe the text shown in this image.",
        "description": "General-purpose transcription prompt"
    },
    "church_slavonic": {
        "name": "Church Slavonic",
        "prompt": "Transcribe the Church Slavonic text shown in this historical manuscript image. Preserve all diacritical marks, titlos, and abbreviations.",
        "description": "Optimized for Church Slavonic manuscripts with special characters"
    },
    "glagolitic": {
        "name": "Glagolitic Script",
        "prompt": "Transcribe the Glagolitic script text shown in this medieval manuscript. Output the text in Glagolitic Unicode characters.",
        "description": "Specialized for Glagolitic alphabet (U+2C00–U+2C5F)"
    },
    "cyrillic_historical": {
        "name": "Historical Cyrillic",
        "prompt": "Transcribe the historical Cyrillic text shown in this manuscript. Preserve archaic letters and orthography.",
        "description": "For pre-reform Russian/Ukrainian manuscripts"
    },
    "latin_medieval": {
        "name": "Medieval Latin",
        "prompt": "Transcribe the Medieval Latin text shown in this manuscript. Preserve abbreviations and ligatures.",
        "description": "For Latin manuscripts with medieval conventions"
    },
    "math": {
        "name": "Mathematical Content",
        "prompt": "Transcribe the mathematical equations, formulas, symbols, and text shown in this handwritten page. Preserve mathematical notation accurately.",
        "description": "For pages with equations and mathematical symbols"
    },
    "degraded": {
        "name": "Degraded/Faded Text",
        "prompt": "Transcribe the faded, low-quality handwritten text in this degraded manuscript image. Use context to infer unclear characters.",
        "description": "Optimized for poor quality or damaged documents"
    },
    "multilingual": {
        "name": "Multilingual",
        "prompt": "Transcribe all text shown in this image, preserving multiple languages and scripts if present.",
        "description": "For documents with mixed languages"
    },
    "detailed": {
        "name": "Detailed Transcription",
        "prompt": "Carefully transcribe all text in this image, preserving line breaks, formatting, and special characters. Include all visible text exactly as it appears.",
        "description": "For detailed, character-perfect transcription"
    },
    "cursive": {
        "name": "Cursive Handwriting",
        "prompt": "Transcribe the cursive handwritten text shown in this image. Pay attention to connected letters and flowing script.",
        "description": "Optimized for cursive/script handwriting"
    },
    "custom": {
        "name": "Custom",
        "prompt": "",
        "description": "User-defined custom prompt"
    }
}


def get_prompt(preset_key: str, custom_text: str = "") -> str:
    """
    Get prompt text for a preset key.

    Args:
        preset_key: Key from QWEN3_PROMPT_PRESETS
        custom_text: Custom prompt text (used when preset_key is "custom")

    Returns:
        Prompt string to use for Qwen3 inference
    """
    if preset_key == "custom":
        return custom_text if custom_text else QWEN3_PROMPT_PRESETS["default"]["prompt"]

    preset = QWEN3_PROMPT_PRESETS.get(preset_key)
    if preset:
        return preset["prompt"]

    return QWEN3_PROMPT_PRESETS["default"]["prompt"]


def get_preset_names() -> list:
    """Get list of all preset names for UI display."""
    return [info["name"] for key, info in QWEN3_PROMPT_PRESETS.items()]


def get_preset_key_by_name(name: str) -> str:
    """Get preset key from display name."""
    for key, info in QWEN3_PROMPT_PRESETS.items():
        if info["name"] == name:
            return key
    return "default"
