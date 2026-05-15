"""
LightOnOCR Model Registry

Extensible registry for LightOnOCR fine-tuned model variants.
Easy for users/community to add new models.

Usage:
    from lighton_models import LIGHTON_MODELS, get_available_models, add_custom_model

    # Get all available models
    models = get_available_models()

    # Add a custom model at runtime
    add_custom_model("My Custom Model", "username/model-id", "Description")
"""

from typing import Dict, Any, List, Optional


# Registry of available LightOnOCR models
# Format: display_name -> {id, description, language, ...}
LIGHTON_MODELS: Dict[str, Dict[str, Any]] = {
    "Base Model (1B)": {
        "id": "lightonai/LightOnOCR-2-1B-base",
        "description": "Base LightOnOCR model (1B params). General multilingual OCR.",
        "language": "multilingual",
        "vram": "~4GB",
        "type": "line",
    },
    "German Shorthand": {
        "id": "wjbmattingly/LightOnOCR-2-1B-german-shorthand-line",
        "description": "Fine-tuned for German Kurrent/Shorthand scripts.",
        "language": "de",
        "vram": "~4GB",
        "type": "line",
    },
    "Church Slavonic (wjbmattingly)": {
        "id": "wjbmattingly/LightOnOCR-2-1B-old-church-slavonic-line",
        "description": "Fine-tuned for Old Church Slavonic manuscript lines.",
        "language": "cu",
        "vram": "~4GB",
        "type": "line",
    },
}


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Returns all available preset models for dropdown.

    Returns:
        Dict mapping display names to model info dicts
    """
    return LIGHTON_MODELS.copy()


def get_model_info(display_name: str) -> Optional[Dict[str, Any]]:
    """
    Get info for a specific model by display name.

    Args:
        display_name: The display name of the model

    Returns:
        Model info dict or None if not found
    """
    return LIGHTON_MODELS.get(display_name)


def get_model_id(display_name: str) -> Optional[str]:
    """
    Get HuggingFace model ID for a display name.

    Args:
        display_name: The display name of the model

    Returns:
        HuggingFace model ID or None if not found
    """
    info = LIGHTON_MODELS.get(display_name)
    return info.get("id") if info else None


def add_custom_model(
    name: str,
    model_id: str,
    description: str = "",
    language: str = "unknown",
    vram: str = "~4GB"
):
    """
    Add a custom model to the registry at runtime.

    Useful for user-defined presets or community models discovered during session.

    Args:
        name: Display name for the model
        model_id: HuggingFace model ID (e.g., "username/model-name")
        description: Brief description of the model
        language: Language code(s) the model supports
        vram: Estimated VRAM requirement
    """
    LIGHTON_MODELS[name] = {
        "id": model_id,
        "description": description,
        "language": language,
        "vram": vram,
        "type": "line",
    }


def get_model_names() -> List[str]:
    """
    Get list of all model display names.

    Returns:
        List of model names for UI dropdowns
    """
    return list(LIGHTON_MODELS.keys())


def is_valid_model(display_name: str) -> bool:
    """
    Check if a display name corresponds to a valid model.

    Args:
        display_name: The display name to check

    Returns:
        True if model exists in registry
    """
    return display_name in LIGHTON_MODELS
