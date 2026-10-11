"""
PyLaia Engine Plugin

Wraps the PyLaia CTC-based HTR inference system as a plugin.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from polyscriptor.core.engine_base import HTREngine, TranscriptionResult


try:
    # Use native Linux implementation (no WSL dependency)
    from polyscriptor.inference.pylaia_native import PyLaiaInference, PYLAIA_MODELS, preset_source, _local_file
    PYLAIA_AVAILABLE = True
    PYLAIA_LM_AVAILABLE = False  # Language model not yet implemented
except ImportError:
    PYLAIA_AVAILABLE = False
    PYLAIA_MODELS = {}
    preset_source = lambda info: None
    _local_file = lambda path: None
    PYLAIA_LM_AVAILABLE = False


class PyLaiaEngine(HTREngine):
    """PyLaia CTC-based HTR engine plugin."""

    def __init__(self):
        self.model: Optional[PyLaiaInference] = None
        self.model_lm: Optional[PyLaiaInferenceLM] = None


    def get_name(self) -> str:
        return "CRNN-CTC (PyLaia-inspired)"

    def get_description(self) -> str:
        return "Puigcerver CRNN-CTC: clean-room PyTorch reimplementation of the PyLaia architecture"

    def get_aliases(self) -> List[str]:
        return ["crnn-ctc", "CRNN-CTC", "PyLaia"]  # "PyLaia" kept for backward compatibility

    def is_available(self) -> bool:
        return PYLAIA_AVAILABLE

    def get_unavailable_reason(self) -> str:
        if not PYLAIA_AVAILABLE:
            return "CRNN-CTC engine not available. Check that inference_pylaia_native.py exists and dependencies are installed."
        return ""









    def load_model(self, config: Dict[str, Any]) -> bool:
        """Load PyLaia model."""
        try:
            model_path = config.get("model_path", "")
            if not model_path or model_path == "No preset models found":
                print("Error loading PyLaia model: no model_path in config")
                return False

            # If it's a preset name, resolve to actual path and syms
            syms_path = None
            if model_path in PYLAIA_MODELS:
                preset_info = PYLAIA_MODELS[model_path]
                if isinstance(preset_info, dict):
                    if preset_source(preset_info) == "hf":
                        # Published model whose files are not here: fetch from the
                        # Hugging Face repo (cached after the first load)
                        try:
                            from huggingface_hub import hf_hub_download
                        except ImportError as exc:
                            raise RuntimeError(
                                "huggingface_hub is required for Hugging Face model presets"
                            ) from exc
                        repo_id = preset_info["repo_id"]
                        model_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=preset_info.get("hf_checkpoint", "best_model.pt"),
                        )
                        syms_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=preset_info.get("hf_syms", "symbols.txt"),
                        )
                    else:
                        model_path = preset_info.get("checkpoint", preset_info.get("path", model_path))
                        syms_path = preset_info.get("syms")
                        # Registry paths are relative to the repository root
                        model_path = str(_local_file(model_path) or model_path)
                        if syms_path:
                            syms_path = str(_local_file(syms_path) or syms_path)
                # If preset_info is just a string, use it as the path
                elif isinstance(preset_info, str):
                    model_path = preset_info

            use_lm = config.get("use_lm", False)

            # Unload previous model
            self.unload_model()

            if use_lm and PYLAIA_LM_AVAILABLE:
                # Load with language model
                lm_weight = config.get("lm_weight", 1.5)
                lm_path = config.get("lm_path")

                self.model_lm = PyLaiaInferenceLM(
                    model_path=model_path,
                    lm_path=lm_path,
                    lm_weight=lm_weight
                )
                self.model = None
            else:
                # Load without language model
                # PyLaiaInference expects checkpoint_path, syms_path, and enable_spaces
                enable_spaces = config.get("enable_spaces", True)
                # batch_processing.py passes --device here; without it the reader
                # picked cuda whenever a GPU was visible, even for --device cpu
                device = config.get("device")
                if device in (None, "", "auto"):
                    device = None
                self.model = PyLaiaInference(
                    checkpoint_path=model_path,
                    syms_path=syms_path,
                    enable_spaces=enable_spaces,
                    device=device
                )
                self.model_lm = None

            return True

        except Exception as e:
            import traceback
            print(f"Error loading PyLaia model: {e}")
            print(traceback.format_exc())
            self.model = None
            self.model_lm = None
            return False

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.model_lm is not None:
            del self.model_lm
            self.model_lm = None

        # Free GPU memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None or self.model_lm is not None

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """Transcribe a line image with PyLaia."""
        if not self.is_model_loaded():
            return TranscriptionResult(text="[Model not loaded]", confidence=0.0)

        try:
            # Convert numpy to PIL
            from PIL import Image as PILImage
            if isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image

            # Flip horizontally for RTL scripts
            if config and config.get("flip_rtl", False):
                pil_image = pil_image.transpose(PILImage.FLIP_LEFT_RIGHT)

            # PyLaiaInferenceWSL uses transcribe() which returns (text, confidence) tuple
            # Use LM version if available (not yet implemented for WSL)
            if self.model_lm is not None:
                # PyLaiaInferenceLM might have different method
                result = self.model_lm.transcribe(pil_image)
            else:
                result = self.model.transcribe(pil_image)

            # Result is a tuple: (text, confidence)
            if isinstance(result, tuple):
                text, confidence = result
            else:
                # Fallback for dict-style results
                text = result.get("text", "")
                confidence = result.get("confidence", 1.0)

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                metadata={"model": "pylaia"}
            )

        except Exception as e:
            import traceback
            print(f"Error in PyLaia transcription: {e}")
            print(traceback.format_exc())
            return TranscriptionResult.error_result(f"{e}")

    def get_capabilities(self) -> Dict[str, bool]:
        """PyLaia capabilities."""
        return {
            "batch": False,  # Could be implemented
            "confidence_scores": True,  # CTC provides confidence
            "beam_search": False,  # CTC uses greedy/beam decoding
            "language_model": PYLAIA_LM_AVAILABLE,  # Optional KenLM
            "preprocessing": False,  # External preprocessing recommended
        }
