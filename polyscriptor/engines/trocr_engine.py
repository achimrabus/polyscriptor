"""
TrOCR Engine Plugin

Wraps the TrOCR HTR inference system as a plugin for the unified GUI.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from polyscriptor.core.engine_base import HTREngine, TranscriptionResult


try:
    from polyscriptor.inference.page import TrOCRInference
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False


# The n-gram repeat block as a user setting. Defined once here so that every front end
# offers the same field with the same default and explanation: the web form and the
# the web config schema are both built from this dict.
REPEAT_BLOCK_FIELD = {
    "key": "no_repeat_ngram_size",
    "type": "number",
    "label": "Repeat block (n-gram size)",
    "min": 0,
    "max": 10,
    "default": 0,
    "hint": ("0 = off (recommended). N forbids any sequence of N tokens from occurring twice "
             "in a line. TrOCR splits Cyrillic into single letters or smaller pieces, so a block "
             "suppresses ordinary letter sequences; for Latin-script models it is untested."),
}


def repeat_block_size(value, default: int = 0) -> int:
    """Turn a form or CLI value into a valid no_repeat_ngram_size.

    Form values arrive as numbers, strings or NaN (an emptied browser number field).
    Anything that is not a non-negative number becomes `default`, i.e. off; a block
    must never be switched on by a malformed value.
    """
    try:
        n = float(str(value).strip()) if value is not None else float("nan")
    except ValueError:
        return default
    if n != n or n < 0 or n == float("inf"):   # NaN, negative, infinite
        return default
    return int(n)


class TrOCREngine(HTREngine):
    """TrOCR HTR engine plugin."""

    def __init__(self):
        self.model: Optional[TrOCRInference] = None
        self._current_model_path: Optional[str] = None  # Store loaded model path
        self._num_beams: int = 4  # Effective decoding beams (set in load_model for headless/batch)
        self._no_repeat_ngram_size: int = 0  # 0 = off; overrides the checkpoint's generation_config


    def get_name(self) -> str:
        return "TrOCR"

    def get_description(self) -> str:
        return "Transformer-based OCR optimized for handwritten manuscripts"

    def is_available(self) -> bool:
        return TROCR_AVAILABLE

    def get_unavailable_reason(self) -> str:
        if not TROCR_AVAILABLE:
            return "TrOCR inference module not available. Check that inference_page.py exists."
        return ""








    def load_model(self, config: Dict[str, Any]) -> bool:
        """Load TrOCR model."""
        try:
            model_path = config.get("model_path", "")
            if not model_path or model_path == "No local models found" or model_path == "No models found":
                return False

            normalize = config.get("normalize_background", False)
            flip_rtl = config.get("flip_rtl", False)
            model_source = config.get("model_source", "local")
            is_hf = (model_source == "huggingface")

            # Capture effective decoding beams from CLI/batch config so that
            # transcribe_line() uses it in headless mode (where get_config()
            # 'beam_search' (GUI) and 'num_beams' (CLI / polyscriptor.cli.batch).
            self._num_beams = int(config.get("beam_search", config.get("num_beams", 4)))
            self._no_repeat_ngram_size = repeat_block_size(config.get("no_repeat_ngram_size", 0))
            print(f"TrOCR decoding: num_beams={self._num_beams} "
                  f"({'greedy' if self._num_beams <= 1 else 'beam search'}), "
                  f"no_repeat_ngram_size={self._no_repeat_ngram_size}")

            self.model = TrOCRInference(
                model_path=model_path,
                normalize_bg=normalize,
                flip_rtl=flip_rtl,
                is_huggingface=is_hf,
                processor_subfolder=config.get("processor_subfolder")
            )

            # Store the model path for metadata
            self._current_model_path = model_path

            return True

        except Exception as e:
            print(f"Error loading TrOCR model: {e}")
            self.model = None
            self._current_model_path = None
            return False

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_path = None

            # Free GPU memory
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """Transcribe a line image with TrOCR."""
        if self.model is None:
            return TranscriptionResult(text="[Model not loaded]", confidence=0.0)

        # Use provided config or fall back to current config
        if config is None:
            config = self.get_config()

        # Prefer an explicit per-call value (GUI passes 'beam_search', CLI may
        # pass 'num_beams'); otherwise use the value captured at load_model time.
        # the self._num_beams fallback this silently defaulted to 4 and ignored
        # the CLI --num-beams entirely.
        beam_search = config.get("beam_search", config.get("num_beams", self._num_beams))
        no_repeat = (repeat_block_size(config["no_repeat_ngram_size"])
                     if "no_repeat_ngram_size" in config else self._no_repeat_ngram_size)

        try:
            # TrOCRInference expects PIL Image, convert from numpy
            from PIL import Image
            pil_image = Image.fromarray(image)

            text, confidence, token_confidences = self.model.transcribe_line(
                pil_image, num_beams=beam_search, return_confidence=True,
                no_repeat_ngram_size=no_repeat)

            # Build metadata with model information
            metadata = {
                "beam_search": beam_search,
                "no_repeat_ngram_size": no_repeat,
                "engine": "TrOCR"
            }

            # Add model path if available
            if self._current_model_path:
                metadata["model"] = self._current_model_path

            # Honest confidence: geometric mean of the generated tokens'
            # probabilities (beam-safe, see TrOCRInference.transcribe_line).
            # min_token_prob flags the weakest spot in the line — a low value
            # means the model was unsure somewhere even if the mean looks fine.
            # None (not 1.0) when no scores are available, so the UI shows
            # "no confidence" instead of a fake 100%.
            if token_confidences:
                metadata["min_token_prob"] = round(min(token_confidences), 4)
                metadata["num_tokens"] = len(token_confidences)

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                metadata=metadata
            )

        except Exception as e:
            print(f"Error in TrOCR transcription: {e}")
            return TranscriptionResult.error_result(f"{e}")

    def get_capabilities(self) -> Dict[str, bool]:
        """TrOCR capabilities."""
        return {
            "batch": False,  # Could be implemented in future
            "confidence_scores": True,  # geometric mean of per-token probabilities (beam-safe)
            "beam_search": True,
            "language_model": False,  # Implicit in decoder, not explicit LM
            "preprocessing": True,  # Has built-in normalization
        }
