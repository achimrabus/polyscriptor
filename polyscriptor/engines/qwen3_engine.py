"""
Qwen3 VLM Engine Plugin

Wraps the Qwen3 Vision-Language Model HTR inference system as a plugin.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from polyscriptor.core.engine_base import HTREngine, TranscriptionResult


try:
    from polyscriptor.inference.qwen3 import Qwen3VLMInference, QWEN3_MODELS
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False
    QWEN3_MODELS = {}

try:
    from polyscriptor.engines.qwen3_prompts import QWEN3_PROMPT_PRESETS, get_prompt
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    QWEN3_PROMPT_PRESETS = {
        "default": {"name": "Default", "prompt": "Transcribe the text shown in this image.", "description": ""}
    }


class Qwen3Engine(HTREngine):
    """Qwen3 Vision-Language Model HTR engine plugin."""

    def __init__(self):
        self.model: Optional[Qwen3VLMInference] = None

        self._use_sampling_checkbox: Optional[QCheckBox] = None

    def get_name(self) -> str:
        return "Qwen3-VL"

    def get_description(self) -> str:
        return "Vision-language model with LoRA fine-tuning support"

    def is_available(self) -> bool:
        return QWEN3_AVAILABLE

    def get_unavailable_reason(self) -> str:
        if not QWEN3_AVAILABLE:
            return "Qwen3 not available. Install with: pip install transformers>=4.37.0 accelerate peft qwen-vl-utils"
        return ""











    def load_model(self, config: Dict[str, Any]) -> bool:
        """Load Qwen3 model."""
        try:
            # Cleanup any existing model first to avoid memory fragmentation
            if self.model is not None:
                print("Cleaning up previous model before loading new one...")
                self.unload_model()

            # Support both 'base_model' (GUI) and 'model_id' (CLI)
            base_model = config.get("base_model") or config.get("model_id") or config.get("model_path")
            if not base_model:
                return False

            adapter = config.get("adapter")
            max_image_size = config.get("max_image_size", 1536)
            device = config.get("device", "auto")

            self.model = Qwen3VLMInference(
                base_model=base_model,
                adapter_model=adapter,
                max_image_size=max_image_size,
                device=device,
            )

            return True

        except Exception as e:
            print(f"Error loading Qwen3 model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            return False

    def unload_model(self):
        """
        Unload model from memory with proper cleanup to prevent CUDA memory fragmentation.

        IMPORTANT: This method uses the cleanup() function from Qwen3VLMInference
        which properly moves the model to CPU, deletes references, and clears CUDA cache.
        """
        if self.model is not None:
            # Use the proper cleanup method to avoid memory fragmentation
            try:
                self.model.cleanup()
            except Exception as e:
                print(f"Warning during model cleanup: {e}")
                # Fallback: manual cleanup
                try:
                    self.model.model.cpu()
                except:
                    pass

            del self.model
            self.model = None

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """Transcribe a line image with Qwen3."""
        if self.model is None:
            return TranscriptionResult(text="[Model not loaded]", confidence=0.0)

        try:
            # Qwen3 uses transcribe_page() for full page transcription
            # Convert numpy to PIL for Qwen3
            from PIL import Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Extract prompt from config
            prompt = "Transcribe the text shown in this image."  # Default
            if config:
                prompt_preset = config.get("prompt_preset", "default")
                prompt_text = config.get("prompt_text", "")

                # Determine which prompt to use
                if PROMPTS_AVAILABLE:
                    prompt = get_prompt(prompt_preset, prompt_text)
                elif prompt_preset == "custom" and prompt_text:
                    prompt = prompt_text

            # Use transcribe_page which handles the full image
            # Pass generation parameters from config
            generation_kwargs = {}
            if config:
                generation_kwargs["do_sample"] = config.get("do_sample", False)
                generation_kwargs["temperature"] = config.get("temperature", 1.0) if config.get("do_sample") else None
                generation_kwargs["max_new_tokens"] = config.get("max_new_tokens", 2048)
                # Note: repetition_penalty is already set in inference_qwen3.py

            result = self.model.transcribe_page(pil_image, prompt=prompt, return_confidence=True, **generation_kwargs)

            # Extract just the text from PageTranscription object
            text = result.text if hasattr(result, 'text') else str(result)
            confidence = result.confidence if hasattr(result, 'confidence') else 1.0

            return TranscriptionResult(
                text=text,
                confidence=confidence if confidence is not None else 1.0,
                metadata={
                    "model": "qwen3-vlm",
                    "prompt": prompt,
                    "prompt_preset": config.get("prompt_preset", "default") if config else "default"
                }
            )

        except Exception as e:
            import traceback
            print(f"Error in Qwen3 transcription: {e}")
            print(traceback.format_exc())
            return TranscriptionResult.error_result(f"{e}")

    def supports_batch(self) -> bool:
        """VLM calls run one image at a time; the base sequential loop applies."""
        return False

    def get_capabilities(self) -> Dict[str, bool]:
        """Qwen3 capabilities."""
        return {
            "batch": False,
            "confidence_scores": True,  # VLM can provide confidence
            "beam_search": False,  # Uses sampling/greedy decoding
            "language_model": True,  # VLM has built-in language understanding
            "preprocessing": True,  # Has built-in vision preprocessing
        }
    def requires_line_segmentation(self) -> bool:
        """Qwen3 can process full pages without segmentation."""
        return False
