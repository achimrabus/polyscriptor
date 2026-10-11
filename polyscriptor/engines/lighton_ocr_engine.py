"""
LightOnOCR Engine Plugin

Wraps the LightOnOCR VLM for line-level HTR.
This is a LINE-LEVEL model (requires line segmentation).

Key features:
- Lightweight (~1B params, ~4GB VRAM)
- Multiple fine-tuned variants (base, German shorthand, etc.)
- Supports custom HuggingFace models
- Uses chat template for prompt handling

IMPORTANT: Requires transformers from git:
    pip install git+https://github.com/huggingface/transformers.git
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

from polyscriptor.core.engine_base import HTREngine, TranscriptionResult


# Check for LightOnOCR dependencies
LIGHTON_AVAILABLE = False
LIGHTON_MISSING_DEPS = []

try:
    import torch
except ImportError:
    LIGHTON_MISSING_DEPS.append("torch")

try:
    from transformers import AutoProcessor
    # LightOnOCR uses a custom model class
    try:
        from transformers import LightOnOcrForConditionalGeneration
        LIGHTON_MODEL_CLASS_AVAILABLE = True
    except ImportError:
        LIGHTON_MODEL_CLASS_AVAILABLE = False
        LIGHTON_MISSING_DEPS.append("transformers (from git - needs LightOnOcrForConditionalGeneration)")
except ImportError:
    LIGHTON_MISSING_DEPS.append("transformers")
    LIGHTON_MODEL_CLASS_AVAILABLE = False

if not LIGHTON_MISSING_DEPS:
    LIGHTON_AVAILABLE = True

# Import model registry
try:
    from polyscriptor.engines.lighton_models import (
        LIGHTON_MODELS, get_available_models, get_model_id, get_model_info
    )
except ImportError:
    LIGHTON_MODELS = {
        "Base Model (1B)": {
            "id": "lightonai/LightOnOCR-2-1B-base",
            "description": "Base LightOnOCR model",
            "vram": "~4GB",
        }
    }
    get_available_models = lambda: LIGHTON_MODELS
    get_model_id = lambda name: LIGHTON_MODELS.get(name, {}).get("id")
    get_model_info = lambda name: LIGHTON_MODELS.get(name)


class LightOnOCREngine(HTREngine):
    """LightOnOCR Vision-Language Model HTR engine plugin (LINE-LEVEL)."""

    def __init__(self):
        self.model = None
        self.processor = None
        self._device = "cuda:0"


    def get_name(self) -> str:
        return "LightOnOCR"

    def get_description(self) -> str:
        return "LightOnOCR: Lightweight line-level VLM with fine-tuned variants"

    def is_available(self) -> bool:
        return LIGHTON_AVAILABLE

    def get_unavailable_reason(self) -> str:
        reasons = []
        if LIGHTON_MISSING_DEPS:
            deps = ", ".join(LIGHTON_MISSING_DEPS)
            reasons.append(f"Missing dependencies: {deps}")
            reasons.append("Install transformers from git: pip install git+https://github.com/huggingface/transformers.git")
        return " | ".join(reasons) if reasons else ""






    def load_model(self, config: Dict[str, Any]) -> bool:
        """Load LightOnOCR model."""
        try:
            # Cleanup any existing model first
            if self.model is not None:
                print("Cleaning up previous model before loading new one...")
                self.unload_model()

            import torch
            from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

            # Support both GUI config keys and CLI keys
            model_id = config.get("model_id") or config.get("model_path") or "lightonai/LightOnOCR-2-1B-base"
            device = config.get("device", "cuda:0")
            self._device = device

            # Determine dtype based on device
            self._dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

            print(f"Loading LightOnOCR from {model_id}...")

            # Load processor - use LightOnOcrProcessor specifically
            self.processor = LightOnOcrProcessor.from_pretrained(model_id)

            # Load model
            self.model = LightOnOcrForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self._dtype,
            )
            self.model = self.model.to(device)
            self.model.eval()

            print(f"LightOnOCR loaded on {device}")
            return True

        except Exception as e:
            print(f"Error loading LightOnOCR model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.processor = None
            return False

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            try:
                import torch
                # Move to CPU first
                self.model.cpu()
                del self.model
                self.model = None

                if self.processor is not None:
                    del self.processor
                    self.processor = None

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning during model cleanup: {e}")

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.processor is not None

    def requires_line_segmentation(self) -> bool:
        """LightOnOCR is a LINE-LEVEL model - requires segmentation."""
        return True

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """Transcribe a single line image with LightOnOCR."""
        if self.model is None or self.processor is None:
            return TranscriptionResult(text="[Model not loaded]", confidence=0.0)

        config = config or {}

        try:
            from PIL import Image
            import torch
            import tempfile

            # Convert numpy to PIL
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Ensure RGB
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Get parameters
            max_new_tokens = config.get("max_new_tokens", 256)

            # Save image to temp file (LightOnOCR processor works best with file paths)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                pil_image.save(tmp_path)

            try:
                # Build conversation - LightOnOCR expects image only, no text prompt
                # The model is specifically trained for OCR so no prompt is needed
                # Note: Use the file path directly, not file:// URL
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": tmp_path}
                        ]
                    }
                ]

                # Apply chat template
                inputs = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                # Move inputs to device with proper dtype
                inputs = {
                    k: v.to(device=self._device, dtype=self._dtype) if v.is_floating_point() else v.to(self._device)
                    for k, v in inputs.items()
                }

                # Generate
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Greedy decoding for consistency
                    )

                # Decode output (skip input tokens)
                input_len = inputs['input_ids'].shape[1]
                generated_ids = output_ids[0, input_len:]
                text = self.processor.decode(generated_ids, skip_special_tokens=True)

                return TranscriptionResult(
                    text=text.strip(),
                    confidence=1.0,  # LightOnOCR doesn't provide confidence
                    metadata={
                        "model": "LightOnOCR",
                        "model_id": config.get("model_id", "unknown"),
                    }
                )

            finally:
                # Clean up temp file
                try:
                    Path(tmp_path).unlink()
                except:
                    pass

        except Exception as e:
            import traceback
            print(f"Error in LightOnOCR transcription: {e}")
            traceback.print_exc()
            return TranscriptionResult.error_result(f"{e}")

    def transcribe_lines(self, images: List[np.ndarray], config: Optional[Dict[str, Any]] = None) -> List[TranscriptionResult]:
        """Transcribe multiple line images."""
        # Process each line individually (LightOnOCR is a line-level model)
        return [self.transcribe_line(img, config) for img in images]

    def supports_batch(self) -> bool:
        """LightOnOCR processes lines individually."""
        return False

    def get_capabilities(self) -> Dict[str, bool]:
        """LightOnOCR capabilities."""
        return {
            "batch": False,
            "confidence_scores": False,
            "beam_search": False,
            "language_model": True,  # VLM has built-in language understanding
            "preprocessing": True,   # Has built-in vision preprocessing
        }
