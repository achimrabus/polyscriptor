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

from htr_engine_base import HTREngine, TranscriptionResult

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QGroupBox, QSlider, QSpinBox, QTextEdit, QLineEdit
    )
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

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
    from lighton_models import (
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
        self._config_widget: Optional[QWidget] = None
        self._device = "cuda:0"

        # Widget references
        self._model_variant_combo: Optional[QComboBox] = None
        self._custom_model_edit: Optional[QLineEdit] = None
        self._longest_edge_slider: Optional[QSlider] = None
        self._longest_edge_label: Optional[QLabel] = None
        self._max_tokens_spin: Optional[QSpinBox] = None
        self._prompt_text_edit: Optional[QTextEdit] = None
        self._device_combo: Optional[QComboBox] = None
        self._model_desc_label: Optional[QLabel] = None

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

    def get_config_widget(self) -> QWidget:
        """Create LightOnOCR configuration panel."""
        if self._config_widget is not None:
            return self._config_widget

        widget = QWidget()
        layout = QVBoxLayout()

        # Model variant selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()

        # Preset models dropdown
        model_layout.addWidget(QLabel("Preset Model:"))
        self._model_variant_combo = QComboBox()

        # Populate with available models
        models = get_available_models()
        for name, info in models.items():
            display = f"{name} ({info.get('vram', '~4GB')})"
            self._model_variant_combo.addItem(display, userData=name)

        # Add "Custom" option
        self._model_variant_combo.addItem("Custom HuggingFace Model", userData="custom")
        self._model_variant_combo.currentIndexChanged.connect(self._on_model_changed)
        model_layout.addWidget(self._model_variant_combo)

        # Model description
        self._model_desc_label = QLabel("")
        self._model_desc_label.setWordWrap(True)
        self._model_desc_label.setStyleSheet("color: gray; font-size: 9pt;")
        model_layout.addWidget(self._model_desc_label)

        # Custom model ID input
        model_layout.addWidget(QLabel("Custom Model ID:"))
        self._custom_model_edit = QLineEdit()
        self._custom_model_edit.setPlaceholderText("e.g., username/LightOnOCR-2-1B-custom")
        self._custom_model_edit.setEnabled(False)  # Disabled until "Custom" selected
        model_layout.addWidget(self._custom_model_edit)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Inference settings
        settings_group = QGroupBox("Inference Settings")
        settings_layout = QVBoxLayout()

        # Longest edge
        edge_layout = QVBoxLayout()
        edge_layout.addWidget(QLabel("Longest Edge (image resize):"))

        edge_slider_layout = QHBoxLayout()
        self._longest_edge_slider = QSlider(Qt.Orientation.Horizontal)
        self._longest_edge_slider.setRange(512, 1024)
        self._longest_edge_slider.setValue(700)
        self._longest_edge_slider.setTickInterval(100)
        self._longest_edge_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._longest_edge_slider.valueChanged.connect(self._on_edge_changed)
        edge_slider_layout.addWidget(self._longest_edge_slider)

        self._longest_edge_label = QLabel("700px")
        self._longest_edge_label.setMinimumWidth(60)
        edge_slider_layout.addWidget(self._longest_edge_label)

        edge_layout.addLayout(edge_slider_layout)
        edge_hint = QLabel("Larger = better quality but slower, more VRAM")
        edge_hint.setStyleSheet("color: gray; font-size: 9pt;")
        edge_layout.addWidget(edge_hint)
        settings_layout.addLayout(edge_layout)

        # Max new tokens
        tokens_layout = QHBoxLayout()
        tokens_layout.addWidget(QLabel("Max New Tokens:"))
        self._max_tokens_spin = QSpinBox()
        self._max_tokens_spin.setRange(64, 512)
        self._max_tokens_spin.setValue(256)
        self._max_tokens_spin.setSingleStep(32)
        tokens_layout.addWidget(self._max_tokens_spin)
        tokens_hint = QLabel("Maximum output length")
        tokens_hint.setStyleSheet("color: gray; font-size: 8pt;")
        tokens_layout.addWidget(tokens_hint)
        settings_layout.addLayout(tokens_layout)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Custom prompt
        prompt_group = QGroupBox("Custom Prompt (Optional)")
        prompt_layout = QVBoxLayout()

        self._prompt_text_edit = QTextEdit()
        self._prompt_text_edit.setPlaceholderText(
            "Leave empty for default prompt, or enter custom prompt...\n"
            "Example: 'Transcribe the German shorthand text.'"
        )
        self._prompt_text_edit.setMaximumHeight(80)
        prompt_layout.addWidget(self._prompt_text_edit)

        prompt_hint = QLabel("Tip: Include language or script type for better results.")
        prompt_hint.setStyleSheet("color: gray; font-size: 9pt;")
        prompt_layout.addWidget(prompt_hint)

        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        # Device selection
        device_group = QGroupBox("Device")
        device_layout = QHBoxLayout()

        device_layout.addWidget(QLabel("GPU:"))
        self._device_combo = QComboBox()

        # Detect available devices
        devices = ["cuda:0"]
        try:
            import torch
            if torch.cuda.is_available():
                devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        except:
            pass
        devices.append("cpu")

        self._device_combo.addItems(devices)
        device_layout.addWidget(self._device_combo)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        layout.addStretch()
        widget.setLayout(layout)

        # Initialize model description
        self._on_model_changed(0)

        self._config_widget = widget
        return widget

    def _on_model_changed(self, index: int):
        """Update UI when model selection changes."""
        if self._model_variant_combo is None:
            return

        model_name = self._model_variant_combo.currentData()
        is_custom = (model_name == "custom")

        # Enable/disable custom model input
        if self._custom_model_edit:
            self._custom_model_edit.setEnabled(is_custom)

        # Update description
        if self._model_desc_label:
            if is_custom:
                self._model_desc_label.setText("Enter a HuggingFace model ID below")
            else:
                info = get_model_info(model_name)
                if info:
                    desc = info.get("description", "")
                    lang = info.get("language", "")
                    if lang:
                        desc += f"\nLanguage: {lang}"
                    self._model_desc_label.setText(desc)
                else:
                    self._model_desc_label.setText("")

    def _on_edge_changed(self, value: int):
        """Update longest edge label."""
        if self._longest_edge_label:
            self._longest_edge_label.setText(f"{value}px")

    def get_config(self) -> Dict[str, Any]:
        """Extract configuration from widget controls."""
        if self._config_widget is None:
            return {}

        model_name = self._model_variant_combo.currentData()

        if model_name == "custom":
            model_id = self._custom_model_edit.text().strip()
        else:
            model_id = get_model_id(model_name) or "lightonai/LightOnOCR-2-1B-base"

        custom_prompt = self._prompt_text_edit.toPlainText().strip()

        return {
            "model_variant": model_name,
            "model_id": model_id,
            "longest_edge": self._longest_edge_slider.value(),
            "max_new_tokens": self._max_tokens_spin.value(),
            "custom_prompt": custom_prompt if custom_prompt else None,
            "device": self._device_combo.currentText(),
        }

    def set_config(self, config: Dict[str, Any]):
        """Restore configuration to widget controls."""
        if self._config_widget is None:
            return

        # Model variant
        model_variant = config.get("model_variant", "Base Model (1B)")
        for i in range(self._model_variant_combo.count()):
            if self._model_variant_combo.itemData(i) == model_variant:
                self._model_variant_combo.setCurrentIndex(i)
                break

        # Custom model ID
        if model_variant == "custom":
            self._custom_model_edit.setText(config.get("model_id", ""))

        # Inference settings
        self._longest_edge_slider.setValue(config.get("longest_edge", 700))
        self._max_tokens_spin.setValue(config.get("max_new_tokens", 256))

        # Custom prompt
        custom_prompt = config.get("custom_prompt")
        if custom_prompt:
            self._prompt_text_edit.setPlainText(custom_prompt)

        # Device
        device = config.get("device", "cuda:0")
        idx = self._device_combo.findText(device)
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)

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
            return TranscriptionResult(text=f"[Error: {e}]", confidence=0.0)

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
            "batch_processing": False,
            "confidence_scores": False,
            "beam_search": False,
            "language_model": True,  # VLM has built-in language understanding
            "preprocessing": True,   # Has built-in vision preprocessing
        }
