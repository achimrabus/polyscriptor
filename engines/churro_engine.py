"""
Churro VLM Engine Plugin

Wraps the Churro (Qwen2.5-VL-3B) historical document OCR model as a plugin.

IMPORTANT: Churro uses Qwen2_5_VLForConditionalGeneration (Qwen2.5), NOT Qwen3VLForConditionalGeneration.
This is a different model class from the Qwen3 engine!
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from htr_engine_base import HTREngine, TranscriptionResult

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QPushButton, QSpinBox, QCheckBox, QLineEdit, QFileDialog,
        QGroupBox, QSlider, QTextEdit
    )
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

try:
    from inference_churro import ChurroInference
    CHURRO_AVAILABLE = True
except ImportError:
    CHURRO_AVAILABLE = False


# Churro-specific prompts for historical manuscripts
CHURRO_PROMPTS = {
    "default": {
        "name": "Default (Generic)",
        "prompt": "Transcribe all text from this historical document.",
        "description": "Generic prompt for any historical document"
    },
    "church_slavonic": {
        "name": "Church Slavonic",
        "prompt": "Transcribe the Church Slavonic text shown in this historical manuscript image. Preserve all diacritical marks, titlos, and abbreviations.",
        "description": "Specialized for Church Slavonic manuscripts with diacritics"
    },
    "glagolitic": {
        "name": "Glagolitic Script",
        "prompt": "Transcribe the Glagolitic script text shown in this medieval manuscript. Output the text in Glagolitic Unicode characters.",
        "description": "For Glagolitic script manuscripts"
    },
    "ukrainian": {
        "name": "Ukrainian Historical",
        "prompt": "Транскрибуйте українськийтекст на цьому зображенні.",
        "description": "For Ukrainian manuscripts (uses training prompt for fine-tuned model)"
    },
    "detailed": {
        "name": "Detailed Preservation",
        "prompt": "Carefully transcribe all text from this historical document, preserving original spelling, abbreviations, and diacritical marks.",
        "description": "Maximum preservation of original text features"
    },
    "custom": {
        "name": "Custom Prompt",
        "prompt": "",
        "description": "Enter your own custom prompt below"
    }
}


class ChurroEngine(HTREngine):
    """Churro VLM (Qwen2.5-VL-3B) HTR engine plugin for historical documents."""

    def __init__(self):
        self.model: Optional[ChurroInference] = None
        self._config_widget: Optional[QWidget] = None

        # Widget references
        self._model_name_edit: Optional[QLineEdit] = None
        self._adapter_path_edit: Optional[QLineEdit] = None
        self._device_combo: Optional[QComboBox] = None
        self._max_image_size_slider: Optional[QSlider] = None
        self._max_image_size_label: Optional[QLabel] = None
        self._max_tokens_spin: Optional[QSpinBox] = None
        self._prompt_preset_combo: Optional[QComboBox] = None
        self._prompt_text_edit: Optional[QTextEdit] = None

    def get_name(self) -> str:
        return "Churro VLM"

    def get_description(self) -> str:
        return "Historical document OCR (Qwen2.5-VL-3B fine-tuned on 155 collections)"

    def is_available(self) -> bool:
        return CHURRO_AVAILABLE and PYQT_AVAILABLE

    def get_unavailable_reason(self) -> str:
        if not CHURRO_AVAILABLE:
            return "Churro not available. Check inference_churro.py and transformers>=4.37.0"
        if not PYQT_AVAILABLE:
            return "PyQt6 not installed. Install with: pip install PyQt6"
        return ""

    def get_config_widget(self) -> QWidget:
        """Create Churro configuration panel."""
        if self._config_widget is not None:
            return self._config_widget

        widget = QWidget()
        layout = QVBoxLayout()

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()

        model_layout.addWidget(QLabel("HuggingFace Model (Base):"))
        self._model_name_edit = QLineEdit()
        self._model_name_edit.setText("stanford-oval/churro-3B")
        self._model_name_edit.setPlaceholderText("e.g., stanford-oval/churro-3B")
        model_layout.addWidget(self._model_name_edit)

        hint = QLabel("💡 Default: stanford-oval/churro-3B (recommended)")
        hint.setStyleSheet("color: gray; font-size: 9pt;")
        hint.setWordWrap(True)
        model_layout.addWidget(hint)

        # Adapter path (optional)
        model_layout.addWidget(QLabel("LoRA Adapter Path (optional):"))
        adapter_layout = QHBoxLayout()
        self._adapter_path_edit = QLineEdit()
        self._adapter_path_edit.setPlaceholderText("Leave empty for base model, or path to fine-tuned adapter")
        adapter_layout.addWidget(self._adapter_path_edit)
        browse_adapter_btn = QPushButton("Browse...")
        browse_adapter_btn.clicked.connect(self._browse_adapter)
        adapter_layout.addWidget(browse_adapter_btn)
        model_layout.addLayout(adapter_layout)

        adapter_hint = QLabel("💡 For fine-tuned models (e.g., models/churro-ukrainian-pages-*/final_model)")
        adapter_hint.setStyleSheet("color: gray; font-size: 9pt;")
        adapter_hint.setWordWrap(True)
        model_layout.addWidget(adapter_hint)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Device selection
        device_group = QGroupBox("Device Settings")
        device_layout = QVBoxLayout()

        device_layout.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItem("Auto (cuda:0 if available)", "auto")
        self._device_combo.addItem("GPU 0 (cuda:0)", "cuda:0")
        self._device_combo.addItem("GPU 1 (cuda:1)", "cuda:1")
        self._device_combo.addItem("CPU (slow!)", "cpu")
        device_layout.addWidget(self._device_combo)

        device_hint = QLabel("💡 Single GPU recommended (avoids multi-GPU issues)")
        device_hint.setStyleSheet("color: gray; font-size: 9pt;")
        device_layout.addWidget(device_hint)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Preprocessing settings
        preproc_group = QGroupBox("Image Preprocessing")
        preproc_layout = QVBoxLayout()

        # Max image size
        preproc_layout.addWidget(QLabel("Max Image Size:"))

        slider_layout = QHBoxLayout()
        self._max_image_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._max_image_size_slider.setRange(1024, 4096)
        self._max_image_size_slider.setValue(2048)
        self._max_image_size_slider.setTickInterval(512)
        self._max_image_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._max_image_size_slider.valueChanged.connect(self._on_image_size_changed)
        slider_layout.addWidget(self._max_image_size_slider)

        self._max_image_size_label = QLabel("2048px")
        self._max_image_size_label.setMinimumWidth(70)
        slider_layout.addWidget(self._max_image_size_label)

        preproc_layout.addLayout(slider_layout)

        size_hint = QLabel("Resizes images while preserving aspect ratio. Larger = slower but better quality.")
        size_hint.setStyleSheet("color: gray; font-size: 9pt;")
        size_hint.setWordWrap(True)
        preproc_layout.addWidget(size_hint)

        preproc_group.setLayout(preproc_layout)
        layout.addWidget(preproc_group)

        # Generation settings
        gen_group = QGroupBox("Generation Settings")
        gen_layout = QVBoxLayout()

        # Max tokens
        tokens_layout = QHBoxLayout()
        tokens_layout.addWidget(QLabel("Max Tokens:"))
        self._max_tokens_spin = QSpinBox()
        self._max_tokens_spin.setRange(256, 8000)
        self._max_tokens_spin.setValue(2000)
        self._max_tokens_spin.setSingleStep(256)
        tokens_layout.addWidget(self._max_tokens_spin)
        tokens_hint = QLabel("Lines: 500-1000, Pages: 2000-4000")
        tokens_hint.setStyleSheet("color: gray; font-size: 8pt;")
        tokens_layout.addWidget(tokens_hint)
        gen_layout.addLayout(tokens_layout)

        gen_group.setLayout(gen_layout)
        layout.addWidget(gen_group)

        # Prompt customization group
        prompt_group = QGroupBox("Transcription Prompt")
        prompt_layout = QVBoxLayout()

        # Prompt presets dropdown
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self._prompt_preset_combo = QComboBox()

        # Populate with Churro-specific presets
        for key, info in CHURRO_PROMPTS.items():
            self._prompt_preset_combo.addItem(info["name"], key)

        self._prompt_preset_combo.currentIndexChanged.connect(self._on_prompt_preset_changed)
        preset_layout.addWidget(self._prompt_preset_combo)
        prompt_layout.addLayout(preset_layout)

        # Preset description
        self._prompt_desc_label = QLabel("")
        self._prompt_desc_label.setWordWrap(True)
        self._prompt_desc_label.setStyleSheet("color: gray; font-size: 9pt;")
        prompt_layout.addWidget(self._prompt_desc_label)

        # Custom prompt text edit
        self._prompt_text_edit = QTextEdit()
        self._prompt_text_edit.setPlaceholderText("Enter custom prompt or select a preset above...")
        self._prompt_text_edit.setMaximumHeight(80)
        self._prompt_text_edit.setEnabled(False)  # Disabled until "Custom" is selected
        prompt_layout.addWidget(self._prompt_text_edit)

        # Prompt hints
        hint_label = QLabel(
            "💡 Tip: Churro is trained on 46 languages and 155 historical collections.\n"
            "Specify the script type (Church Slavonic, Glagolitic) for best results."
        )
        hint_label.setStyleSheet("color: gray; font-size: 9pt; padding: 5px;")
        hint_label.setWordWrap(True)
        prompt_layout.addWidget(hint_label)

        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        layout.addStretch()
        widget.setLayout(layout)

        # Initialize prompt to default
        self._on_prompt_preset_changed(0)

        self._config_widget = widget
        return widget

    def _browse_adapter(self):
        """Open file browser to select adapter directory."""
        if self._adapter_path_edit is None:
            return

        current_path = self._adapter_path_edit.text().strip()
        if not current_path:
            current_path = str(Path.cwd() / "models")

        directory = QFileDialog.getExistingDirectory(
            self._config_widget,
            "Select LoRA Adapter Directory",
            current_path,
            QFileDialog.Option.ShowDirsOnly
        )

        if directory:
            self._adapter_path_edit.setText(directory)

    def _on_image_size_changed(self, value: int):
        """Update image size label."""
        if self._max_image_size_label:
            self._max_image_size_label.setText(f"{value}px")

    def _on_prompt_preset_changed(self, index: int):
        """Update prompt text and description when preset changes."""
        if self._prompt_preset_combo is None or self._prompt_text_edit is None:
            return

        preset_key = self._prompt_preset_combo.currentData()
        preset_info = CHURRO_PROMPTS.get(preset_key, CHURRO_PROMPTS["default"])

        # Update description
        if self._prompt_desc_label:
            self._prompt_desc_label.setText(preset_info["description"])

        if preset_key == "custom":
            # Enable custom editing
            self._prompt_text_edit.setEnabled(True)
            self._prompt_text_edit.setFocus()
        else:
            # Load preset and disable editing
            self._prompt_text_edit.setEnabled(False)
            self._prompt_text_edit.setPlainText(preset_info["prompt"])

    def get_config(self) -> Dict[str, Any]:
        """Extract configuration from widget controls."""
        if self._config_widget is None:
            return {}

        adapter_path = self._adapter_path_edit.text().strip()

        config = {
            "model_name": self._model_name_edit.text().strip() or "stanford-oval/churro-3B",
            "adapter_path": adapter_path if adapter_path else None,
            "device": self._device_combo.currentData(),
            "max_image_size": self._max_image_size_slider.value(),
            "max_new_tokens": self._max_tokens_spin.value(),
            # Prompt configuration
            "prompt_preset": self._prompt_preset_combo.currentData(),
            "prompt_text": self._prompt_text_edit.toPlainText().strip()
        }

        return config

    def set_config(self, config: Dict[str, Any]):
        """Restore configuration to widget controls."""
        if self._config_widget is None:
            return

        self._model_name_edit.setText(config.get("model_name", "stanford-oval/churro-3B"))

        adapter_path = config.get("adapter_path", "")
        self._adapter_path_edit.setText(adapter_path if adapter_path else "")

        # Set device
        device = config.get("device", "auto")
        for i in range(self._device_combo.count()):
            if self._device_combo.itemData(i) == device:
                self._device_combo.setCurrentIndex(i)
                break

        self._max_image_size_slider.setValue(config.get("max_image_size", 2048))
        self._max_tokens_spin.setValue(config.get("max_new_tokens", 2000))

        # Prompt configuration
        prompt_preset = config.get("prompt_preset", "default")
        for i in range(self._prompt_preset_combo.count()):
            if self._prompt_preset_combo.itemData(i) == prompt_preset:
                self._prompt_preset_combo.setCurrentIndex(i)
                break

        # If custom prompt, restore text
        if prompt_preset == "custom":
            custom_text = config.get("prompt_text", "")
            self._prompt_text_edit.setPlainText(custom_text)

    def load_model(self, config: Dict[str, Any]) -> bool:
        """Load Churro model."""
        try:
            # Cleanup any existing model first
            if self.model is not None:
                print("Cleaning up previous model before loading new one...")
                self.unload_model()

            model_name = config.get("model_name", "stanford-oval/churro-3B")
            adapter_path = config.get("adapter_path", None)
            device = config.get("device", "auto")
            max_image_size = config.get("max_image_size", 2048)

            print(f"Loading Churro model: {model_name}")
            if adapter_path:
                print(f"  with adapter: {adapter_path}")
            print(f"Device: {device}, Max image size: {max_image_size}px")

            # CRITICAL: Uses ChurroInference (Qwen2.5-VL), NOT Qwen3VLMInference
            self.model = ChurroInference(
                model_name=model_name,
                adapter_path=adapter_path,
                device=device,
                max_image_size=max_image_size
            )

            print("✓ Churro model loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading Churro model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            return False

    def unload_model(self):
        """Unload model from memory with proper cleanup."""
        if self.model is not None:
            try:
                self.model.cleanup()
            except Exception as e:
                print(f"Warning during model cleanup: {e}")

            del self.model
            self.model = None
            print("✓ Churro model unloaded")

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _get_prompt_from_config(self, config: Optional[Dict[str, Any]]) -> str:
        """Extract prompt from config."""
        if not config:
            return CHURRO_PROMPTS["default"]["prompt"]

        prompt_preset = config.get("prompt_preset", "default")
        prompt_text = config.get("prompt_text", "")

        if prompt_preset == "custom" and prompt_text:
            return prompt_text
        else:
            return CHURRO_PROMPTS.get(prompt_preset, CHURRO_PROMPTS["default"])["prompt"]

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """Transcribe a line image with Churro."""
        if self.model is None:
            return TranscriptionResult(text="[Model not loaded]", confidence=0.0)

        try:
            # Convert numpy to PIL
            from PIL import Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Get prompt and settings
            prompt = self._get_prompt_from_config(config)
            max_tokens = config.get("max_new_tokens", 500) if config else 500

            # Use transcribe_line method (shorter max_tokens for lines)
            result = self.model.transcribe_line(
                pil_image,
                prompt=prompt,
                max_new_tokens=min(max_tokens, 1000)  # Cap at 1000 for single lines
            )

            return TranscriptionResult(
                text=result.text,
                confidence=result.confidence,
                metadata={
                    "model": "churro-vlm",
                    "model_name": self.model.model_name,
                    "processing_time": result.processing_time,
                    "prompt": prompt,
                    "prompt_preset": config.get("prompt_preset", "default") if config else "default"
                }
            )

        except Exception as e:
            import traceback
            print(f"Error in Churro transcription: {e}")
            print(traceback.format_exc())
            return TranscriptionResult(text=f"[Error: {e}]", confidence=0.0)

    def transcribe_lines(self, images: list[np.ndarray], config: Optional[Dict[str, Any]] = None) -> list[TranscriptionResult]:
        """Batch transcription with Churro."""
        if self.model is None:
            return [TranscriptionResult(text="[Model not loaded]", confidence=0.0) for _ in images]

        try:
            from PIL import Image

            # Get prompt once for all images
            prompt = self._get_prompt_from_config(config)
            max_tokens = config.get("max_new_tokens", 500) if config else 500

            results = []

            for img_array in images:
                # Convert to PIL
                if isinstance(img_array, np.ndarray):
                    pil_img = Image.fromarray(img_array)
                else:
                    pil_img = img_array

                # Transcribe with custom prompt
                result = self.model.transcribe_line(
                    pil_img,
                    prompt=prompt,
                    max_new_tokens=min(max_tokens, 1000)
                )

                results.append(TranscriptionResult(
                    text=result.text,
                    confidence=result.confidence,
                    metadata={
                        "model": "churro-vlm",
                        "model_name": self.model.model_name,
                        "processing_time": result.processing_time,
                        "prompt": prompt,
                        "prompt_preset": config.get("prompt_preset", "default") if config else "default"
                    }
                ))

            return results

        except Exception as e:
            print(f"Error in Churro batch transcription: {e}")
            import traceback
            traceback.print_exc()
            return [TranscriptionResult(text=f"[Error: {e}]", confidence=0.0) for _ in images]

    def supports_batch(self) -> bool:
        """Churro supports batch processing."""
        return True

    def get_capabilities(self) -> Dict[str, bool]:
        """Churro capabilities."""
        return {
            "batch_processing": True,
            "confidence_scores": False,  # Not implemented yet
            "beam_search": False,  # Uses greedy decoding
            "language_model": True,  # VLM has built-in language understanding
            "preprocessing": True,  # Has built-in vision preprocessing
        }

    def requires_line_segmentation(self) -> bool:
        """Churro can process full pages without segmentation (like Qwen3)."""
        return False  # VLM can process full pages directly

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        if not self.is_model_loaded():
            return {
                "status": "Not loaded",
                "model": "N/A"
            }

        try:
            import torch
            memory_info = "N/A"
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                memory_info = f"{allocated:.1f}GB / {total:.1f}GB"

            return {
                "status": "Loaded",
                "model": self.model.model_name,
                "device": self.model.device,
                "max_image_size": f"{self.model.max_image_size}px",
                "type": "Churro VLM (Qwen2.5-VL-3B)",
                "specialization": "Historical documents (155 collections, 46 languages)",
                "memory": memory_info
            }
        except Exception as e:
            return {
                "status": "Loaded (info unavailable)",
                "error": str(e)
            }
