"""
Qwen3 VLM Engine Plugin

Wraps the Qwen3 Vision-Language Model HTR inference system as a plugin.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from htr_engine_base import HTREngine, TranscriptionResult

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QPushButton, QSpinBox, QCheckBox, QLineEdit, QFileDialog,
        QGroupBox, QRadioButton, QButtonGroup, QSlider, QTextEdit
    )
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

try:
    from inference_qwen3 import Qwen3VLMInference, QWEN3_MODELS
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False
    QWEN3_MODELS = {}

try:
    from qwen3_prompts import QWEN3_PROMPT_PRESETS, get_prompt
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
        self._config_widget: Optional[QWidget] = None

        # Widget references
        self._model_source_combo: Optional[QComboBox] = None
        self._preset_combo: Optional[QComboBox] = None
        self._custom_base_edit: Optional[QLineEdit] = None
        self._custom_adapter_edit: Optional[QLineEdit] = None
        self._max_image_size_slider: Optional[QSlider] = None
        self._max_image_size_label: Optional[QLabel] = None
        self._use_sampling_checkbox: Optional[QCheckBox] = None
        self._temperature_spin: Optional[QSpinBox] = None
        self._repetition_penalty_spin: Optional[QSpinBox] = None
        self._max_tokens_spin: Optional[QSpinBox] = None

    def get_name(self) -> str:
        return "Qwen3-VL"

    def get_description(self) -> str:
        return "Vision-language model with LoRA fine-tuning support"

    def is_available(self) -> bool:
        return QWEN3_AVAILABLE and PYQT_AVAILABLE

    def get_unavailable_reason(self) -> str:
        if not QWEN3_AVAILABLE:
            return "Qwen3 not available. Install with: pip install transformers>=4.37.0 accelerate peft qwen-vl-utils"
        if not PYQT_AVAILABLE:
            return "PyQt6 not installed. Install with: pip install PyQt6"
        return ""

    def get_config_widget(self) -> QWidget:
        """Create Qwen3 configuration panel."""
        if self._config_widget is not None:
            return self._config_widget

        widget = QWidget()
        layout = QVBoxLayout()

        # Model source selection
        source_group = QGroupBox("Model Source")
        source_layout = QVBoxLayout()

        self._model_source_combo = QComboBox()
        self._model_source_combo.addItems(["Preset Models", "Custom Base + Adapter"])
        self._model_source_combo.currentTextChanged.connect(self._on_model_source_changed)
        source_layout.addWidget(self._model_source_combo)

        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # Preset models group
        self._preset_group = QGroupBox("Preset Model")
        preset_layout = QVBoxLayout()

        self._preset_combo = QComboBox()
        self._populate_preset_models()
        preset_layout.addWidget(QLabel("Model:"))
        preset_layout.addWidget(self._preset_combo)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)

        # Preset description
        self._preset_desc_label = QLabel("")
        self._preset_desc_label.setWordWrap(True)
        self._preset_desc_label.setStyleSheet("color: gray; font-size: 9pt;")
        preset_layout.addWidget(self._preset_desc_label)

        self._preset_group.setLayout(preset_layout)
        layout.addWidget(self._preset_group)

        # Custom model group
        self._custom_group = QGroupBox("Custom Model")
        custom_layout = QVBoxLayout()

        # Base model
        custom_layout.addWidget(QLabel("Base Model:"))
        base_layout = QHBoxLayout()
        self._custom_base_edit = QLineEdit()
        self._custom_base_edit.setPlaceholderText("e.g., Qwen/Qwen3-VL-8B-Instruct")
        self._custom_base_edit.setText("Qwen/Qwen3-VL-8B-Instruct")
        base_layout.addWidget(self._custom_base_edit)
        custom_layout.addLayout(base_layout)

        # Adapter (optional)
        custom_layout.addWidget(QLabel("Adapter Path (optional):"))
        adapter_layout = QHBoxLayout()
        self._custom_adapter_edit = QLineEdit()
        self._custom_adapter_edit.setPlaceholderText("Leave empty for base model, or path to LoRA adapter")
        adapter_layout.addWidget(self._custom_adapter_edit)
        browse_adapter_btn = QPushButton("Browse...")
        browse_adapter_btn.clicked.connect(self._browse_adapter)
        adapter_layout.addWidget(browse_adapter_btn)
        custom_layout.addLayout(adapter_layout)

        self._custom_group.setLayout(custom_layout)
        self._custom_group.setVisible(False)  # Hidden by default
        layout.addWidget(self._custom_group)

        # Inference settings
        settings_group = QGroupBox("Inference Settings")
        settings_layout = QVBoxLayout()

        # Max image size
        size_layout = QVBoxLayout()
        size_layout.addWidget(QLabel("Max Image Size:"))

        slider_layout = QHBoxLayout()
        self._max_image_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._max_image_size_slider.setRange(512, 2048)
        self._max_image_size_slider.setValue(1536)
        self._max_image_size_slider.setTickInterval(256)
        self._max_image_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._max_image_size_slider.valueChanged.connect(self._on_image_size_changed)
        slider_layout.addWidget(self._max_image_size_slider)

        self._max_image_size_label = QLabel("1536px")
        self._max_image_size_label.setMinimumWidth(60)
        slider_layout.addWidget(self._max_image_size_label)

        size_layout.addLayout(slider_layout)
        size_hint = QLabel("Larger = better quality but slower, more VRAM")
        size_hint.setStyleSheet("color: gray; font-size: 9pt;")
        size_layout.addWidget(size_hint)

        settings_layout.addLayout(size_layout)

        # Generation settings
        gen_group = QGroupBox("Generation Parameters")
        gen_layout = QVBoxLayout()

        # Temperature
        self._use_sampling_checkbox = QCheckBox("Enable Sampling (vs Greedy)")
        self._use_sampling_checkbox.toggled.connect(self._on_sampling_toggled)
        gen_layout.addWidget(self._use_sampling_checkbox)

        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self._temperature_spin = QSpinBox()
        self._temperature_spin.setRange(1, 20)
        self._temperature_spin.setValue(10)
        self._temperature_spin.setSuffix(" (0.1-2.0)")
        self._temperature_spin.setEnabled(False)
        temp_layout.addWidget(self._temperature_spin)
        temp_hint = QLabel("Higher = more creative, lower = more focused")
        temp_hint.setStyleSheet("color: gray; font-size: 8pt;")
        temp_layout.addWidget(temp_hint)
        gen_layout.addLayout(temp_layout)

        # Repetition penalty
        rep_layout = QHBoxLayout()
        rep_layout.addWidget(QLabel("Repetition Penalty:"))
        self._repetition_penalty_spin = QSpinBox()
        self._repetition_penalty_spin.setRange(10, 30)
        self._repetition_penalty_spin.setValue(12)
        self._repetition_penalty_spin.setSuffix(" (1.0-3.0)")
        rep_layout.addWidget(self._repetition_penalty_spin)
        rep_hint = QLabel("Higher = more diverse output")
        rep_hint.setStyleSheet("color: gray; font-size: 8pt;")
        rep_layout.addWidget(rep_hint)
        gen_layout.addLayout(rep_layout)

        # Max tokens
        tokens_layout = QHBoxLayout()
        tokens_layout.addWidget(QLabel("Max Tokens:"))
        self._max_tokens_spin = QSpinBox()
        self._max_tokens_spin.setRange(256, 4096)
        self._max_tokens_spin.setValue(1024)  # Default to 1024 for faster inference
        self._max_tokens_spin.setSingleStep(256)
        tokens_layout.addWidget(self._max_tokens_spin)
        tokens_hint = QLabel("Maximum length of output")
        tokens_hint.setStyleSheet("color: gray; font-size: 8pt;")
        tokens_layout.addWidget(tokens_hint)
        gen_layout.addLayout(tokens_layout)

        gen_group.setLayout(gen_layout)
        settings_layout.addWidget(gen_group)

        # Prompt customization group
        prompt_group = QGroupBox("Transcription Prompt")
        prompt_layout = QVBoxLayout()

        # Prompt presets dropdown
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self._prompt_preset_combo = QComboBox()

        # Populate with presets
        for key, info in QWEN3_PROMPT_PRESETS.items():
            self._prompt_preset_combo.addItem(info["name"], key)

        self._prompt_preset_combo.currentIndexChanged.connect(self._on_prompt_preset_changed)
        preset_layout.addWidget(self._prompt_preset_combo)
        prompt_layout.addLayout(preset_layout)

        # Custom prompt text edit
        self._prompt_text_edit = QTextEdit()
        self._prompt_text_edit.setPlaceholderText("Enter custom prompt or select a preset above...")
        self._prompt_text_edit.setMaximumHeight(80)
        self._prompt_text_edit.setEnabled(False)  # Disabled until "Custom" is selected
        prompt_layout.addWidget(self._prompt_text_edit)

        # Prompt hints
        hint_label = QLabel(
            "💡 Tip: Specify language, script type, or content domain for better results.\n"
            "Examples: 'Church Slavonic', 'mathematical formulas', 'degraded manuscript'"
        )
        hint_label.setStyleSheet("color: gray; font-size: 9pt; padding: 5px;")
        hint_label.setWordWrap(True)
        prompt_layout.addWidget(hint_label)

        prompt_group.setLayout(prompt_layout)
        settings_layout.addWidget(prompt_group)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        layout.addStretch()
        widget.setLayout(layout)

        # Initialize prompt to default
        self._on_prompt_preset_changed(0)

        # Initialize preset description
        self._on_preset_changed(self._preset_combo.currentText())

        self._config_widget = widget
        return widget

    def _populate_preset_models(self):
        """Populate preset models dropdown."""
        if self._preset_combo is None:
            return

        self._preset_combo.clear()

        if not QWEN3_MODELS:
            self._preset_combo.addItem("No presets available")
            return

        for model_id, info in QWEN3_MODELS.items():
            # Format: "model-id (description - VRAM)"
            display_name = f"{model_id} ({info.get('vram', 'Unknown VRAM')})"
            self._preset_combo.addItem(display_name, userData=model_id)

    def _on_preset_changed(self, display_name: str):
        """Update description when preset changes."""
        if not self._preset_combo or not self._preset_desc_label:
            return

        model_id = self._preset_combo.currentData()
        if model_id and model_id in QWEN3_MODELS:
            info = QWEN3_MODELS[model_id]
            desc = info.get('description', '')
            speed = info.get('speed', '')
            self._preset_desc_label.setText(f"{desc}\nSpeed: {speed}")
        else:
            self._preset_desc_label.setText("")

    def _on_model_source_changed(self, source: str):
        """Toggle between preset and custom model selection."""
        is_preset = (source == "Preset Models")
        self._preset_group.setVisible(is_preset)
        self._custom_group.setVisible(not is_preset)

    def _browse_adapter(self):
        """Open file dialog to select adapter directory."""
        directory = QFileDialog.getExistingDirectory(
            self._config_widget,
            "Select Adapter Directory",
            "models"
        )

        if directory:
            self._custom_adapter_edit.setText(directory)

    def _on_image_size_changed(self, value: int):
        """Update image size label."""
        if self._max_image_size_label:
            self._max_image_size_label.setText(f"{value}px")

    def _on_sampling_toggled(self, checked: bool):
        """Enable/disable temperature when sampling is toggled."""
        if self._temperature_spin:
            self._temperature_spin.setEnabled(checked)

    def _on_prompt_preset_changed(self, index: int):
        """Update prompt text when preset changes."""
        if self._prompt_preset_combo is None or self._prompt_text_edit is None:
            return

        preset_key = self._prompt_preset_combo.currentData()

        if preset_key == "custom":
            # Enable custom editing
            self._prompt_text_edit.setEnabled(True)
            self._prompt_text_edit.setFocus()
        else:
            # Load preset and disable editing
            self._prompt_text_edit.setEnabled(False)
            preset_info = QWEN3_PROMPT_PRESETS.get(preset_key, QWEN3_PROMPT_PRESETS["default"])
            self._prompt_text_edit.setPlainText(preset_info["prompt"])

    def get_config(self) -> Dict[str, Any]:
        """Extract configuration from widget controls."""
        if self._config_widget is None:
            return {}

        is_preset = (self._model_source_combo.currentText() == "Preset Models")

        config = {
            "model_source": "preset" if is_preset else "custom",
            "max_image_size": self._max_image_size_slider.value(),
            "do_sample": self._use_sampling_checkbox.isChecked(),
            "temperature": self._temperature_spin.value() / 10.0,  # Convert to 0.1-2.0 range
            "repetition_penalty": self._repetition_penalty_spin.value() / 10.0,  # Convert to 1.0-3.0 range
            "max_new_tokens": self._max_tokens_spin.value(),
            # Prompt configuration
            "prompt_preset": self._prompt_preset_combo.currentData(),
            "prompt_text": self._prompt_text_edit.toPlainText().strip()
        }

        if is_preset:
            model_id = self._preset_combo.currentData()
            if model_id and model_id in QWEN3_MODELS:
                preset_info = QWEN3_MODELS[model_id]
                config["base_model"] = preset_info["base"]
                config["adapter"] = preset_info.get("adapter")
                config["preset_id"] = model_id
        else:
            config["base_model"] = self._custom_base_edit.text()
            adapter_text = self._custom_adapter_edit.text().strip()
            config["adapter"] = adapter_text if adapter_text else None

        return config

    def set_config(self, config: Dict[str, Any]):
        """Restore configuration to widget controls."""
        if self._config_widget is None:
            return

        model_source = config.get("model_source", "preset")
        self._model_source_combo.setCurrentText("Preset Models" if model_source == "preset" else "Custom Base + Adapter")

        if model_source == "preset":
            preset_id = config.get("preset_id", "")
            for i in range(self._preset_combo.count()):
                if self._preset_combo.itemData(i) == preset_id:
                    self._preset_combo.setCurrentIndex(i)
                    break
        else:
            self._custom_base_edit.setText(config.get("base_model", ""))
            self._custom_adapter_edit.setText(config.get("adapter", "") or "")

        self._max_image_size_slider.setValue(config.get("max_image_size", 1536))

        # Generation parameters
        self._use_sampling_checkbox.setChecked(config.get("do_sample", False))
        self._temperature_spin.setValue(int(config.get("temperature", 1.0) * 10))
        self._repetition_penalty_spin.setValue(int(config.get("repetition_penalty", 1.2) * 10))
        self._max_tokens_spin.setValue(config.get("max_new_tokens", 2048))

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

            self.model = Qwen3VLMInference(
                base_model=base_model,
                adapter_model=adapter,
                max_image_size=max_image_size
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
            return TranscriptionResult(text=f"[Error: {e}]", confidence=0.0)

    def transcribe_lines(self, images: list[np.ndarray], config: Optional[Dict[str, Any]] = None) -> list[TranscriptionResult]:
        """Batch transcription with Qwen3 (processes each line with custom prompt)."""
        if self.model is None:
            return [TranscriptionResult(text="[Model not loaded]", confidence=0.0) for _ in images]

        try:
            # Extract prompt from config (same logic as transcribe_line)
            prompt = "Transcribe the text shown in this image."  # Default
            if config:
                prompt_preset = config.get("prompt_preset", "default")
                prompt_text = config.get("prompt_text", "")

                # Determine which prompt to use
                if PROMPTS_AVAILABLE:
                    prompt = get_prompt(prompt_preset, prompt_text)
                elif prompt_preset == "custom" and prompt_text:
                    prompt = prompt_text

            # Process each image with the custom prompt
            from PIL import Image
            results = []

            for img_array in images:
                # Convert to PIL
                if isinstance(img_array, np.ndarray):
                    pil_img = Image.fromarray(img_array)
                else:
                    pil_img = img_array

                # Transcribe with custom prompt
                generation_kwargs = {}
                if config:
                    generation_kwargs["do_sample"] = config.get("do_sample", False)
                    generation_kwargs["temperature"] = config.get("temperature", 1.0) if config.get("do_sample") else None
                    generation_kwargs["max_new_tokens"] = config.get("max_new_tokens", 2048)

                result = self.model.transcribe_page(pil_img, prompt=prompt, return_confidence=True, **generation_kwargs)

                # Extract text and confidence
                text = result.text if hasattr(result, 'text') else str(result)
                confidence = result.confidence if hasattr(result, 'confidence') else 1.0

                results.append(TranscriptionResult(
                    text=text,
                    confidence=confidence if confidence is not None else 1.0,
                    metadata={
                        "model": "qwen3-vlm",
                        "prompt": prompt,
                        "prompt_preset": config.get("prompt_preset", "default") if config else "default"
                    }
                ))

            return results

        except Exception as e:
            print(f"Error in Qwen3 batch transcription: {e}")
            import traceback
            traceback.print_exc()
            return [TranscriptionResult(text=f"[Error: {e}]", confidence=0.0) for _ in images]

    def supports_batch(self) -> bool:
        """Qwen3 supports batch processing."""
        return True

    def get_capabilities(self) -> Dict[str, bool]:
        """Qwen3 capabilities."""
        return {
            "batch_processing": True,
            "confidence_scores": True,  # VLM can provide confidence
            "beam_search": False,  # Uses sampling/greedy decoding
            "language_model": True,  # VLM has built-in language understanding
            "preprocessing": True,  # Has built-in vision preprocessing
        }
    def requires_line_segmentation(self) -> bool:
        """Qwen3 can process full pages without segmentation."""
        return False
