"""
Kraken HTR Engine Plugin

Wraps the Kraken OCR system as a plugin for the unified GUI.
Kraken is specialized for historical document OCR with robust segmentation and recognition.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from htr_engine_base import HTREngine, TranscriptionResult

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QPushButton, QLineEdit, QFileDialog, QGroupBox, QCheckBox
    )
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

try:
    from kraken import rpred
    from kraken.lib import vgsl, models
    KRAKEN_AVAILABLE = True
except ImportError:
    KRAKEN_AVAILABLE = False


# Preset Kraken models (can be extended)
KRAKEN_MODELS = {
    "default": {
        "path": None,  # Will use get_model from Kraken
        "description": "Default Kraken model (requires download)",
        "language": "multi"
    },
}


class KrakenEngine(HTREngine):
    """Kraken HTR engine plugin."""

    def __init__(self):
        self.model: Optional[Any] = None  # TorchSeqRecognizer
        self._config_widget: Optional[QWidget] = None

        # Widget references
        self._model_source_combo: Optional[QComboBox] = None
        self._preset_combo: Optional[QComboBox] = None
        self._custom_model_edit: Optional[QLineEdit] = None
        self._bidi_reorder_check: Optional[QCheckBox] = None

    def get_name(self) -> str:
        return "Kraken"

    def get_description(self) -> str:
        return "Kraken OCR - Specialized for historical documents with .mlmodel support"

    def is_available(self) -> bool:
        return KRAKEN_AVAILABLE and PYQT_AVAILABLE

    def get_unavailable_reason(self) -> str:
        if not KRAKEN_AVAILABLE:
            return "Kraken not installed. Install with: pip install kraken"
        if not PYQT_AVAILABLE:
            return "PyQt6 not installed. Install with: pip install PyQt6"
        return ""

    def get_config_widget(self) -> QWidget:
        """Create Kraken configuration panel."""
        if self._config_widget is not None:
            return self._config_widget

        widget = QWidget()
        layout = QVBoxLayout()

        # Model source selection
        source_group = QGroupBox("Model Source")
        source_layout = QVBoxLayout()

        self._model_source_combo = QComboBox()
        self._model_source_combo.addItems(["Preset Models", "Custom Model File"])
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

        preset_hint = QLabel("Note: Preset models may require download on first use")
        preset_hint.setStyleSheet("color: gray; font-size: 9pt;")
        preset_layout.addWidget(preset_hint)

        self._preset_group.setLayout(preset_layout)
        layout.addWidget(self._preset_group)

        # Custom model group
        self._custom_group = QGroupBox("Custom Model")
        custom_layout = QVBoxLayout()

        custom_layout.addWidget(QLabel("Model File (.mlmodel):"))
        model_layout = QHBoxLayout()
        self._custom_model_edit = QLineEdit()
        self._custom_model_edit.setPlaceholderText("Path to .mlmodel file")
        model_layout.addWidget(self._custom_model_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_model)
        model_layout.addWidget(browse_btn)
        custom_layout.addLayout(model_layout)

        self._custom_group.setLayout(custom_layout)
        self._custom_group.setVisible(False)  # Hidden by default
        layout.addWidget(self._custom_group)

        # Recognition settings
        settings_group = QGroupBox("Recognition Settings")
        settings_layout = QVBoxLayout()

        self._bidi_reorder_check = QCheckBox("Bidirectional Text Reordering")
        self._bidi_reorder_check.setChecked(True)
        self._bidi_reorder_check.setToolTip("Enable for RTL languages (Arabic, Hebrew, etc.)")
        settings_layout.addWidget(self._bidi_reorder_check)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        layout.addStretch()
        widget.setLayout(layout)

        self._config_widget = widget
        return widget

    def _populate_preset_models(self):
        """Populate preset models dropdown."""
        if self._preset_combo is None:
            return

        self._preset_combo.clear()

        if not KRAKEN_MODELS:
            self._preset_combo.addItem("No presets available")
            return

        for model_id, info in KRAKEN_MODELS.items():
            desc = info.get('description', model_id)
            self._preset_combo.addItem(f"{model_id} - {desc}", userData=model_id)

    def _on_model_source_changed(self, source: str):
        """Toggle between preset and custom model selection."""
        is_preset = (source == "Preset Models")
        self._preset_group.setVisible(is_preset)
        self._custom_group.setVisible(not is_preset)

    def _browse_model(self):
        """Open file dialog to select model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self._config_widget,
            "Select Kraken Model",
            "models",
            "Kraken Models (*.mlmodel);;All Files (*)"
        )

        if file_path:
            self._custom_model_edit.setText(file_path)

    def get_config(self) -> Dict[str, Any]:
        """Extract configuration from widget controls."""
        if self._config_widget is None:
            return {}

        is_preset = (self._model_source_combo.currentText() == "Preset Models")

        config = {
            "model_source": "preset" if is_preset else "custom",
            "bidi_reordering": self._bidi_reorder_check.isChecked(),
        }

        if is_preset:
            model_id = self._preset_combo.currentData()
            if model_id and model_id in KRAKEN_MODELS:
                config["preset_id"] = model_id
                config["model_path"] = KRAKEN_MODELS[model_id].get("path")
        else:
            config["model_path"] = self._custom_model_edit.text()

        return config

    def set_config(self, config: Dict[str, Any]):
        """Restore configuration to widget controls."""
        if self._config_widget is None:
            return

        model_source = config.get("model_source", "preset")
        self._model_source_combo.setCurrentText("Preset Models" if model_source == "preset" else "Custom Model File")

        if model_source == "preset":
            preset_id = config.get("preset_id", "")
            for i in range(self._preset_combo.count()):
                if self._preset_combo.itemData(i) == preset_id:
                    self._preset_combo.setCurrentIndex(i)
                    break
        else:
            self._custom_model_edit.setText(config.get("model_path", ""))

        self._bidi_reorder_check.setChecked(config.get("bidi_reordering", True))

    def load_model(self, config: Dict[str, Any]) -> bool:
        """Load Kraken model."""
        try:
            model_path = config.get("model_path")

            # If model_path is None (preset), try to get default model
            if model_path is None:
                # Kraken can download default models automatically
                # For now, we'll require an explicit path
                print("Error: No model path specified. Please provide a .mlmodel file path.")
                return False

            if not model_path or not Path(model_path).exists():
                print(f"Error: Model file not found: {model_path}")
                return False

            # Load model using Kraken's vgsl module
            vgsl_model = vgsl.TorchVGSLModel.load_model(model_path)

            # Wrap in TorchSeqRecognizer for use with rpred
            from kraken.lib.models import TorchSeqRecognizer
            self.model = TorchSeqRecognizer(vgsl_model, device='cpu')

            print(f"Kraken model loaded from: {model_path}")

            return True

        except Exception as e:
            import traceback
            print(f"Error loading Kraken model: {e}")
            print(traceback.format_exc())
            self.model = None
            return False

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None

            # Free GPU memory
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """Transcribe a line image with Kraken."""
        if self.model is None:
            return TranscriptionResult(text="[Model not loaded]", confidence=0.0)

        if config is None:
            config = self.get_config()

        try:
            # Import numpy at the start
            import numpy as np

            # Convert numpy to PIL
            from PIL import Image as PILImage
            if isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image

            # Convert to grayscale first
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')

            # IMPORTANT: Do NOT binarize! Kraken models work better with grayscale
            # Modern Kraken models are trained on grayscale images and binarization
            # destroys character details, especially in historical manuscripts
            # The previous median threshold was causing poor recognition quality
            binary_image = pil_image  # Keep original grayscale

            # Create a simple segmentation boundary for the full line image
            # Kraken's rpred needs a Segmentation object with line boundaries
            from kraken.containers import BaselineLine, Segmentation

            height, width = binary_image.height, binary_image.width

            # Create a baseline (horizontal line through the middle)
            # Use 0-indexed coordinates (width-1, height-1 as maximum)
            baseline = [[0, height // 2], [width - 1, height // 2]]

            # Create a boundary polygon (rectangle around the entire image)
            # Use 0-indexed coordinates to avoid "outside of image bounds" error
            boundary = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]

            # Create a BaselineLine (not BBoxLine - that doesn't support baselines)
            line = BaselineLine(
                id='line_0',
                baseline=baseline,
                boundary=boundary,
                text='',
                tags=None,
                split=None
            )

            # Create Segmentation container
            seg = Segmentation(
                type='baselines',
                imagename='line',
                text_direction='horizontal-lr',
                script_detection=False,
                lines=[line],
                regions={},
                line_orders=[]
            )

            # Run recognition
            bidi = config.get("bidi_reordering", True)

            # Model is already wrapped as TorchSeqRecognizer in load_model()
            # rpred returns a generator
            results = list(rpred.rpred(
                network=self.model,
                im=binary_image,
                bounds=seg,
                bidi_reordering=bidi
            ))

            # Extract text from first result
            if results and len(results) > 0:
                text = results[0].prediction
                confidence = results[0].confidences
                avg_confidence = sum(confidence) / len(confidence) if confidence else 1.0

                return TranscriptionResult(
                    text=text,
                    confidence=avg_confidence,
                    metadata={"model": "kraken"}
                )
            else:
                return TranscriptionResult(text="", confidence=0.0)

        except Exception as e:
            import traceback
            print(f"Error in Kraken transcription: {e}")
            print(traceback.format_exc())
            return TranscriptionResult(text=f"[Error: {e}]", confidence=0.0)

    def get_capabilities(self) -> Dict[str, bool]:
        """Kraken capabilities."""
        return {
            "batch_processing": False,  # Could be implemented
            "confidence_scores": True,  # Kraken provides per-character confidence
            "beam_search": False,  # Internal to Kraken
            "language_model": False,  # Not explicitly exposed
            "preprocessing": False,  # External binarization recommended
        }
