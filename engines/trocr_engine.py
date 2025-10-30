"""
TrOCR Engine Plugin

Wraps the TrOCR HTR inference system as a plugin for the unified GUI.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from htr_engine_base import HTREngine, TranscriptionResult

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QPushButton, QSpinBox, QCheckBox, QLineEdit, QFileDialog,
        QGroupBox, QRadioButton, QButtonGroup
    )
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

try:
    from inference_page import TrOCRInference
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False


class TrOCREngine(HTREngine):
    """TrOCR HTR engine plugin."""

    def __init__(self):
        self.model: Optional[TrOCRInference] = None
        self._config_widget: Optional[QWidget] = None

        # Widget references (set when config widget is created)
        self._model_source_combo: Optional[QComboBox] = None
        self._local_model_combo: Optional[QComboBox] = None
        self._hf_model_edit: Optional[QLineEdit] = None
        self._beam_spin: Optional[QSpinBox] = None
        self._normalize_check: Optional[QCheckBox] = None

    def get_name(self) -> str:
        return "TrOCR"

    def get_description(self) -> str:
        return "Transformer-based OCR optimized for handwritten manuscripts"

    def is_available(self) -> bool:
        return TROCR_AVAILABLE and PYQT_AVAILABLE

    def get_unavailable_reason(self) -> str:
        if not TROCR_AVAILABLE:
            return "TrOCR inference module not available. Check that inference_page.py exists."
        if not PYQT_AVAILABLE:
            return "PyQt6 not installed. Install with: pip install PyQt6"
        return ""

    def get_config_widget(self) -> QWidget:
        """Create TrOCR configuration panel."""
        if self._config_widget is not None:
            return self._config_widget

        widget = QWidget()
        layout = QVBoxLayout()

        # Model source selection
        source_group = QGroupBox("Model Source")
        source_layout = QVBoxLayout()

        self._model_source_combo = QComboBox()
        self._model_source_combo.addItems(["Local Models", "HuggingFace Hub"])
        self._model_source_combo.currentTextChanged.connect(self._on_model_source_changed)
        source_layout.addWidget(self._model_source_combo)

        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # Local models group
        self._local_group = QGroupBox("Local Model")
        local_layout = QVBoxLayout()

        self._local_model_combo = QComboBox()
        self._populate_local_models()
        local_layout.addWidget(QLabel("Model:"))
        local_layout.addWidget(self._local_model_combo)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_local_model)
        local_layout.addWidget(browse_btn)

        self._local_group.setLayout(local_layout)
        layout.addWidget(self._local_group)

        # HuggingFace models group
        self._hf_group = QGroupBox("HuggingFace Model")
        hf_layout = QVBoxLayout()

        self._hf_model_edit = QLineEdit()
        self._hf_model_edit.setPlaceholderText("e.g., kazars24/trocr-base-handwritten-ru")
        self._hf_model_edit.setText("kazars24/trocr-base-handwritten-ru")
        hf_layout.addWidget(QLabel("Model ID:"))
        hf_layout.addWidget(self._hf_model_edit)

        self._hf_group.setLayout(hf_layout)
        self._hf_group.setVisible(False)  # Hidden by default
        layout.addWidget(self._hf_group)

        # Inference settings
        settings_group = QGroupBox("Inference Settings")
        settings_layout = QVBoxLayout()

        # Beam search
        beam_layout = QHBoxLayout()
        beam_layout.addWidget(QLabel("Beam Search:"))
        self._beam_spin = QSpinBox()
        self._beam_spin.setRange(1, 10)
        self._beam_spin.setValue(4)
        self._beam_spin.setToolTip("Higher = better quality but slower (1 = greedy, 4 = recommended)")
        beam_layout.addWidget(self._beam_spin)
        beam_layout.addStretch()
        settings_layout.addLayout(beam_layout)

        # Normalize background
        self._normalize_check = QCheckBox("Normalize Background")
        self._normalize_check.setToolTip("Apply CLAHE normalization (use if model was trained with it)")
        settings_layout.addWidget(self._normalize_check)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        layout.addStretch()
        widget.setLayout(layout)

        self._config_widget = widget
        return widget

    def _populate_local_models(self):
        """Scan models directory for TrOCR checkpoints."""
        if self._local_model_combo is None:
            return

        self._local_model_combo.clear()

        models_dir = Path("models")
        if not models_dir.exists():
            self._local_model_combo.addItem("No models found")
            return

        # Find checkpoint directories
        checkpoints = []
        for model_dir in models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Check if it's a TrOCR model (has pytorch_model.bin or model.safetensors)
            if (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists():
                checkpoints.append(str(model_dir))

            # Also check for checkpoint subdirectories
            for subdir in model_dir.glob("checkpoint-*"):
                if subdir.is_dir():
                    if (subdir / "pytorch_model.bin").exists() or (subdir / "model.safetensors").exists():
                        checkpoints.append(str(subdir))

        if checkpoints:
            self._local_model_combo.addItems(sorted(checkpoints))
        else:
            self._local_model_combo.addItem("No local models found")

    def _browse_local_model(self):
        """Open file dialog to select model directory."""
        directory = QFileDialog.getExistingDirectory(
            self._config_widget,
            "Select Model Directory",
            "models"
        )

        if directory:
            # Add to combo box if not already present
            if self._local_model_combo.findText(directory) == -1:
                self._local_model_combo.addItem(directory)
            self._local_model_combo.setCurrentText(directory)

    def _on_model_source_changed(self, source: str):
        """Toggle between local and HuggingFace model selection."""
        is_local = (source == "Local Models")
        self._local_group.setVisible(is_local)
        self._hf_group.setVisible(not is_local)

    def get_config(self) -> Dict[str, Any]:
        """Extract configuration from widget controls."""
        if self._config_widget is None:
            return {}

        is_local = (self._model_source_combo.currentText() == "Local Models")

        return {
            "model_source": "local" if is_local else "huggingface",
            "model_path": self._local_model_combo.currentText() if is_local else self._hf_model_edit.text(),
            "beam_search": self._beam_spin.value(),
            "normalize_background": self._normalize_check.isChecked(),
        }

    def set_config(self, config: Dict[str, Any]):
        """Restore configuration to widget controls."""
        if self._config_widget is None:
            return

        model_source = config.get("model_source", "local")
        self._model_source_combo.setCurrentText("Local Models" if model_source == "local" else "HuggingFace Hub")

        model_path = config.get("model_path", "")
        if model_source == "local":
            idx = self._local_model_combo.findText(model_path)
            if idx >= 0:
                self._local_model_combo.setCurrentIndex(idx)
        else:
            self._hf_model_edit.setText(model_path)

        self._beam_spin.setValue(config.get("beam_search", 4))
        self._normalize_check.setChecked(config.get("normalize_background", False))

    def load_model(self, config: Dict[str, Any]) -> bool:
        """Load TrOCR model."""
        try:
            model_path = config.get("model_path", "")
            if not model_path or model_path == "No local models found" or model_path == "No models found":
                return False

            normalize = config.get("normalize_background", False)
            model_source = config.get("model_source", "local")
            is_hf = (model_source == "huggingface")

            self.model = TrOCRInference(
                model_path=model_path,
                normalize_bg=normalize,
                is_huggingface=is_hf
            )

            return True

        except Exception as e:
            print(f"Error loading TrOCR model: {e}")
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
        """Transcribe a line image with TrOCR."""
        if self.model is None:
            return TranscriptionResult(text="[Model not loaded]", confidence=0.0)

        # Use provided config or fall back to current config
        if config is None:
            config = self.get_config()

        beam_search = config.get("beam_search", 4)

        try:
            # TrOCRInference expects PIL Image, convert from numpy
            from PIL import Image
            pil_image = Image.fromarray(image)

            text = self.model.transcribe_line(pil_image, num_beams=beam_search)

            return TranscriptionResult(
                text=text,
                confidence=1.0,  # TrOCR doesn't provide confidence scores
                metadata={"beam_search": beam_search}
            )

        except Exception as e:
            print(f"Error in TrOCR transcription: {e}")
            return TranscriptionResult(text=f"[Error: {e}]", confidence=0.0)

    def get_capabilities(self) -> Dict[str, bool]:
        """TrOCR capabilities."""
        return {
            "batch_processing": False,  # Could be implemented in future
            "confidence_scores": False,  # TrOCR doesn't provide per-token confidence
            "beam_search": True,
            "language_model": False,  # Implicit in decoder, not explicit LM
            "preprocessing": True,  # Has built-in normalization
        }
