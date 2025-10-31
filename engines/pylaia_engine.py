"""
PyLaia Engine Plugin

Wraps the PyLaia CTC-based HTR inference system as a plugin.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from htr_engine_base import HTREngine, TranscriptionResult

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QPushButton, QCheckBox, QLineEdit, QFileDialog,
        QGroupBox, QDoubleSpinBox
    )
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

try:
    # Use native Linux implementation (no WSL dependency)
    from inference_pylaia_native import PyLaiaInference, PYLAIA_MODELS
    PYLAIA_AVAILABLE = True
    PYLAIA_LM_AVAILABLE = False  # Language model not yet implemented
except ImportError:
    PYLAIA_AVAILABLE = False
    PYLAIA_MODELS = {}
    PYLAIA_LM_AVAILABLE = False


class PyLaiaEngine(HTREngine):
    """PyLaia CTC-based HTR engine plugin."""

    def __init__(self):
        self.model: Optional[PyLaiaInference] = None
        self.model_lm: Optional[PyLaiaInferenceLM] = None
        self._config_widget: Optional[QWidget] = None

        # Widget references
        self._model_combo: Optional[QComboBox] = None
        self._use_lm_check: Optional[QCheckBox] = None
        self._lm_weight_spin: Optional[QDoubleSpinBox] = None
        self._custom_model_edit: Optional[QLineEdit] = None
        self._custom_lm_edit: Optional[QLineEdit] = None

    def get_name(self) -> str:
        return "PyLaia"

    def get_description(self) -> str:
        return "CTC-based HTR with optional KenLM language model"

    def is_available(self) -> bool:
        return PYLAIA_AVAILABLE and PYQT_AVAILABLE

    def get_unavailable_reason(self) -> str:
        if not PYLAIA_AVAILABLE:
            return "PyLaia not available. Check that inference_pylaia.py exists and dependencies are installed."
        if not PYQT_AVAILABLE:
            return "PyQt6 not installed. Install with: pip install PyQt6"
        return ""

    def get_config_widget(self) -> QWidget:
        """Create PyLaia configuration panel."""
        if self._config_widget is not None:
            return self._config_widget

        widget = QWidget()
        layout = QVBoxLayout()

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()

        # Preset models
        model_layout.addWidget(QLabel("Preset Model:"))
        self._model_combo = QComboBox()
        self._populate_preset_models()
        self._model_combo.currentTextChanged.connect(self._on_preset_changed)
        model_layout.addWidget(self._model_combo)

        # Custom model path
        model_layout.addWidget(QLabel("Custom Model Path:"))
        custom_layout = QHBoxLayout()
        self._custom_model_edit = QLineEdit()
        self._custom_model_edit.setPlaceholderText("Leave empty to use preset model")
        custom_layout.addWidget(self._custom_model_edit)
        browse_model_btn = QPushButton("Browse...")
        browse_model_btn.clicked.connect(self._browse_model)
        custom_layout.addWidget(browse_model_btn)
        model_layout.addLayout(custom_layout)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Language model settings
        lm_group = QGroupBox("Language Model (Optional)")
        lm_layout = QVBoxLayout()

        self._use_lm_check = QCheckBox("Use Language Model")
        self._use_lm_check.setChecked(False)
        self._use_lm_check.toggled.connect(self._on_lm_toggled)
        if not PYLAIA_LM_AVAILABLE:
            self._use_lm_check.setEnabled(False)
            self._use_lm_check.setToolTip("KenLM not available. Install with: pip install kenlm")
        lm_layout.addWidget(self._use_lm_check)

        # LM weight
        weight_layout = QHBoxLayout()
        weight_layout.addWidget(QLabel("LM Weight:"))
        self._lm_weight_spin = QDoubleSpinBox()
        self._lm_weight_spin.setRange(0.0, 10.0)
        self._lm_weight_spin.setValue(1.5)
        self._lm_weight_spin.setSingleStep(0.1)
        self._lm_weight_spin.setToolTip("Higher = more influence from language model")
        self._lm_weight_spin.setEnabled(False)
        weight_layout.addWidget(self._lm_weight_spin)
        weight_layout.addStretch()
        lm_layout.addLayout(weight_layout)

        # Custom LM path
        lm_layout.addWidget(QLabel("Custom LM Path:"))
        custom_lm_layout = QHBoxLayout()
        self._custom_lm_edit = QLineEdit()
        self._custom_lm_edit.setPlaceholderText("Leave empty for auto-detection")
        self._custom_lm_edit.setEnabled(False)
        custom_lm_layout.addWidget(self._custom_lm_edit)
        browse_lm_btn = QPushButton("Browse...")
        browse_lm_btn.clicked.connect(self._browse_lm)
        browse_lm_btn.setEnabled(False)
        self._browse_lm_btn = browse_lm_btn
        custom_lm_layout.addWidget(browse_lm_btn)
        lm_layout.addLayout(custom_lm_layout)

        lm_group.setLayout(lm_layout)
        layout.addWidget(lm_group)

        layout.addStretch()
        widget.setLayout(layout)

        self._config_widget = widget
        return widget

    def _populate_preset_models(self):
        """Populate preset models dropdown."""
        if self._model_combo is None:
            return

        self._model_combo.clear()

        if not PYLAIA_MODELS:
            self._model_combo.addItem("No preset models found")
            return

        for model_id in PYLAIA_MODELS.keys():
            self._model_combo.addItem(model_id)

    def _on_preset_changed(self, preset_name: str):
        """Update when preset changes."""
        # Could add description display here
        pass

    def _on_lm_toggled(self, checked: bool):
        """Enable/disable LM controls."""
        self._lm_weight_spin.setEnabled(checked)
        self._custom_lm_edit.setEnabled(checked)
        self._browse_lm_btn.setEnabled(checked)

    def _browse_model(self):
        """Open file dialog to select model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self._config_widget,
            "Select PyLaia Model",
            "models",
            "PyLaia Models (*.ckpt *.pth *.pt);;All Files (*)"
        )

        if file_path:
            self._custom_model_edit.setText(file_path)

    def _browse_lm(self):
        """Open file dialog to select LM file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self._config_widget,
            "Select KenLM Model",
            "models",
            "KenLM Models (*.arpa *.klm *.bin);;All Files (*)"
        )

        if file_path:
            self._custom_lm_edit.setText(file_path)

    def get_config(self) -> Dict[str, Any]:
        """Extract configuration from widget controls."""
        if self._config_widget is None:
            return {}

        custom_model = self._custom_model_edit.text().strip()
        preset_model = self._model_combo.currentText()

        config = {
            "model_path": custom_model if custom_model else preset_model,
            "use_lm": self._use_lm_check.isChecked(),
            "lm_weight": self._lm_weight_spin.value(),
        }

        if config["use_lm"]:
            custom_lm = self._custom_lm_edit.text().strip()
            if custom_lm:
                config["lm_path"] = custom_lm

        return config

    def set_config(self, config: Dict[str, Any]):
        """Restore configuration to widget controls."""
        if self._config_widget is None:
            return

        model_path = config.get("model_path", "")

        # Try to find in presets
        idx = self._model_combo.findText(model_path)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
            self._custom_model_edit.clear()
        else:
            self._custom_model_edit.setText(model_path)

        self._use_lm_check.setChecked(config.get("use_lm", False))
        self._lm_weight_spin.setValue(config.get("lm_weight", 1.5))

        if "lm_path" in config:
            self._custom_lm_edit.setText(config["lm_path"])

    def load_model(self, config: Dict[str, Any]) -> bool:
        """Load PyLaia model."""
        try:
            model_path = config.get("model_path", "")
            if not model_path or model_path == "No preset models found":
                return False

            # If it's a preset name, resolve to actual path and syms
            syms_path = None
            if model_path in PYLAIA_MODELS:
                preset_info = PYLAIA_MODELS[model_path]
                if isinstance(preset_info, dict):
                    model_path = preset_info.get("checkpoint", preset_info.get("path", model_path))
                    syms_path = preset_info.get("syms")
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
                # PyLaiaInference expects checkpoint_path and syms_path
                self.model = PyLaiaInference(
                    checkpoint_path=model_path,
                    syms_path=syms_path
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
            return TranscriptionResult(text=f"[Error: {e}]", confidence=0.0)

    def get_capabilities(self) -> Dict[str, bool]:
        """PyLaia capabilities."""
        return {
            "batch_processing": False,  # Could be implemented
            "confidence_scores": True,  # CTC provides confidence
            "beam_search": False,  # CTC uses greedy/beam decoding
            "language_model": PYLAIA_LM_AVAILABLE,  # Optional KenLM
            "preprocessing": False,  # External preprocessing recommended
        }
