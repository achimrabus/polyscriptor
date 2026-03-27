"""
Kraken HTR Engine Plugin

Wraps the Kraken OCR system as a plugin for the unified GUI.
Kraken is specialized for historical document OCR with robust segmentation and recognition.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


def _print(msg: str) -> None:
    """Print with graceful fallback if console can't encode the message (e.g. Windows CP-1252)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))

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


# Local model (included in repo)
LOCAL_BLLA_MODEL = "pagexml/blla.mlmodel"

# Preset Kraken models — local + Zenodo community models (auto-download on first use)
KRAKEN_MODELS = {
    "blla-local": {
        "path": LOCAL_BLLA_MODEL,
        "description": "BLLA Segmentation Model (Local, Default)",
        "language": "multi",
        "source": "local"
    },
    # --- VERIFIED ZENODO MODELS ---
    # CATMuS-Print: printed text, multilingual, verified DOI 10.5281/zenodo.10592716
    "catmus-print": {
        "zenodo_id": "10.5281/zenodo.10592716",
        "description": "CATMuS-Print (Modern Printed Text, multilingual)",
        "language": "multi",
        "source": "zenodo"
    },
    # Arabic handwritten segmentation (Muharaf Corpus), verified DOI 10.5281/zenodo.14295555
    "arabic-muharaf": {
        "zenodo_id": "10.5281/zenodo.14295555",
        "description": "Arabic Handwritten Segmentation (Muharaf Corpus)",
        "language": "arabic",
        "source": "zenodo"
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
        return KRAKEN_AVAILABLE

    def get_unavailable_reason(self) -> str:
        if not KRAKEN_AVAILABLE:
            return "Kraken not installed. Install with: pip install kraken"
        return ""

    def get_config_widget(self) -> QWidget:
        """Create Kraken configuration panel."""
        if not PYQT_AVAILABLE:
            raise RuntimeError("PyQt6 not installed. Install with: pip install PyQt6")
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
        self._preset_combo.currentIndexChanged.connect(self._on_preset_model_changed)
        preset_layout.addWidget(QLabel("Model:"))
        preset_layout.addWidget(self._preset_combo)

        preset_hint = QLabel("Note: Zenodo models (⬇️) auto-download on first use")
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
        """Populate preset models dropdown with local and Zenodo models."""
        if self._preset_combo is None:
            return

        self._preset_combo.clear()

        if not KRAKEN_MODELS:
            self._preset_combo.addItem("No presets available")
            return

        # Local model first
        for model_id, info in KRAKEN_MODELS.items():
            if info.get("source") == "local":
                desc = info.get('description', model_id)
                self._preset_combo.addItem(f"📁 {desc}", userData=model_id)
                break

        self._preset_combo.insertSeparator(self._preset_combo.count())

        # Zenodo models
        for model_id, info in KRAKEN_MODELS.items():
            if info.get("source") == "zenodo":
                desc = info.get('description', model_id)
                lang = info.get('language', '')
                self._preset_combo.addItem(f"⬇️  {desc} ({lang})", userData=model_id)

        self._preset_combo.insertSeparator(self._preset_combo.count())
        self._preset_combo.addItem("📂 Browse Custom File...", userData="__custom__")

    def _on_model_source_changed(self, source: str):
        """Toggle between preset and custom model selection."""
        is_preset = (source == "Preset Models")
        self._preset_group.setVisible(is_preset)
        self._custom_group.setVisible(not is_preset)

    def _on_preset_model_changed(self, index: int):
        """Handle preset selection — open file browser for custom option."""
        model_id = self._preset_combo.currentData()
        if model_id == "__custom__":
            file_path, _ = QFileDialog.getOpenFileName(
                self._config_widget,
                "Select Kraken Model File",
                "",
                "Kraken Models (*.mlmodel);;All Files (*)"
            )
            if file_path:
                self._model_source_combo.setCurrentText("Custom Model File")
                self._custom_model_edit.setText(file_path)
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentIndex(0)
            self._preset_combo.blockSignals(False)

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
        """Load Kraken model (local or Zenodo auto-download)."""
        try:
            model_path = config.get("model_path")
            preset_id = config.get("preset_id")

            # Resolve Zenodo preset: download if needed
            if preset_id and preset_id in KRAKEN_MODELS:
                model_info = KRAKEN_MODELS[preset_id]
                if model_info.get("source") == "zenodo":
                    zenodo_id = model_info.get("zenodo_id")
                    model_path = self._download_zenodo_model(zenodo_id, preset_id)
                    if not model_path:
                        print(f"Error: Failed to download Zenodo model '{preset_id}'")
                        return False
                elif model_info.get("source") == "local":
                    model_path = model_info.get("path")

            # Fall back to default local blla model
            if not model_path:
                model_path = LOCAL_BLLA_MODEL
                print(f"No model specified, using default: {model_path}")

            if not Path(model_path).exists():
                print(f"Error: Model file not found: {model_path}")
                print("For Zenodo models, run: kraken get <zenodo_id>")
                return False

            vgsl_model = vgsl.TorchVGSLModel.load_model(model_path)
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

    def _download_zenodo_model(self, zenodo_id: str, model_name: str) -> Optional[str]:
        """Download a Kraken model from Zenodo via `kraken get`.

        Models are cached in `kraken_models/` inside the repo root.
        Returns local path on success, None on failure.
        """
        import subprocess
        import shutil
        import sys
        import time

        # Prefer the kraken binary from the same venv as this Python process
        # (shutil.which only searches PATH, which may not include the venv bin/ in
        # systemd services that invoke uvicorn directly without activating the venv).
        venv_kraken = Path(sys.executable).parent / "kraken"
        kraken_cmd = str(venv_kraken) if venv_kraken.exists() else shutil.which("kraken")
        if not kraken_cmd:
            _print("❌ 'kraken' command not found. Install with: pip install kraken")
            _print(f"💡 Manual download: https://zenodo.org/record/{zenodo_id.split('/')[-1]}")
            return None

        repo_root = Path(__file__).parent.parent
        models_dir = repo_root / "kraken_models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"{model_name}.mlmodel"

        if model_path.exists():
            _print(f"✅ Using cached Zenodo model: {model_path}")
            return str(model_path)

        # Check for any existing name-matched file
        for existing in models_dir.glob("*.mlmodel"):
            if model_name.lower() in existing.stem.lower():
                _print(f"✅ Found existing model: {existing}")
                return str(existing)

        _print(f"📥 Downloading Zenodo model {zenodo_id} …")
        _print(f"📂 Will save to: {model_path}")
        _print("⏳ This may take a few minutes on first use …")

        try:
            result = subprocess.run(
                [kraken_cmd, "get", zenodo_id],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                # Find freshly downloaded .mlmodel (modified within last 2 min)
                search_dirs = [
                    Path.home() / "Library" / "Application Support" / "htrmopo",
                    Path.home() / ".kraken",
                ]
                downloaded = None
                for d in search_dirs:
                    if not d.exists():
                        continue
                    for p in d.rglob("*.mlmodel"):
                        if time.time() - p.stat().st_mtime < 120:
                            downloaded = p
                            break
                    if downloaded:
                        break
                if downloaded and downloaded.exists():
                    shutil.copy2(downloaded, model_path)
                    _print(f"✅ Model saved to: {model_path}")
                    return str(model_path)
                else:
                    _print("⚠️  Download succeeded but couldn't locate the file")
            else:
                _print(f"❌ kraken get failed (exit {result.returncode}): {result.stderr}")
                _print(f"💡 Manual: kraken get {zenodo_id}  then copy to {models_dir}/")
        except subprocess.TimeoutExpired:
            _print("⏱️  Download timeout (>5 min). Try manually: kraken get " + zenodo_id)
        except Exception as e:
            _print(f"❌ Download error: {e}")

        return None

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


def download_preset_model(preset_name: str) -> Optional[str]:
    """Module-level helper: resolve and (if needed) download a Kraken preset model.

    Used by batch_processing.py and the web server without instantiating KrakenEngine.
    Returns local file path, or None on failure.
    """
    if preset_name not in KRAKEN_MODELS:
        print(f"Unknown Kraken preset: '{preset_name}'. Available: {list(KRAKEN_MODELS)}")
        return None
    info = KRAKEN_MODELS[preset_name]
    if info.get("source") == "local":
        return info.get("path")
    if info.get("source") == "zenodo":
        engine = KrakenEngine.__new__(KrakenEngine)
        return engine._download_zenodo_model(info["zenodo_id"], preset_name)
    return None
