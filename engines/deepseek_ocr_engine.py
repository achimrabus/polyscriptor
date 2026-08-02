"""
DeepSeek-OCR-2 Engine Plugin (subprocess mode)

Wraps DeepSeek-OCR-2 as a page-level OCR engine for Polyscriptor.
DeepSeek-OCR-2 requires transformers==4.46.3, which conflicts with LightOnOCR's
transformers 5.x dependency in the main venv.  The engine therefore runs in an
isolated venv (venv_deepseek) via a subprocess worker — the main venv never
imports DeepSeek or its conflicting transformers version.

Setup (venv_deepseek):
    python -m venv /path/to/dhlab-slavistik/venv_deepseek
    source venv_deepseek/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install "transformers==4.46.3" einops addict easydict pillow
    # Optional (faster inference, requires CUDA toolkit headers):
    pip install flash-attn --no-build-isolation
    deactivate

Key features:
- Two OCR modes: Document OCR (with layout, markdown) and Free OCR (plain text)
- Visual Causal Flow architecture with Flash Attention 2
- ~3B parameters, ~6 GB VRAM
- subprocess isolation: no transformers version conflicts with main venv
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from htr_engine_base import HTREngine, TranscriptionResult

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWidgets import (
        QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox,
        QHBoxLayout, QLabel, QLineEdit, QPushButton,
        QSlider, QSpinBox, QVBoxLayout, QWidget,
    )
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

# Default venv path (sibling of the project root, same convention as venv_paddle)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_VENV = _PROJECT_ROOT / "venv_deepseek"

# Worker script lives in the same engines/ directory as this file
_WORKER_SCRIPT = Path(__file__).resolve().parent / "deepseek_worker.py"


# ---------------------------------------------------------------------------
# Venv helpers (filesystem-only, no subprocess, no imports)
# ---------------------------------------------------------------------------

def _find_venv_python(venv_path: Path) -> Optional[Path]:
    """Return the Python interpreter inside a venv, or None if not found."""
    import sys as _sys
    if _sys.platform == "win32":
        candidates = [venv_path / "Scripts" / "python.exe"]
    else:
        candidates = [venv_path / "bin" / "python", venv_path / "bin" / "python3"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _venv_has_transformers(venv_path: Path) -> bool:
    """Check that transformers is installed in the venv (filesystem, no subprocess)."""
    import sys as _sys
    if _sys.platform == "win32":
        candidates = [(venv_path / "Lib" / "site-packages" / "transformers",)]
    else:
        candidates = list((venv_path / "lib").glob("python*/site-packages/transformers"))
    for sp_dir in candidates:
        if sp_dir.is_dir():
            return True
    return False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DeepSeekOCREngine(HTREngine):
    """
    DeepSeek-OCR-2 page-level OCR engine (subprocess mode).

    Calls deepseek_worker.py via the venv_deepseek Python interpreter so
    DeepSeek's transformers==4.46.3 dependency lives in an isolated venv and
    never conflicts with the main venv's transformers 5.x (used by LightOnOCR).
    """

    def __init__(self):
        self._venv_path: Path = _DEFAULT_VENV
        self._venv_python: Optional[Path] = None
        self._is_loaded: bool = False

        # Config widget references
        self._config_widget: Optional[QWidget] = None
        self._venv_edit: Optional[QLineEdit] = None
        self._ocr_mode_combo: Optional[QComboBox] = None
        self._strip_markdown_checkbox: Optional[QCheckBox] = None
        self._base_size_slider: Optional[QSlider] = None
        self._base_size_label: Optional[QLabel] = None
        self._image_size_slider: Optional[QSlider] = None
        self._image_size_label: Optional[QLabel] = None
        self._crop_mode_checkbox: Optional[QCheckBox] = None
        self._device_combo: Optional[QComboBox] = None
        self._repetition_penalty_spin: Optional[QDoubleSpinBox] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "DeepSeek-OCR"

    def get_description(self) -> str:
        return "DeepSeek-OCR-2: Page-level VLM for document OCR with layout support"

    def get_aliases(self) -> List[str]:
        return ["deepseek", "deepseek-ocr"]

    # ------------------------------------------------------------------
    # Availability (filesystem checks only — no subprocess, no imports)
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if _find_venv_python(self._venv_path) is None:
            return False
        return _venv_has_transformers(self._venv_path)

    def get_unavailable_reason(self) -> str:
        if _find_venv_python(self._venv_path) is None:
            return (
                f"venv_deepseek not found at: {self._venv_path}\n\n"
                "Create it with:\n"
                f"  python -m venv {self._venv_path}\n"
                f"  source {self._venv_path}/bin/activate\n"
                '  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n'
                '  pip install "transformers==4.46.3" einops addict easydict pillow\n'
                "  # Optional: pip install flash-attn --no-build-isolation\n"
                "  deactivate"
            )
        if not _venv_has_transformers(self._venv_path):
            return (
                f"transformers not installed in {self._venv_path}\n\n"
                "Install with:\n"
                f"  source {self._venv_path}/bin/activate\n"
                '  pip install "transformers==4.46.3" einops addict easydict pillow\n'
                "  deactivate"
            )
        return ""

    # ------------------------------------------------------------------
    # Configuration widget
    # ------------------------------------------------------------------

    def get_config_widget(self) -> QWidget:
        """Create DeepSeek-OCR configuration panel."""
        if self._config_widget is not None:
            return self._config_widget

        widget = QWidget()
        layout = QVBoxLayout()

        # Venv path
        venv_group = QGroupBox("DeepSeek venv")
        venv_layout = QVBoxLayout()
        venv_layout.addWidget(QLabel("Path to venv_deepseek:"))
        venv_row = QHBoxLayout()
        self._venv_edit = QLineEdit(str(self._venv_path))
        self._venv_edit.setToolTip(
            "Path to the isolated Python venv with DeepSeek-OCR dependencies.\n"
            "Default: <project>/venv_deepseek\n"
            'Create with: python -m venv venv_deepseek\n'
            '  source venv_deepseek/bin/activate\n'
            '  pip install "transformers==4.46.3" einops addict easydict pillow'
        )
        venv_row.addWidget(self._venv_edit)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_venv)
        venv_row.addWidget(btn_browse)
        venv_layout.addLayout(venv_row)
        venv_group.setLayout(venv_layout)
        layout.addWidget(venv_group)

        # OCR Mode selection
        mode_group = QGroupBox("OCR Mode")
        mode_layout = QVBoxLayout()

        self._ocr_mode_combo = QComboBox()
        self._ocr_mode_combo.addItems(["Document OCR", "Free OCR"])
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self._ocr_mode_combo)

        mode_hint = QLabel(
            "Document OCR: Preserves layout with markdown formatting\n"
            "Free OCR: Plain text without layout structure"
        )
        mode_hint.setStyleSheet("color: gray; font-size: 9pt;")
        mode_hint.setWordWrap(True)
        mode_layout.addWidget(mode_hint)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()

        self._strip_markdown_checkbox = QCheckBox("Strip Markdown formatting")
        self._strip_markdown_checkbox.setChecked(False)
        strip_hint = QLabel("Remove headers, bold, lists from output (plain text only)")
        strip_hint.setStyleSheet("color: gray; font-size: 9pt;")
        output_layout.addWidget(self._strip_markdown_checkbox)
        output_layout.addWidget(strip_hint)

        # Repetition penalty
        rep_row = QHBoxLayout()
        rep_row.addWidget(QLabel("Repetition penalty:"))
        self._repetition_penalty_spin = QDoubleSpinBox()
        self._repetition_penalty_spin.setRange(1.0, 2.0)
        self._repetition_penalty_spin.setSingleStep(0.05)
        self._repetition_penalty_spin.setDecimals(2)
        self._repetition_penalty_spin.setValue(1.3)
        self._repetition_penalty_spin.setToolTip(
            "Penalise repeated tokens during generation.\n"
            "1.0 = disabled (risk of loops), 1.3 = recommended,\n"
            "1.5+ = strong suppression (may hurt accuracy)."
        )
        rep_row.addWidget(self._repetition_penalty_spin)
        rep_row.addStretch()
        output_layout.addLayout(rep_row)
        rep_hint = QLabel("1.3 recommended — reduces hallucination/repetition loops")
        rep_hint.setStyleSheet("color: gray; font-size: 9pt;")
        output_layout.addWidget(rep_hint)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Image processing settings
        image_group = QGroupBox("Image Processing")
        image_layout = QVBoxLayout()

        # Base Size
        base_layout = QVBoxLayout()
        base_layout.addWidget(QLabel("Base Size (patch resolution):"))

        base_slider_layout = QHBoxLayout()
        self._base_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._base_size_slider.setRange(512, 2048)
        self._base_size_slider.setValue(1024)
        self._base_size_slider.setTickInterval(256)
        self._base_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._base_size_slider.valueChanged.connect(self._on_base_size_changed)
        base_slider_layout.addWidget(self._base_size_slider)

        self._base_size_label = QLabel("1024px")
        self._base_size_label.setMinimumWidth(60)
        base_slider_layout.addWidget(self._base_size_label)

        base_layout.addLayout(base_slider_layout)
        image_layout.addLayout(base_layout)

        # Image Size
        img_layout = QVBoxLayout()
        img_layout.addWidget(QLabel("Image Size (output resolution):"))

        img_slider_layout = QHBoxLayout()
        self._image_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._image_size_slider.setRange(512, 1024)
        self._image_size_slider.setValue(640)
        self._image_size_slider.setTickInterval(128)
        self._image_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._image_size_slider.valueChanged.connect(self._on_image_size_changed)
        img_slider_layout.addWidget(self._image_size_slider)

        self._image_size_label = QLabel("640px")
        self._image_size_label.setMinimumWidth(60)
        img_slider_layout.addWidget(self._image_size_label)

        img_layout.addLayout(img_slider_layout)
        image_layout.addLayout(img_layout)

        # Crop Mode
        self._crop_mode_checkbox = QCheckBox("Enable Crop Mode")
        self._crop_mode_checkbox.setChecked(True)
        crop_hint = QLabel("Crop to content region (recommended for documents)")
        crop_hint.setStyleSheet("color: gray; font-size: 9pt;")
        image_layout.addWidget(self._crop_mode_checkbox)
        image_layout.addWidget(crop_hint)

        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        # Device selection
        device_group = QGroupBox("Device")
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["cuda:0", "cuda:1", "cpu"])
        self._device_combo.setToolTip(
            "GPU device for inference (runs inside venv_deepseek subprocess).\n"
            "cuda:0 is recommended; use cpu only as fallback (very slow)."
        )
        device_layout.addWidget(self._device_combo)
        device_layout.addStretch()
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Info
        info = QLabel(
            "DeepSeek-OCR-2 runs in an isolated venv (venv_deepseek) to avoid\n"
            "transformers version conflicts with LightOnOCR.\n"
            "First run downloads model weights (~6 GB). Each call spawns a\n"
            "subprocess — expect ~30–120 s per page depending on GPU."
        )
        info.setStyleSheet("color: gray; font-size: 9pt; padding: 8px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()
        widget.setLayout(layout)
        self._config_widget = widget
        return widget

    def _browse_venv(self):
        folder = QFileDialog.getExistingDirectory(
            self._config_widget, "Select venv_deepseek directory", str(self._venv_path)
        )
        if folder:
            self._venv_edit.setText(folder)

    def _on_base_size_changed(self, value: int):
        if self._base_size_label:
            self._base_size_label.setText(f"{value}px")

    def _on_image_size_changed(self, value: int):
        if self._image_size_label:
            self._image_size_label.setText(f"{value}px")

    # ------------------------------------------------------------------
    # Config get / set
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        if self._config_widget is None:
            return {
                "venv_path": str(self._venv_path),
                "model_id": "deepseek-ai/DeepSeek-OCR-2",
                "ocr_mode": "document",
                "strip_markdown": False,
                "base_size": 1024,
                "image_size": 640,
                "crop_mode": True,
                "device": "cuda:0",
                "repetition_penalty": 1.3,
            }

        ocr_mode_text = self._ocr_mode_combo.currentText()
        ocr_mode = "document" if ocr_mode_text == "Document OCR" else "free"

        return {
            "venv_path": self._venv_edit.text().strip() or str(self._venv_path),
            "model_id": "deepseek-ai/DeepSeek-OCR-2",
            "ocr_mode": ocr_mode,
            "strip_markdown": self._strip_markdown_checkbox.isChecked(),
            "base_size": self._base_size_slider.value(),
            "image_size": self._image_size_slider.value(),
            "crop_mode": self._crop_mode_checkbox.isChecked(),
            "device": self._device_combo.currentText(),
            "repetition_penalty": self._repetition_penalty_spin.value(),
        }

    def set_config(self, config: Dict[str, Any]):
        if self._config_widget is None:
            return

        if "venv_path" in config:
            self._venv_edit.setText(config["venv_path"])

        ocr_mode = config.get("ocr_mode", "document")
        self._ocr_mode_combo.setCurrentText(
            "Document OCR" if ocr_mode == "document" else "Free OCR"
        )

        self._strip_markdown_checkbox.setChecked(config.get("strip_markdown", False))
        self._base_size_slider.setValue(config.get("base_size", 1024))
        self._image_size_slider.setValue(config.get("image_size", 640))
        self._crop_mode_checkbox.setChecked(config.get("crop_mode", True))
        self._repetition_penalty_spin.setValue(config.get("repetition_penalty", 1.3))

        device = config.get("device", "cuda:0")
        idx = self._device_combo.findText(device)
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Model loading (validates venv + worker; actual model loads lazily in subprocess)
    # ------------------------------------------------------------------

    def load_model(self, config: Dict[str, Any]) -> bool:
        """Validate venv + worker script exist. Actual model loading is lazy (in subprocess)."""
        venv_path = Path(config.get("venv_path", str(self._venv_path)))
        self._venv_path = venv_path

        # Sync venv_edit if widget is open
        if self._venv_edit is not None:
            self._venv_edit.setText(str(venv_path))

        python = _find_venv_python(venv_path)
        if python is None:
            logger.error("[DeepSeek-OCR] venv Python not found at %s", venv_path)
            return False

        if not _WORKER_SCRIPT.exists():
            logger.error("[DeepSeek-OCR] Worker script not found: %s", _WORKER_SCRIPT)
            return False

        self._venv_python = python
        self._is_loaded = True
        logger.info("[DeepSeek-OCR] Ready — venv: %s, worker: %s", python, _WORKER_SCRIPT)
        return True

    def unload_model(self):
        self._venv_python = None
        self._is_loaded = False

    def is_model_loaded(self) -> bool:
        return self._is_loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def requires_line_segmentation(self) -> bool:
        return False  # Page-level model — no pre-segmentation needed

    def transcribe_line(
        self, image: np.ndarray, config: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """
        Transcribe a full page image via DeepSeek-OCR-2 subprocess.

        Despite the method name, page-based engines receive the full page here.
        """
        if not self._is_loaded or self._venv_python is None:
            return TranscriptionResult(text="[DeepSeek-OCR not loaded]", confidence=0.0)

        if config is None:
            config = self.get_config()

        # Write image to a temp file so the subprocess can read it
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            pil_img.convert("RGB").save(tmp_path)

            worker_config = json.dumps({
                "model_id": config.get("model_id", "deepseek-ai/DeepSeek-OCR-2"),
                "device": config.get("device", "cuda:0"),
                "ocr_mode": config.get("ocr_mode", "document"),
                "strip_markdown": config.get("strip_markdown", False),
                "base_size": config.get("base_size", 1024),
                "image_size": config.get("image_size", 640),
                "crop_mode": config.get("crop_mode", True),
                "repetition_penalty": config.get("repetition_penalty", 1.3),
            })

            result = subprocess.run(
                [str(self._venv_python), str(_WORKER_SCRIPT), worker_config, tmp_path],
                capture_output=True, text=True, timeout=300,
                start_new_session=True,  # isolate from terminal SIGINT
            )

            def _parse_worker_output(raw: str) -> dict:
                """Worker stdout may contain model debug lines before the JSON result.
                The JSON is always the last non-empty line."""
                last_line = next(
                    (l for l in reversed(raw.splitlines()) if l.strip()), ""
                )
                return json.loads(last_line)

            if result.returncode != 0:
                try:
                    output = _parse_worker_output(result.stdout)
                    if "error" in output:
                        err_msg = output["error"]
                        tb = output.get("traceback", "")
                        logger.error("[DeepSeek-OCR] Worker error: %s", err_msg)
                        if tb:
                            logger.debug("[DeepSeek-OCR] Traceback:\n%s", tb)
                        return TranscriptionResult(text=f"[Error: {err_msg}]", confidence=0.0)
                except (json.JSONDecodeError, ValueError):
                    pass
                stderr = result.stderr[-2000:] if result.stderr else "(no stderr)"
                logger.error("[DeepSeek-OCR] Worker exited %d: %s", result.returncode, stderr)
                return TranscriptionResult(text="[DeepSeek-OCR error — see log]", confidence=0.0)

            output = _parse_worker_output(result.stdout)

            if "error" in output:
                logger.error("[DeepSeek-OCR] Worker error: %s", output["error"])
                return TranscriptionResult(
                    text=f"[Error: {output['error']}]", confidence=0.0
                )

            text = output.get("text", "")
            return TranscriptionResult(
                text=text,
                confidence=1.0,  # DeepSeek-OCR-2 does not return per-token confidence
                metadata={
                    "engine": "DeepSeek-OCR",
                    "model": output.get("model_id", "deepseek-ai/DeepSeek-OCR-2"),
                    "ocr_mode": output.get("ocr_mode", config.get("ocr_mode", "document")),
                },
            )

        except subprocess.TimeoutExpired:
            logger.error("[DeepSeek-OCR] Subprocess timed out after 300s")
            return TranscriptionResult(text="[DeepSeek-OCR timed out]", confidence=0.0)
        except json.JSONDecodeError as e:
            logger.error("[DeepSeek-OCR] Failed to parse worker output: %s", e)
            return TranscriptionResult(text="[DeepSeek-OCR output parse error]", confidence=0.0)
        except Exception as e:
            logger.error("[DeepSeek-OCR] Unexpected error: %s", e)
            return TranscriptionResult(text=f"[Error: {e}]", confidence=0.0)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def transcribe_lines(
        self, images: List[np.ndarray], config: Optional[Dict[str, Any]] = None
    ) -> List[TranscriptionResult]:
        """Transcribe multiple images (each as a full page)."""
        return [self.transcribe_line(img, config) for img in images]

    def supports_batch(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def get_capabilities(self) -> Dict[str, bool]:
        return {
            "batch_processing": False,
            "confidence_scores": False,
            "beam_search": False,
            "language_model": True,
            "preprocessing": True,
            "layout_detection": True,
        }
