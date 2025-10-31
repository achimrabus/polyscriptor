"""
Party HTR Engine Plugin (Linux Native)

Party OCR integration for the plugin-based GUI system.
Based on working code from transcription_gui_party.py (proof-of-concept).

This engine uses Party's PAGE XML workflow:
1. Convert line images to PAGE XML format
2. Call Party OCR via subprocess
3. Parse output PAGE XML to extract transcriptions

No WSL needed - runs natively on Linux.
"""

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PIL import Image

from htr_engine_base import HTREngine, TranscriptionResult

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QPushButton, QLineEdit, QGroupBox, QFileDialog
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


class PartyEngine(HTREngine):
    """
    Party HTR Engine - Linux native implementation.

    Party processes entire pages using PAGE XML format, not individual lines.
    This engine buffers line images and processes them as a batch.
    """

    def __init__(self):
        """Initialize Party engine."""
        self.model_path: Optional[str] = None
        self.device: str = "cuda:0"
        self.project_root = Path.home() / "htr_gui" / "dhlab-slavistik"

        # Default model path
        self.default_model_path = str(self.project_root / "models/party_models/party_european_langs.safetensors")

        # Config widget references
        self._config_widget: Optional[QWidget] = None
        self._model_combo: Optional[QComboBox] = None
        self._model_path_label: Optional[QLabel] = None
        self._device_combo: Optional[QComboBox] = None

        # Batch processing state
        self._is_loaded = False

    def get_name(self) -> str:
        return "Party OCR"

    def get_description(self) -> str:
        return "PyTorch Lightning HTR with PAGE XML workflow"

    def is_available(self) -> bool:
        """Check if Party is installed and model exists."""
        try:
            # Check if party command is available
            result = subprocess.run(
                ["bash", "-c", "which party"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return False

            # Check if default model exists
            if not Path(self.default_model_path).exists():
                return False

            return True

        except Exception:
            return False

    def get_unavailable_reason(self) -> str:
        """Get reason why Party is unavailable."""
        try:
            result = subprocess.run(
                ["bash", "-c", "which party"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return "Party not installed. Install with: pip install party-ocr"

            if not Path(self.default_model_path).exists():
                return f"Party model not found at: {self.default_model_path}"

            return "Unknown error"

        except Exception as e:
            return f"Error checking Party availability: {e}"

    def get_config_widget(self) -> QWidget:
        """Create Party configuration widget."""
        if self._config_widget is not None:
            return self._config_widget

        widget = QWidget()
        layout = QVBoxLayout()

        # Model selection group
        model_group = QGroupBox("Party Model")
        model_layout = QVBoxLayout()

        # Model selector
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Model:"))

        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "European Languages (default)",
            "Custom model..."
        ])
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        model_select_layout.addWidget(self._model_combo)
        model_layout.addLayout(model_select_layout)

        # Model path display
        self._model_path_label = QLabel(self.default_model_path)
        self._model_path_label.setStyleSheet("color: gray; font-size: 9pt; padding: 5px;")
        self._model_path_label.setWordWrap(True)
        model_layout.addWidget(self._model_path_label)

        # Browse button
        btn_browse = QPushButton("Browse for model...")
        btn_browse.clicked.connect(self._browse_model)
        model_layout.addWidget(btn_browse)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Device selection
        device_group = QGroupBox("Device")
        device_layout = QVBoxLayout()

        device_select_layout = QHBoxLayout()
        device_select_layout.addWidget(QLabel("Device:"))

        self._device_combo = QComboBox()
        self._device_combo.addItems(["cuda:0", "cuda:1", "cpu"])
        device_select_layout.addWidget(self._device_combo)
        device_layout.addLayout(device_select_layout)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Info label
        info_label = QLabel(
            "Party processes entire pages using PAGE XML format.\n"
            "Batch processing is recommended for best performance."
        )
        info_label.setStyleSheet("color: gray; font-size: 9pt; padding: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addStretch()
        widget.setLayout(layout)

        self._config_widget = widget
        return widget

    def _on_model_changed(self, model_name: str):
        """Handle model selection change."""
        if model_name == "European Languages (default)":
            self._model_path_label.setText(self.default_model_path)
        # Custom model is handled by browse button

    def _browse_model(self):
        """Browse for Party model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self._config_widget,
            "Select Party Model",
            str(self.project_root / "models/party_models"),
            "Model Files (*.safetensors *.pt *.pth);;All Files (*)"
        )

        if file_path:
            self._model_path_label.setText(file_path)
            self._model_combo.setCurrentText("Custom model...")

    def get_config(self) -> Dict[str, Any]:
        """Extract configuration from widget."""
        if self._config_widget is None:
            return {
                "model_path": self.default_model_path,
                "device": "cuda:0"
            }

        # Get model path from label (handles both default and custom)
        model_path = self._model_path_label.text()
        device = self._device_combo.currentText()

        return {
            "model_path": model_path,
            "device": device
        }

    def set_config(self, config: Dict[str, Any]):
        """Restore configuration to widget."""
        if self._config_widget is None:
            return

        model_path = config.get("model_path", self.default_model_path)
        device = config.get("device", "cuda:0")

        # Update model path label
        self._model_path_label.setText(model_path)

        # Update model combo
        if model_path == self.default_model_path:
            self._model_combo.setCurrentText("European Languages (default)")
        else:
            self._model_combo.setCurrentText("Custom model...")

        # Update device combo
        idx = self._device_combo.findText(device)
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)

    def load_model(self, config: Dict[str, Any]) -> bool:
        """Load Party model (validate paths)."""
        try:
            model_path = config.get("model_path", self.default_model_path)
            device = config.get("device", "cuda:0")

            # Validate model file exists
            if not Path(model_path).exists():
                print(f"Error: Model file not found: {model_path}")
                return False

            # Store config
            self.model_path = model_path
            self.device = device
            self._is_loaded = True

            print(f"[PartyEngine] Loaded model: {model_path}")
            print(f"[PartyEngine] Device: {device}")

            return True

        except Exception as e:
            print(f"Error loading Party model: {e}")
            self._is_loaded = False
            return False

    def unload_model(self):
        """Unload model (cleanup)."""
        self.model_path = None
        self._is_loaded = False

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """
        Transcribe a single line image.

        Note: Party is designed for batch processing. Single-line transcription
        is less efficient. Use transcribe_lines() for better performance.
        """
        # Convert to batch of one
        results = self.transcribe_lines([image], config)
        return results[0] if results else TranscriptionResult(text="", confidence=0.0)

    def transcribe_lines(self, images: List[np.ndarray], config: Optional[Dict[str, Any]] = None) -> List[TranscriptionResult]:
        """
        Batch transcription using Party's PAGE XML workflow.

        This is the recommended way to use Party - processes entire page at once.
        """
        if not self._is_loaded:
            return [TranscriptionResult(text="[Model not loaded]", confidence=0.0) for _ in images]

        if config is None:
            config = {"model_path": self.model_path, "device": self.device}

        try:
            # Create temporary directory for images and XML
            with tempfile.TemporaryDirectory(prefix="party_") as temp_dir:
                temp_path = Path(temp_dir)

                # Save images to temp directory
                image_paths = []
                for i, img_array in enumerate(images):
                    # Convert numpy to PIL
                    if isinstance(img_array, np.ndarray):
                        pil_img = Image.fromarray(img_array)
                    else:
                        pil_img = img_array

                    # Save to temp file
                    img_path = temp_path / f"line_{i:04d}.png"
                    pil_img.save(img_path)
                    image_paths.append(img_path)

                # Create PAGE XML from images
                xml_path = self._create_page_xml(image_paths, temp_path)

                # Call Party OCR
                output_xml_path = self._call_party(xml_path, config, temp_path)

                # Parse output XML
                transcriptions = self._parse_party_output(output_xml_path)

                # Convert to TranscriptionResult objects
                results = []
                for text, confidence in transcriptions:
                    results.append(TranscriptionResult(
                        text=text,
                        confidence=confidence,
                        metadata={
                            "engine": "party",
                            "model": config.get("model_path", "unknown")
                        }
                    ))

                # Pad with empty results if we got fewer than expected
                while len(results) < len(images):
                    results.append(TranscriptionResult(text="", confidence=0.0))

                return results

        except Exception as e:
            print(f"Error in Party batch transcription: {e}")
            import traceback
            traceback.print_exc()

            # Return empty results for all images
            return [TranscriptionResult(text=f"[Error: {str(e)}]", confidence=0.0) for _ in images]

    def _create_page_xml(self, image_paths: List[Path], temp_dir: Path) -> Path:
        """
        Generate PAGE XML from line images.

        Based on PartyWorker._create_page_xml() from transcription_gui_party.py
        """
        from page_xml_exporter import PageXMLExporter
        from inference_page import LineSegment

        # Use first image as reference for page dimensions
        first_img = Image.open(image_paths[0])
        page_width = first_img.width
        page_height = sum(Image.open(p).height for p in image_paths) + (len(image_paths) - 1) * 10

        # Create line segments (stacked vertically with 10px gap)
        segments = []
        y_offset = 0

        for img_path in image_paths:
            img = Image.open(img_path)
            width, height = img.size

            segment = LineSegment(
                bbox=(0, y_offset, width, y_offset + height),
                coords=None,
                confidence=None,
                text=None  # Will be filled by Party
            )
            segments.append(segment)
            y_offset += height + 10  # 10px gap between lines

        # Create dummy image reference (Party needs a page image)
        # We'll use the first line image as the page image
        page_image_path = temp_dir / "page.png"

        # Create a composite image by stacking all lines
        composite_img = Image.new('RGB', (page_width, page_height), color='white')
        y_pos = 0
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            composite_img.paste(img, (0, y_pos))
            y_pos += img.height + 10

        composite_img.save(page_image_path)

        # Generate PAGE XML
        xml_path = temp_dir / "input.xml"
        exporter = PageXMLExporter(str(page_image_path), page_width, page_height)
        exporter.export(
            segments,
            str(xml_path),
            creator="PartyEngine-Plugin",
            comments="Temporary PAGE XML for Party OCR batch processing"
        )

        return xml_path

    def _call_party(self, input_xml: Path, config: Dict[str, Any], temp_dir: Path) -> Path:
        """
        Execute Party OCR subprocess.

        Based on PartyWorker.run() from transcription_gui_party.py (lines 80-122)
        """
        model_path = config.get("model_path", self.model_path)
        device = config.get("device", self.device)

        # Output XML path
        output_xml = temp_dir / "output.xml"

        # Build Party command
        # Note: Must run from image directory so Party can find the image file
        cmd = (
            f"cd {temp_dir} && "
            f"party -d {device} ocr -i {input_xml.name} {output_xml.name} -mi {model_path}"
        )

        # Execute Party
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            # Save error log
            error_log = self.project_root / "party_error.log"
            with open(error_log, 'w') as f:
                f.write("=== PARTY ERROR LOG ===\n")
                f.write(f"Command: {cmd}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

            raise RuntimeError(
                f"Party OCR failed. Error log saved to {error_log}\n\n"
                f"Error preview:\n{result.stderr[-500:]}"
            )

        # Check if output file was created
        if not output_xml.exists():
            raise FileNotFoundError(f"Party did not create output file: {output_xml}")

        if output_xml.stat().st_size == 0:
            raise ValueError(f"Party output XML is empty: {output_xml}")

        return output_xml

    def _parse_party_output(self, xml_path: Path) -> List[Tuple[str, float]]:
        """
        Parse Party's output PAGE XML to extract transcriptions.

        Based on PartyWorker._parse_party_xml() from transcription_gui_party.py (lines 150-181)
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse Party output XML: {e}")

        # PAGE XML namespace
        ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

        results = []

        # Find all TextLine elements
        for textline in root.findall('.//p:TextLine', ns):
            # Look for TextEquiv > Unicode
            text_equiv = textline.find('.//p:TextEquiv', ns)
            unicode_elem = textline.find('.//p:Unicode', ns)

            # Extract text
            text = ""
            if unicode_elem is not None and unicode_elem.text:
                text = unicode_elem.text

            # Extract confidence
            confidence = 1.0
            if text_equiv is not None and 'conf' in text_equiv.attrib:
                try:
                    confidence = float(text_equiv.attrib['conf'])
                except ValueError:
                    pass

            results.append((text, confidence))

        return results

    def get_capabilities(self) -> Dict[str, bool]:
        """Party engine capabilities."""
        return {
            "batch_processing": True,   # Party excels at batch processing
            "confidence_scores": True,  # Party provides confidence scores
            "beam_search": False,       # Internal to Party
            "language_model": True,     # Party uses language models
            "preprocessing": True,      # Party handles preprocessing
        }

    def requires_line_segmentation(self) -> bool:
        """Party requires line segmentation but processes all lines at once."""
        return True

    def supports_batch(self) -> bool:
        """Party has optimized batch processing."""
        return True


# Model registry
PARTY_MODELS = {
    "European Languages": {
        "path": "models/party_models/party_european_langs.safetensors",
        "description": "Party model for European languages (Glagolitic, Church Slavonic, Latin)"
    }
}
