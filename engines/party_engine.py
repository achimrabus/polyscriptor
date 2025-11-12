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
from inference_page import LineSegment
from page_xml_exporter import PageXMLExporter

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QPushButton, QLineEdit, QGroupBox, QFileDialog, QCheckBox, QSpinBox
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object


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

        # Find party executable (check multiple locations)
        self.party_exe = self._find_party_executable()

        # Config widget references
        self._config_widget: Optional[QWidget] = None
        self._model_combo: Optional[QComboBox] = None
        self._model_path_label: Optional[QLabel] = None
        self._device_combo: Optional[QComboBox] = None

        # Batch processing state
        self._is_loaded = False

    def _find_party_executable(self) -> Optional[str]:
        """Find party executable in multiple possible locations."""
        # Check locations in order of preference
        possible_locations = [
            Path.home() / "htr_gui/bin/party",  # htr_gui venv
            Path.home() / "party/party_env/bin/party",  # party_env
            "party",  # System PATH (fallback)
        ]

        for location in possible_locations:
            if isinstance(location, Path) and location.exists():
                return str(location)
            elif location == "party":
                # Check if party is in PATH
                try:
                    result = subprocess.run(
                        ["which", "party"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        return "party"  # Use command directly from PATH
                except:
                    pass

        return None

    def get_name(self) -> str:
        return "Party"

    def get_description(self) -> str:
        return "PyTorch Lightning HTR with PAGE XML workflow"

    def is_available(self) -> bool:
        """Check if Party is installed and model exists."""
        try:
            # Check if party executable was found
            if self.party_exe is None:
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
            if self.party_exe is None:
                return "Party executable not found. Checked: ~/htr_gui/bin/party, ~/party/party_env/bin/party, and system PATH"

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

        # Performance optimization group
        perf_group = QGroupBox("Performance Optimization")
        perf_layout = QVBoxLayout()

        # Compilation toggle
        self._compile_checkbox = QCheckBox("Use torch.compile (faster inference, slower startup)")
        self._compile_checkbox.setChecked(True)  # Default: enabled
        self._compile_checkbox.setToolTip(
            "Enable torch.compile() for faster inference.\n"
            "First run will be slow (10-20s startup), but subsequent inference is faster.\n"
            "Disable for faster startup with slightly slower inference."
        )
        perf_layout.addWidget(self._compile_checkbox)

        # Quantization toggle
        self._quantize_checkbox = QCheckBox("Use quantization (lower VRAM, faster)")
        self._quantize_checkbox.setChecked(False)  # Default: disabled
        self._quantize_checkbox.setToolTip(
            "Enable post-training quantization for faster inference and lower VRAM usage.\n"
            "May slightly reduce quality, but significantly improves speed."
        )
        perf_layout.addWidget(self._quantize_checkbox)

        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self._batch_size_spin = QSpinBox()
        self._batch_size_spin.setRange(1, 32)
        self._batch_size_spin.setValue(8)  # Default: 8
        self._batch_size_spin.setToolTip(
            "Number of lines to process simultaneously.\n"
            "Higher = faster for large batches, but uses more VRAM.\n"
            "Lower = slower but uses less VRAM."
        )
        batch_layout.addWidget(self._batch_size_spin)
        batch_layout.addStretch()
        perf_layout.addLayout(batch_layout)

        # Line encoding method
        encoding_layout = QHBoxLayout()
        encoding_layout.addWidget(QLabel("Line Encoding:"))
        self._encoding_combo = QComboBox()
        self._encoding_combo.addItem("Curves (more accurate)", "curves")
        self._encoding_combo.addItem("Boxes (faster)", "boxes")
        self._encoding_combo.setCurrentIndex(0)  # Default: curves
        self._encoding_combo.setToolTip(
            "Method for encoding line regions in PAGE XML.\n"
            "Curves: More accurate, preserves baseline curvature\n"
            "Boxes: Faster, uses simple bounding boxes"
        )
        encoding_layout.addWidget(self._encoding_combo)
        encoding_layout.addStretch()
        perf_layout.addLayout(encoding_layout)

        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

        # Post-processing group
        postproc_group = QGroupBox("Post-Processing")
        postproc_layout = QVBoxLayout()

        # Remove duplicates toggle
        self._remove_duplicates_checkbox = QCheckBox("Remove consecutive duplicate lines")
        self._remove_duplicates_checkbox.setChecked(True)  # Default: enabled
        self._remove_duplicates_checkbox.setToolTip(
            "Remove lines that are exact duplicates of the previous line.\n"
            "Useful when Party generates repetitive transcriptions.\n"
            "Only removes consecutive duplicates, not all duplicates."
        )
        postproc_layout.addWidget(self._remove_duplicates_checkbox)

        postproc_group.setLayout(postproc_layout)
        layout.addWidget(postproc_group)

        # Info label
        info_label = QLabel(
            "💡 For fastest results: Disable torch.compile and enable quantization.\n"
            "For best quality: Enable torch.compile, disable quantization.\n"
            "Party processes entire pages using PAGE XML format."
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
                "device": "cuda:0",
                "compile": True,
                "quantize": False,
                "batch_size": 8,
                "encoding": "curves"
            }

        # Get model path from label (handles both default and custom)
        model_path = self._model_path_label.text()
        device = self._device_combo.currentText()

        # Get optimization parameters
        compile_enabled = self._compile_checkbox.isChecked()
        quantize_enabled = self._quantize_checkbox.isChecked()
        batch_size = self._batch_size_spin.value()
        encoding = self._encoding_combo.currentData()

        # Get post-processing parameters
        remove_duplicates = self._remove_duplicates_checkbox.isChecked()

        return {
            "model_path": model_path,
            "device": device,
            "compile": compile_enabled,
            "quantize": quantize_enabled,
            "batch_size": batch_size,
            "encoding": encoding,
            "remove_duplicates": remove_duplicates
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

        # Update optimization parameters
        self._compile_checkbox.setChecked(config.get("compile", True))
        self._quantize_checkbox.setChecked(config.get("quantize", False))
        self._batch_size_spin.setValue(config.get("batch_size", 8))

        # Update encoding combo
        encoding = config.get("encoding", "curves")
        for i in range(self._encoding_combo.count()):
            if self._encoding_combo.itemData(i) == encoding:
                self._encoding_combo.setCurrentIndex(i)
                break

        # Update post-processing parameters
        self._remove_duplicates_checkbox.setChecked(config.get("remove_duplicates", True))

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

    def prefers_batch_with_context(self) -> bool:
        """
        Indicate that Party engine works best with batch processing using original image.

        This signals to the GUI that it should call transcribe_lines() with
        original_image_path and line_bboxes instead of processing lines individually.

        CRITICAL for language recognition: Party needs the original page image to
        correctly detect scripts like Glagolitic, Church Slavonic, etc.
        """
        return True

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """
        Transcribe a single line image.

        Note: Party is designed for batch processing. Single-line transcription
        is less efficient. Use transcribe_lines() for better performance.
        """
        # Convert to batch of one
        results = self.transcribe_lines([image], config)
        return results[0] if results else TranscriptionResult(text="", confidence=0.0)

    def transcribe_lines(
        self,
        images: List[np.ndarray],
        config: Optional[Dict[str, Any]] = None,
        original_image_path: Optional[str] = None,
        line_bboxes: Optional[List[tuple]] = None
    ) -> List[TranscriptionResult]:
        """
        Batch transcription using Party's PAGE XML workflow.

        This is the recommended way to use Party - processes entire page at once.

        Args:
            images: List of line images (np.ndarray)
            config: Engine configuration
            original_image_path: Path to original page image (CRITICAL for language recognition!)
                                If provided, Party will use the original image instead of
                                creating a synthetic composite, preserving page-level context
                                needed for proper script/language detection (e.g., Glagolitic)
            line_bboxes: List of (x1, y1, x2, y2) bounding boxes for each line in original image
                        Required when original_image_path is provided
        """
        if not self._is_loaded:
            return [TranscriptionResult(text="[Model not loaded]", confidence=0.0) for _ in images]

        if config is None:
            config = {"model_path": self.model_path, "device": self.device}

        try:
            # Create temporary directory for images and XML
            with tempfile.TemporaryDirectory(prefix="party_") as temp_dir:
                temp_path = Path(temp_dir)

                # CRITICAL: Use original image if provided (preserves language recognition)
                if original_image_path and line_bboxes:
                    # Load and convert original image to RGB (Party requires 3 channels)
                    original_path = Path(original_image_path)
                    img = Image.open(original_image_path)

                    # Convert to RGB if needed (handles RGBA, grayscale, etc.)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Save RGB image to temp directory
                    page_image_path = temp_path / original_path.name
                    img.save(page_image_path)

                    width, height = img.size

                    # Create segments with original coordinates
                    segments = []
                    for i, bbox in enumerate(line_bboxes):
                        x1, y1, x2, y2 = bbox
                        segments.append(LineSegment(
                            image=img.crop(bbox),  # Cropped for metadata
                            bbox=bbox,
                            coords=None,
                            text=None,
                            confidence=None
                        ))

                    # Generate PAGE XML with original image reference
                    xml_path = temp_path / "input.xml"
                    exporter = PageXMLExporter(str(page_image_path), width, height)
                    exporter.export(
                        segments,
                        str(xml_path),
                        creator="PartyEngine-Plugin-OriginalImage",
                        comments="PAGE XML with original image (preserves language recognition)"
                    )
                else:
                    # Fallback: Save individual line images (old behavior, creates synthetic composite)
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

                    # Create PAGE XML from images (synthetic composite)
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

                # Optional: Remove consecutive duplicate transcriptions
                # (Party sometimes generates identical text for adjacent lines)
                if config.get("remove_duplicates", True):
                    results = self._remove_consecutive_duplicates(results)

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
                image=img,  # LineSegment requires image as first parameter
                bbox=(0, y_offset, width, y_offset + height),
                coords=None,
                text=None,  # Will be filled by Party
                confidence=None
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
        Execute Party OCR subprocess with optimization parameters.

        Based on PartyWorker.run() from transcription_gui_party.py (lines 80-122)
        Enhanced with CLI parameters from party ocr --help
        """
        model_path = config.get("model_path", self.model_path)
        device = config.get("device", self.device)

        # Get optimization parameters
        compile_enabled = config.get("compile", True)
        quantize_enabled = config.get("quantize", False)
        batch_size = config.get("batch_size", 8)
        encoding = config.get("encoding", "curves")

        # Build optimization flags
        compile_flag = "--compile" if compile_enabled else "--no-compile"
        quantize_flag = "--quantize" if quantize_enabled else "--no-quantize"

        # Output XML path
        output_xml = temp_dir / "output.xml"

        # Build enhanced Party command with all optimization parameters
        # Note: Must run from image directory so Party can find the image file
        cmd = (
            f"cd {temp_dir} && "
            f"{self.party_exe} -d {device} ocr "
            f"-i {input_xml.name} {output_xml.name} "
            f"-mi {model_path} "
            f"{compile_flag} {quantize_flag} "
            f"-b {batch_size} "
            f"--{encoding}"
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

    def _remove_consecutive_duplicates(self, results: List[TranscriptionResult]) -> List[TranscriptionResult]:
        """
        Remove consecutive duplicate transcriptions.

        Party sometimes generates identical text for adjacent lines, especially
        when lines are similar or there's repetitive content in the manuscript.
        This method removes exact consecutive duplicates while preserving
        the original sequence.

        Args:
            results: List of transcription results

        Returns:
            Filtered list with consecutive duplicates removed
        """
        if len(results) <= 1:
            return results

        filtered = [results[0]]  # Always keep first result

        for i in range(1, len(results)):
            current = results[i]
            previous = filtered[-1]

            # Only remove if text is EXACTLY the same and both are non-empty
            if current.text and current.text == previous.text:
                # Skip this duplicate
                continue
            else:
                filtered.append(current)

        return filtered

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
