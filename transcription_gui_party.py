"""
Party HTR GUI - Proof of Concept

Simple GUI to test Party integration with automatic PAGE XML generation.

Workflow:
1. Load image
2. Segment lines with Kraken
3. Generate temp PAGE XML internally
4. Call Party via WSL
5. Display transcriptions

Usage:
    python transcription_gui_party.py
"""

import sys
import os
import tempfile
import subprocess
from pathlib import Path
from typing import List
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox,
    QProgressBar, QComboBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor

# Import segmentation
from kraken_segmenter import KrakenLineSegmenter
from inference_page import LineSegment

# Import PAGE XML exporter
from page_xml_exporter import PageXMLExporter


class PartyWorker(QThread):
    """Background worker for Party OCR processing."""

    finished = pyqtSignal(list)  # List of transcriptions
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, segments: List[LineSegment], image_path: str, model_path: str):
        super().__init__()
        self.segments = segments
        self.image_path = image_path
        self.model_path = model_path
        self.project_root = os.path.expanduser("~/htr_gui/dhlab-slavistik")

    def run(self):
        temp_xml_path = None
        output_xml_path = None

        try:
            self.progress.emit("Creating temporary PAGE XML...")

            # Create temporary PAGE XML in same directory as image
            # This ensures Party can find the image file
            image_path = Path(self.image_path)
            temp_xml_path = str(image_path.parent / f"temp_party_{image_path.stem}.xml")

            # Get image dimensions
            img = Image.open(self.image_path)
            width, height = img.size

            # Export to PAGE XML (will use relative path - just filename)
            exporter = PageXMLExporter(self.image_path, width, height)
            exporter.export(
                self.segments,
                temp_xml_path,
                creator="Party-GUI-PoC",
                comments="Temporary PAGE XML for Party OCR"
            )

            self.progress.emit(f"Calling Party OCR on {len(self.segments)} lines...")

            # Use native Linux paths - no conversion needed
            xml_path = temp_xml_path
            model_path = self.model_path
            image_dir = str(image_path.parent)

            # Create separate output path to avoid in-place modification issues
            output_xml_path = str(image_path.parent / f"temp_party_{image_path.stem}_output.xml")

            # Build Party command
            # Syntax: party -d cuda:0 ocr -i input.xml output.xml -mi model.safetensors
            # Run from image directory so Party can find the image file
            # Note: Language is determined by the model, not a command-line option
            # Party is now installed in the main htr_gui venv, no separate venv needed
            cmd = (
                f"cd {image_dir} && "
                f"party -d cuda:0 ocr -i {xml_path} {output_xml_path} -mi {model_path}"
            )

            # Execute natively on Linux
            self.progress.emit("Running Party (this may take 30-60 seconds)...")

            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                # Save full error to file for debugging
                error_log = Path(self.project_root) / "party_error.log"
                with open(error_log, 'w') as f:
                    f.write("=== PARTY ERROR LOG ===\n")
                    f.write(f"Command: {cmd}\n\n")
                    f.write("=== STDOUT ===\n")
                    f.write(result.stdout)
                    f.write("\n\n=== STDERR ===\n")
                    f.write(result.stderr)

                self.error.emit(f"Party failed. Full error saved to {error_log}\n\nError preview:\n{result.stderr[-500:]}")
                return

            self.progress.emit("Parsing Party output...")

            # Check if output file exists before parsing
            if not Path(output_xml_path).exists():
                self.error.emit(f"Party did not create output file: {output_xml_path}")
                return

            # Parse output XML
            transcriptions = self._parse_party_xml(output_xml_path)

            self.progress.emit(f"Complete! Found {len(transcriptions)} transcriptions")
            self.finished.emit(transcriptions)

        except subprocess.TimeoutExpired:
            self.error.emit("Party timed out (>5 minutes). Try a smaller image or check GPU availability.")
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")
        finally:
            # Cleanup temp files (both input and output)
            for path in [temp_xml_path, output_xml_path]:
                if path and Path(path).exists():
                    try:
                        Path(path).unlink()
                    except:
                        pass

    def _parse_party_xml(self, xml_path: str) -> List[str]:
        """Parse Party's output PAGE XML to extract transcriptions."""
        import xml.etree.ElementTree as ET

        # Check if file exists and is not empty
        if not Path(xml_path).exists():
            raise FileNotFoundError(f"Output XML not found: {xml_path}")

        if Path(xml_path).stat().st_size == 0:
            raise ValueError(f"Output XML is empty: {xml_path}")

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse XML: {e}. The file may be corrupted or incomplete.")

        # PAGE XML namespace
        ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

        transcriptions = []

        # Find all TextLine elements
        for textline in root.findall('.//p:TextLine', ns):
            # Look for TextEquiv > Unicode
            unicode_elem = textline.find('.//p:Unicode', ns)
            if unicode_elem is not None and unicode_elem.text:
                transcriptions.append(unicode_elem.text)
            else:
                transcriptions.append("")  # Empty line

        return transcriptions


class PartyGUI(QMainWindow):
    """Simple GUI for Party OCR with Kraken segmentation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Party HTR - Proof of Concept")
        self.resize(1000, 700)

        # Project root
        self.project_root = os.path.expanduser("~/htr_gui/dhlab-slavistik")

        # State
        self.current_image_path = None
        self.current_image = None
        self.segments = []
        self.transcriptions = []
        self.worker = None

        # Default model path (Linux)
        self.party_model_path = os.path.join(self.project_root, "models/party_models/party_european_langs.safetensors")

        self._init_ui()

    def _init_ui(self):
        """Initialize UI components."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Title
        title = QLabel("Party HTR - Proof of Concept")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "1. Load image  |  2. Segment lines with Kraken  |  3. Process with Party"
        )
        instructions.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(instructions)

        # Top buttons
        btn_layout = QHBoxLayout()

        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self._load_image)
        btn_layout.addWidget(btn_load)

        btn_segment = QPushButton("Segment Lines (Kraken)")
        btn_segment.clicked.connect(self._segment_lines)
        btn_layout.addWidget(btn_segment)

        btn_process = QPushButton("Process with Party")
        btn_process.clicked.connect(self._process_party)
        btn_layout.addWidget(btn_process)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Model path selector
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Party Model:"))

        self.model_path_input = QLabel(self.party_model_path)
        self.model_path_input.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
        model_layout.addWidget(self.model_path_input, stretch=1)

        btn_browse_model = QPushButton("Browse...")
        btn_browse_model.clicked.connect(self._browse_model)
        model_layout.addWidget(btn_browse_model)

        layout.addLayout(model_layout)

        # Horizontal splitter for left-image, right-transcription layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, stretch=1)

        # Left panel - Image preview
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("Image Preview:"))
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        left_layout.addWidget(self.image_label, stretch=1)

        # Status info under image
        info_layout = QVBoxLayout()
        self.lbl_image = QLabel("Image: -")
        self.lbl_lines = QLabel("Lines: 0")
        self.lbl_status = QLabel("Status: Ready")
        info_layout.addWidget(self.lbl_image)
        info_layout.addWidget(self.lbl_lines)
        info_layout.addWidget(self.lbl_status)
        left_layout.addLayout(info_layout)

        splitter.addWidget(left_widget)

        # Right panel - Transcription results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("Transcription Results:"))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Transcriptions will appear here...")
        self.results_text.setStyleSheet("font-size: 14pt;")
        right_layout.addWidget(self.results_text, stretch=1)

        splitter.addWidget(right_widget)

        # Set initial splitter sizes (50-50 split)
        splitter.setSizes([500, 500])

        # Progress bar below splitter
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Export button
        btn_export = QPushButton("Export Results (TXT)")
        btn_export.clicked.connect(self._export_results)
        layout.addWidget(btn_export)

    def _load_image(self):
        """Load image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)"
        )

        if not file_path:
            return

        try:
            self.current_image_path = file_path
            self.current_image = Image.open(file_path).convert('RGB')

            # Show preview
            pixmap = QPixmap(file_path)
            scaled = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

            # Update status
            self.lbl_image.setText(f"Image: {Path(file_path).name}")
            self.lbl_status.setText("Status: Image loaded")

            # Clear previous results
            self.segments = []
            self.transcriptions = []
            self.results_text.clear()
            self.lbl_lines.setText("Lines: 0")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    def _segment_lines(self):
        """Segment lines using Kraken."""
        if not self.current_image:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        try:
            self.lbl_status.setText("Status: Segmenting with Kraken...")
            QApplication.processEvents()

            # Create Kraken segmenter
            segmenter = KrakenLineSegmenter()

            # Segment lines
            self.segments = segmenter.segment_lines(
                self.current_image,
                text_direction='horizontal-lr',
                use_binarization=True
            )

            # Update status
            self.lbl_lines.setText(f"Lines: {len(self.segments)}")
            self.lbl_status.setText(f"Status: Detected {len(self.segments)} lines")

            # Draw segmentation boxes on image
            self._draw_segments()

            if len(self.segments) == 0:
                QMessageBox.warning(self, "No Lines", "Kraken did not detect any text lines")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Segmentation failed:\n{str(e)}")
            self.lbl_status.setText("Status: Segmentation failed")

    def _browse_model(self):
        """Browse for Party model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Party Model",
            "party_models",
            "Model Files (*.safetensors *.pt *.pth);;All Files (*)"
        )

        if file_path:
            self.party_model_path = file_path
            self.model_path_input.setText(file_path)

    def _process_party(self):
        """Process with Party OCR."""
        if not self.segments:
            QMessageBox.warning(self, "No Segments", "Please segment lines first")
            return

        if not Path(self.party_model_path).exists():
            QMessageBox.critical(
                self,
                "Model Not Found",
                f"Party model not found:\n{self.party_model_path}\n\n"
                "Please download the model or select a different path."
            )
            return

        # Start Party worker thread
        self.worker = PartyWorker(self.segments, self.current_image_path, self.party_model_path)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        self.worker.start()

    def _on_progress(self, message: str):
        """Handle progress updates."""
        self.lbl_status.setText(f"Status: {message}")

    def _on_finished(self, transcriptions: List[str]):
        """Handle completed transcription."""
        self.transcriptions = transcriptions

        # Display results (just the transcriptions, no line numbers)
        result_text = "\n".join(transcriptions)
        self.results_text.setText(result_text)

        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"Status: Complete! {len(transcriptions)} lines transcribed")

        QMessageBox.information(
            self,
            "Success",
            f"Party processing complete!\n\nTranscribed {len(transcriptions)} lines"
        )

    def _on_error(self, error_message: str):
        """Handle error."""
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("Status: Error")
        QMessageBox.critical(self, "Error", error_message)

    def _export_results(self):
        """Export transcription results to file."""
        if not self.transcriptions:
            QMessageBox.warning(self, "No Results", "No transcriptions to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Transcriptions",
            "party_transcription.txt",
            "Text Files (*.txt);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for i, text in enumerate(self.transcriptions, 1):
                    f.write(f"{text}\n")

            QMessageBox.information(self, "Success", f"Exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")

    def _draw_segments(self):
        """Draw green boxes around detected line segments on the image."""
        if not self.current_image_path or not self.segments:
            return

        try:
            # Load original image with PIL and draw boxes
            from PIL import ImageDraw
            img = Image.open(self.current_image_path).convert('RGB')
            draw = ImageDraw.Draw(img)

            # Draw each segment as a green box
            for segment in self.segments:
                x1, y1, x2, y2 = segment.bbox
                # Draw green rectangle with 2px width
                draw.rectangle([x1, y1, x2, y2], outline='green', width=2)

            # Convert PIL image to QPixmap
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)

            pixmap = QPixmap()
            pixmap.loadFromData(buffer.read())

            # Scale and display
            scaled = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

        except Exception as e:
            print(f"Warning: Could not draw segments: {e}")


def main():
    app = QApplication(sys.argv)
    window = PartyGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
