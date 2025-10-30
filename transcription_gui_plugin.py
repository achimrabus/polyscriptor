"""
HTR Transcription GUI - Plugin-Based Version

Unified interface for multiple HTR engines using the plugin system.

Features:
- Dropdown engine selection (TrOCR, Qwen3, PyLaia, Commercial APIs)
- Dynamic configuration panels per engine
- Seamless zoom/pan with QGraphicsView
- Drag & drop + file dialog import
- Automatic line segmentation
- Export to TXT/CSV

Usage:
    python transcription_gui_plugin.py
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QPushButton, QLabel, QTextEdit, QLineEdit, QComboBox,
    QFileDialog, QProgressBar, QStatusBar, QMessageBox,
    QListWidget, QListWidgetItem, QGroupBox, QScrollArea, QSlider, QSpinBox, QCheckBox,
    QFontDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QAction

# Import segmentation components
from inference_page import LineSegmenter, PageXMLSegmenter, LineSegment
from page_xml_exporter import PageXMLExporter

# Import HTR Engine Plugin System
from htr_engine_base import get_global_registry, HTREngine, TranscriptionResult

# Get available engines
engine_registry = get_global_registry()
available_engines = engine_registry.get_available_engines()

print(f"HTR Engine Plugin System: {len(available_engines)} engines available")
for engine in available_engines:
    print(f"  - {engine.get_name()}: {engine.get_description()}")


class ZoomableGraphicsView(QGraphicsView):
    """Graphics view with smooth zoom and pan capabilities."""

    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.line_items = []

    def has_image(self):
        return not self._empty

    def fit_in_view(self):
        """Fit image to view."""
        rect = QRectF(self._scene.itemsBoundingRect())
        if not rect.isNull():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 0

    def set_image(self, pixmap: QPixmap):
        """Load image into view."""
        self._scene.clear()
        self.line_items = []
        self._scene.addPixmap(pixmap)
        self._empty = False
        self.fit_in_view()

    def draw_line_boxes(self, lines: List[LineSegment], color: QColor = QColor(0, 255, 0)):
        """Draw bounding boxes for detected lines."""
        # Remove old boxes
        for item in self.line_items:
            self._scene.removeItem(item)
        self.line_items = []

        # Draw new boxes
        pen = QPen(color)
        pen.setWidth(2)

        for line in lines:
            x1, y1, x2, y2 = line.bbox
            rect_item = self._scene.addRect(x1, y1, x2 - x1, y2 - y1, pen)
            self.line_items.append(rect_item)

    def wheelEvent(self, event):
        """Zoom with mouse wheel."""
        if self.has_image():
            factor = 1.25 if event.angleDelta().y() > 0 else 0.8
            self.scale(factor, factor)
            self._zoom += 1 if factor > 1 else -1


class TranscriptionWorker(QThread):
    """Background worker for HTR transcription."""

    progress = pyqtSignal(int, int, str)  # current, total, text
    finished = pyqtSignal(list)  # List of transcriptions
    error = pyqtSignal(str)

    def __init__(self, engine: HTREngine, line_segments: List[LineSegment], image: Image.Image):
        super().__init__()
        self.engine = engine
        self.line_segments = line_segments
        self.image = image

    def run(self):
        """Process all line segments."""
        try:
            transcriptions = []

            for i, line_seg in enumerate(self.line_segments):
                # Crop line from image
                x1, y1, x2, y2 = line_seg.bbox
                line_img = self.image.crop((x1, y1, x2, y2))
                line_array = np.array(line_img)

                # Transcribe with engine
                result = self.engine.transcribe_line(line_array)

                # Ensure we get the text as a string
                text = str(result.text) if hasattr(result, 'text') else str(result)
                transcriptions.append(text)
                self.progress.emit(i + 1, len(self.line_segments), text)

            self.finished.emit(transcriptions)

        except Exception as e:
            self.error.emit(str(e))


class TranscriptionGUI(QMainWindow):
    """Main GUI window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HTR Transcription Tool - Plugin System")
        self.setGeometry(100, 100, 1400, 900)

        # State
        self.current_image_path: Optional[Path] = None
        self.current_image: Optional[Image.Image] = None
        self.line_segments: List[LineSegment] = []
        self.transcriptions: List[str] = []
        self.current_engine: Optional[HTREngine] = None
        self.worker: Optional[TranscriptionWorker] = None

        # Cache config widgets to prevent deletion
        self.config_widgets_cache: Dict[str, QWidget] = {}

        # Settings file
        self.settings_file = Path.home() / ".trocr_gui" / "settings_plugin.json"
        self.settings_file.parent.mkdir(exist_ok=True)

        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Setup user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel: Image view
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Image view
        self.image_view = ZoomableGraphicsView()
        left_layout.addWidget(self.image_view)

        # Image controls
        img_controls = QHBoxLayout()
        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self.load_image)
        img_controls.addWidget(btn_load)

        btn_segment = QPushButton("Segment Lines")
        btn_segment.clicked.connect(self.segment_lines)
        img_controls.addWidget(btn_segment)

        btn_fit = QPushButton("Fit to View")
        btn_fit.clicked.connect(self.image_view.fit_in_view)
        img_controls.addWidget(btn_fit)

        left_layout.addLayout(img_controls)

        main_layout.addWidget(left_panel, stretch=2)

        # Segmentation settings
        seg_group = QGroupBox("Segmentation Settings")
        seg_layout = QVBoxLayout()

        # Check if Kraken is available
        try:
            from kraken_segmenter import KrakenLineSegmenter
            KRAKEN_AVAILABLE = True
        except ImportError:
            KRAKEN_AVAILABLE = False

        # Method selector
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.seg_method_combo = QComboBox()
        self.seg_method_combo.addItem("HPP (Fast)", "HPP")
        if KRAKEN_AVAILABLE:
            self.seg_method_combo.addItem("Kraken (Robust)", "Kraken")
            self.seg_method_combo.setCurrentIndex(1)  # Default to Kraken if available
        else:
            self.seg_method_combo.addItem("Kraken (Not installed)", None)
            self.seg_method_combo.model().item(1).setEnabled(False)
        self.seg_method_combo.currentIndexChanged.connect(self._on_seg_method_changed)
        method_layout.addWidget(self.seg_method_combo)
        seg_layout.addLayout(method_layout)

        # HPP-specific parameters
        self.hpp_params_widget = QWidget()
        hpp_layout = QVBoxLayout()
        hpp_layout.setContentsMargins(0, 0, 0, 0)

        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(QLabel("Threshold:"))
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(5, 150)  # 0.5% to 15%
        self.sensitivity_slider.setValue(50)  # Default 5%
        self.sensitivity_label = QLabel("5.0%")
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(f"{v/10:.1f}%")
        )
        self.sensitivity_slider.setToolTip("Detection threshold: Higher = more selective (0.5-15%)")
        sensitivity_layout.addWidget(self.sensitivity_slider)
        sensitivity_layout.addWidget(self.sensitivity_label)
        hpp_layout.addLayout(sensitivity_layout)

        min_height_layout = QHBoxLayout()
        min_height_layout.addWidget(QLabel("Min Height:"))
        self.min_height_spin = QSpinBox()
        self.min_height_spin.setRange(5, 100)
        self.min_height_spin.setValue(10)
        self.min_height_spin.setSuffix(" px")
        self.min_height_spin.setToolTip("Minimum line height in pixels")
        min_height_layout.addWidget(self.min_height_spin)
        min_height_layout.addStretch()
        hpp_layout.addLayout(min_height_layout)

        self.use_morph_check = QCheckBox("Morph. Ops")
        self.use_morph_check.setChecked(True)
        self.use_morph_check.setToolTip("Apply morphological operations to connect broken characters")
        hpp_layout.addWidget(self.use_morph_check)

        self.hpp_params_widget.setLayout(hpp_layout)
        seg_layout.addWidget(self.hpp_params_widget)

        # Kraken-specific parameters
        self.kraken_params_widget = QWidget()
        kraken_layout = QVBoxLayout()
        kraken_layout.setContentsMargins(0, 0, 0, 0)

        self.use_binarization_check = QCheckBox("Use Binarization")
        self.use_binarization_check.setChecked(True)
        self.use_binarization_check.setToolTip("Apply neural binarization preprocessing (recommended for degraded documents)")
        kraken_layout.addWidget(self.use_binarization_check)

        self.kraken_params_widget.setLayout(kraken_layout)
        seg_layout.addWidget(self.kraken_params_widget)

        # Set initial visibility based on default method
        self._on_seg_method_changed(self.seg_method_combo.currentIndex())

        seg_group.setLayout(seg_layout)
        left_layout.addWidget(seg_group)    

        # Right panel: Engine selection and transcription
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Engine selection
        engine_group = QGroupBox("HTR Engine")
        engine_layout = QVBoxLayout()

        self.engine_combo = QComboBox()
        if not available_engines:
            self.engine_combo.addItem("No engines available")
        else:
            for engine in available_engines:
                self.engine_combo.addItem(engine.get_name())

        self.engine_combo.currentTextChanged.connect(self.on_engine_changed)
        engine_layout.addWidget(self.engine_combo)

        # Engine description
        self.engine_desc_label = QLabel("")
        self.engine_desc_label.setWordWrap(True)
        self.engine_desc_label.setStyleSheet("color: gray; font-size: 12pt;")
        engine_layout.addWidget(self.engine_desc_label)

        engine_group.setLayout(engine_layout)
        right_layout.addWidget(engine_group)

        # Dynamic engine configuration panel
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setMinimumHeight(300)

        self.config_container = QWidget()
        self.config_layout = QVBoxLayout(self.config_container)
        self.config_layout.addStretch()

        config_scroll.setWidget(self.config_container)
        right_layout.addWidget(config_scroll)

        # Load/Process buttons
        process_layout = QHBoxLayout()

        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.clicked.connect(self.load_model)
        process_layout.addWidget(self.btn_load_model)

        self.btn_process = QPushButton("Process Image")
        self.btn_process.clicked.connect(self.process_image)
        self.btn_process.setEnabled(False)
        process_layout.addWidget(self.btn_process)

        right_layout.addLayout(process_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Transcription results
        results_group = QGroupBox("Transcriptions")
        results_layout = QVBoxLayout()

        self.transcription_text = QTextEdit()
        self.transcription_text.setReadOnly(True)
        results_layout.addWidget(self.transcription_text)

        # Export buttons
        export_layout = QHBoxLayout()

        btn_export_txt = QPushButton("Export TXT")
        btn_export_txt.clicked.connect(self.export_txt)
        export_layout.addWidget(btn_export_txt)

        btn_export_csv = QPushButton("Export CSV")
        btn_export_csv.clicked.connect(self.export_csv)
        export_layout.addWidget(btn_export_csv)

        btn_export_xml = QPushButton("Export PAGE XML")
        btn_export_xml.clicked.connect(self.export_xml)
        export_layout.addWidget(btn_export_xml)

        results_layout.addLayout(export_layout)

        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group, stretch=1)

        main_layout.addWidget(right_panel, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Initialize first engine
        if available_engines:
            self.on_engine_changed(self.engine_combo.currentText())

    def on_engine_changed(self, engine_name: str):
        """Handle engine selection change."""
        # Get selected engine
        engine = engine_registry.get_engine_by_name(engine_name)
        if not engine:
            self.status_bar.showMessage("Engine not found")
            return

        self.current_engine = engine

        # Update description
        self.engine_desc_label.setText(engine.get_description())

        # Hide previous config widgets (but don't delete them)
        while self.config_layout.count() > 1:  # Keep the stretch
            item = self.config_layout.takeAt(0)
            if item.widget():
                item.widget().hide()

        # Get or create config widget (cached to prevent deletion bug)
        if engine_name not in self.config_widgets_cache:
            try:
                config_widget = engine.get_config_widget()
                self.config_widgets_cache[engine_name] = config_widget
            except Exception as e:
                self.status_bar.showMessage(f"Error loading config: {e}")
                QMessageBox.warning(self, "Error", f"Failed to load engine config: {e}")
                return

        # Show cached widget
        config_widget = self.config_widgets_cache[engine_name]
        config_widget.show()
        self.config_layout.insertWidget(0, config_widget)
        self.status_bar.showMessage(f"Engine: {engine_name}")

    def update_process_button_state(self):
        """Update process button enabled state based on current conditions."""
        has_model = self.current_engine and self.current_engine.is_model_loaded()
        has_image = self.current_image is not None
        has_lines = len(self.line_segments) > 0

    # Check if engine requires segmentation
        needs_segmentation = self.current_engine and self.current_engine.requires_line_segmentation()

    # Debug output
        print(f"Button state: model={has_model}, image={has_image}, lines={has_lines}, needs_seg={needs_segmentation}")

    # Enable if: model loaded AND image loaded AND (lines segmented OR doesn't need segmentation)
        should_enable = has_model and has_image and (has_lines or not needs_segmentation)
        self.btn_process.setEnabled(should_enable)

        if not should_enable:
            missing = []
            if not has_model: missing.append("model not loaded")
            if not has_image: missing.append("no image")
            if needs_segmentation and not has_lines: missing.append("need to segment lines")
            self.status_bar.showMessage(f"Cannot process: {', '.join(missing)}")

    
    def load_model(self):
        """Load the selected HTR model."""
        if not self.current_engine:
            QMessageBox.warning(self, "No Engine", "Please select an HTR engine")
            return

        # Get configuration from engine
        try:
            config = self.current_engine.get_config()
        except Exception as e:
            QMessageBox.warning(self, "Config Error", f"Failed to get engine config: {e}")
            return

        # Load model
        self.status_bar.showMessage("Loading model...")
        QApplication.processEvents()

        success = self.current_engine.load_model(config)

        if success:
            self.status_bar.showMessage(f"Model loaded: {self.current_engine.get_name()}")
            self.update_process_button_state()
            QMessageBox.information(self, "Success", "Model loaded successfully!")
        else:
            self.status_bar.showMessage("Failed to load model")
            QMessageBox.warning(self, "Error", "Failed to load model. Check console for details.")

    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)"
        )

        if not file_path:
            return

        try:
            self.current_image_path = Path(file_path)
            self.current_image = Image.open(file_path).convert("RGB")

            # Display image
            img_array = np.array(self.current_image)
            height, width = img_array.shape[:2]
            bytes_per_line = 3 * width
            q_img = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            self.image_view.set_image(pixmap)

            self.status_bar.showMessage(f"Loaded: {file_path}")
            self.line_segments = []
            self.transcriptions = []
            self.transcription_text.clear()

            # Update button state after loading image
            self.update_process_button_state()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image: {e}")

    def _on_seg_method_changed(self, index):
        """Handle segmentation method change."""
        method = self.seg_method_combo.currentData()

        if method == "Kraken":
            # Show Kraken parameters, hide HPP parameters
            self.kraken_params_widget.setVisible(True)
            self.hpp_params_widget.setVisible(False)
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage("Switched to Kraken segmentation (slower but more robust)")
        else:  # HPP
            # Show HPP parameters, hide Kraken parameters
            self.hpp_params_widget.setVisible(True)
            self.kraken_params_widget.setVisible(False)
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage("Switched to HPP segmentation (fast)")

    def segment_lines(self):
        """Segment lines in the current image."""
        if self.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        try:
            method = self.seg_method_combo.currentData()

            if method == "Kraken":
                # Use Kraken segmentation
                self.status_bar.showMessage("Detecting lines with Kraken (this may take 3-8 seconds)...")
                QApplication.processEvents()

                from kraken_segmenter import KrakenLineSegmenter

                use_binarization = self.use_binarization_check.isChecked()
                segmenter = KrakenLineSegmenter()
                kraken_segments = segmenter.segment_lines(
                    self.current_image,
                    text_direction='horizontal-lr',
                    use_binarization=use_binarization
                )

                # Convert Kraken segments to LineSegment format
                from inference_page import LineSegment
                self.line_segments = []
                for seg in kraken_segments:
                    self.line_segments.append(LineSegment(
                        image=seg.image,
                        bbox=seg.bbox,
                        coords=seg.baseline  # LineSegment uses 'coords' not 'baseline'
                    ))

            else:  # HPP method
                self.status_bar.showMessage("Detecting lines with HPP...")
                QApplication.processEvents()

                # Get HPP settings
                sensitivity = self.sensitivity_slider.value() / 1000.0
                min_height = self.min_height_spin.value()
                use_morph = self.use_morph_check.isChecked()

                segmenter = LineSegmenter(
                    sensitivity=sensitivity,
                    min_line_height=min_height,
                    use_morph=use_morph
                )
                self.line_segments = segmenter.segment_lines(self.current_image)

            # Draw boxes
            self.image_view.draw_line_boxes(self.line_segments)

            num_lines = len(self.line_segments)
            self.status_bar.showMessage(f"Found {num_lines} lines with {method}")

            # Warn if no lines or only 1 line detected
            if num_lines == 0:
                QMessageBox.warning(
                    self,
                    "No Lines Detected",
                    f"{method} did not detect any text lines.\n\n"
                    f"Possible solutions:\n"
                    f"- Try switching to {'HPP' if method == 'Kraken' else 'Kraken'} method\n"
                    f"- Check if the document contains text"
                )
            elif num_lines == 1:
                QMessageBox.information(
                    self,
                    "Only 1 Line Detected",
                    f"Only 1 text line was detected using {method}.\n\n"
                    f"If the page contains multiple lines, try:\n"
                    + ("- Switching to HPP method with adjusted parameters"
                       if method == "Kraken"
                       else "- INCREASING Threshold (e.g., 8-10%) to be more selective\n"
                            "- Reduce Min Height if lines are close together\n"
                            "- Enable Morph. Ops checkbox\n"
                            "- Or try switching to Kraken method")
                )

            # Update process button state
            self.update_process_button_state()

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"Segmentation error:\n{error_detail}")
            QMessageBox.warning(self, "Error", f"Failed to segment lines: {e}\n\nCheck console for details.")
    
    def process_image(self):
        """Transcribe all detected lines or full page (for VLMs)."""
        if not self.current_engine or not self.current_engine.is_model_loaded():
            QMessageBox.warning(self, "Model Not Loaded", "Please load a model first")
            return

        if not self.current_image:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        # For VLMs that don't need segmentation, create a fake line segment with the full image
        line_segments = self.line_segments
        if not line_segments and not self.current_engine.requires_line_segmentation():
            # Create a single "line" representing the full page
            h, w = self.current_image.height, self.current_image.width
            full_page_segment = LineSegment(
                image=self.current_image,
                bbox=(0, 0, w, h),
                coords=None
            )
            line_segments = [full_page_segment]

        if not line_segments:
            QMessageBox.warning(self, "No Lines", "Please segment lines first")
            return

        # Start worker thread
        self.worker = TranscriptionWorker(
            self.current_engine,
            line_segments,
            self.current_image
        )

        self.worker.progress.connect(self.on_transcription_progress)
        self.worker.finished.connect(self.on_transcription_finished)
        self.worker.error.connect(self.on_transcription_error)

        # UI updates
        self.btn_process.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.line_segments))
        self.status_bar.showMessage("Transcribing...")

        self.worker.start()

    def on_transcription_progress(self, current: int, total: int, text: str):
        """Update progress during transcription."""
        self.progress_bar.setValue(current)
        self.status_bar.showMessage(f"Transcribing line {current}/{total}...")

    def on_transcription_finished(self, transcriptions: List[str]):
        """Handle completion of transcription."""
        self.transcriptions = transcriptions

        # Display results (without "Line X:" prefix)
        result_text = "\n".join(transcriptions)
        self.transcription_text.setPlainText(result_text)

        # UI updates
        self.btn_process.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"Transcription complete ({len(transcriptions)} lines)")

    def on_transcription_error(self, error_msg: str):
        """Handle transcription error."""
        self.btn_process.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Transcription failed")
        QMessageBox.warning(self, "Error", f"Transcription failed: {error_msg}")

    def export_txt(self):
        """Export transcriptions to TXT file."""
        if not self.transcriptions:
            QMessageBox.warning(self, "No Data", "No transcriptions to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export TXT",
            str(self.current_image_path.with_suffix(".txt")) if self.current_image_path else "transcription.txt",
            "Text Files (*.txt);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.transcriptions))

            self.status_bar.showMessage(f"Exported to: {file_path}")
            QMessageBox.information(self, "Success", f"Exported to: {file_path}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export: {e}")

    def export_csv(self):
        """Export transcriptions to CSV file."""
        if not self.transcriptions:
            QMessageBox.warning(self, "No Data", "No transcriptions to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            str(self.current_image_path.with_suffix(".csv")) if self.current_image_path else "transcription.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        try:
            import csv
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["line_number", "text"])
                for i, text in enumerate(self.transcriptions, 1):
                    writer.writerow([i, text])

            self.status_bar.showMessage(f"Exported to: {file_path}")
            QMessageBox.information(self, "Success", f"Exported to: {file_path}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export: {e}")

    def export_xml(self):
        """Export transcriptions to PAGE XML file."""
        if not self.line_segments:
            QMessageBox.warning(self, "No Data", "No segmentation data to export. Please segment lines first.")
            return

        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "PAGE XML requires an image reference")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export PAGE XML",
            str(self.current_image_path.with_suffix(".xml")) if self.current_image_path else "transcription.xml",
            "PAGE XML Files (*.xml);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Get image dimensions
            img = Image.open(self.current_image_path)
            width, height = img.size

            # If transcriptions exist, add them to segments
            segments_to_export = self.line_segments.copy()
            if self.transcriptions and len(self.transcriptions) == len(self.line_segments):
                # Add transcriptions to segments
                for i, (seg, text) in enumerate(zip(segments_to_export, self.transcriptions)):
                    seg.text = text

            # Create exporter and export
            exporter = PageXMLExporter(str(self.current_image_path), width, height)
            exporter.export(
                segments_to_export,
                file_path,
                creator="HTR-Transcription-GUI-Plugin",
                comments=f"Engine: {self.current_engine.get_name() if self.current_engine else 'None'}"
            )

            self.status_bar.showMessage(f"Exported PAGE XML to: {file_path}")
            QMessageBox.information(self, "Success", f"Exported PAGE XML to:\n{file_path}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export PAGE XML: {e}")

    def load_settings(self):
        """Load saved settings."""
        if not self.settings_file.exists():
            return

        try:
            with open(self.settings_file, "r") as f:
                settings = json.load(f)

            # Restore last engine
            last_engine = settings.get("last_engine")
            if last_engine:
                idx = self.engine_combo.findText(last_engine)
                if idx >= 0:
                    self.engine_combo.setCurrentIndex(idx)

            # Restore engine configs
            engine_configs = settings.get("engine_configs", {})
            if self.current_engine and self.current_engine.get_name() in engine_configs:
                self.current_engine.set_config(engine_configs[self.current_engine.get_name()])

        except Exception as e:
            print(f"Warning: Failed to load settings: {e}")

    def save_settings(self):
        """Save current settings."""
        try:
            settings = {
                "last_engine": self.engine_combo.currentText(),
                "engine_configs": {}
            }

            # Save all engine configs
            for engine in available_engines:
                config = engine.get_config()
                settings["engine_configs"][engine.get_name()] = config

            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=2)

        except Exception as e:
            print(f"Warning: Failed to save settings: {e}")

    def closeEvent(self, event):
        """Handle window close."""
        # Save settings
        self.save_settings()

        # Unload models
        if self.current_engine:
            self.current_engine.unload_model()

        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("HTR Transcription Tool")

    if not available_engines:
        QMessageBox.critical(
            None,
            "No Engines Available",
            "No HTR engines found. Please install at least one engine:\n\n"
            "- TrOCR: Already included\n"
            "- Qwen3: pip install transformers accelerate peft qwen-vl-utils\n"
            "- PyLaia: See Documentation/PYLAIA_INSTALLATION_ISSUES.md\n"
            "- Commercial APIs: pip install openai google-generativeai anthropic"
        )
        sys.exit(1)

    window = TranscriptionGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
