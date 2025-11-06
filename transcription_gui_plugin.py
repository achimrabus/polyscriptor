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
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
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

# Import comparison widget
from comparison_widget import ComparisonWidget

# Import logo handler
from logo_handler import get_logo_handler

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


class StatisticsPanel(QWidget):
    """Panel displaying transcription metadata and statistics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.clear()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Model Information Group
        model_group = QGroupBox("Model Information")
        model_layout = QGridLayout()
        model_layout.setColumnStretch(1, 1)
        model_layout.setSpacing(5)

        model_layout.addWidget(QLabel("Engine:"), 0, 0)
        self.lbl_engine = QLabel("N/A")
        self.lbl_engine.setStyleSheet("font-weight: bold;")
        model_layout.addWidget(self.lbl_engine, 0, 1)

        model_layout.addWidget(QLabel("Model:"), 1, 0)
        self.lbl_model = QLabel("N/A")
        self.lbl_model.setWordWrap(True)
        self.lbl_model.setStyleSheet("font-size: 10pt;")
        model_layout.addWidget(self.lbl_model, 1, 1)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Transcription Statistics Group
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout()
        stats_layout.setColumnStretch(1, 1)
        stats_layout.setSpacing(5)

        stats_layout.addWidget(QLabel("Lines:"), 0, 0)
        self.lbl_lines = QLabel("0")
        stats_layout.addWidget(self.lbl_lines, 0, 1)

        stats_layout.addWidget(QLabel("Characters:"), 1, 0)
        self.lbl_chars = QLabel("0")
        stats_layout.addWidget(self.lbl_chars, 1, 1)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Timing Statistics Group
        timing_group = QGroupBox("Performance")
        timing_layout = QGridLayout()
        timing_layout.setColumnStretch(1, 1)
        timing_layout.setSpacing(5)

        timing_layout.addWidget(QLabel("Time:"), 0, 0)
        self.lbl_time = QLabel("0.0s")
        timing_layout.addWidget(self.lbl_time, 0, 1)

        timing_layout.addWidget(QLabel("Speed:"), 1, 0)
        self.lbl_speed = QLabel("0.0 l/s")
        timing_layout.addWidget(self.lbl_speed, 1, 1)

        timing_group.setLayout(timing_layout)
        layout.addWidget(timing_group)

        # Confidence Statistics Group
        conf_group = QGroupBox("Confidence")
        conf_layout = QGridLayout()
        conf_layout.setColumnStretch(1, 1)
        conf_layout.setSpacing(5)

        conf_layout.addWidget(QLabel("Average:"), 0, 0)
        self.lbl_avg_conf = QLabel("N/A")
        conf_layout.addWidget(self.lbl_avg_conf, 0, 1)

        conf_layout.addWidget(QLabel("Range:"), 1, 0)
        self.lbl_conf_range = QLabel("N/A")
        conf_layout.addWidget(self.lbl_conf_range, 1, 1)

        conf_layout.addWidget(QLabel("Low (<80%):"), 2, 0)
        self.lbl_low_conf = QLabel("0")
        conf_layout.addWidget(self.lbl_low_conf, 2, 1)

        conf_group.setLayout(conf_layout)
        layout.addWidget(conf_group)

        layout.addStretch()

    def clear(self):
        """Clear all statistics."""
        self.lbl_engine.setText("N/A")
        self.lbl_model.setText("N/A")
        self.lbl_lines.setText("0")
        self.lbl_chars.setText("0")
        self.lbl_time.setText("0.0s")
        self.lbl_speed.setText("0.0 l/s")
        self.lbl_avg_conf.setText("N/A")
        self.lbl_conf_range.setText("N/A")
        self.lbl_low_conf.setText("0")

    def update_statistics(self, stats: Dict[str, Any]):
        """Update statistics display with new data."""
        # Model information
        if "engine" in stats:
            self.lbl_engine.setText(stats["engine"])
        if "model_name" in stats:
            model_name = stats["model_name"]
            # Truncate long paths
            if len(model_name) > 40:
                model_name = "..." + model_name[-37:]
            self.lbl_model.setText(model_name)

        # Transcription statistics
        if "line_count" in stats:
            self.lbl_lines.setText(str(stats["line_count"]))
        if "char_count" in stats:
            self.lbl_chars.setText(f"{stats['char_count']:,}")

        # Timing statistics
        if "inference_time" in stats:
            time_val = stats["inference_time"]
            self.lbl_time.setText(f"{time_val:.2f}s")

            # Calculate speed
            line_count = stats.get("line_count", 0)
            if line_count > 0 and time_val > 0:
                speed = line_count / time_val
                self.lbl_speed.setText(f"{speed:.2f} l/s")

        # Confidence statistics
        if "avg_confidence" in stats:
            avg = stats["avg_confidence"]
            if avg is not None:
                self.lbl_avg_conf.setText(f"{avg*100:.1f}%")
                # Color code by confidence
                if avg >= 0.9:
                    color = "green"
                elif avg >= 0.75:
                    color = "orange"
                else:
                    color = "red"
                self.lbl_avg_conf.setStyleSheet(f"color: {color}; font-weight: bold;")

        if "min_confidence" in stats and "max_confidence" in stats:
            min_conf = stats["min_confidence"]
            max_conf = stats["max_confidence"]
            if min_conf is not None and max_conf is not None:
                self.lbl_conf_range.setText(f"{min_conf*100:.0f}% - {max_conf*100:.0f}%")

        if "low_confidence_lines" in stats:
            self.lbl_low_conf.setText(str(stats["low_confidence_lines"]))


class TranscriptionWorker(QThread):
    """Background worker for HTR transcription."""

    progress = pyqtSignal(int, int, str)  # current, total, text
    finished = pyqtSignal(list, dict)  # List of transcriptions, metadata dict
    error = pyqtSignal(str)

    def __init__(self, engine: HTREngine, line_segments: List[LineSegment], image: Image.Image, image_path: Optional[Path] = None):
        super().__init__()
        self.engine = engine
        self.line_segments = line_segments
        self.image = image
        self.image_path = image_path  # Store original image path for Party

    def run(self):
        """Process all line segments."""
        import time
        start_time = time.time()

        try:
            transcriptions = []
            results_with_confidence = []  # Store full results for confidence stats

            # Check if engine prefers batch processing with original image context
            # (CRITICAL for Party to correctly recognize scripts like Glagolitic)
            if hasattr(self.engine, 'prefers_batch_with_context') and self.engine.prefers_batch_with_context() and self.image_path:
                # Batch processing with original image (Party-specific)
                line_images = []
                line_bboxes = []

                for line_seg in self.line_segments:
                    x1, y1, x2, y2 = line_seg.bbox
                    line_img = self.image.crop((x1, y1, x2, y2))
                    line_array = np.array(line_img)
                    line_images.append(line_array)
                    line_bboxes.append((x1, y1, x2, y2))

                # Call batch transcription with original image context
                results = self.engine.transcribe_lines(
                    line_images,
                    config=None,
                    original_image_path=str(self.image_path),
                    line_bboxes=line_bboxes
                )

                # Extract text from results
                for i, result in enumerate(results):
                    text = str(result.text) if hasattr(result, 'text') else str(result)
                    transcriptions.append(text)
                    results_with_confidence.append(result)
                    self.progress.emit(i + 1, len(self.line_segments), text)
            else:
                # Line-by-line processing (default behavior)
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
                    results_with_confidence.append(result)
                    self.progress.emit(i + 1, len(self.line_segments), text)

            # Calculate statistics
            elapsed_time = time.time() - start_time

            # Build metadata dictionary
            metadata = {
                "inference_time": elapsed_time,
                "line_count": len(transcriptions),
                "char_count": sum(len(t) for t in transcriptions),
                "engine": self.engine.get_name() if hasattr(self.engine, 'get_name') else "Unknown"
            }

            # Extract confidence statistics if available
            confidences = []
            for result in results_with_confidence:
                if hasattr(result, 'confidence') and result.confidence is not None:
                    # Ensure confidence is a valid number (0-1 range)
                    conf = float(result.confidence)
                    if 0 <= conf <= 1:
                        confidences.append(conf)
                    elif conf > 1:
                        # If confidence is in percentage (0-100), normalize it
                        confidences.append(conf / 100.0)

            if confidences:
                metadata["avg_confidence"] = sum(confidences) / len(confidences)
                metadata["min_confidence"] = min(confidences)
                metadata["max_confidence"] = max(confidences)
                metadata["low_confidence_lines"] = sum(1 for c in confidences if c < 0.8)
            else:
                # No confidence data available
                metadata["avg_confidence"] = None
                metadata["min_confidence"] = None
                metadata["max_confidence"] = None
                metadata["low_confidence_lines"] = 0

            # Add model name from engine metadata
            if results_with_confidence and hasattr(results_with_confidence[0], 'metadata'):
                result_meta = results_with_confidence[0].metadata
                if isinstance(result_meta, dict) and 'model' in result_meta:
                    metadata["model_name"] = result_meta['model']

            self.finished.emit(transcriptions, metadata)

        except Exception as e:
            self.error.emit(str(e))


class TranscriptionGUI(QMainWindow):
    """Main GUI window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polyscriptor - Multi-Engine HTR Tool")
        self.setGeometry(100, 100, 1400, 900)

        # Set application icon
        logo_handler = get_logo_handler()
        self.setWindowIcon(logo_handler.get_icon())

        # State
        self.current_image_path: Optional[Path] = None
        self.current_image: Optional[Image.Image] = None
        self.line_segments: List[LineSegment] = []
        self.transcriptions: List[str] = []
        self.current_engine: Optional[HTREngine] = None
        self.worker: Optional[TranscriptionWorker] = None
        self.comparison_widget: Optional[ComparisonWidget] = None
        self.comparison_mode_active: bool = False

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

        # Logo display at top
        logo_handler = get_logo_handler()
        logo_label = QLabel()
        logo_pixmap = logo_handler.get_logo_pixmap(width=300)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("padding: 10px;")
        left_layout.addWidget(logo_label)

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

        # Transcription results (horizontal split: text + statistics)
        results_group = QGroupBox("Transcriptions")
        results_layout = QVBoxLayout()

        # Horizontal splitter for text and statistics
        self.results_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Transcription text
        self.text_container = QWidget()
        text_layout = QVBoxLayout(self.text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)

        self.transcription_text = QTextEdit()
        self.transcription_text.setReadOnly(True)
        text_layout.addWidget(self.transcription_text)

        self.results_splitter.addWidget(self.text_container)

        # Right: Statistics panel (compact, scrollable)
        self.stats_scroll = QScrollArea()
        self.stats_scroll.setWidgetResizable(True)
        self.stats_scroll.setMinimumWidth(200)
        self.stats_scroll.setMaximumWidth(300)

        self.stats_panel = StatisticsPanel()
        self.stats_scroll.setWidget(self.stats_panel)

        self.results_splitter.addWidget(self.stats_scroll)

        # Set initial sizes (70% text, 30% stats)
        self.results_splitter.setStretchFactor(0, 7)  # Text gets more space
        self.results_splitter.setStretchFactor(1, 3)  # Stats gets less space

        results_layout.addWidget(self.results_splitter)

        # Export and comparison buttons (below splitter)
        export_layout = QHBoxLayout()

        # Compare button (checkable - toggles comparison mode)
        self.btn_compare = QPushButton("⚖ Compare")
        self.btn_compare.setCheckable(True)
        self.btn_compare.setStyleSheet("""
            QPushButton {
                min-height: 30px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
            }
        """)
        self.btn_compare.toggled.connect(self.toggle_comparison_mode)
        export_layout.addWidget(self.btn_compare)

        export_layout.addStretch()  # Push export buttons to the right

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

    def toggle_comparison_mode(self, enabled: bool):
        """Toggle comparison mode on/off."""
        if enabled:
            # Validate prerequisites
            if not self.current_engine or not self.current_engine.is_model_loaded():
                QMessageBox.warning(self, "No Model",
                                   "Please load an engine and model first!")
                self.btn_compare.setChecked(False)
                return

            if not self.transcriptions:
                QMessageBox.warning(self, "No Transcriptions",
                                   "Please process the image first!")
                self.btn_compare.setChecked(False)
                return

            # Prepare line segments and images for comparison
            # Handle both line-based models (PyLaia, TrOCR) and page-based models (Qwen, APIs)
            if self.line_segments:
                # Line-based: use actual segments
                line_images = []
                if self.current_image:
                    img_np = np.array(self.current_image)
                    for segment in self.line_segments:
                        x, y, w, h = segment.bbox
                        line_img = img_np[y:y+h, x:x+w]
                        line_images.append(line_img)
            else:
                # Page-based: treat whole page as single "line"
                from inference_page import LineSegment
                self.line_segments = [LineSegment(
                    image=self.current_image,
                    bbox=(0, 0, self.current_image.width, self.current_image.height),
                    coords=None,
                    text=None,
                    confidence=None,
                    char_confidences=None
                )]
                line_images = [np.array(self.current_image)]

            # Create comparison widget
            self.comparison_widget = ComparisonWidget(
                self.current_engine,
                self.line_segments,
                line_images,
                self
            )

            # Connect signals
            self.comparison_widget.comparison_closed.connect(
                lambda: self.btn_compare.setChecked(False)
            )
            self.comparison_widget.status_message.connect(
                self.status_bar.showMessage
            )

            # Set base transcriptions
            self.comparison_widget.set_base_transcriptions(self.transcriptions)

            # Hide statistics panel and replace with comparison widget
            self.stats_scroll.hide()
            self.results_splitter.replaceWidget(1, self.comparison_widget)
            self.comparison_widget.show()

            # Update button text
            self.btn_compare.setText("⚖ Comparison Active")

            self.comparison_mode_active = True
            self.status_bar.showMessage("Comparison mode activated")

        else:
            # Close comparison mode
            if self.comparison_widget:
                # Clean up comparison widget
                self.comparison_widget.unload_comparison_engine()
                self.comparison_widget.hide()

                # Restore statistics panel
                self.results_splitter.replaceWidget(1, self.stats_scroll)
                self.stats_scroll.show()

                # Delete comparison widget
                self.comparison_widget.deleteLater()
                self.comparison_widget = None

            # Update button text
            self.btn_compare.setText("⚖ Compare")

            self.comparison_mode_active = False
            self.status_bar.showMessage("Comparison mode closed")

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
            self.current_image,
            self.current_image_path  # Pass original image path for Party context
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

    def on_transcription_finished(self, transcriptions: List[str], metadata: Dict[str, Any] = None):
        """Handle completion of transcription."""
        self.transcriptions = transcriptions

        # Display results (without "Line X:" prefix)
        result_text = "\n".join(transcriptions)
        self.transcription_text.setPlainText(result_text)

        # Update statistics panel
        if metadata:
            self.stats_panel.update_statistics(metadata)

        # UI updates
        self.btn_process.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Status bar message with timing
        status_msg = f"Transcription complete ({len(transcriptions)} lines)"
        if metadata and "inference_time" in metadata:
            status_msg += f" in {metadata['inference_time']:.1f}s"
        self.status_bar.showMessage(status_msg)

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
