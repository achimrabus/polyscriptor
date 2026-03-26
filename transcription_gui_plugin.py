"""
HTR Transcription GUI - Plugin-Based Version

Unified interface for multiple HTR engines using the plugin system.

Features:
- Dropdown engine selection (TrOCR, Qwen3, CRNN-CTC, Commercial APIs)
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

# Windows console defaults to CP-1252; reconfigure to UTF-8 so Cyrillic output
# and emoji from engine code don't raise UnicodeEncodeError. No-op on Linux/macOS.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 — reconfigure not available
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image

# Disable PIL DecompressionBomb protection for large manuscript images
Image.MAX_IMAGE_PIXELS = None

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSplitter, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QPushButton, QLabel, QTextEdit, QLineEdit, QComboBox,
    QFileDialog, QProgressBar, QStatusBar, QMessageBox,
    QListWidget, QListWidgetItem, QGroupBox, QScrollArea, QSlider, QSpinBox, QCheckBox,
    QFontDialog, QToolButton, QSizePolicy, QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QSettings
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QAction, QShortcut, QKeySequence, QPalette

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


class CollapsibleSection(QWidget):
    """Section with a clickable header that collapses/expands its content."""

    def __init__(self, title, expanded=True, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header button (full-width, checkable, with arrow indicator)
        self._header = QToolButton()
        self._header.setText(title)
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._header.setStyleSheet(
            "QToolButton { font-weight: bold; border: 1px solid palette(mid); "
            "border-radius: 3px; padding: 4px 8px; background: palette(button); "
            "color: palette(buttonText); }"
            "QToolButton:hover { background: palette(midlight); }"
        )
        self._header.toggled.connect(self._on_toggled)
        layout.addWidget(self._header)

        # Content area
        self._content = QWidget()
        self.content_layout = QVBoxLayout(self._content)
        self.content_layout.setContentsMargins(2, 4, 2, 4)
        self._content.setVisible(expanded)
        layout.addWidget(self._content)

    def _on_toggled(self, checked):
        self._content.setVisible(checked)
        self._header.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        # Notify parent layout that our size hint changed
        self.updateGeometry()

    def set_expanded(self, expanded):
        self._header.setChecked(expanded)

    def is_expanded(self):
        return self._header.isChecked()


class ZoomableGraphicsView(QGraphicsView):
    """Graphics view with smooth zoom and pan capabilities."""

    rects_changed = pyqtSignal(int)   # emits len(drawn_rects) after add/clear
    escape_pressed = pyqtSignal()

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

        # Draw mode state
        self.draw_mode: bool = False
        self.drawn_rects: List[Tuple[int, int, int, int]] = []
        self.drawn_rect_items: List = []
        self._drag_start = None    # QPointF (scene coords) while dragging
        self._drag_item = None     # live QGraphicsRectItem during drag

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
        self.drawn_rect_items = []
        self.drawn_rects = []
        self._drag_start = None
        self._drag_item = None
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

    def set_draw_mode(self, enabled: bool):
        """Toggle rectangle draw mode on/off."""
        self.draw_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            if self._drag_item is not None:
                self._scene.removeItem(self._drag_item)
                self._drag_item = None
            self._drag_start = None
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def clear_drawn_rects(self):
        """Remove all drawn rectangles from the scene."""
        for item in self.drawn_rect_items:
            self._scene.removeItem(item)
        self.drawn_rect_items = []
        self.drawn_rects = []
        if self._drag_item is not None:
            self._scene.removeItem(self._drag_item)
            self._drag_item = None
        self.rects_changed.emit(0)

    def mousePressEvent(self, event):
        if self.draw_mode and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = self.mapToScene(event.pos())
            pen = QPen(QColor(180, 0, 0))
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            self._drag_item = self._scene.addRect(
                self._drag_start.x(), self._drag_start.y(), 0, 0, pen
            )
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.draw_mode and self._drag_start is not None:
            pos = self.mapToScene(event.pos())
            x1 = min(self._drag_start.x(), pos.x())
            y1 = min(self._drag_start.y(), pos.y())
            x2 = max(self._drag_start.x(), pos.x())
            y2 = max(self._drag_start.y(), pos.y())
            self._drag_item.setRect(x1, y1, x2 - x1, y2 - y1)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.draw_mode and event.button() == Qt.MouseButton.LeftButton:
            if self._drag_start is not None:
                pos = self.mapToScene(event.pos())
                x1 = int(min(self._drag_start.x(), pos.x()))
                y1 = int(min(self._drag_start.y(), pos.y()))
                x2 = int(max(self._drag_start.x(), pos.x()))
                y2 = int(max(self._drag_start.y(), pos.y()))
                if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                    self.drawn_rects.append((x1, y1, x2, y2))
                    self.drawn_rect_items.append(self._drag_item)
                    self.rects_changed.emit(len(self.drawn_rects))
                else:
                    self._scene.removeItem(self._drag_item)
            self._drag_start = None
            self._drag_item = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if self.draw_mode and event.key() == Qt.Key.Key_Escape:
            self.escape_pressed.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


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
        self.transcription_engine: Optional[HTREngine] = None  # engine that produced self.transcriptions
        self.worker: Optional[TranscriptionWorker] = None
        self.comparison_widget: Optional[ComparisonWidget] = None
        self.comparison_mode_active: bool = False
        self.regions: list = []  # SegRegion list from blla segmentation

        # Multi-page navigation state
        self.image_list: List[Path] = []
        self.current_image_index: int = -1

        # Cache config widgets to prevent deletion
        self.config_widgets_cache: Dict[str, QWidget] = {}

        self.setup_ui()
        self.restore_settings()

    def setup_ui(self):
        """Setup user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main container layout
        container_layout = QVBoxLayout(central_widget)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # Top-level splitter: Image panel (left) ↔ Controls+Results (right)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(4)
        self.main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: palette(mid);
            }
            QSplitter::handle:hover {
                background-color: palette(midlight);
            }
        """)
        container_layout.addWidget(self.main_splitter)

        # Left panel: Image view
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Logo display at top (smaller on FHD to save space)
        logo_handler = get_logo_handler()
        logo_label = QLabel()
        screen_height = QApplication.primaryScreen().availableGeometry().height()
        logo_width = 200 if screen_height < 1200 else 300
        logo_pixmap = logo_handler.get_logo_pixmap(width=logo_width)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("padding: 5px;" if screen_height < 1200 else "padding: 10px;")
        left_layout.addWidget(logo_label)

        # Image view
        self.image_view = ZoomableGraphicsView()
        left_layout.addWidget(self.image_view)

        # Image controls - row 1: load and segmentation
        img_controls = QHBoxLayout()
        btn_load = QPushButton("Load Images...")
        btn_load.clicked.connect(self._load_images)
        btn_load.setToolTip("Select one or more images to process (Ctrl+O)")
        QShortcut(QKeySequence("Ctrl+O"), self, self._load_images)
        img_controls.addWidget(btn_load)

        btn_segment = QPushButton("Segment Lines")
        btn_segment.clicked.connect(self.segment_lines)
        img_controls.addWidget(btn_segment)

        btn_fit = QPushButton("Fit to View")
        btn_fit.clicked.connect(self.image_view.fit_in_view)
        img_controls.addWidget(btn_fit)

        left_layout.addLayout(img_controls)

        # Draw mode controls: rectangle drawing for manual re-segmentation
        # Hidden until blla segmentation populates self.regions
        self.draw_controls_widget = QWidget()
        draw_controls = QHBoxLayout(self.draw_controls_widget)
        draw_controls.setContentsMargins(0, 0, 0, 0)

        self.btn_draw_mode = QPushButton("Draw Regions")
        self.btn_draw_mode.setCheckable(True)
        self.btn_draw_mode.setToolTip(
            "Toggle draw mode: click and drag to draw rectangles on the image.\n"
            "Each rectangle marks an area to re-segment with blla.\n"
            "Press Escape or click again to exit draw mode.")
        draw_controls.addWidget(self.btn_draw_mode)

        self.btn_resegment_drawn = QPushButton("Re-segment Drawn")
        self.btn_resegment_drawn.setEnabled(False)
        self.btn_resegment_drawn.setToolTip(
            "Run blla segmentation within each drawn rectangle.\n"
            "Replaces existing regions that overlap the drawn areas.")
        draw_controls.addWidget(self.btn_resegment_drawn)

        btn_clear_drawn = QPushButton("Clear Drawn")
        btn_clear_drawn.setToolTip("Remove all drawn rectangles.")
        draw_controls.addWidget(btn_clear_drawn)

        self.draw_controls_widget.setVisible(False)
        left_layout.addWidget(self.draw_controls_widget)

        self.btn_draw_mode.toggled.connect(self._toggle_draw_mode)
        self.btn_resegment_drawn.clicked.connect(self._resegment_drawn_regions)
        btn_clear_drawn.clicked.connect(self._clear_drawn_regions)
        self.image_view.rects_changed.connect(
            lambda n: self.btn_resegment_drawn.setEnabled(n > 0))
        self.image_view.escape_pressed.connect(
            lambda: self.btn_draw_mode.setChecked(False))

        # Image navigation - row 2: prev/next buttons and page counter
        nav_controls = QHBoxLayout()

        self.btn_prev_image = QPushButton("< Prev")
        self.btn_prev_image.clicked.connect(self._load_previous_image)
        self.btn_prev_image.setEnabled(False)
        self.btn_prev_image.setToolTip("Previous image (PageUp)")
        nav_controls.addWidget(self.btn_prev_image)

        self.lbl_image_index = QLabel("No images loaded")
        self.lbl_image_index.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image_index.setMinimumWidth(100)
        nav_controls.addWidget(self.lbl_image_index, stretch=1)

        self.btn_next_image = QPushButton("Next >")
        self.btn_next_image.clicked.connect(self._load_next_image)
        self.btn_next_image.setEnabled(False)
        self.btn_next_image.setToolTip("Next image (PageDown)")
        nav_controls.addWidget(self.btn_next_image)

        left_layout.addLayout(nav_controls)

        # Add left panel to main splitter
        self.main_splitter.addWidget(left_panel)

        # Right panel: Controls and Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # --- Segmentation Settings (collapsible) ---
        self.seg_section = CollapsibleSection("Segmentation Settings")
        seg_layout = self.seg_section.content_layout

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
            self.seg_method_combo.addItem("Kraken Classical", "Kraken")
            self.seg_method_combo.addItem("Kraken Neural (blla)", "KrakenBLLA")
            self.seg_method_combo.setCurrentIndex(1)  # Default to Kraken Classical if available
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

        # Kraken Neural (blla) parameters
        self.blla_params_widget = QWidget()
        blla_layout = QVBoxLayout()
        blla_layout.setContentsMargins(0, 0, 0, 0)

        blla_device_layout = QHBoxLayout()
        blla_device_layout.addWidget(QLabel("Device:"))
        self.blla_device_combo = QComboBox()
        self.blla_device_combo.addItem("CPU", "cpu")
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    self.blla_device_combo.addItem(f"GPU {i}: {name}", f"cuda:{i}")
                # Default to first GPU (only if GPUs were actually added)
                if torch.cuda.device_count() > 0:
                    self.blla_device_combo.setCurrentIndex(1)
        except ImportError:
            pass
        blla_device_layout.addWidget(self.blla_device_combo)
        blla_layout.addLayout(blla_device_layout)

        # Max columns
        blla_cols_layout = QHBoxLayout()
        blla_cols_layout.addWidget(QLabel("Max Columns:"))
        self.blla_max_columns_spin = QSpinBox()
        self.blla_max_columns_spin.setRange(1, 8)
        self.blla_max_columns_spin.setValue(4)
        self.blla_max_columns_spin.setToolTip(
            "Maximum number of text columns to detect per region (1-8).\n"
            "Set to 2 for single-page two-column, 4 for double-page two-column.")
        blla_cols_layout.addWidget(self.blla_max_columns_spin)
        blla_cols_layout.addStretch()
        blla_layout.addLayout(blla_cols_layout)

        # Split width threshold
        blla_split_layout = QHBoxLayout()
        blla_split_layout.addWidget(QLabel("Split Width:"))
        self.blla_split_slider = QSlider(Qt.Orientation.Horizontal)
        self.blla_split_slider.setRange(10, 80)  # 10% to 80%
        self.blla_split_slider.setValue(40)       # default 40%
        self.blla_split_label = QLabel("40%")
        self.blla_split_slider.valueChanged.connect(
            lambda v: self.blla_split_label.setText(f"{v}%")
        )
        self.blla_split_slider.setToolTip(
            "Regions wider than this % of page width get sub-split into columns.\n"
            "Lower = more aggressive splitting.\n"
            "Portrait single-page: 40% (default)\n"
            "Landscape double-page: try 20-25%")
        blla_split_layout.addWidget(self.blla_split_slider)
        blla_split_layout.addWidget(self.blla_split_label)
        blla_layout.addLayout(blla_split_layout)

        # Min lines to split
        blla_minlines_layout = QHBoxLayout()
        blla_minlines_layout.addWidget(QLabel("Min Lines:"))
        self.blla_min_lines_spin = QSpinBox()
        self.blla_min_lines_spin.setRange(3, 50)
        self.blla_min_lines_spin.setValue(10)
        self.blla_min_lines_spin.setToolTip(
            "Minimum lines in a region before attempting column split.\n"
            "Lower = split regions with fewer lines.")
        blla_minlines_layout.addWidget(self.blla_min_lines_spin)
        blla_minlines_layout.addStretch()
        blla_layout.addLayout(blla_minlines_layout)

        # Text direction (reading order)
        blla_dir_layout = QHBoxLayout()
        blla_dir_layout.addWidget(QLabel("Direction:"))
        self.blla_direction_combo = QComboBox()
        self.blla_direction_combo.addItem("LTR — Latin, Cyrillic, …", "horizontal-lr")
        self.blla_direction_combo.addItem("RTL — Arabic, Ottoman, Hebrew, …", "horizontal-rl")
        self.blla_direction_combo.addItem("Vertical LTR", "vertical-lr")
        self.blla_direction_combo.addItem("Vertical RTL", "vertical-rl")
        self.blla_direction_combo.setToolTip(
            "Controls column reading order.\n"
            "Use RTL for Arabic, Ottoman Turkish, Hebrew manuscripts.\n"
            "Affects which column is listed first in the output.")
        blla_dir_layout.addWidget(self.blla_direction_combo)
        blla_layout.addLayout(blla_dir_layout)

        # Custom model path
        blla_model_layout = QHBoxLayout()
        blla_model_layout.addWidget(QLabel("Model:"))
        self.blla_model_edit = QLineEdit()
        self.blla_model_edit.setPlaceholderText("Default (pagexml/blla.mlmodel)")
        self.blla_model_edit.setToolTip(
            "Path to a custom kraken blla .mlmodel segmentation file.\n"
            "Leave blank to use the built-in default (pagexml/blla.mlmodel).\n"
            "Any kraken blla-compatible model can be used here.")
        blla_model_btn = QPushButton("Browse…")
        blla_model_btn.clicked.connect(self._browse_blla_model)
        blla_model_layout.addWidget(self.blla_model_edit)
        blla_model_layout.addWidget(blla_model_btn)
        blla_layout.addLayout(blla_model_layout)

        self.blla_params_widget.setLayout(blla_layout)
        seg_layout.addWidget(self.blla_params_widget)

        # Set initial visibility based on default method
        self._on_seg_method_changed(self.seg_method_combo.currentIndex())

        right_layout.addWidget(self.seg_section)

        # --- HTR Engine (collapsible) ---
        self.engine_section = CollapsibleSection("HTR Engine", expanded=False)
        engine_content = self.engine_section.content_layout

        self.engine_combo = QComboBox()
        if not available_engines:
            self.engine_combo.addItem("No engines available")
        else:
            for engine in available_engines:
                self.engine_combo.addItem(engine.get_name())

        self.engine_combo.currentTextChanged.connect(self.on_engine_changed)
        engine_content.addWidget(self.engine_combo)

        # Engine description
        self.engine_desc_label = QLabel("")
        self.engine_desc_label.setWordWrap(True)
        self.engine_desc_label.setStyleSheet("color: gray; font-size: 12pt;")
        engine_content.addWidget(self.engine_desc_label)

        # Dynamic engine configuration container
        # Wrapped in a scroll area so tall configs don't push export buttons off-screen.
        # No minimum height (collapses if empty); max height caps growth.
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setFrameShape(config_scroll.Shape.NoFrame)

        self.config_container = QWidget()
        self.config_layout = QVBoxLayout(self.config_container)
        self.config_layout.addStretch()
        config_scroll.setWidget(self.config_container)
        engine_content.addWidget(config_scroll)

        right_layout.addWidget(self.engine_section)

        # Load/Process buttons — outside collapsible section so always visible
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

        # Transcription results — splitter fills all remaining space
        self.results_group = QGroupBox("Transcriptions")
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

        # Right: Statistics panel (hidden by default on FHD, toggle via button below)
        self.stats_scroll = QScrollArea()
        self.stats_scroll.setWidgetResizable(True)
        self.stats_scroll.setMinimumWidth(180)

        self.stats_panel = StatisticsPanel()
        self.stats_scroll.setWidget(self.stats_panel)

        self.results_splitter.addWidget(self.stats_scroll)

        self.transcription_text.setMinimumWidth(400)
        self.results_splitter.setCollapsible(0, False)
        self.results_splitter.setCollapsible(1, True)

        # Stats panel hidden by default; user clicks Stats button to show
        self.results_splitter.setSizes([1, 0])

        results_layout.addWidget(self.results_splitter)
        self.results_group.setLayout(results_layout)

        # Wrap results_group in a QStackedWidget so comparison can take over the full area
        self.results_stack = QStackedWidget()
        self.results_stack.addWidget(self.results_group)  # page 0: normal view
        # comparison widget will be added as page 1 dynamically
        right_layout.addWidget(self.results_stack, stretch=1)

        # When engine section expands, hide results area (gives config room to breathe)
        self.engine_section._header.toggled.connect(
            lambda checked: self.results_stack.setVisible(not checked)
        )
        # Set initial state to match engine section (collapsed → results visible)
        self.results_stack.setVisible(not self.engine_section.is_expanded())

        # Export + comparison + stats toggle row — pinned at the very bottom of right_layout
        # (outside results_group so it is always visible regardless of section heights)
        export_layout = QHBoxLayout()

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

        export_layout.addStretch()

        btn_export_txt = QPushButton("Export TXT")
        btn_export_txt.clicked.connect(self.export_txt)
        export_layout.addWidget(btn_export_txt)

        btn_export_csv = QPushButton("Export CSV")
        btn_export_csv.clicked.connect(self.export_csv)
        export_layout.addWidget(btn_export_csv)

        btn_export_xml = QPushButton("Export PAGE XML")
        btn_export_xml.clicked.connect(self.export_xml)
        export_layout.addWidget(btn_export_xml)

        self.btn_stats_toggle = QPushButton("Stats")
        self.btn_stats_toggle.setCheckable(True)
        self.btn_stats_toggle.setChecked(False)  # always off at startup
        self.btn_stats_toggle.setToolTip("Show/hide statistics panel")
        self.btn_stats_toggle.toggled.connect(self._toggle_stats_panel)
        export_layout.addWidget(self.btn_stats_toggle)

        right_layout.addLayout(export_layout)

        # Add right panel to main splitter
        self.main_splitter.addWidget(right_panel)

        # Set minimum widths to prevent crushing
        left_panel.setMinimumWidth(400)   # Image needs at least 400px
        right_panel.setMinimumWidth(600)  # Controls need at least 600px

        # Set initial sizes (40% image, 60% controls+results at FHD 1920px)
        # At 1920px width: 768px image, 1152px controls
        self.main_splitter.setSizes([768, 1152])

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Menu bar
        self.setup_menu_bar()

        # Initialize first engine
        if available_engines:
            self.on_engine_changed(self.engine_combo.currentText())

        # Auto-collapse seg settings on FHD (< 1200px height)
        # engine_section always starts collapsed (expanding hides transcription area)
        if screen_height < 1200:
            self.seg_section.set_expanded(False)

    def setup_menu_bar(self):
        """Setup menu bar with view presets."""
        menu_bar = self.menuBar()

        # View menu
        view_menu = menu_bar.addMenu("&View")

        # Layout presets submenu
        layout_menu = view_menu.addMenu("Layout Presets")

        # Default layout (40/60)
        preset_default = QAction("Default (40/60)", self)
        preset_default.setShortcut("Ctrl+1")
        preset_default.setStatusTip("Balanced image/text layout")
        preset_default.triggered.connect(lambda: self.apply_layout_preset(768, 1152, "Default"))
        layout_menu.addAction(preset_default)

        # Image focus (60/40)
        preset_image = QAction("Image Focus (60/40)", self)
        preset_image.setShortcut("Ctrl+2")
        preset_image.setStatusTip("Large image for detailed inspection")
        preset_image.triggered.connect(lambda: self.apply_layout_preset(1152, 768, "Image Focus"))
        layout_menu.addAction(preset_image)

        # Transcription focus (25/75)
        preset_text = QAction("Transcription Focus (25/75)", self)
        preset_text.setShortcut("Ctrl+3")
        preset_text.setStatusTip("Large text area for editing")
        preset_text.triggered.connect(lambda: self.apply_layout_preset(480, 1440, "Transcription Focus"))
        layout_menu.addAction(preset_text)

    def apply_layout_preset(self, image_width: int, controls_width: int, preset_name: str):
        """Apply predefined layout preset."""
        try:
            self.main_splitter.setSizes([image_width, controls_width])
            self.status_bar.showMessage(f"{preset_name} layout applied", 2000)
        except Exception as e:
            print(f"Error applying layout preset: {e}")

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
            if not self.transcriptions:
                QMessageBox.warning(self, "No Transcriptions",
                                   "Please process the image first!")
                self.btn_compare.setChecked(False)
                return

            if not self.transcription_engine:
                QMessageBox.warning(self, "No Base Engine",
                                   "Cannot determine which engine produced the transcription.")
                self.btn_compare.setChecked(False)
                return

            # Prepare line segments and images for comparison
            # Handle both line-based models (CRNN-CTC, TrOCR) and page-based models (Qwen, APIs)
            if self.line_segments:
                # Line-based: use pre-cropped images from segments (already properly cropped)
                line_images = []
                for segment in self.line_segments:
                    if segment.image is not None:
                        line_images.append(np.array(segment.image))
                    elif self.current_image:
                        # Fallback: crop from full image using (x1,y1,x2,y2) bbox
                        x1, y1, x2, y2 = segment.bbox
                        img_np = np.array(self.current_image)
                        line_images.append(img_np[y1:y2, x1:x2])
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

            # Create comparison widget — use transcription_engine (the engine that
            # produced self.transcriptions), not current_engine (which may have changed)
            self.comparison_widget = ComparisonWidget(
                self.transcription_engine,
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

            # Add comparison widget to stack and switch to it (full content area)
            self.results_stack.addWidget(self.comparison_widget)
            self.results_stack.setCurrentWidget(self.comparison_widget)
            self.btn_stats_toggle.setEnabled(False)

            # Update button text
            self.btn_compare.setText("⚖ Comparison Active")

            self.comparison_mode_active = True
            self.status_bar.showMessage("Comparison mode activated")

        else:
            # Close comparison mode
            if self.comparison_widget:
                # Terminate worker if still running (closed via main button, not ✕)
                if (self.comparison_widget.comparison_worker and
                        self.comparison_widget.comparison_worker.isRunning()):
                    self.comparison_widget.comparison_worker.terminate()
                    self.comparison_widget.comparison_worker.wait()
                self.comparison_widget.unload_comparison_engine()

                # Switch back to normal view and remove comparison from stack
                self.results_stack.setCurrentWidget(self.results_group)
                self.results_stack.removeWidget(self.comparison_widget)
                self.comparison_widget.deleteLater()
                self.comparison_widget = None

            self.btn_stats_toggle.setEnabled(True)
            self.btn_compare.setText("⚖ Compare")
            self.comparison_mode_active = False
            self.status_bar.showMessage("Comparison mode closed")

    def _toggle_stats_panel(self, show: bool):
        """Show or hide the statistics panel in the results splitter."""
        if show:
            self.results_splitter.setSizes([700, 250])
        else:
            self.results_splitter.setSizes([1, 0])

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

    def _load_images(self):
        """Open file dialog to select one or more images or PDFs."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images or PDFs",
            "",
            "Images & PDFs (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.pdf);;"
            "PDF Files (*.pdf);;"
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;"
            "All Files (*)"
        )

        if not file_paths:
            return

        expanded = []
        for p_str in file_paths:
            p = Path(p_str)
            if p.suffix.lower() == '.pdf':
                try:
                    import fitz as _fitz
                    import tempfile
                    doc = _fitz.open(str(p))
                    n = len(doc)
                    doc.close()
                    load_all = True
                    if n > 1:
                        reply = QMessageBox.question(
                            self, "PDF",
                            f"'{p.name}' has {n} pages. Load all pages?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                        )
                        load_all = (reply == QMessageBox.StandardButton.Yes)
                    tmp = Path(tempfile.mkdtemp(prefix="polyscriptor_pdf_"))
                    mat = _fitz.Matrix(150 / 72, 150 / 72)
                    doc = _fitz.open(str(p))
                    pages_to_render = range(len(doc)) if load_all else range(1)
                    for i in pages_to_render:
                        page = doc[i]
                        pix = page.get_pixmap(matrix=mat, colorspace=_fitz.csRGB)
                        from PIL import Image as _PIL
                        img = _PIL.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        out = tmp / f"{p.stem}_page{i+1:03d}.png"
                        img.save(str(out))
                        expanded.append(out)
                    doc.close()
                except ImportError:
                    QMessageBox.warning(self, "PDF Error",
                        "PyMuPDF not installed. Install with:\npip install pymupdf")
                except Exception as e:
                    QMessageBox.warning(self, "PDF Error", f"Could not open PDF: {e}")
            else:
                expanded.append(p)

        if not expanded:
            return

        self.image_list = expanded
        self.current_image_index = 0
        self._display_current_image()
        self._update_navigation_ui()

    def _display_current_image(self):
        """Display the image at current_image_index."""
        if not self.image_list or self.current_image_index < 0:
            return

        file_path = self.image_list[self.current_image_index]

        try:
            self.current_image_path = file_path
            self.current_image = Image.open(file_path).convert("RGB")

            # Display image
            img_array = np.array(self.current_image)
            height, width = img_array.shape[:2]
            bytes_per_line = 3 * width
            q_img = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            self.image_view.set_image(pixmap)

            self.status_bar.showMessage(f"Loaded: {file_path.name}")
            self.line_segments = []
            self.regions = []
            self.transcriptions = []
            self.transcription_text.clear()
            self.stats_panel.clear()

            # Update button state after loading image
            self.update_process_button_state()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image: {e}")

    def _load_previous_image(self):
        """Navigate to the previous image in the list."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self._display_current_image()
            self._update_navigation_ui()

    def _load_next_image(self):
        """Navigate to the next image in the list."""
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self._display_current_image()
            self._update_navigation_ui()

    def _update_navigation_ui(self):
        """Update navigation buttons and page counter label."""
        if not self.image_list:
            self.lbl_image_index.setText("No images loaded")
            self.btn_prev_image.setEnabled(False)
            self.btn_next_image.setEnabled(False)
            return

        total = len(self.image_list)
        current = self.current_image_index + 1

        self.lbl_image_index.setText(f"{current} / {total}")
        self.btn_prev_image.setEnabled(self.current_image_index > 0)
        self.btn_next_image.setEnabled(self.current_image_index < total - 1)

    def load_image(self):
        """Legacy method - redirects to _load_images for backwards compatibility."""
        self._load_images()

    def _browse_blla_model(self):
        """Open file dialog to select a custom blla .mlmodel file."""
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Segmentation Model", "",
            "Kraken Models (*.mlmodel);;All Files (*)"
        )
        if path:
            self.blla_model_edit.setText(path)

    def _on_seg_method_changed(self, index):
        """Handle segmentation method change."""
        method = self.seg_method_combo.currentData()

        # Hide all parameter widgets first
        self.hpp_params_widget.setVisible(False)
        self.kraken_params_widget.setVisible(False)
        if hasattr(self, 'blla_params_widget'):
            self.blla_params_widget.setVisible(False)

        # Show/hide draw controls based on whether blla is selected
        if hasattr(self, 'draw_controls_widget'):
            self.draw_controls_widget.setVisible(method == "KrakenBLLA")

        if method == "Kraken":
            self.kraken_params_widget.setVisible(True)
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage("Switched to Kraken Classical segmentation")
        elif method == "KrakenBLLA":
            if hasattr(self, 'blla_params_widget'):
                self.blla_params_widget.setVisible(True)
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage("Switched to Kraken Neural (blla) — supports multi-column layouts")
        else:  # HPP
            self.hpp_params_widget.setVisible(True)
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage("Switched to HPP segmentation (fast)")

    def segment_lines(self):
        """Segment lines in the current image."""
        if self.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        try:
            method = self.seg_method_combo.currentData()

            if method == "KrakenBLLA":
                # Use Kraken Neural (blla) segmentation — regions + baselines
                self.status_bar.showMessage("Detecting regions & lines with Kraken Neural (blla)...")
                QApplication.processEvents()

                from kraken_segmenter import KrakenLineSegmenter
                from inference_page import LineSegment, sort_lines_by_region

                device = self.blla_device_combo.currentData() if hasattr(self, 'blla_device_combo') else 'cpu'
                segmenter = KrakenLineSegmenter(device=device)
                max_cols = self.blla_max_columns_spin.value() if hasattr(self, 'blla_max_columns_spin') else 4
                split_frac = (self.blla_split_slider.value() / 100.0) if hasattr(self, 'blla_split_slider') else 0.40
                min_lines = self.blla_min_lines_spin.value() if hasattr(self, 'blla_min_lines_spin') else 10
                custom_model = self.blla_model_edit.text().strip() if hasattr(self, 'blla_model_edit') else None
                text_dir = self.blla_direction_combo.currentData() if hasattr(self, 'blla_direction_combo') else 'horizontal-lr'
                regions, kraken_lines = segmenter.segment_with_regions(
                    self.current_image, device=device,
                    model_path=custom_model or None,
                    max_columns=max_cols,
                    split_width_fraction=split_frac,
                    min_lines_to_split=min_lines,
                    text_direction=text_dir,
                )

                # Store regions for later use (visualization, export)
                self.regions = regions

                # Convert kraken LineSegments to inference_page LineSegment format
                self.line_segments = []
                for seg in kraken_lines:
                    self.line_segments.append(LineSegment(
                        image=seg.image,
                        bbox=seg.bbox,
                        coords=seg.baseline,
                    ))

                num_regions = len(regions)
                num_lines = len(self.line_segments)
                self.status_bar.showMessage(
                    f"blla: {num_regions} region(s), {num_lines} lines"
                )

            elif method == "Kraken":
                # Use Kraken Classical segmentation
                self.status_bar.showMessage("Detecting lines with Kraken Classical (this may take 3-8 seconds)...")
                QApplication.processEvents()

                from kraken_segmenter import KrakenLineSegmenter

                use_binarization = self.use_binarization_check.isChecked()
                segmenter = KrakenLineSegmenter()
                kraken_segments = segmenter.segment_lines(
                    self.current_image,
                    text_direction='horizontal-lr',
                    use_binarization=use_binarization
                )

                self.regions = []  # Classical has no regions

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
                self.regions = []  # HPP has no regions
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

            # Draw boxes (color-code by region if blla was used)
            if method == "KrakenBLLA" and self.regions:
                self._draw_region_line_boxes()
            else:
                self.image_view.draw_line_boxes(self.line_segments)

            num_lines = len(self.line_segments)
            method_display = {"HPP": "HPP", "Kraken": "Kraken Classical", "KrakenBLLA": "Kraken Neural (blla)"}.get(method, method)
            self.status_bar.showMessage(f"Found {num_lines} lines with {method_display}")

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
    
    def _draw_region_line_boxes(self):
        """Draw color-coded line boxes grouped by region after blla segmentation."""
        # Region color palette
        region_colors = [
            QColor(0, 200, 0),      # green
            QColor(0, 100, 255),     # blue
            QColor(255, 128, 0),     # orange
            QColor(180, 0, 180),     # purple
            QColor(255, 0, 0),       # red
        ]

        # Remove old line items
        for item in self.image_view.line_items:
            self.image_view._scene.removeItem(item)
        self.image_view.line_items = []

        # Build region-to-color mapping and assign lines to regions by bbox containment
        line_idx = 0
        for ri, region in enumerate(self.regions):
            color = region_colors[ri % len(region_colors)]
            pen = QPen(color)
            pen.setWidth(2)

            # Draw region polygon/bbox
            if region.polygon and len(region.polygon) >= 3:
                region_pen = QPen(color)
                region_pen.setWidth(3)
                region_pen.setStyle(Qt.PenStyle.DashLine)
                for i in range(len(region.polygon)):
                    p1 = region.polygon[i]
                    p2 = region.polygon[(i + 1) % len(region.polygon)]
                    line_item = self.image_view._scene.addLine(
                        p1[0], p1[1], p2[0], p2[1], region_pen
                    )
                    self.image_view.line_items.append(line_item)
            else:
                region_pen = QPen(color)
                region_pen.setWidth(3)
                region_pen.setStyle(Qt.PenStyle.DashLine)
                x1, y1, x2, y2 = region.bbox
                rect_item = self.image_view._scene.addRect(
                    x1, y1, x2 - x1, y2 - y1, region_pen
                )
                self.image_view.line_items.append(rect_item)

            # Draw lines belonging to this region
            n_lines = len(region.line_ids)
            for _ in range(n_lines):
                if line_idx < len(self.line_segments):
                    seg = self.line_segments[line_idx]
                    x1, y1, x2, y2 = seg.bbox
                    rect_item = self.image_view._scene.addRect(
                        x1, y1, x2 - x1, y2 - y1, pen
                    )
                    self.image_view.line_items.append(rect_item)
                    line_idx += 1

        # Any remaining lines (shouldn't happen, but be safe)
        default_pen = QPen(QColor(0, 255, 0))
        default_pen.setWidth(2)
        while line_idx < len(self.line_segments):
            seg = self.line_segments[line_idx]
            x1, y1, x2, y2 = seg.bbox
            rect_item = self.image_view._scene.addRect(
                x1, y1, x2 - x1, y2 - y1, default_pen
            )
            self.image_view.line_items.append(rect_item)
            line_idx += 1

    def _toggle_draw_mode(self, checked: bool):
        """Enable or disable rectangle draw mode on the image view."""
        self.image_view.set_draw_mode(checked)

    def _clear_drawn_regions(self):
        """Remove all drawn rectangles from the image view."""
        self.image_view.clear_drawn_rects()

    def _resegment_drawn_regions(self):
        """Re-run blla segmentation inside each drawn rectangle, replacing overlapping regions."""
        drawn = self.image_view.drawn_rects
        if not drawn or self.current_image is None:
            return

        from inference_page import LineSegment as InfLineSegment

        try:
            from kraken_segmenter import KrakenLineSegmenter
        except ImportError:
            QMessageBox.warning(self, "Not Available", "Kraken segmenter not installed.")
            return

        W, H = self.current_image.size

        def _overlap(b1, b2):
            return not (b1[2] <= b2[0] or b2[2] <= b1[0] or
                        b1[3] <= b2[1] or b2[3] <= b1[1])

        # Build (region, lines) pairs for KEPT regions (no overlap with any drawn rect)
        kept_pairs = []
        line_offset = 0
        for region in self.regions:
            n = len(region.line_ids)
            region_lines = list(self.line_segments[line_offset:line_offset + n])
            line_offset += n
            if not any(_overlap(region.bbox, dr) for dr in drawn):
                kept_pairs.append((region, region_lines))

        # Get segmenter params (same guards as segment_lines)
        device = self.blla_device_combo.currentData() if hasattr(self, 'blla_device_combo') else 'cpu'
        max_cols = self.blla_max_columns_spin.value() if hasattr(self, 'blla_max_columns_spin') else 4
        split_frac = (self.blla_split_slider.value() / 100.0) if hasattr(self, 'blla_split_slider') else 0.40
        min_lines = self.blla_min_lines_spin.value() if hasattr(self, 'blla_min_lines_spin') else 10
        custom_model = self.blla_model_edit.text().strip() if hasattr(self, 'blla_model_edit') else None
        text_dir = self.blla_direction_combo.currentData() if hasattr(self, 'blla_direction_combo') else 'horizontal-lr'

        segmenter = KrakenLineSegmenter(device=device)
        new_pairs = []

        for ri, (dx1, dy1, dx2, dy2) in enumerate(drawn):
            dx1, dy1 = max(0, dx1), max(0, dy1)
            dx2, dy2 = min(W, dx2), min(H, dy2)
            if dx2 <= dx1 or dy2 <= dy1:
                continue

            self.status_bar.showMessage(f"Re-segmenting drawn rect {ri + 1}/{len(drawn)}...")
            QApplication.processEvents()

            crop = self.current_image.crop((dx1, dy1, dx2, dy2))
            try:
                seg_regions, seg_lines = segmenter.segment_with_regions(
                    crop, device=device,
                    model_path=custom_model or None,
                    max_columns=max_cols,
                    split_width_fraction=split_frac,
                    min_lines_to_split=min_lines,
                    text_direction=text_dir,
                )
            except Exception as e:
                self.status_bar.showMessage(f"Error re-segmenting rect {ri + 1}: {e}")
                continue

            # Adjust all coordinates by (+dx1, +dy1) to convert crop → page space
            for sr in seg_regions:
                sr.id = f"r_draw_{ri}_{sr.id}"
                bx1, by1, bx2, by2 = sr.bbox
                sr.bbox = (bx1 + dx1, by1 + dy1, bx2 + dx1, by2 + dy1)
                if sr.polygon:
                    sr.polygon = [(px + dx1, py + dy1) for px, py in sr.polygon]

            # Convert kraken LineSegments to InfLineSegment with adjusted coords
            inf_lines = []
            for seg in seg_lines:
                sbx1, sby1, sbx2, sby2 = seg.bbox
                adj_bbox = (sbx1 + dx1, sby1 + dy1, sbx2 + dx1, sby2 + dy1)
                adj_baseline = (
                    [(bpx + dx1, bpy + dy1) for bpx, bpy in seg.baseline]
                    if seg.baseline else None
                )
                inf_lines.append(InfLineSegment(
                    image=seg.image,
                    bbox=adj_bbox,
                    coords=adj_baseline,
                ))

            # Pair each region with its slice of inf_lines
            line_offset2 = 0
            for sr in seg_regions:
                n2 = len(sr.line_ids)
                new_pairs.append((sr, inf_lines[line_offset2:line_offset2 + n2]))
                line_offset2 += n2

        # Merge all pairs, sort left-to-right by region x-center
        all_pairs = kept_pairs + new_pairs
        all_pairs.sort(key=lambda p: (p[0].bbox[0] + p[0].bbox[2]) / 2)

        self.regions = [r for r, _ in all_pairs]
        self.line_segments = [l for _, ls in all_pairs for l in ls]

        # Re-index line_ids so they are contiguous
        flat_idx = 0
        for region in self.regions:
            n = len(region.line_ids)
            region.line_ids = [f"l_{flat_idx + i}" for i in range(n)]
            flat_idx += n

        # Clear drawn rects, exit draw mode, refresh canvas
        self.image_view.clear_drawn_rects()
        self.btn_draw_mode.setChecked(False)
        self._draw_region_line_boxes()
        self.status_bar.showMessage(
            f"{len(self.regions)} region(s), {len(self.line_segments)} lines after re-segmentation"
        )

    def process_image(self):
        """Transcribe all detected lines or full page (for VLMs)."""
        if not self.current_engine or not self.current_engine.is_model_loaded():
            QMessageBox.warning(self, "Model Not Loaded", "Please load a model first")
            return

        if not self.current_image:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        # Collapse engine section so progress bar and transcription area are visible
        self.engine_section.set_expanded(False)

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
        self.transcription_engine = self.current_engine  # snapshot which engine produced these

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
                if self.regions and len(self.regions) > 1:
                    # Region-aware: separate output per region with headers
                    offset = 0
                    for ri, region in enumerate(self.regions):
                        n = len(region.line_ids) if hasattr(region, 'line_ids') else 0
                        region_lines = self.transcriptions[offset:offset + n]
                        offset += n
                        if ri > 0:
                            f.write("\n")
                        f.write(f"=== Region {ri + 1} ===\n")
                        f.write("\n".join(region_lines))
                        f.write("\n")
                else:
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

            transcriptions = (
                self.transcriptions
                if self.transcriptions and len(self.transcriptions) == len(self.line_segments)
                else None
            )
            engine_name = self.current_engine.get_name() if self.current_engine else 'None'
            exporter = PageXMLExporter(str(self.current_image_path), width, height)

            if self.regions and len(self.regions) > 1:
                # Region-aware export: one TextRegion per detected column/region,
                # TextLines nested inside, actual baseline polylines used.
                exporter.export_with_regions(
                    self.regions,
                    self.line_segments,
                    file_path,
                    transcriptions=transcriptions,
                    creator="HTR-Transcription-GUI-Plugin",
                    comments=f"Engine: {engine_name}; regions: {len(self.regions)}"
                )
                msg = (f"Exported PAGE XML to:\n{file_path}\n"
                       f"({len(self.regions)} regions, {len(self.line_segments)} lines)")
            else:
                # Single-region fallback: all lines in one TextRegion
                segments_to_export = list(self.line_segments)
                if transcriptions:
                    from inference_page import LineSegment as InfLineSegment
                    for i, (seg, text) in enumerate(zip(segments_to_export, transcriptions)):
                        if isinstance(seg, InfLineSegment):
                            seg.text = text
                exporter.export(
                    segments_to_export,
                    file_path,
                    creator="HTR-Transcription-GUI-Plugin",
                    comments=f"Engine: {engine_name}"
                )
                msg = f"Exported PAGE XML to:\n{file_path}"

            self.status_bar.showMessage(f"Exported PAGE XML to: {file_path}")
            QMessageBox.information(self, "Success", msg)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export PAGE XML: {e}")

    def save_settings(self):
        """Save window geometry and splitter positions."""
        try:
            settings = QSettings('Polyscriptor', 'HTR_GUI')

            # Save window geometry
            settings.setValue('window/geometry', self.saveGeometry())
            settings.setValue('window/state', self.saveState())

            # Save splitter positions
            settings.setValue('splitter/main', self.main_splitter.saveState())
            settings.setValue('splitter/results', self.results_splitter.saveState())

            # Save last used engine
            if self.current_engine:
                settings.setValue('engine/last_used', self.current_engine.get_name())

            # Save engine configs (as JSON for compatibility)
            engine_configs = {}
            for engine in available_engines:
                try:
                    config = engine.get_config()
                    engine_configs[engine.get_name()] = config
                except:
                    pass
            settings.setValue('engine/configs', json.dumps(engine_configs))

        except Exception as e:
            print(f"Warning: Failed to save settings: {e}")

    def restore_settings(self):
        """Restore window geometry and splitter positions."""
        try:
            settings = QSettings('Polyscriptor', 'HTR_GUI')

            # Restore window geometry (with defaults)
            geometry = settings.value('window/geometry')
            if geometry:
                self.restoreGeometry(geometry)
            else:
                # Default: 1600×900 centered on screen
                self.resize(1600, 900)
                screen = QApplication.primaryScreen().geometry()
                x = (screen.width() - self.width()) // 2
                y = (screen.height() - self.height()) // 2
                self.move(x, y)

            state = settings.value('window/state')
            if state:
                self.restoreState(state)

            # Restore splitter positions (with defaults from setup_ui)
            main_state = settings.value('splitter/main')
            if main_state:
                self.main_splitter.restoreState(main_state)
            # else: use initial sizes set in setup_ui (768, 1152)

            results_state = settings.value('splitter/results')
            if results_state:
                self.results_splitter.restoreState(results_state)
            # Always enforce stats panel state to match button (button starts OFF)
            self._toggle_stats_panel(self.btn_stats_toggle.isChecked())

            # Restore last used engine
            last_engine = settings.value('engine/last_used')
            if last_engine:
                idx = self.engine_combo.findText(last_engine)
                if idx >= 0:
                    self.engine_combo.setCurrentIndex(idx)

            # Restore engine configs
            engine_configs_json = settings.value('engine/configs')
            if engine_configs_json:
                try:
                    engine_configs = json.loads(engine_configs_json)
                    if self.current_engine and self.current_engine.get_name() in engine_configs:
                        self.current_engine.set_config(engine_configs[self.current_engine.get_name()])
                except:
                    pass

        except Exception as e:
            print(f"Warning: Failed to restore settings: {e}")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for navigation."""
        key = event.key()

        # Only handle PageUp/PageDown for navigation (arrow keys are reserved
        # for text cursor movement in the transcription QTextEdit)
        if key == Qt.Key.Key_PageUp:
            if self.btn_prev_image.isEnabled():
                self._load_previous_image()
        elif key == Qt.Key.Key_PageDown:
            if self.btn_next_image.isEnabled():
                self._load_next_image()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window close."""
        # Save settings
        self.save_settings()

        # Unload models
        if self.current_engine:
            self.current_engine.unload_model()

        event.accept()


def _setup_dark_mode(app: QApplication) -> None:
    """Apply Fusion style + dark palette when the system is in dark mode.

    Qt's native Windows style has limited dark-mode support.  Switching to the
    platform-independent Fusion style and providing an explicit palette ensures
    all widgets (including custom ones) render correctly in both light and dark
    system themes on Windows, Linux, and macOS.
    """
    # Detect dark mode BEFORE calling setStyle() — setStyle("Fusion") resets
    # the palette to Fusion's light default, so the probe must happen first.
    is_dark = False
    try:
        # Qt 6.5+ (PyQt6 >= 6.6.0): reliable cross-platform detection
        is_dark = (app.styleHints().colorScheme() == Qt.ColorScheme.Dark)
    except AttributeError:
        # Fallback: read OS palette before any style change
        bg = app.palette().color(QPalette.ColorRole.Window)
        is_dark = bg.lightness() < 128

    # Switch to Fusion — platform-independent rendering, honours QPalette
    app.setStyle("Fusion")

    if not is_dark:
        return  # Light mode — Fusion default palette is fine

    dark = QPalette()
    c = QColor
    dark.setColor(QPalette.ColorRole.Window,          c(45,  45,  48))
    dark.setColor(QPalette.ColorRole.WindowText,      c(220, 220, 220))
    dark.setColor(QPalette.ColorRole.Base,            c(30,  30,  32))
    dark.setColor(QPalette.ColorRole.AlternateBase,   c(45,  45,  48))
    dark.setColor(QPalette.ColorRole.ToolTipBase,     c(45,  45,  48))
    dark.setColor(QPalette.ColorRole.ToolTipText,     c(220, 220, 220))
    dark.setColor(QPalette.ColorRole.Text,            c(220, 220, 220))
    dark.setColor(QPalette.ColorRole.Button,          c(60,  60,  64))
    dark.setColor(QPalette.ColorRole.ButtonText,      c(220, 220, 220))
    dark.setColor(QPalette.ColorRole.BrightText,      c(255, 255, 255))
    dark.setColor(QPalette.ColorRole.Link,            c(100, 160, 220))
    dark.setColor(QPalette.ColorRole.Highlight,       c(42,  130, 218))
    dark.setColor(QPalette.ColorRole.HighlightedText, c(255, 255, 255))
    dark.setColor(QPalette.ColorRole.Midlight,        c(75,  75,  80))
    dark.setColor(QPalette.ColorRole.Mid,             c(80,  80,  85))
    dark.setColor(QPalette.ColorRole.Dark,            c(25,  25,  27))
    dark.setColor(QPalette.ColorRole.Shadow,          c(10,  10,  12))
    # Disabled-state variants (dimmed)
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        dark.setColor(QPalette.ColorGroup.Disabled, role, c(120, 120, 120))
    app.setPalette(dark)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("HTR Transcription Tool")
    _setup_dark_mode(app)

    if not available_engines:
        QMessageBox.critical(
            None,
            "No Engines Available",
            "No HTR engines found. Please install at least one engine:\n\n"
            "- TrOCR: Already included\n"
            "- Qwen3: pip install transformers accelerate peft qwen-vl-utils\n"
            "- CRNN-CTC: See Documentation/PYLAIA_INSTALLATION_ISSUES.md\n"
            "- Commercial APIs: pip install openai google-generativeai anthropic"
        )
        sys.exit(1)

    window = TranscriptionGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
