"""
Professional TrOCR Transcription GUI using PyQt6

Features:
- Seamless zoom/pan with QGraphicsView
- Drag & drop + file dialog import
- Local & HuggingFace model selection
- Automatic line segmentation
- OCR processing with progress
- Text editor with font selection
- Comparison mode (side-by-side models)
- Export to TXT/CSV

Usage:
    python transcription_gui_qt.py
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QPushButton, QLabel, QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFileDialog, QProgressBar, QStatusBar, QToolBar, QFontDialog,
    QMessageBox, QListWidget, QListWidgetItem, QGroupBox, QGridLayout,
    QScrollArea, QTabWidget, QButtonGroup, QRadioButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QWheelEvent, QMouseEvent, QPen,
    QColor, QFont, QAction, QKeySequence, QTextCharFormat, QTextCursor
)

# Import inference components
from inference_page import (
    LineSegmenter, PageXMLSegmenter, TrOCRInference,
    LineSegment, normalize_background
)
from page_xml_exporter import PageXMLExporter

# Import Kraken segmenter (optional - will handle import error gracefully)
try:
    from kraken_segmenter import KrakenLineSegmenter
    KRAKEN_AVAILABLE = True
except ImportError:
    KRAKEN_AVAILABLE = False
    KrakenLineSegmenter = None

# Import Qwen3 VLM (optional - will handle import error gracefully)
try:
    from inference_qwen3 import Qwen3VLMInference, QWEN3_MODELS
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False
    QWEN3_MODELS = {}
    print("WARNING: Qwen3 not available. Install with: pip install transformers>=4.37.0 accelerate peft")

# Import PyLaia (optional - will handle import error gracefully)
try:
    from inference_pylaia import PyLaiaInference, PYLAIA_MODELS
    from inference_pylaia_lm import PyLaiaInferenceLM, check_lm_availability
    PYLAIA_AVAILABLE = True
    PYLAIA_LM_AVAILABLE = check_lm_availability()
except ImportError:
    PYLAIA_AVAILABLE = False
    PYLAIA_MODELS = {}
    PYLAIA_LM_AVAILABLE = False
    print("WARNING: PyLaia not available. Check train_pylaia.py is present.")

# Import commercial APIs (optional - will handle import errors gracefully)
try:
    from inference_commercial_api import (
        OpenAIInference, GeminiInference, ClaudeInference,
        check_api_availability,
        OPENAI_MODELS, GEMINI_MODELS, CLAUDE_MODELS
    )
    API_AVAILABILITY = check_api_availability()
    COMMERCIAL_API_AVAILABLE = any(API_AVAILABILITY.values())
except ImportError:
    COMMERCIAL_API_AVAILABLE = False
    API_AVAILABILITY = {"openai": False, "gemini": False, "claude": False}
    OPENAI_MODELS = []
    GEMINI_MODELS = []
    CLAUDE_MODELS = []
    print("WARNING: Commercial API support not available. Install with: pip install openai google-generativeai anthropic")


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

        # Line overlays
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
        """Set the image to display."""
        self._scene.clear()
        self.line_items.clear()

        if pixmap and not pixmap.isNull():
            self._scene.addPixmap(pixmap)
            self._empty = False
            self.fit_in_view()
        else:
            self._empty = True

    def draw_line_boxes(self, segments: List[LineSegment]):
        """Draw bounding boxes for detected lines."""
        # Clear existing line overlays
        for item in self.line_items:
            self._scene.removeItem(item)
        self.line_items.clear()

        # Draw new boxes
        pen = QPen(QColor(0, 255, 0, 180))  # Green with transparency
        pen.setWidth(2)

        for idx, segment in enumerate(segments):
            x1, y1, x2, y2 = segment.bbox
            rect_item = self._scene.addRect(x1, y1, x2-x1, y2-y1, pen)
            self.line_items.append(rect_item)

            # Add line number label
            text_item = self._scene.addText(f"{idx+1}", QFont("Arial", 12))
            text_item.setPos(x1, y1 - 20)
            text_item.setDefaultTextColor(QColor(0, 255, 0))
            self.line_items.append(text_item)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if self.has_image():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1

            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fit_in_view()
            else:
                self._zoom = 0


class OCRWorker(QThread):
    """Background thread for OCR processing."""

    progress = pyqtSignal(int, str)  # progress, status message
    finished = pyqtSignal(list)  # list of transcribed segments
    error = pyqtSignal(str)  # error message
    aborted = pyqtSignal()  # emitted when aborted

    def __init__(self, segments: List[LineSegment], ocr,
                 num_beams: int = 1, max_length: int = 128,
                 return_confidence: bool = True, is_pylaia: bool = False):
        """
        Initialize OCR worker.

        Args:
            segments: List of line segments to process
            ocr: Either TrOCRInference or PyLaiaInference instance
            num_beams: Beam search size (TrOCR only)
            max_length: Max sequence length (TrOCR only)
            return_confidence: Whether to return confidence scores
            is_pylaia: True if using PyLaia, False if using TrOCR
        """
        super().__init__()
        self.segments = segments
        self.ocr = ocr
        self.num_beams = num_beams
        self.max_length = max_length
        self.return_confidence = return_confidence
        self.is_pylaia = is_pylaia
        self._is_aborted = False

    def abort(self):
        """Request abortion of OCR processing."""
        self._is_aborted = True

    def run(self):
        """Process all line segments."""
        try:
            total = len(self.segments)
            results = []

            for idx, segment in enumerate(self.segments):
                # Check if aborted
                if self._is_aborted:
                    self.aborted.emit()
                    return

                self.progress.emit(int((idx / total) * 100),
                                 f"Processing line {idx+1}/{total}...")

                if self.is_pylaia:
                    # PyLaia inference
                    result_dict = self.ocr.recognize_line(
                        segment.image,
                        return_confidence=self.return_confidence
                    )
                    text = result_dict['text']
                    confidence = result_dict.get('confidence', None)
                    char_confidences = result_dict.get('char_confidences', [])
                else:
                    # TrOCR inference
                    text, confidence, char_confidences = self.ocr.transcribe_line(
                        segment.image,
                        num_beams=self.num_beams,
                        max_length=self.max_length,
                        return_confidence=self.return_confidence
                    )

                # Create new LineSegment with text and confidence
                result = LineSegment(
                    image=segment.image,
                    bbox=segment.bbox,
                    text=text,
                    confidence=confidence,
                    char_confidences=char_confidences
                )
                results.append(result)

            self.progress.emit(100, "Processing complete!")
            self.finished.emit(results)

        except Exception as e:
            if not self._is_aborted:
                self.error.emit(f"OCR Error: {str(e)}")


class TranscriptionGUI(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrOCR Transcription Tool")
        self.setGeometry(100, 100, 1400, 900)

        # State
        self.current_image_path: Optional[Path] = None
        self.current_pixmap: Optional[QPixmap] = None
        self.segments: List[LineSegment] = []
        self.ocr: Optional[TrOCRInference] = None
        self.ocr_worker: Optional[OCRWorker] = None

        # Image list for navigation
        self.image_list: List[Path] = []
        self.current_image_index: int = -1

        # Settings
        # Default to GPU if available (much faster - 5-20x speedup)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.normalize_bg = False
        self.num_beams = 4
        self.max_length = 128

        # Segmentation settings
        self.segmentation_method = "Kraken" if KRAKEN_AVAILABLE else "HPP"  # Default to Kraken if available
        self.kraken_model_path = None  # Path to custom Kraken model

        # Display settings
        self.show_confidence = True  # Show confidence scores by default

        # Model history
        self.hf_model_history_file = Path(".hf_model_history.json")
        self.hf_model_history = self._load_hf_model_history()

        # Processing statistics
        self.processing_start_time = None
        self.last_model_info = {}  # Store info about last processing run

        # Initialize Qwen3 as None
        self.qwen3 = None
        self._current_qwen3_model = None  # Track current model to avoid reinitialization

        # Initialize PyLaia as None
        self.pylaia: Optional[PyLaiaInference] = None
        self._current_pylaia_model = None  # Track current model to avoid reinitialization

        self._setup_ui()
        self._create_menu_bar()
        self._create_toolbar()
        self._setup_connections()

    def _setup_ui(self):
        """Create the main UI layout."""
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Main splitter (image viewer | transcription panel)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Image viewer
        left_panel = self._create_image_panel()
        self.main_splitter.addWidget(left_panel)

        # Right panel: Transcription editor
        right_panel = self._create_transcription_panel()
        self.main_splitter.addWidget(right_panel)

        self.main_splitter.setSizes([700, 700])
        main_layout.addWidget(self.main_splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(300)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _create_image_panel(self) -> QWidget:
        """Create the image viewing panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Image navigation controls
        nav_layout = QHBoxLayout()

        btn_open_images = QPushButton("Load Images...")
        btn_open_images.clicked.connect(self._open_multiple_images)
        nav_layout.addWidget(btn_open_images)

        self.btn_prev_image = QPushButton("< Previous")
        self.btn_prev_image.clicked.connect(self._load_previous_image)
        self.btn_prev_image.setEnabled(False)
        nav_layout.addWidget(self.btn_prev_image)

        self.lbl_image_index = QLabel("No images loaded")
        nav_layout.addWidget(self.lbl_image_index)

        self.btn_next_image = QPushButton("Next >")
        self.btn_next_image.clicked.connect(self._load_next_image)
        self.btn_next_image.setEnabled(False)
        nav_layout.addWidget(self.btn_next_image)

        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        # Zoom controls
        controls_layout = QHBoxLayout()

        btn_fit = QPushButton("Fit to Window")
        btn_fit.clicked.connect(self._fit_to_window)
        controls_layout.addWidget(btn_fit)

        btn_zoom_in = QPushButton("Zoom In (+)")
        btn_zoom_in.clicked.connect(self._zoom_in)
        controls_layout.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("Zoom Out (-)")
        btn_zoom_out.clicked.connect(self._zoom_out)
        controls_layout.addWidget(btn_zoom_out)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Graphics view
        self.graphics_view = ZoomableGraphicsView()
        layout.addWidget(self.graphics_view)

        # Segmentation controls (store as instance variable for visibility control)
        self.seg_group = QGroupBox("Line Segmentation")
        seg_layout = QVBoxLayout()

        # Row 0: Method selection
        seg_row0 = QHBoxLayout()
        seg_row0.addWidget(QLabel("Method:"))

        self.combo_seg_method = QComboBox()
        self.combo_seg_method.addItem("HPP (Fast)", "HPP")
        if KRAKEN_AVAILABLE:
            self.combo_seg_method.addItem("Kraken (Robust)", "Kraken")
            # Set Kraken as default (index 1)
            self.combo_seg_method.setCurrentIndex(1)
        else:
            self.combo_seg_method.addItem("Kraken (Not installed)", None)
            self.combo_seg_method.model().item(1).setEnabled(False)
        self.combo_seg_method.currentIndexChanged.connect(self._on_seg_method_changed)
        seg_row0.addWidget(self.combo_seg_method)

        # Kraken model selection (visible if Kraken is available and default)
        self.lbl_kraken_model = QLabel("Model:")
        self.lbl_kraken_model.setVisible(KRAKEN_AVAILABLE)
        seg_row0.addWidget(self.lbl_kraken_model)

        self.combo_kraken_model = QComboBox()
        self.combo_kraken_model.addItem("Default (built-in)", None)
        self.combo_kraken_model.setVisible(KRAKEN_AVAILABLE)
        seg_row0.addWidget(self.combo_kraken_model)

        self.btn_browse_kraken = QPushButton("Browse...")
        self.btn_browse_kraken.setVisible(KRAKEN_AVAILABLE)
        self.btn_browse_kraken.clicked.connect(self._browse_kraken_model)
        seg_row0.addWidget(self.btn_browse_kraken)

        seg_row0.addStretch()
        seg_layout.addLayout(seg_row0)

        # First row: Detect button and line count
        seg_row1 = QHBoxLayout()
        self.btn_segment = QPushButton("Detect Lines")
        self.btn_segment.clicked.connect(self._segment_lines)
        self.btn_segment.setEnabled(False)
        seg_row1.addWidget(self.btn_segment)

        self.lbl_lines_count = QLabel("Lines: 0")
        seg_row1.addWidget(self.lbl_lines_count)
        seg_row1.addStretch()
        seg_layout.addLayout(seg_row1)

        # Second row: Segmentation parameters (HPP-specific, hidden if Kraken is default)
        seg_row2 = QHBoxLayout()

        # Threshold slider (using double spinbox for decimal precision)
        self.lbl_threshold = QLabel("Threshold:")
        self.lbl_threshold.setVisible(not KRAKEN_AVAILABLE)  # Hide if Kraken is default
        seg_row2.addWidget(self.lbl_threshold)
        self.spin_sensitivity = QDoubleSpinBox()
        self.spin_sensitivity.setRange(0.5, 15.0)
        self.spin_sensitivity.setValue(5.0)  # 5% default (matches original working algorithm)
        self.spin_sensitivity.setSingleStep(0.5)
        self.spin_sensitivity.setDecimals(1)
        self.spin_sensitivity.setSuffix("%")
        self.spin_sensitivity.setToolTip("Detection threshold: Higher values = more selective (0.5-15%). Default: 5%")
        self.spin_sensitivity.setVisible(not KRAKEN_AVAILABLE)  # Hide if Kraken is default
        seg_row2.addWidget(self.spin_sensitivity)
        seg_row2.addSpacing(10)

        # Min line height
        self.lbl_min_height = QLabel("Min Height:")
        self.lbl_min_height.setVisible(not KRAKEN_AVAILABLE)  # Hide if Kraken is default
        seg_row2.addWidget(self.lbl_min_height)
        self.spin_min_height = QSpinBox()
        self.spin_min_height.setRange(5, 50)
        self.spin_min_height.setValue(10)  # Lowered from 15 to 10 for tighter spacing
        self.spin_min_height.setSuffix(" px")
        self.spin_min_height.setToolTip("Minimum line height in pixels. Default: 10px")
        self.spin_min_height.setVisible(not KRAKEN_AVAILABLE)  # Hide if Kraken is default
        seg_row2.addWidget(self.spin_min_height)
        seg_row2.addSpacing(10)

        # Morphological operations checkbox
        self.chk_morph = QCheckBox("Morph. Ops")
        self.chk_morph.setChecked(True)
        self.chk_morph.setToolTip("Apply morphological operations to connect broken characters")
        self.chk_morph.setVisible(not KRAKEN_AVAILABLE)  # Hide if Kraken is default
        seg_row2.addWidget(self.chk_morph)

        seg_row2.addStretch()
        seg_layout.addLayout(seg_row2)

        self.seg_group.setLayout(seg_layout)
        layout.addWidget(self.seg_group)

        return panel

    def _create_transcription_panel(self) -> QWidget:
        """Create the transcription editing panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Model settings
        settings_group = QGroupBox("Model & Settings")
        settings_layout = QVBoxLayout()

        # Model selection tabs
        self.model_tabs = QTabWidget()

        # TrOCR tab (unified Local + HuggingFace)
        trocr_tab = QWidget()
        trocr_layout = QGridLayout(trocr_tab)

        # Model source selector
        trocr_layout.addWidget(QLabel("Model Source:"), 0, 0)
        self.combo_trocr_source = QComboBox()
        self.combo_trocr_source.addItem("Local Models", "local")
        self.combo_trocr_source.addItem("HuggingFace Hub", "huggingface")
        self.combo_trocr_source.currentIndexChanged.connect(self._on_trocr_source_changed)
        trocr_layout.addWidget(self.combo_trocr_source, 0, 1, 1, 2)

        # Local model selection (visible by default)
        self.lbl_local_model = QLabel("Model:")
        trocr_layout.addWidget(self.lbl_local_model, 1, 0)
        self.combo_model = QComboBox()
        self._populate_models()
        trocr_layout.addWidget(self.combo_model, 1, 1, 1, 2)

        self.btn_browse_model = QPushButton("Browse...")
        self.btn_browse_model.clicked.connect(self._browse_model)
        trocr_layout.addWidget(self.btn_browse_model, 1, 3)

        # HuggingFace model selection (hidden by default)
        self.lbl_hf_model = QLabel("Model ID:")
        self.lbl_hf_model.setVisible(False)
        trocr_layout.addWidget(self.lbl_hf_model, 2, 0)

        self.combo_hf_model = QComboBox()
        self.combo_hf_model.setEditable(True)
        self.combo_hf_model.setPlaceholderText("e.g., kazars24/trocr-base-handwritten-ru")
        self._populate_hf_model_history()
        self.combo_hf_model.setVisible(False)
        trocr_layout.addWidget(self.combo_hf_model, 2, 1, 1, 2)

        self.btn_validate_hf = QPushButton("Validate")
        self.btn_validate_hf.clicked.connect(self._validate_hf_model)
        self.btn_validate_hf.setVisible(False)
        trocr_layout.addWidget(self.btn_validate_hf, 2, 3)

        # Model info display (for HuggingFace)
        self.lbl_model_info = QLabel("Model Info:")
        self.lbl_model_info.setVisible(False)
        trocr_layout.addWidget(self.lbl_model_info, 3, 0, Qt.AlignmentFlag.AlignTop)
        self.txt_model_info = QTextEdit()
        self.txt_model_info.setReadOnly(True)
        self.txt_model_info.setPlaceholderText("Model information will appear here...")
        self.txt_model_info.setMaximumHeight(80)
        self.txt_model_info.setVisible(False)
        trocr_layout.addWidget(self.txt_model_info, 3, 1, 1, 3)

        # Info box
        info_group = QGroupBox("About TrOCR")
        info_layout = QVBoxLayout()
        info_text = QLabel(
            "• Transformer-based OCR with high accuracy\n"
            "• Best for: complex handwriting, mixed scripts\n"
            "• Slower than PyLaia but more versatile"
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        trocr_layout.addWidget(info_group, 4, 0, 1, 4)

        # Add spacer
        trocr_layout.setRowStretch(5, 1)

        self.model_tabs.addTab(trocr_tab, "TrOCR")

        # Qwen3 VLM tab
        if QWEN3_AVAILABLE:
            qwen3_tab = QWidget()
            qwen3_layout = QGridLayout(qwen3_tab)

            # Model selection mode
            qwen3_layout.addWidget(QLabel("Model Source:"), 0, 0)
            self.combo_qwen3_source = QComboBox()
            self.combo_qwen3_source.addItem("Preset Models", "preset")
            self.combo_qwen3_source.addItem("Custom HuggingFace", "custom")
            self.combo_qwen3_source.currentIndexChanged.connect(self._on_qwen3_source_changed)
            qwen3_layout.addWidget(self.combo_qwen3_source, 0, 1)

            # Preset model dropdown
            self.lbl_qwen3_preset = QLabel("Preset:")
            qwen3_layout.addWidget(self.lbl_qwen3_preset, 1, 0)
            self.combo_qwen3_model = QComboBox()

            # Populate with available models
            for model_id, info in QWEN3_MODELS.items():
                display_name = f"{model_id} ({info['vram']})"
                self.combo_qwen3_model.addItem(display_name, model_id)

            self.combo_qwen3_model.setToolTip("Select preset Qwen3 VLM model")
            qwen3_layout.addWidget(self.combo_qwen3_model, 1, 1, 1, 2)

            # Custom HuggingFace model (hidden by default)
            self.lbl_qwen3_base = QLabel("Base Model:")
            self.lbl_qwen3_base.setVisible(False)
            qwen3_layout.addWidget(self.lbl_qwen3_base, 2, 0)
            self.txt_qwen3_base = QLineEdit()
            self.txt_qwen3_base.setPlaceholderText("e.g., Qwen/Qwen3-VL-8B-Instruct")
            self.txt_qwen3_base.setVisible(False)
            qwen3_layout.addWidget(self.txt_qwen3_base, 2, 1, 1, 2)

            # Custom adapter path (optional)
            qwen3_layout.addWidget(QLabel("Adapter:"), 3, 0)
            self.txt_qwen3_adapter = QLineEdit()
            self.txt_qwen3_adapter.setPlaceholderText("Optional: your-username/qwen3-ukrainian-adapter")
            qwen3_layout.addWidget(self.txt_qwen3_adapter, 3, 1, 1, 2)

            # Prompt customization
            qwen3_layout.addWidget(QLabel("Prompt:"), 4, 0, Qt.AlignmentFlag.AlignTop)
            self.txt_qwen3_prompt = QTextEdit()
            self.txt_qwen3_prompt.setPlainText("Transcribe the text shown in this image.")
            self.txt_qwen3_prompt.setMaximumHeight(80)
            qwen3_layout.addWidget(self.txt_qwen3_prompt, 4, 1, 1, 2)

            # Advanced settings
            advanced_group = QGroupBox("Advanced Settings")
            advanced_layout = QGridLayout()

            advanced_layout.addWidget(QLabel("Max Tokens:"), 0, 0)
            self.spin_qwen3_max_tokens = QSpinBox()
            self.spin_qwen3_max_tokens.setRange(512, 8192)
            self.spin_qwen3_max_tokens.setValue(2048)
            advanced_layout.addWidget(self.spin_qwen3_max_tokens, 0, 1)

            advanced_layout.addWidget(QLabel("Image Size:"), 0, 2)
            self.spin_qwen3_img_size = QSpinBox()
            self.spin_qwen3_img_size.setRange(512, 2048)
            self.spin_qwen3_img_size.setValue(1536)
            advanced_layout.addWidget(self.spin_qwen3_img_size, 0, 3)

            # Confidence estimation checkbox
            self.chk_qwen3_confidence = QCheckBox("Estimate Confidence (slower)")
            self.chk_qwen3_confidence.setChecked(False)
            self.chk_qwen3_confidence.setToolTip("Extract token probabilities for confidence estimation")
            advanced_layout.addWidget(self.chk_qwen3_confidence, 1, 0, 1, 2)

            advanced_group.setLayout(advanced_layout)
            qwen3_layout.addWidget(advanced_group, 5, 0, 1, 3)

            self.model_tabs.addTab(qwen3_tab, "Qwen3 VLM")

        # PyLaia tab
        if PYLAIA_AVAILABLE:
            pylaia_tab = QWidget()
            pylaia_layout = QGridLayout(pylaia_tab)

            # Model source selector
            pylaia_layout.addWidget(QLabel("Model Source:"), 0, 0)
            self.combo_pylaia_source = QComboBox()
            self.combo_pylaia_source.addItem("Local Models", "local")
            self.combo_pylaia_source.addItem("HuggingFace Hub", "huggingface")
            self.combo_pylaia_source.currentIndexChanged.connect(self._on_pylaia_source_changed)
            pylaia_layout.addWidget(self.combo_pylaia_source, 0, 1, 1, 2)

            # Local model selection (visible by default)
            self.lbl_pylaia_local = QLabel("Model:")
            pylaia_layout.addWidget(self.lbl_pylaia_local, 1, 0)
            self.combo_pylaia_model = QComboBox()

            # Populate with available models
            for model_name, model_path in PYLAIA_MODELS.items():
                self.combo_pylaia_model.addItem(model_name, model_path)

            self.combo_pylaia_model.setToolTip("Select PyLaia model")
            pylaia_layout.addWidget(self.combo_pylaia_model, 1, 1, 1, 2)

            # Browse button for custom models
            self.btn_pylaia_browse = QPushButton("Browse...")
            self.btn_pylaia_browse.setToolTip("Load a custom PyLaia model directory")
            self.btn_pylaia_browse.clicked.connect(self._browse_pylaia_model)
            pylaia_layout.addWidget(self.btn_pylaia_browse, 1, 3)

            # HuggingFace model selection (hidden by default)
            self.lbl_pylaia_hf = QLabel("Model ID:")
            self.lbl_pylaia_hf.setVisible(False)
            pylaia_layout.addWidget(self.lbl_pylaia_hf, 2, 0)

            self.txt_pylaia_hf = QLineEdit()
            self.txt_pylaia_hf.setPlaceholderText("e.g., user/pylaia-ukrainian-model (Note: PyLaia HF models rare)")
            self.txt_pylaia_hf.setVisible(False)
            pylaia_layout.addWidget(self.txt_pylaia_hf, 2, 1, 1, 3)

            # Note about HF support
            self.lbl_pylaia_hf_note = QLabel("⚠️ Note: PyLaia models on HuggingFace are rare. Most models are local only.")
            self.lbl_pylaia_hf_note.setWordWrap(True)
            self.lbl_pylaia_hf_note.setVisible(False)
            pylaia_layout.addWidget(self.lbl_pylaia_hf_note, 3, 0, 1, 4)

            # Info box with key features
            info_group = QGroupBox("About PyLaia")
            info_layout = QVBoxLayout()

            info_text = QLabel(
                "• CNN-RNN-CTC architecture for handwritten text recognition\n"
                "• ~100x faster than TrOCR for line-based transcription\n"
                "• Requires line segmentation (use 'Detect Lines' first)\n"
                "• Best for: manuscripts with consistent script style"
            )
            info_text.setWordWrap(True)
            info_text.setStyleSheet("font-size: 10px;")
            info_layout.addWidget(info_text)
            info_group.setLayout(info_layout)
            pylaia_layout.addWidget(info_group, 4, 0, 1, 4)

            # Confidence checkbox
            self.chk_pylaia_confidence = QCheckBox("Calculate Confidence Scores")
            self.chk_pylaia_confidence.setChecked(True)
            self.chk_pylaia_confidence.setToolTip("Compute line and character-level confidence scores")
            pylaia_layout.addWidget(self.chk_pylaia_confidence, 5, 0, 1, 2)

            # Language model options
            lm_group = QGroupBox("Language Model Post-Correction (Optional)")
            lm_layout = QGridLayout()

            # Enable LM checkbox
            self.chk_pylaia_use_lm = QCheckBox("Use Language Model")
            self.chk_pylaia_use_lm.setChecked(False)
            self.chk_pylaia_use_lm.setToolTip("Enable CTC beam search with n-gram language model for improved accuracy")
            self.chk_pylaia_use_lm.stateChanged.connect(self._on_pylaia_lm_changed)
            lm_layout.addWidget(self.chk_pylaia_use_lm, 0, 0, 1, 2)

            # LM path selection (hidden by default)
            self.lbl_pylaia_lm_path = QLabel("LM File:")
            self.lbl_pylaia_lm_path.setVisible(False)
            lm_layout.addWidget(self.lbl_pylaia_lm_path, 1, 0)

            self.txt_pylaia_lm_path = QLineEdit()
            self.txt_pylaia_lm_path.setPlaceholderText("language_models/ukrainian_char.arpa")
            self.txt_pylaia_lm_path.setVisible(False)
            lm_layout.addWidget(self.txt_pylaia_lm_path, 1, 1, 1, 2)

            self.btn_browse_lm = QPushButton("Browse...")
            self.btn_browse_lm.setVisible(False)
            self.btn_browse_lm.clicked.connect(self._browse_language_model)
            lm_layout.addWidget(self.btn_browse_lm, 1, 3)

            # Beam width
            self.lbl_beam_width = QLabel("Beam Width:")
            self.lbl_beam_width.setVisible(False)
            lm_layout.addWidget(self.lbl_beam_width, 2, 0)

            self.spin_beam_width = QSpinBox()
            self.spin_beam_width.setRange(10, 500)
            self.spin_beam_width.setValue(100)
            self.spin_beam_width.setToolTip("Larger = better quality but slower (100 is good default)")
            self.spin_beam_width.setVisible(False)
            lm_layout.addWidget(self.spin_beam_width, 2, 1)

            lm_group.setLayout(lm_layout)
            pylaia_layout.addWidget(lm_group, 6, 0, 1, 4)

            # Add spacer
            pylaia_layout.setRowStretch(7, 1)

            self.model_tabs.addTab(pylaia_tab, "PyLaia")

        # Commercial API tab (OpenAI, Gemini, Claude)
        if COMMERCIAL_API_AVAILABLE:
            api_tab = QWidget()
            api_layout = QGridLayout(api_tab)

            # Provider selection
            api_layout.addWidget(QLabel("Provider:"), 0, 0)
            self.combo_api_provider = QComboBox()

            # Add available providers
            if API_AVAILABILITY["openai"]:
                self.combo_api_provider.addItem("OpenAI (GPT-4o)", "openai")
            if API_AVAILABILITY["gemini"]:
                self.combo_api_provider.addItem("Google Gemini", "gemini")
            if API_AVAILABILITY["claude"]:
                self.combo_api_provider.addItem("Anthropic Claude", "claude")

            self.combo_api_provider.currentIndexChanged.connect(self._on_api_provider_changed)
            api_layout.addWidget(self.combo_api_provider, 0, 1, 1, 2)

            # API Key input
            api_layout.addWidget(QLabel("API Key:"), 1, 0)
            self.txt_api_key = QLineEdit()
            self.txt_api_key.setPlaceholderText("Enter your API key...")
            self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.txt_api_key.textChanged.connect(self._on_api_key_changed)
            api_layout.addWidget(self.txt_api_key, 1, 1, 1, 2)

            self.btn_show_api_key = QPushButton("👁")
            self.btn_show_api_key.setMaximumWidth(40)
            self.btn_show_api_key.setCheckable(True)
            self.btn_show_api_key.setToolTip("Show/hide API key")
            self.btn_show_api_key.toggled.connect(self._toggle_api_key_visibility)
            api_layout.addWidget(self.btn_show_api_key, 1, 3)

            # Validate API Key button
            self.btn_validate_api_key = QPushButton("Validate & Check Models")
            self.btn_validate_api_key.setToolTip("Validate API key and check available models")
            self.btn_validate_api_key.clicked.connect(self._validate_api_key)
            api_layout.addWidget(self.btn_validate_api_key, 1, 4)

            # API validation status label
            self.lbl_api_validation_status = QLabel("")
            self.lbl_api_validation_status.setStyleSheet("font-size: 10px;")
            api_layout.addWidget(self.lbl_api_validation_status, 1, 5)

            # Model selection (provider-specific)
            api_layout.addWidget(QLabel("Model:"), 2, 0)
            self.combo_api_model = QComboBox()
            self._populate_api_models()  # Populate based on initial provider
            api_layout.addWidget(self.combo_api_model, 2, 1, 1, 3)

            # Custom prompt
            api_layout.addWidget(QLabel("Prompt:"), 3, 0, Qt.AlignmentFlag.AlignTop)
            self.txt_api_prompt = QTextEdit()
            self.txt_api_prompt.setPlaceholderText(
                "Transcribe all handwritten text in this manuscript image. "
                "Preserve the original language (Cyrillic, Latin, etc.) and layout. "
                "Output only the transcribed text without any additional commentary."
            )
            self.txt_api_prompt.setMaximumHeight(80)
            api_layout.addWidget(self.txt_api_prompt, 3, 1, 1, 3)

            # Advanced settings
            advanced_group = QGroupBox("Advanced Settings")
            advanced_layout = QGridLayout()

            advanced_layout.addWidget(QLabel("Max Tokens:"), 0, 0)
            self.spin_api_max_tokens = QSpinBox()
            self.spin_api_max_tokens.setRange(100, 2000)
            self.spin_api_max_tokens.setValue(500)
            self.spin_api_max_tokens.setToolTip("Maximum tokens to generate")
            advanced_layout.addWidget(self.spin_api_max_tokens, 0, 1)

            advanced_layout.addWidget(QLabel("Temperature:"), 0, 2)
            self.spin_api_temperature = QDoubleSpinBox()
            self.spin_api_temperature.setRange(0.0, 1.0)
            self.spin_api_temperature.setSingleStep(0.1)
            self.spin_api_temperature.setValue(0.0)
            self.spin_api_temperature.setToolTip("0.0 = deterministic, 1.0 = creative")
            advanced_layout.addWidget(self.spin_api_temperature, 0, 3)

            advanced_group.setLayout(advanced_layout)
            api_layout.addWidget(advanced_group, 4, 0, 1, 4)

            # Info box
            info_group = QGroupBox("About Commercial APIs")
            info_layout = QVBoxLayout()
            info_text = QLabel(
                "• Best accuracy for complex manuscripts\n"
                "• Pay-per-use (check provider pricing)\n"
                "• Requires internet connection\n"
                "• OpenAI GPT-4o: Excellent general-purpose\n"
                "• Gemini 2.0 Flash: Fast and cost-effective\n"
                "• Claude 3.5 Sonnet: Best for text correction"
            )
            info_text.setWordWrap(True)
            info_text.setStyleSheet("font-size: 10px;")
            info_layout.addWidget(info_text)
            info_group.setLayout(info_layout)
            api_layout.addWidget(info_group, 5, 0, 1, 4)

            # Spacer
            api_layout.setRowStretch(6, 1)

            self.model_tabs.addTab(api_tab, "Commercial APIs")

            # Initialize API client storage
            self.api_client = None
            self._current_api_config = None

        settings_layout.addWidget(self.model_tabs)

        # Connect tab change handler
        self.model_tabs.currentChanged.connect(self._on_model_tab_changed)

        # Device and settings row
        device_settings_layout = QHBoxLayout()

        # Device selection (CPU/GPU)
        device_settings_layout.addWidget(QLabel("Device:"))

        self.radio_gpu = QRadioButton("GPU")
        self.radio_cpu = QRadioButton("CPU")

        # Default to CPU (more stable for batch processing)
        self.radio_cpu.setChecked(True)

        if not torch.cuda.is_available():
            self.radio_gpu.setEnabled(False)
            self.radio_gpu.setToolTip("No CUDA-capable GPU detected")
        else:
            self.radio_gpu.setToolTip("Faster but may have CUDA memory issues with large images")

        self.radio_gpu.toggled.connect(self._on_device_changed)
        self.radio_cpu.toggled.connect(self._on_device_changed)

        device_settings_layout.addWidget(self.radio_gpu)
        device_settings_layout.addWidget(self.radio_cpu)
        device_settings_layout.addSpacing(20)

        # Background normalization (TrOCR-specific)
        self.chk_normalize = QCheckBox("Normalize Background")
        self.chk_normalize.setChecked(False)
        self.chk_normalize.stateChanged.connect(self._on_normalize_changed)
        self.chk_normalize.setToolTip("Apply CLAHE normalization (TrOCR only)")
        device_settings_layout.addWidget(self.chk_normalize)
        device_settings_layout.addStretch()

        settings_layout.addLayout(device_settings_layout)

        # Inference parameters row (TrOCR-specific)
        self.trocr_inference_layout = QHBoxLayout()

        # Beam search
        self.lbl_beams = QLabel("Beam Search:")
        self.trocr_inference_layout.addWidget(self.lbl_beams)
        self.spin_beams = QSpinBox()
        self.spin_beams.setRange(1, 10)
        self.spin_beams.setValue(4)
        self.spin_beams.valueChanged.connect(self._on_beams_changed)
        self.spin_beams.setToolTip("Higher values = better quality but slower (TrOCR only)")
        self.trocr_inference_layout.addWidget(self.spin_beams)
        self.trocr_inference_layout.addSpacing(20)

        # Max length
        self.lbl_max_length = QLabel("Max Length:")
        self.trocr_inference_layout.addWidget(self.lbl_max_length)
        self.spin_max_length = QSpinBox()
        self.spin_max_length.setRange(64, 256)
        self.spin_max_length.setValue(128)
        self.spin_max_length.valueChanged.connect(self._on_max_length_changed)
        self.spin_max_length.setToolTip("Maximum sequence length (TrOCR only)")
        self.trocr_inference_layout.addWidget(self.spin_max_length)
        self.trocr_inference_layout.addStretch()

        settings_layout.addLayout(self.trocr_inference_layout)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Processing controls
        process_layout = QHBoxLayout()

        self.btn_process = QPushButton("Process All Lines")
        self.btn_process.clicked.connect(self._process_all_lines)
        self.btn_process.setEnabled(False)
        process_layout.addWidget(self.btn_process)

        self.btn_abort = QPushButton("Abort")
        self.btn_abort.clicked.connect(self._abort_processing)
        self.btn_abort.setEnabled(False)
        self.btn_abort.setStyleSheet("QPushButton { background-color: #d32f2f; color: white; }")
        process_layout.addWidget(self.btn_abort)

        process_layout.addStretch()
        layout.addLayout(process_layout)

        # Font and display options
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("Font:"))

        btn_font = QPushButton("Select Font...")
        btn_font.clicked.connect(self._select_font)
        font_layout.addWidget(btn_font)

        font_layout.addSpacing(20)

        # Confidence display toggle
        self.chk_show_confidence = QCheckBox("Show Confidence:")
        self.chk_show_confidence.setChecked(True)
        self.chk_show_confidence.stateChanged.connect(self._on_confidence_display_changed)
        self.chk_show_confidence.setToolTip("Display confidence scores with color-coded highlighting")
        font_layout.addWidget(self.chk_show_confidence)

        # Confidence granularity dropdown
        self.combo_confidence_granularity = QComboBox()
        self.combo_confidence_granularity.addItem("Line Average", "line")
        self.combo_confidence_granularity.addItem("Per Token", "token")
        self.combo_confidence_granularity.setToolTip("Choose how confidence is displayed:\n- Line Average: Single score for the whole line\n- Per Token: Individual color for each token/character")
        self.combo_confidence_granularity.currentIndexChanged.connect(self._on_confidence_display_changed)
        font_layout.addWidget(self.combo_confidence_granularity)

        font_layout.addStretch()
        layout.addLayout(font_layout)

        # Text editor
        self.text_editor = QTextEdit()
        self.text_editor.setPlaceholderText("Transcription will appear here...")

        # Set default font to 14pt for better readability
        default_font = QFont()
        default_font.setPointSize(14)
        self.text_editor.setFont(default_font)

        layout.addWidget(self.text_editor)

        # Character count
        self.lbl_char_count = QLabel("Characters: 0 | Words: 0")
        self.text_editor.textChanged.connect(self._update_char_count)
        layout.addWidget(self.lbl_char_count)

        # Statistics panel
        stats_group = QGroupBox("Processing Statistics")
        stats_layout = QGridLayout()

        # Row 0: Model info
        stats_layout.addWidget(QLabel("Model:"), 0, 0)
        self.lbl_model_used = QLabel("N/A")
        self.lbl_model_used.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.lbl_model_used, 0, 1, 1, 3)

        # Row 1: Line count and confidence
        stats_layout.addWidget(QLabel("Lines Processed:"), 1, 0)
        self.lbl_lines_processed = QLabel("0")
        stats_layout.addWidget(self.lbl_lines_processed, 1, 1)

        stats_layout.addWidget(QLabel("Avg Confidence:"), 1, 2)
        self.lbl_avg_confidence = QLabel("N/A")
        stats_layout.addWidget(self.lbl_avg_confidence, 1, 3)

        # Row 2: Processing time and segmentation
        stats_layout.addWidget(QLabel("Processing Time:"), 2, 0)
        self.lbl_processing_time = QLabel("N/A")
        stats_layout.addWidget(self.lbl_processing_time, 2, 1)

        stats_layout.addWidget(QLabel("Segmentation:"), 2, 2)
        self.lbl_seg_method = QLabel("N/A")
        stats_layout.addWidget(self.lbl_seg_method, 2, 3)

        # Row 3: Model parameters (initially hidden)
        stats_layout.addWidget(QLabel("Parameters:"), 3, 0)
        self.lbl_model_params = QLabel("N/A")
        self.lbl_model_params.setWordWrap(True)
        self.lbl_model_params.setStyleSheet("font-size: 9px;")
        stats_layout.addWidget(self.lbl_model_params, 3, 1, 1, 3)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        return panel

    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Image...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_image)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("&Save Transcription...", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_transcription)
        file_menu.addAction(save_action)

        export_action = QAction("&Export...", self)
        export_action.triggered.connect(self._export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        fit_action = QAction("Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self._fit_to_window)
        view_menu.addAction(fit_action)

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self._zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self._zoom_out)
        view_menu.addAction(zoom_out_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        open_btn = QPushButton("Open Image")
        open_btn.clicked.connect(self._open_image)
        toolbar.addWidget(open_btn)

        toolbar.addSeparator()

        segment_btn = QPushButton("Detect Lines")
        segment_btn.clicked.connect(self._segment_lines)
        toolbar.addWidget(segment_btn)

        process_btn = QPushButton("Process All")
        process_btn.clicked.connect(self._process_all_lines)
        toolbar.addWidget(process_btn)

        toolbar.addSeparator()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_transcription)
        toolbar.addWidget(save_btn)

    def _setup_connections(self):
        """Setup signal/slot connections."""
        # Enable drag and drop
        self.setAcceptDrops(True)

        # Load saved API keys (if any)
        if COMMERCIAL_API_AVAILABLE:
            self._load_api_keys()

        # Trigger initial tab change to set correct button state
        # This ensures the button shows correctly on first load
        self._on_model_tab_changed(0)  # TrOCR tab is index 0

    def _load_api_keys(self):
        """Load saved API keys from JSON file (base64 encoded for basic obfuscation)."""
        if not COMMERCIAL_API_AVAILABLE:
            return

        api_keys_file = Path.home() / ".trocr_gui" / "api_keys.json"
        try:
            if api_keys_file.exists():
                with open(api_keys_file, 'r', encoding='utf-8') as f:
                    encoded_keys = json.load(f)
                    # Store the encoded keys for later use when provider is selected
                    self._saved_api_keys = encoded_keys
        except Exception as e:
            print(f"Failed to load API keys: {e}")
            self._saved_api_keys = {}

    def _save_api_keys(self):
        """Save API keys to JSON file (base64 encoded for basic obfuscation)."""
        if not COMMERCIAL_API_AVAILABLE:
            return

        if not hasattr(self, 'txt_api_key'):
            return

        api_keys_file = Path.home() / ".trocr_gui" / "api_keys.json"
        api_keys_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            import base64

            # Get current provider and key
            provider = self.combo_api_provider.currentData()
            api_key = self.txt_api_key.text().strip()

            # Load existing keys
            encoded_keys = getattr(self, '_saved_api_keys', {})

            # Update with current key (base64 encode for basic obfuscation)
            if api_key:
                encoded_keys[provider] = base64.b64encode(api_key.encode()).decode()

            # Save to file
            with open(api_keys_file, 'w', encoding='utf-8') as f:
                json.dump(encoded_keys, f)

            self._saved_api_keys = encoded_keys
        except Exception as e:
            print(f"Failed to save API keys: {e}")

    def _load_hf_model_history(self) -> List[str]:
        """Load HuggingFace model history from JSON file."""
        try:
            if self.hf_model_history_file.exists():
                with open(self.hf_model_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    return history if isinstance(history, list) else []
        except Exception as e:
            print(f"Failed to load HF model history: {e}")
        return []

    def _save_hf_model_history(self):
        """Save HuggingFace model history to JSON file."""
        try:
            with open(self.hf_model_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.hf_model_history, f, indent=2)
        except Exception as e:
            print(f"Failed to save HF model history: {e}")

    def _add_to_hf_model_history(self, model_id: str):
        """Add a model ID to history (most recent first, max 10 items)."""
        if model_id and model_id.strip():
            model_id = model_id.strip()
            # Remove if already exists
            if model_id in self.hf_model_history:
                self.hf_model_history.remove(model_id)
            # Add to beginning
            self.hf_model_history.insert(0, model_id)
            # Keep only last 10
            self.hf_model_history = self.hf_model_history[:10]
            # Save to file
            self._save_hf_model_history()
            # Refresh UI
            self._populate_hf_model_history()

    def _populate_hf_model_history(self):
        """Populate HF model combo box with history."""
        current_text = self.combo_hf_model.currentText()
        self.combo_hf_model.clear()

        # Add history items
        for model_id in self.hf_model_history:
            self.combo_hf_model.addItem(model_id)

        # Restore current text if it was entered
        if current_text and current_text not in self.hf_model_history:
            self.combo_hf_model.setCurrentText(current_text)

    def _populate_models(self):
        """Populate model dropdown with available checkpoints."""
        self.combo_model.clear()

        # Find local models
        models_dir = Path("./models")
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "config.json").exists():
                    self.combo_model.addItem(str(model_dir), model_dir)

        if self.combo_model.count() == 0:
            self.combo_model.addItem("No models found", None)

    def _browse_model(self):
        """Browse for model checkpoint directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Model Checkpoint Directory", "./models"
        )
        if dir_path:
            self.combo_model.addItem(dir_path, Path(dir_path))
            self.combo_model.setCurrentIndex(self.combo_model.count() - 1)

    def _validate_hf_model(self):
        """Validate HuggingFace model ID."""
        model_id = self.combo_hf_model.currentText().strip()

        if not model_id:
            QMessageBox.warning(self, "Warning", "Please enter a HuggingFace model ID!")
            return

        self.status_bar.showMessage(f"Validating model: {model_id}...")
        self.txt_model_info.setPlainText("Checking model on HuggingFace Hub...")

        try:
            # Try to fetch model info from HuggingFace Hub
            from huggingface_hub import model_info

            info = model_info(model_id)

            # Display model information
            model_card = f"Model: {info.modelId}\n"
            model_card += f"Author: {info.author or 'Unknown'}\n"
            model_card += f"Downloads: {info.downloads:,}\n"
            model_card += f"Likes: {info.likes}\n"
            if info.lastModified:
                model_card += f"Last Modified: {info.lastModified.strftime('%Y-%m-%d')}\n"
            if info.pipeline_tag:
                model_card += f"Task: {info.pipeline_tag}\n"
            if info.tags:
                model_card += f"Tags: {', '.join(info.tags[:5])}\n"

            self.txt_model_info.setPlainText(model_card)
            self.status_bar.showMessage(f"Model '{model_id}' validated successfully!", 5000)

            # Add to history
            self._add_to_hf_model_history(model_id)

            QMessageBox.information(
                self,
                "Model Validated",
                f"Model '{model_id}' found on HuggingFace Hub!\n\n"
                "You can now use this model for OCR processing."
            )

        except Exception as e:
            error_msg = f"Failed to validate model: {str(e)}"
            self.txt_model_info.setPlainText(error_msg)
            self.status_bar.showMessage("Model validation failed!", 5000)

            QMessageBox.warning(
                self,
                "Validation Failed",
                f"Could not find or access model '{model_id}' on HuggingFace Hub.\n\n"
                f"Error: {str(e)}\n\n"
                "Please check the model ID and try again."
            )

    def _open_image(self):
        """Open an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif);;All Files (*)"
        )

        if file_path:
            self._load_image(Path(file_path))

    def _load_image(self, image_path: Path):
        """Load and display an image."""
        try:
            self.current_image_path = image_path

            # Load image
            pil_image = Image.open(image_path).convert('RGB')

            # Convert to QPixmap
            img_array = np.array(pil_image)
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_array.data, width, height, bytes_per_line,
                           QImage.Format.Format_RGB888)
            self.current_pixmap = QPixmap.fromImage(q_image)

            # Display
            self.graphics_view.set_image(self.current_pixmap)

            # Enable controls based on current mode
            tab_index = self.model_tabs.currentIndex()

            # Determine which model tab is active
            if QWEN3_AVAILABLE and PYLAIA_AVAILABLE:
                is_qwen3 = (tab_index == 1)
            elif QWEN3_AVAILABLE:
                is_qwen3 = (tab_index == 1)
            elif PYLAIA_AVAILABLE:
                is_qwen3 = False
            else:
                is_qwen3 = False

            if is_qwen3:
                # Qwen3 mode: Enable process button directly (no segmentation needed)
                self.btn_process.setEnabled(True)
                self.btn_segment.setEnabled(False)
            else:
                # TrOCR/PyLaia mode: Enable segment button
                self.btn_segment.setEnabled(True)
                self.btn_process.setEnabled(False)  # Will be enabled after segmentation

            self.status_bar.showMessage(f"Loaded: {image_path.name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    def _segment_lines(self):
        """Segment image into text lines."""
        if not self.current_image_path:
            return

        try:
            # Load image
            image = Image.open(self.current_image_path).convert('RGB')

            # Segment based on selected method
            if self.segmentation_method == "Kraken":
                self.status_bar.showMessage("Detecting lines with Kraken (this may take 3-8 seconds)...")

                # Get Kraken model path
                kraken_model_data = self.combo_kraken_model.currentData()

                # Create Kraken segmenter
                segmenter = KrakenLineSegmenter(
                    model_path=kraken_model_data,
                    device=self.device
                )

                # Segment lines
                self.segments = segmenter.segment_lines(
                    image,
                    text_direction='horizontal-lr',
                    use_binarization=True
                )

            else:  # HPP method
                self.status_bar.showMessage("Detecting lines with HPP...")

                # Get parameters from GUI
                sensitivity = self.spin_sensitivity.value() / 100.0  # Convert % to decimal
                min_height = self.spin_min_height.value()
                use_morph = self.chk_morph.isChecked()

                # Segment with configured parameters
                segmenter = LineSegmenter(
                    min_line_height=min_height,
                    min_gap=5,
                    sensitivity=sensitivity,
                    use_morph=use_morph
                )
                self.segments = segmenter.segment_lines(image, debug=False)

            # Draw boxes
            self.graphics_view.draw_line_boxes(self.segments)

            # Update UI
            num_lines = len(self.segments)
            self.lbl_lines_count.setText(f"Lines: {num_lines}")
            self.btn_process.setEnabled(num_lines > 0)

            # Update statistics panel
            self.lbl_seg_method.setText(self.segmentation_method)

            # Provide feedback based on results
            if num_lines == 0:
                if self.segmentation_method == "Kraken":
                    self.status_bar.showMessage("No lines detected by Kraken!")
                    QMessageBox.warning(
                        self,
                        "No Lines Detected",
                        "Kraken did not detect any text lines.\n\n"
                        "Possible solutions:\n"
                        "- Try switching to HPP method\n"
                        "- Check if the document contains text\n"
                        "- Try a different Kraken model"
                    )
                else:
                    self.status_bar.showMessage("No lines detected! Try adjusting sensitivity.")
                    QMessageBox.warning(
                        self,
                        "No Lines Detected",
                        "Line segmentation did not detect any text lines.\n\n"
                        "Try adjusting the segmentation parameters:\n"
                        "- DECREASE Threshold (try 1-2%) to detect fainter lines\n"
                        "- Lower Min Height if text is small\n"
                        "- Enable Morph. Ops to connect broken characters\n"
                        "- Or try switching to Kraken method"
                    )
            elif num_lines == 1:
                self.status_bar.showMessage(f"Detected 1 line - Check if this is correct")
                # Show warning but don't block - might be legitimate single line
                QMessageBox.information(
                    self,
                    "Single Line Detected",
                    f"Only 1 text line was detected using {self.segmentation_method}.\n\n"
                    "If the page contains multiple lines, try:\n"
                    + ("- Switching to HPP method with adjusted parameters\n"
                       if self.segmentation_method == "Kraken"
                       else "- INCREASING Threshold (e.g., 8-10%) to be more selective\n"
                            "- Reduce Min Height if lines are close together\n"
                            "- Enable Morph. Ops checkbox\n"
                            "- Or try switching to Kraken method\n")
                    + "\nYou can still process this line or re-detect with different settings."
                )
            else:
                self.status_bar.showMessage(f"Detected {num_lines} lines using {self.segmentation_method}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Line segmentation failed:\n{str(e)}\n\n"
                               f"Method: {self.segmentation_method}")
            self.lbl_lines_count.setText("Lines: 0")
            self.btn_process.setEnabled(False)

    def _on_trocr_source_changed(self, index):
        """Handle TrOCR source selection change."""
        is_hf = (self.combo_trocr_source.currentData() == "huggingface")

        # Show/hide appropriate fields based on source
        self.lbl_local_model.setVisible(not is_hf)
        self.combo_model.setVisible(not is_hf)
        self.btn_browse_model.setVisible(not is_hf)

        self.lbl_hf_model.setVisible(is_hf)
        self.combo_hf_model.setVisible(is_hf)
        self.btn_validate_hf.setVisible(is_hf)
        self.lbl_model_info.setVisible(is_hf)
        self.txt_model_info.setVisible(is_hf)

    def _on_pylaia_source_changed(self, index):
        """Handle PyLaia source selection change."""
        if not PYLAIA_AVAILABLE:
            return

        is_hf = (self.combo_pylaia_source.currentData() == "huggingface")

        # Show/hide appropriate fields based on source
        self.lbl_pylaia_local.setVisible(not is_hf)
        self.combo_pylaia_model.setVisible(not is_hf)
        self.btn_pylaia_browse.setVisible(not is_hf)

        self.lbl_pylaia_hf.setVisible(is_hf)
        self.txt_pylaia_hf.setVisible(is_hf)
        self.lbl_pylaia_hf_note.setVisible(is_hf)

    def _on_pylaia_lm_changed(self, state):
        """Handle PyLaia language model toggle."""
        if not PYLAIA_AVAILABLE:
            return

        use_lm = (state == 2)  # Qt.CheckState.Checked

        # Show/hide LM options
        self.lbl_pylaia_lm_path.setVisible(use_lm)
        self.txt_pylaia_lm_path.setVisible(use_lm)
        self.btn_browse_lm.setVisible(use_lm)
        self.lbl_beam_width.setVisible(use_lm)
        self.spin_beam_width.setVisible(use_lm)

    def _browse_language_model(self):
        """Browse for language model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Language Model File",
            "",
            "Language Model Files (*.arpa *.bin);;All Files (*)"
        )

        if file_path:
            self.txt_pylaia_lm_path.setText(file_path)
            self.status_bar.showMessage(f"Selected language model: {Path(file_path).name}")

    def _on_api_provider_changed(self, index):
        """Handle commercial API provider change."""
        if not COMMERCIAL_API_AVAILABLE:
            return

        # Update model list based on provider
        self._populate_api_models()

        # Load saved API key for this provider (if any)
        provider = self.combo_api_provider.currentData()
        if hasattr(self, '_saved_api_keys') and provider in self._saved_api_keys:
            import base64
            try:
                # Decode the saved key (base64 for basic obfuscation)
                decoded_key = base64.b64decode(self._saved_api_keys[provider]).decode()
                self.txt_api_key.setText(decoded_key)
            except Exception as e:
                print(f"Failed to load saved API key for {provider}: {e}")

        # Reset API client (will be recreated with new provider)
        self.api_client = None
        self._current_api_config = None

    def _populate_api_models(self):
        """Populate model dropdown based on selected API provider."""
        if not COMMERCIAL_API_AVAILABLE:
            return

        self.combo_api_model.clear()

        provider = self.combo_api_provider.currentData()

        if provider == "openai":
            for model in OPENAI_MODELS:
                self.combo_api_model.addItem(model)
        elif provider == "gemini":
            for model in GEMINI_MODELS:
                self.combo_api_model.addItem(model)
        elif provider == "claude":
            for model in CLAUDE_MODELS:
                self.combo_api_model.addItem(model)

    def _on_api_key_changed(self, text):
        """Handle API key change."""
        if not COMMERCIAL_API_AVAILABLE:
            return

        # Reset API client when key changes
        self.api_client = None
        self._current_api_config = None

        # Save API key when it changes (if not empty)
        if text.strip():
            self._save_api_keys()

        # Enable/disable process button based on API key and image availability
        # Only do this if we're on the Commercial API tab
        tab_index = self.model_tabs.currentIndex()
        tab_count = 1
        if QWEN3_AVAILABLE:
            tab_count += 1
        if PYLAIA_AVAILABLE:
            tab_count += 1
        api_tab_index = tab_count if COMMERCIAL_API_AVAILABLE else -1

        if tab_index == api_tab_index:
            if self.current_image_path is not None and text.strip():
                self.btn_process.setEnabled(True)
                self.status_bar.showMessage("API key entered. Ready to transcribe.")
            else:
                self.btn_process.setEnabled(False)
                if not text.strip():
                    self.status_bar.showMessage("Please enter API key to enable transcription.")

    def _toggle_api_key_visibility(self, checked):
        """Toggle API key visibility."""
        if not COMMERCIAL_API_AVAILABLE:
            return

        if checked:
            self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Password)

    def _validate_api_key(self):
        """Validate API key and check available models."""
        if not COMMERCIAL_API_AVAILABLE:
            return

        provider = self.combo_api_provider.currentData()
        api_key = self.txt_api_key.text().strip()

        if not api_key:
            QMessageBox.warning(self, "No API Key", "Please enter an API key first.")
            return

        self.btn_validate_api_key.setEnabled(False)
        self.btn_validate_api_key.setText("Validating...")
        self.lbl_api_validation_status.setText("⏳ Validating...")
        self.lbl_api_validation_status.setStyleSheet("color: orange; font-size: 10px;")
        QApplication.processEvents()

        try:
            # Attempt to initialize client and make a minimal test call
            if provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=api_key)

                # List models to validate key
                models = client.models.list()
                available_models = [model.id for model in models.data if 'gpt-4' in model.id or 'vision' in model.id]

                # Update model dropdown with available models
                if available_models:
                    self.combo_api_model.clear()
                    for model_id in available_models:
                        self.combo_api_model.addItem(model_id)
                    success_msg = f"✓ Valid! Found {len(available_models)} models"
                else:
                    # Fallback to default list
                    self._populate_api_models()
                    success_msg = "✓ Valid! Using default models"

            elif provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_key)

                # List models to validate key
                models = genai.list_models()

                # Get all Gemini models that support generateContent (for vision)
                available_models = []
                for m in models:
                    if 'gemini' in m.name.lower():
                        # Check if model supports generateContent
                        if 'generateContent' in m.supported_generation_methods:
                            model_id = m.name.replace('models/', '')
                            available_models.append(model_id)

                # Sort models - put 2.5/2.0 flash first, then others
                available_models.sort(key=lambda x: (
                    '2.5' not in x,  # 2.5 models first
                    '2.0' not in x,  # then 2.0 models
                    'flash' not in x,  # then flash models
                    x  # then alphabetically
                ))

                # Update model dropdown
                if available_models:
                    self.combo_api_model.clear()
                    for model_id in available_models:
                        self.combo_api_model.addItem(model_id)
                    success_msg = f"✓ Valid! Found {len(available_models)} models"
                else:
                    # Fallback to default list
                    self._populate_api_models()
                    success_msg = "✓ Valid! Using default models"

            elif provider == "claude":
                from anthropic import Anthropic
                client = Anthropic(api_key=api_key)

                # Make a minimal test call to validate (Claude doesn't have a list models endpoint)
                # We'll just try to initialize - if key is invalid, it will fail on first use
                # For now, just use default model list
                self._populate_api_models()
                success_msg = "✓ Key format valid! Using default models"

            else:
                raise ValueError(f"Unknown provider: {provider}")

            # Success
            self.lbl_api_validation_status.setText(success_msg)
            self.lbl_api_validation_status.setStyleSheet("color: green; font-size: 10px;")

            # Enable process button if image is loaded
            if self.current_image_path is not None:
                self.btn_process.setEnabled(True)

            QMessageBox.information(
                self, "API Key Valid",
                f"{provider.capitalize()} API key is valid!\n\n"
                f"Available models have been loaded into the dropdown.\n"
                f"You can now transcribe images."
            )

        except Exception as e:
            # Failure
            error_msg = str(e)
            self.lbl_api_validation_status.setText("✗ Invalid")
            self.lbl_api_validation_status.setStyleSheet("color: red; font-size: 10px;")

            QMessageBox.critical(
                self, "API Key Invalid",
                f"{provider.capitalize()} API key validation failed:\n\n{error_msg}\n\n"
                "Common issues:\n"
                "- Incorrect API key format\n"
                "- API key not activated\n"
                "- Insufficient permissions\n"
                "- Network connection problem"
            )

        finally:
            self.btn_validate_api_key.setEnabled(True)
            self.btn_validate_api_key.setText("Validate & Check Models")

    def _on_qwen3_source_changed(self, index):
        """Handle Qwen3 source selection change."""
        if not QWEN3_AVAILABLE:
            return

        is_custom = (self.combo_qwen3_source.currentData() == "custom")

        # Show/hide appropriate fields based on source
        self.lbl_qwen3_preset.setVisible(not is_custom)
        self.combo_qwen3_model.setVisible(not is_custom)

        self.lbl_qwen3_base.setVisible(is_custom)
        self.txt_qwen3_base.setVisible(is_custom)

    def _on_model_tab_changed(self, index):
        """Handle model tab changes - show/hide appropriate controls."""
        # Determine which tab is active
        # Tab order: TrOCR (0), Qwen3 (1 if available), PyLaia (2/1), Commercial API (last)
        is_trocr = (index == 0)
        is_qwen3 = False
        is_pylaia = False
        is_commercial_api = False

        # Calculate tab indices based on availability
        tab_count = 1  # TrOCR is always first
        qwen3_index = -1
        pylaia_index = -1
        api_index = -1

        if QWEN3_AVAILABLE:
            qwen3_index = tab_count
            tab_count += 1
        if PYLAIA_AVAILABLE:
            pylaia_index = tab_count
            tab_count += 1
        if COMMERCIAL_API_AVAILABLE:
            api_index = tab_count
            tab_count += 1

        # Determine active tab
        if index == qwen3_index:
            is_qwen3 = True
        elif index == pylaia_index:
            is_pylaia = True
        elif index == api_index:
            is_commercial_api = True

        # Hide/show segmentation group
        # Qwen3 and Commercial APIs don't need segmentation (they process full pages)
        # PyLaia and TrOCR need segmentation
        if hasattr(self, 'seg_group'):
            self.seg_group.setVisible(not is_qwen3 and not is_commercial_api)

        # Show/hide TrOCR-specific settings
        if hasattr(self, 'chk_normalize'):
            self.chk_normalize.setVisible(is_trocr)
        if hasattr(self, 'lbl_beams'):
            self.lbl_beams.setVisible(is_trocr)
        if hasattr(self, 'spin_beams'):
            self.spin_beams.setVisible(is_trocr)
        if hasattr(self, 'lbl_max_length'):
            self.lbl_max_length.setVisible(is_trocr)
        if hasattr(self, 'spin_max_length'):
            self.spin_max_length.setVisible(is_trocr)

        # Hide/show device selection (Commercial APIs don't use local GPU)
        if hasattr(self, 'radio_gpu'):
            self.radio_gpu.setVisible(not is_commercial_api)
        if hasattr(self, 'radio_cpu'):
            self.radio_cpu.setVisible(not is_commercial_api)

        # Grey out confidence checkboxes for commercial APIs (they don't provide confidence)
        if hasattr(self, 'chk_qwen3_confidence'):
            # Qwen3 confidence is always under user control
            pass
        if hasattr(self, 'chk_pylaia_confidence'):
            # PyLaia confidence checkbox - no need to disable for APIs (different tab)
            pass

        # Update button text and state based on model
        if is_commercial_api:
            provider_name = self.combo_api_provider.currentText().split()[0] if hasattr(self, 'combo_api_provider') else "API"
            self.btn_process.setText(f"☁️ Transcribe with {provider_name}")
            self.btn_process.setToolTip(f"Transcribe using {provider_name} vision model (requires API key)")
            # Enable if image loaded and API key provided
            if self.current_image_path is not None and hasattr(self, 'txt_api_key') and self.txt_api_key.text().strip():
                self.btn_process.setEnabled(True)
            else:
                self.btn_process.setEnabled(False)
        elif is_qwen3:
            self.btn_process.setText("🔍 Transcribe Entire Page")
            self.btn_process.setToolTip("Qwen3 VLM processes the entire page at once (no segmentation needed)")
            # If an image is already loaded, enable process button
            if self.current_image_path is not None:
                self.btn_process.setEnabled(True)
        elif is_pylaia:
            self.btn_process.setText("⚡ Transcribe Lines (PyLaia)")
            self.btn_process.setToolTip("Fast CNN-RNN-CTC transcription (requires line segmentation first)")
            # In PyLaia mode, only enable if lines are segmented
            if hasattr(self, 'segments') and len(self.segments) > 0:
                self.btn_process.setEnabled(True)
            else:
                self.btn_process.setEnabled(False)
        else:
            self.btn_process.setText("📝 Transcribe Lines (TrOCR)")
            self.btn_process.setToolTip("Transformer-based transcription with high accuracy (requires line segmentation first)")
            # In TrOCR mode, only enable if lines are segmented
            if hasattr(self, 'segments') and len(self.segments) > 0:
                self.btn_process.setEnabled(True)
            else:
                self.btn_process.setEnabled(False)

    def _process_with_qwen3(self):
        """Process entire page with Qwen3 VLM (no segmentation needed)."""
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return

        # Get Qwen3 settings
        prompt = self.txt_qwen3_prompt.toPlainText().strip()
        max_tokens = self.spin_qwen3_max_tokens.value()
        max_img_size = self.spin_qwen3_img_size.value()
        estimate_confidence = self.chk_qwen3_confidence.isChecked()

        # Determine if using preset or custom model
        is_custom = self.combo_qwen3_source.currentData() == "custom"

        if is_custom:
            # Custom HuggingFace model
            base_model = self.txt_qwen3_base.text().strip()
            adapter = self.txt_qwen3_adapter.text().strip() if self.txt_qwen3_adapter.text().strip() else None

            if not base_model:
                QMessageBox.warning(self, "Warning", "Please enter a base model ID (e.g., Qwen/Qwen3-VL-8B-Instruct)!")
                return

            model_display_name = base_model
            if adapter:
                model_display_name += f" + {adapter}"
        else:
            # Preset model
            model_id = self.combo_qwen3_model.currentData()
            if not model_id:
                QMessageBox.warning(self, "Warning", "Please select a Qwen3 model!")
                return

            model_config = QWEN3_MODELS[model_id]
            base_model = model_config["base"]

            # Allow custom adapter to override preset adapter
            custom_adapter = self.txt_qwen3_adapter.text().strip()
            adapter = custom_adapter if custom_adapter else model_config["adapter"]

            model_display_name = model_id

        try:
            self.status_bar.showMessage(f"Loading Qwen3 VLM: {model_display_name}...")
            self.btn_process.setEnabled(False)
            self.btn_segment.setEnabled(False)

            # Save model to history
            if is_custom and base_model:
                self._add_to_hf_model_history(base_model)
                if adapter:
                    self._add_to_hf_model_history(adapter)
            elif not is_custom:
                # Save preset model's base and adapter
                self._add_to_hf_model_history(base_model)
                if adapter:
                    self._add_to_hf_model_history(adapter)

            # Check if we need to reinitialize (model changed)
            need_reinit = False
            if not hasattr(self, 'qwen3') or self.qwen3 is None:
                need_reinit = True
            elif not hasattr(self, '_current_qwen3_model'):
                need_reinit = True
            elif self._current_qwen3_model != (base_model, adapter):
                need_reinit = True
                print(f"Model changed, reinitializing: {base_model} + {adapter}")

            # Initialize or reuse Qwen3
            if need_reinit:
                self.qwen3 = Qwen3VLMInference(
                    base_model=base_model,
                    adapter_model=adapter,
                    device="auto" if self.device == "cuda" else "cpu",
                    max_image_size=max_img_size
                )
                self._current_qwen3_model = (base_model, adapter)

            # Load full page image
            from PIL import Image
            page_image = Image.open(self.current_image_path)

            self.status_bar.showMessage("Transcribing full page with Qwen3 VLM...")
            QApplication.processEvents()  # Update UI

            # Store model info for statistics
            params_list = [f"MaxTokens: {max_tokens}", f"ImgSize: {max_img_size}"]
            if estimate_confidence:
                params_list.append("ConfEst")
            self.last_model_info = {
                'type': 'Qwen3 VLM',
                'name': model_display_name,
                'params': ", ".join(params_list)
            }

            # Transcribe entire page
            result = self.qwen3.transcribe_page(
                page_image,
                prompt=prompt,
                max_new_tokens=max_tokens,
                return_confidence=estimate_confidence
            )

            # Display result
            self.text_editor.setPlainText(result.text)

            # Update statistics panel for Qwen3
            self.lbl_lines_processed.setText("1 (full page)")
            if result.confidence is not None:
                self.lbl_avg_confidence.setText(f"{result.confidence*100:.1f}%")
            else:
                self.lbl_avg_confidence.setText("N/A")

            if result.processing_time > 0:
                if result.processing_time < 60:
                    self.lbl_processing_time.setText(f"{result.processing_time:.1f}s")
                else:
                    minutes = int(result.processing_time // 60)
                    seconds = int(result.processing_time % 60)
                    self.lbl_processing_time.setText(f"{minutes}m {seconds}s")
            else:
                self.lbl_processing_time.setText("N/A")

            self.lbl_seg_method.setText("None (full page)")
            self.lbl_model_used.setText(f"{self.last_model_info['type']}: {self.last_model_info['name']}")
            self.lbl_model_params.setText(self.last_model_info['params'])

            # Update status with timing and confidence
            status_msg = f"Qwen3 transcription complete! Time: {result.processing_time:.2f}s"
            if result.confidence is not None:
                status_msg += f", Confidence: {result.confidence*100:.1f}%"
            self.status_bar.showMessage(status_msg)

            # Show memory usage
            if QWEN3_AVAILABLE:
                memory_usage = self.qwen3.get_memory_usage()
                if memory_usage:
                    memory_str = ", ".join([f"{gpu}: {stats['utilization']}"
                                          for gpu, stats in memory_usage.items()])
                    print(f"GPU Usage: {memory_str}")

            self.btn_process.setEnabled(True)
            self.btn_segment.setEnabled(False)  # No segmentation for Qwen3

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(
                self, "Qwen3 Error",
                f"Qwen3 processing failed:\n{str(e)}\n\nDetails:\n{error_details}"
            )
            self.btn_process.setEnabled(True)
            self.btn_segment.setEnabled(False)

    def _process_with_commercial_api(self):
        """Process entire page with commercial API (OpenAI, Gemini, or Claude)."""
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return

        # Get API settings
        provider = self.combo_api_provider.currentData()
        api_key = self.txt_api_key.text().strip()
        model = self.combo_api_model.currentText()
        prompt = self.txt_api_prompt.toPlainText().strip()
        max_tokens = self.spin_api_max_tokens.value()
        temperature = self.spin_api_temperature.value()

        # Validate API key
        if not api_key:
            QMessageBox.warning(
                self, "API Key Required",
                "Please enter your API key in the Commercial API tab."
            )
            return

        try:
            self.status_bar.showMessage(f"Initializing {provider.upper()} API...")
            self.btn_process.setEnabled(False)
            self.btn_segment.setEnabled(False)

            # Initialize API client if needed or if settings changed
            api_config = (provider, api_key, model)
            if self.api_client is None or self._current_api_config != api_config:
                if provider == "openai":
                    self.api_client = OpenAIInference(
                        api_key=api_key,
                        model=model,
                        default_prompt=prompt if prompt else None
                    )
                elif provider == "gemini":
                    self.api_client = GeminiInference(
                        api_key=api_key,
                        model=model,
                        default_prompt=prompt if prompt else None
                    )
                elif provider == "claude":
                    self.api_client = ClaudeInference(
                        api_key=api_key,
                        model=model,
                        default_prompt=prompt if prompt else None
                    )
                else:
                    QMessageBox.critical(self, "Error", f"Unknown provider: {provider}")
                    return

                self._current_api_config = api_config

            # Load full page image
            from PIL import Image
            import time
            page_image = Image.open(self.current_image_path).convert("RGB")

            self.status_bar.showMessage(f"Transcribing with {provider.upper()} {model}...")
            QApplication.processEvents()  # Update UI

            # Store model info for statistics
            self.last_model_info = {
                'type': f'{provider.capitalize()} API',
                'name': model,
                'params': f"Temp: {temperature}, MaxTok: {max_tokens}"
            }

            # Transcribe entire page
            start_time = time.time()

            # Use provider-specific parameter names
            if provider == "gemini":
                result_text = self.api_client.transcribe(
                    page_image,
                    prompt=prompt if prompt else None,
                    max_output_tokens=max_tokens,  # Gemini uses max_output_tokens
                    temperature=temperature
                )
            else:
                # OpenAI and Claude use max_tokens
                result_text = self.api_client.transcribe(
                    page_image,
                    prompt=prompt if prompt else None,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

            processing_time = time.time() - start_time

            # Display result
            self.text_editor.setPlainText(result_text)

            # Update statistics panel
            self.lbl_lines_processed.setText("1 (full page)")
            self.lbl_avg_confidence.setText("N/A")  # APIs don't provide confidence scores

            if processing_time < 60:
                self.lbl_processing_time.setText(f"{processing_time:.1f}s")
            else:
                minutes = int(processing_time // 60)
                seconds = int(processing_time % 60)
                self.lbl_processing_time.setText(f"{minutes}m {seconds}s")

            self.lbl_seg_method.setText("None (full page)")
            self.lbl_model_used.setText(f"{self.last_model_info['type']}: {self.last_model_info['name']}")
            self.lbl_model_params.setText(self.last_model_info['params'])

            # Update status
            status_msg = f"{provider.capitalize()} transcription complete! Time: {processing_time:.2f}s"
            self.status_bar.showMessage(status_msg)

            self.btn_process.setEnabled(True)
            self.btn_segment.setEnabled(False)  # No segmentation for API models

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(
                self, f"{provider.capitalize()} API Error",
                f"{provider.capitalize()} API processing failed:\n{str(e)}\n\nDetails:\n{error_details}\n\n"
                "Common issues:\n"
                "- Invalid API key\n"
                "- Insufficient API credits\n"
                "- Rate limit exceeded\n"
                "- Network connection problem"
            )
            self.btn_process.setEnabled(True)
            self.btn_segment.setEnabled(False)

    def _process_all_lines(self):
        """Process all detected lines with OCR (or entire page with Qwen3/Commercial API)."""

        # Determine which model tab is active
        tab_index = self.model_tabs.currentIndex()

        # Calculate tab indices
        is_qwen3 = False
        is_pylaia = False
        is_commercial_api = False

        tab_count = 1  # TrOCR is always first
        qwen3_index = -1
        pylaia_index = -1
        api_index = -1

        if QWEN3_AVAILABLE:
            qwen3_index = tab_count
            tab_count += 1
        if PYLAIA_AVAILABLE:
            pylaia_index = tab_count
            tab_count += 1
        if COMMERCIAL_API_AVAILABLE:
            api_index = tab_count
            tab_count += 1

        # Determine active tab
        if tab_index == qwen3_index:
            is_qwen3 = True
        elif tab_index == pylaia_index:
            is_pylaia = True
        elif tab_index == api_index:
            is_commercial_api = True

        # Handle full-page models (Qwen3 and Commercial APIs)
        if is_qwen3:
            self._process_with_qwen3()
            return

        if is_commercial_api:
            self._process_with_commercial_api()
            return

        # Standard line-based processing (TrOCR or PyLaia)
        if not self.segments:
            QMessageBox.warning(
                self,
                "No Lines Available",
                "No text lines have been detected yet.\n\n"
                "Please click 'Detect Lines' first to segment the image into text lines."
            )
            return

        try:
            if is_pylaia:
                # PyLaia model handling
                if not PYLAIA_AVAILABLE:
                    QMessageBox.critical(self, "Error", "PyLaia is not available!")
                    return

                # Check which source is selected (Local or HuggingFace)
                is_hf = (self.combo_pylaia_source.currentData() == "huggingface")

                if is_hf:
                    # HuggingFace model (experimental - most PyLaia models are local only)
                    model_id = self.txt_pylaia_hf.text().strip()
                    if not model_id:
                        QMessageBox.warning(self, "Warning", "Please enter a HuggingFace model ID!")
                        return

                    QMessageBox.warning(
                        self, "Experimental Feature",
                        "HuggingFace support for PyLaia is experimental.\n\n"
                        "Most PyLaia models are local only and not available on HuggingFace.\n"
                        "This feature requires the model to have the correct structure:\n"
                        "- best_model.pt (or model.pt)\n"
                        "- model_config.json\n"
                        "- symbols.txt"
                    )
                    # Note: Would need to implement HF download logic here
                    QMessageBox.critical(self, "Not Implemented",
                                       "HuggingFace model loading for PyLaia is not yet implemented.\n"
                                       "Please use local models or download manually.")
                    return
                else:
                    # Local model
                    model_path = self.combo_pylaia_model.currentData()
                    if not model_path:
                        QMessageBox.warning(self, "Warning", "No PyLaia model selected!")
                        return
                    model_display_name = self.combo_pylaia_model.currentText()

                # Check if language model should be used
                use_lm = self.chk_pylaia_use_lm.isChecked()
                lm_path = self.txt_pylaia_lm_path.text().strip() if use_lm else None
                beam_width = self.spin_beam_width.value() if use_lm else 1

                # Initialize PyLaia (with or without LM) if needed or if settings changed
                model_key = (model_path, lm_path, use_lm)
                if self.pylaia is None or self._current_pylaia_model != model_key:
                    self.status_bar.showMessage(f"Loading PyLaia model on {self.device.upper()}...")

                    if use_lm and lm_path and PYLAIA_LM_AVAILABLE:
                        # Use LM-enhanced inference
                        if not Path(lm_path).exists():
                            QMessageBox.warning(
                                self, "Language Model Not Found",
                                f"Language model file not found:\n{lm_path}\n\n"
                                "Train a language model using:\n"
                                "python train_character_lm.py --input_dir data --output ukrainian_char.arpa"
                            )
                            return

                        self.pylaia = PyLaiaInferenceLM(
                            model_path=model_path,
                            lm_path=lm_path,
                            device=self.device
                        )
                        self.status_bar.showMessage(f"PyLaia model loaded with language model on {self.device.upper()}")
                    elif use_lm and not PYLAIA_LM_AVAILABLE:
                        QMessageBox.warning(
                            self, "LM Not Available",
                            "Language model support requires pyctcdecode.\n\n"
                            "Install with: pip install pyctcdecode"
                        )
                        return
                    else:
                        # Use standard inference without LM
                        self.pylaia = PyLaiaInference(
                            model_path=model_path,
                            device=self.device
                        )

                    self._current_pylaia_model = model_key

                # Get confidence setting
                return_confidence = self.chk_pylaia_confidence.isChecked()

                # Store model info for statistics
                model_name = Path(model_path).stem if not model_path.startswith("models/") else self.combo_pylaia_model.currentText()
                params_list = [f"Confidence: {return_confidence}"]
                if use_lm and lm_path:
                    params_list.append(f"LM: {Path(lm_path).stem}")
                    params_list.append(f"Beam: {beam_width}")

                self.last_model_info = {
                    'type': 'PyLaia',
                    'name': model_name,
                    'params': ", ".join(params_list)
                }

                # Start processing in background thread
                self.ocr_worker = OCRWorker(
                    self.segments, self.pylaia,
                    return_confidence=return_confidence,
                    is_pylaia=True
                )

            else:
                # TrOCR model handling
                # Check which source is selected (Local or HuggingFace)
                is_hf = (self.combo_trocr_source.currentData() == "huggingface")

                if is_hf:
                    # HuggingFace model
                    model_id = self.combo_hf_model.currentText().strip()
                    if not model_id:
                        QMessageBox.warning(self, "Warning", "Please enter a HuggingFace model ID!")
                        return
                    model_path = model_id
                    is_huggingface = True
                    model_display_name = model_id
                    # Add to history when processing
                    self._add_to_hf_model_history(model_id)
                else:
                    # Local model
                    model_data = self.combo_model.currentData()
                    if not model_data:
                        QMessageBox.warning(self, "Warning", "No model selected!")
                        return
                    model_path = str(model_data)
                    is_huggingface = False
                    model_display_name = self.combo_model.currentText()

                # Initialize OCR if needed (or if model changed)
                if self.ocr is None or self.ocr.model_path != model_path:
                    self.status_bar.showMessage(f"Loading TrOCR model on {self.device.upper()}...")
                    self.ocr = TrOCRInference(
                        model_path,
                        device=self.device,
                        normalize_bg=self.normalize_bg,
                        is_huggingface=is_huggingface
                    )

                # Store model info for statistics
                params_list = [f"Beams: {self.num_beams}", f"MaxLen: {self.max_length}"]
                if self.normalize_bg:
                    params_list.append("NormBG")
                self.last_model_info = {
                    'type': 'TrOCR',
                    'name': model_display_name,
                    'params': ", ".join(params_list)
                }

                # Start processing in background thread
                self.ocr_worker = OCRWorker(
                    self.segments, self.ocr,
                    self.num_beams, self.max_length,
                    is_pylaia=False
                )
            self.ocr_worker.progress.connect(self._on_ocr_progress)
            self.ocr_worker.finished.connect(self._on_ocr_finished)
            self.ocr_worker.error.connect(self._on_ocr_error)
            self.ocr_worker.aborted.connect(self._on_ocr_aborted)

            self.btn_process.setEnabled(False)
            self.btn_abort.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Record start time
            self.processing_start_time = time.time()

            self.ocr_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"OCR initialization failed:\n{str(e)}")

    def _on_ocr_progress(self, value: int, message: str):
        """Handle OCR progress updates."""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)

    def _on_ocr_finished(self, segments: List[LineSegment]):
        """Handle OCR completion."""
        self.segments = segments

        # Update text editor with confidence scores
        self._display_transcription_with_confidence()

        self.btn_process.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.progress_bar.setVisible(False)

        # Calculate processing time
        processing_time = 0
        if self.processing_start_time:
            processing_time = time.time() - self.processing_start_time

        # Calculate and display average confidence
        confidences = [seg.confidence for seg in segments if seg.confidence is not None]
        avg_confidence = 0
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            self.status_bar.showMessage(f"Transcription complete! Avg confidence: {avg_confidence*100:.1f}%")
        else:
            self.status_bar.showMessage("Transcription complete!")

        # Update statistics panel
        self.lbl_lines_processed.setText(str(len(segments)))
        if confidences:
            self.lbl_avg_confidence.setText(f"{avg_confidence*100:.1f}%")
        else:
            self.lbl_avg_confidence.setText("N/A")

        if processing_time > 0:
            if processing_time < 60:
                self.lbl_processing_time.setText(f"{processing_time:.1f}s")
            else:
                minutes = int(processing_time // 60)
                seconds = int(processing_time % 60)
                self.lbl_processing_time.setText(f"{minutes}m {seconds}s")
        else:
            self.lbl_processing_time.setText("N/A")

        # Display model info
        if self.last_model_info:
            model_type = self.last_model_info.get('type', 'Unknown')
            model_name = self.last_model_info.get('name', 'Unknown')
            self.lbl_model_used.setText(f"{model_type}: {model_name}")
            self.lbl_model_params.setText(self.last_model_info.get('params', 'N/A'))
        else:
            self.lbl_model_used.setText("N/A")
            self.lbl_model_params.setText("N/A")

    def _on_ocr_error(self, error_msg: str):
        """Handle OCR errors."""
        QMessageBox.critical(self, "OCR Error", error_msg)
        self.btn_process.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.progress_bar.setVisible(False)

    def _on_ocr_aborted(self):
        """Handle OCR abortion."""
        self.btn_process.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Processing aborted by user")
        QMessageBox.information(self, "Aborted", "OCR processing was aborted.\n\nPartial results may be available.")

    def _abort_processing(self):
        """Abort ongoing OCR processing."""
        if self.ocr_worker and self.ocr_worker.isRunning():
            self.ocr_worker.abort()
            self.status_bar.showMessage("Aborting...")

    def _get_confidence_color(self, confidence: float, is_dark_mode: bool) -> QColor:
        """Get background color based on confidence level."""
        if confidence >= 0.95:
            return None  # No highlighting
        elif confidence >= 0.85:
            return QColor(0, 80, 0) if is_dark_mode else QColor(200, 255, 200)  # Green
        elif confidence >= 0.75:
            return QColor(100, 80, 0) if is_dark_mode else QColor(255, 255, 200)  # Yellow
        else:
            return QColor(100, 0, 0) if is_dark_mode else QColor(255, 200, 200)  # Red

    def _display_transcription_with_confidence(self):
        """Display transcription in text editor with confidence-based color coding."""
        if not self.segments:
            return

        # Clear text editor
        self.text_editor.clear()
        cursor = self.text_editor.textCursor()

        # Detect dark mode
        bg_color = self.text_editor.palette().color(self.text_editor.palette().ColorRole.Base)
        is_dark_mode = bg_color.lightness() < 128

        # Get granularity setting
        granularity = self.combo_confidence_granularity.currentData()

        for idx, segment in enumerate(self.segments):
            if not segment.text:
                continue

            # Check if we should show confidence
            if self.show_confidence and segment.confidence is not None:
                if granularity == "token" and segment.char_confidences:
                    # Per-token coloring
                    # Note: char_confidences are per BPE token, not per character
                    # We'll approximate by distributing tokens across the text
                    text = segment.text
                    num_tokens = len(segment.char_confidences)

                    if num_tokens > 0:
                        chars_per_token = max(1, len(text) // num_tokens)

                        for token_idx, token_conf in enumerate(segment.char_confidences):
                            # Calculate character range for this token
                            start_char = token_idx * chars_per_token
                            end_char = min(start_char + chars_per_token, len(text))

                            # Get the text chunk for this token
                            token_text = text[start_char:end_char]

                            # Create format
                            text_format = QTextCharFormat()
                            color = self._get_confidence_color(token_conf, is_dark_mode)
                            if color:
                                text_format.setBackground(color)

                            cursor.insertText(token_text, text_format)

                    # Add average confidence at end
                    avg_conf = segment.confidence
                    confidence_format = QTextCharFormat()
                    confidence_format.setForeground(QColor(150, 150, 150) if is_dark_mode else QColor(128, 128, 128))
                    cursor.insertText(f" ({avg_conf*100:.1f}%)", confidence_format)
                else:
                    # Line-level coloring (original behavior)
                    text_format = QTextCharFormat()
                    color = self._get_confidence_color(segment.confidence, is_dark_mode)
                    if color:
                        text_format.setBackground(color)

                    cursor.insertText(segment.text, text_format)

                    # Add confidence percentage
                    confidence_format = QTextCharFormat()
                    confidence_format.setForeground(QColor(150, 150, 150) if is_dark_mode else QColor(128, 128, 128))
                    cursor.insertText(f" ({segment.confidence*100:.1f}%)", confidence_format)
            else:
                # Plain text without confidence
                cursor.insertText(segment.text)

            # Add newline between lines
            if idx < len(self.segments) - 1:
                cursor.insertText("\n")

        # Move cursor to beginning
        cursor.setPosition(0)
        self.text_editor.setTextCursor(cursor)

    def _on_confidence_display_changed(self, state):
        """Handle confidence display checkbox toggle."""
        self.show_confidence = (state == Qt.CheckState.Checked.value)
        # Re-display transcription with updated settings
        if self.segments:
            self._display_transcription_with_confidence()

    def _save_transcription(self):
        """Save transcription to file."""
        if not self.text_editor.toPlainText():
            QMessageBox.warning(self, "Warning", "No transcription to save!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Transcription", "",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.text_editor.toPlainText())
                self.status_bar.showMessage(f"Saved: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")

    def _export(self):
        """Export transcription in various formats."""
        if not self.segments:
            QMessageBox.warning(self, "Warning", "No transcription to export!")
            return

        # Ask user for export format
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Options")
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Select export format:"))

        format_group = QGroupBox("Format")
        format_layout = QVBoxLayout()

        radio_txt = QRadioButton("Plain Text (.txt)")
        radio_txt.setChecked(True)
        radio_csv = QRadioButton("CSV with confidence (.csv)")
        radio_tsv = QRadioButton("TSV with confidence (.tsv)")
        radio_xml = QRadioButton("PAGE XML (.xml) - for Party and other processors")

        format_layout.addWidget(radio_txt)
        format_layout.addWidget(radio_csv)
        format_layout.addWidget(radio_tsv)
        format_layout.addWidget(radio_xml)
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Options
        chk_include_confidence = QCheckBox("Include confidence scores")
        chk_include_confidence.setChecked(True)
        chk_include_confidence.setEnabled(False)  # Will be enabled for CSV/TSV
        layout.addWidget(chk_include_confidence)

        # Update checkbox state based on format selection
        def on_format_changed():
            is_structured = radio_csv.isChecked() or radio_tsv.isChecked()
            chk_include_confidence.setEnabled(is_structured)

        radio_txt.toggled.connect(on_format_changed)
        radio_csv.toggled.connect(on_format_changed)
        radio_tsv.toggled.connect(on_format_changed)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != 1:  # 1 = Accepted
            return

        # Determine export format
        if radio_csv.isChecked():
            file_filter = "CSV Files (*.csv)"
            default_ext = ".csv"
            delimiter = ","
            export_type = "csv"
        elif radio_tsv.isChecked():
            file_filter = "TSV Files (*.tsv)"
            default_ext = ".tsv"
            delimiter = "\t"
            export_type = "tsv"
        elif radio_xml.isChecked():
            file_filter = "PAGE XML Files (*.xml)"
            default_ext = ".xml"
            delimiter = None
            export_type = "xml"
        else:
            file_filter = "Text Files (*.txt)"
            default_ext = ".txt"
            delimiter = None
            export_type = "txt"

        # Get save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Transcription", "",
            f"{file_filter};;All Files (*)"
        )

        if not file_path:
            return

        # Ensure correct extension
        if not file_path.endswith(default_ext):
            file_path += default_ext

        try:
            if export_type == "xml":
                # PAGE XML export
                if not self.current_image_path:
                    QMessageBox.warning(self, "Warning", "No image loaded. PAGE XML requires image reference.")
                    return

                # Get image dimensions
                img = Image.open(self.current_image_path)
                width, height = img.size

                # Create exporter and export
                exporter = PageXMLExporter(self.current_image_path, width, height)
                exporter.export(
                    self.segments,
                    file_path,
                    creator="TrOCR-GUI",
                    comments=f"Segmentation method: {self.segmentation_method}"
                )

            else:
                # Text-based export (TXT, CSV, TSV)
                with open(file_path, 'w', encoding='utf-8', newline='') as f:
                    if delimiter:  # CSV or TSV format
                        import csv
                        writer = csv.writer(f, delimiter=delimiter)

                        # Write header
                        if chk_include_confidence.isChecked():
                            writer.writerow(['Line', 'Text', 'Confidence'])
                            for idx, seg in enumerate(self.segments, 1):
                                conf_str = f"{seg.confidence*100:.2f}%" if seg.confidence is not None else "N/A"
                                writer.writerow([idx, seg.text or "", conf_str])
                        else:
                            writer.writerow(['Line', 'Text'])
                            for idx, seg in enumerate(self.segments, 1):
                                writer.writerow([idx, seg.text or ""])
                    else:  # Plain text format
                        for seg in self.segments:
                            if seg.text:
                                f.write(seg.text + "\n")

            self.status_bar.showMessage(f"Exported: {file_path}")
            QMessageBox.information(self, "Export Complete", f"Transcription exported to:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{str(e)}")

    def _select_font(self):
        """Open font selection dialog."""
        font, ok = QFontDialog.getFont(self.text_editor.font(), self)
        if ok:
            self.text_editor.setFont(font)

    def _update_char_count(self):
        """Update character and word count."""
        text = self.text_editor.toPlainText()
        char_count = len(text)
        word_count = len(text.split())
        self.lbl_char_count.setText(f"Characters: {char_count} | Words: {word_count}")

    def _on_normalize_changed(self, state):
        """Handle background normalization checkbox change."""
        self.normalize_bg = (state == Qt.CheckState.Checked.value)
        # Reset OCR instance to reload with new settings
        self.ocr = None

    def _on_beams_changed(self, value):
        """Handle beam search value change."""
        self.num_beams = value

    def _on_max_length_changed(self, value):
        """Handle max length value change."""
        self.max_length = value

    def _on_device_changed(self):
        """Handle device selection change."""
        if self.radio_gpu.isChecked():
            self.device = "cuda"
        else:
            self.device = "cpu"
        # Reset OCR instance to reload on new device
        self.ocr = None
        self.status_bar.showMessage(f"Device set to: {self.device.upper()}")

    def _on_seg_method_changed(self, index):
        """Handle segmentation method selection change."""
        method = self.combo_seg_method.currentData()

        if method == "Kraken":
            # Show Kraken model selection controls
            self.lbl_kraken_model.setVisible(True)
            self.combo_kraken_model.setVisible(True)
            self.btn_browse_kraken.setVisible(True)
            # Hide HPP-specific parameters
            self.spin_sensitivity.setVisible(False)
            self.spin_min_height.setVisible(False)
            self.chk_morph.setVisible(False)
            # Hide labels for HPP parameters
            for i in range(self.spin_sensitivity.parent().layout().count()):
                widget = self.spin_sensitivity.parent().layout().itemAt(i).widget()
                if isinstance(widget, QLabel) and widget.text() in ["Threshold:", "Min Height:"]:
                    widget.setVisible(False)

            self.segmentation_method = "Kraken"
            self.status_bar.showMessage("Switched to Kraken segmentation (slower but more robust)")
        else:
            # Show HPP parameters
            self.lbl_kraken_model.setVisible(False)
            self.combo_kraken_model.setVisible(False)
            self.btn_browse_kraken.setVisible(False)
            self.spin_sensitivity.setVisible(True)
            self.spin_min_height.setVisible(True)
            self.chk_morph.setVisible(True)
            # Show labels for HPP parameters
            for i in range(self.spin_sensitivity.parent().layout().count()):
                widget = self.spin_sensitivity.parent().layout().itemAt(i).widget()
                if isinstance(widget, QLabel) and widget.text() in ["Threshold:", "Min Height:"]:
                    widget.setVisible(True)

            self.segmentation_method = "HPP"
            self.status_bar.showMessage("Switched to HPP segmentation (fast)")

    def _browse_kraken_model(self):
        """Browse for Kraken model file (.mlmodel)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Kraken Model File",
            "",
            "Kraken Models (*.mlmodel);;All Files (*)"
        )

        if file_path:
            # Add to combo box
            model_name = Path(file_path).stem
            self.combo_kraken_model.addItem(f"Custom: {model_name}", file_path)
            self.combo_kraken_model.setCurrentIndex(self.combo_kraken_model.count() - 1)
            self.kraken_model_path = file_path
            self.status_bar.showMessage(f"Selected Kraken model: {model_name}")

    def _browse_pylaia_model(self):
        """Browse for PyLaia model directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select PyLaia Model Directory",
            ""
        )

        if dir_path:
            # Verify it contains required files
            model_dir = Path(dir_path)
            required_files = ['best_model.pt', 'model_config.json', 'symbols.txt']
            missing_files = [f for f in required_files if not (model_dir / f).exists()]

            if missing_files:
                QMessageBox.warning(
                    self,
                    "Invalid PyLaia Model",
                    f"Selected directory is missing required files:\n{', '.join(missing_files)}\n\n"
                    f"A valid PyLaia model directory must contain:\n• best_model.pt\n• model_config.json\n• symbols.txt"
                )
                return

            # Add to combo box
            model_name = model_dir.stem
            self.combo_pylaia_model.addItem(f"Custom: {model_name}", str(dir_path))
            self.combo_pylaia_model.setCurrentIndex(self.combo_pylaia_model.count() - 1)
            self.status_bar.showMessage(f"Selected PyLaia model: {model_name}")

    def _open_multiple_images(self):
        """Open multiple images for batch processing."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Images",
            "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif);;All Files (*)"
        )

        if file_paths:
            self.image_list = [Path(p) for p in file_paths]
            self.current_image_index = 0
            self._load_image(self.image_list[0])
            self._update_navigation_ui()

    def _load_previous_image(self):
        """Load the previous image in the list."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self._load_image(self.image_list[self.current_image_index])
            self._update_navigation_ui()

    def _load_next_image(self):
        """Load the next image in the list."""
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self._load_image(self.image_list[self.current_image_index])
            self._update_navigation_ui()

    def _update_navigation_ui(self):
        """Update navigation buttons and label."""
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

    def _fit_to_window(self):
        """Fit image to window."""
        self.graphics_view.fit_in_view()

    def _zoom_in(self):
        """Zoom in."""
        if self.graphics_view.has_image():
            self.graphics_view.scale(1.25, 1.25)
            self.graphics_view._zoom += 1

    def _zoom_out(self):
        """Zoom out."""
        if self.graphics_view.has_image():
            self.graphics_view.scale(0.8, 0.8)
            self.graphics_view._zoom -= 1

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About TrOCR Transcription Tool",
            "<h3>TrOCR Transcription Tool</h3>"
            "<p>Professional handwritten text transcription using TrOCR models.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Seamless zoom and pan</li>"
            "<li>Automatic line segmentation</li>"
            "<li>Background normalization</li>"
            "<li>Font customization</li>"
            "<li>Multiple export formats</li>"
            "</ul>"
            "<p>Built with PyQt6 and Transformers</p>"
        )

    def dragEnterEvent(self, event):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle drop event."""
        urls = event.mimeData().urls()
        if urls:
            # Filter for supported image formats
            image_paths = [
                Path(url.toLocalFile()) for url in urls
                if Path(url.toLocalFile()).suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
            ]

            if image_paths:
                if len(image_paths) == 1:
                    # Single image - just load it
                    self._load_image(image_paths[0])
                else:
                    # Multiple images - setup navigation
                    self.image_list = image_paths
                    self.current_image_index = 0
                    self._load_image(self.image_list[0])
                    self._update_navigation_ui()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        # Navigation shortcuts
        if event.key() == Qt.Key.Key_Left or event.key() == Qt.Key.Key_PageUp:
            if self.btn_prev_image.isEnabled():
                self._load_previous_image()
        elif event.key() == Qt.Key.Key_Right or event.key() == Qt.Key.Key_PageDown:
            if self.btn_next_image.isEnabled():
                self._load_next_image()
        else:
            super().keyPressEvent(event)


def main():
    """Run the application."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look

    window = TranscriptionGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
