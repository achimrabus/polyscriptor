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

    def __init__(self, segments: List[LineSegment], ocr: TrOCRInference,
                 num_beams: int, max_length: int):
        super().__init__()
        self.segments = segments
        self.ocr = ocr
        self.num_beams = num_beams
        self.max_length = max_length
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

                # Transcribe line with confidence scores
                text, confidence, char_confidences = self.ocr.transcribe_line(
                    segment.image,
                    num_beams=self.num_beams,
                    max_length=self.max_length,
                    return_confidence=True
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

        # Initialize Qwen3 as None
        self.qwen3 = None
        self._current_qwen3_model = None  # Track current model to avoid reinitialization

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

        # Local models tab
        local_tab = QWidget()
        local_layout = QGridLayout(local_tab)

        local_layout.addWidget(QLabel("Model:"), 0, 0)
        self.combo_model = QComboBox()
        self._populate_models()
        local_layout.addWidget(self.combo_model, 0, 1, 1, 2)

        btn_browse_model = QPushButton("Browse...")
        btn_browse_model.clicked.connect(self._browse_model)
        local_layout.addWidget(btn_browse_model, 0, 3)

        self.model_tabs.addTab(local_tab, "Local")

        # HuggingFace models tab
        hf_tab = QWidget()
        hf_layout = QGridLayout(hf_tab)

        hf_layout.addWidget(QLabel("Model ID:"), 0, 0)

        # Editable combo box with history
        self.combo_hf_model = QComboBox()
        self.combo_hf_model.setEditable(True)
        self.combo_hf_model.setPlaceholderText("e.g., kazars24/trocr-base-handwritten-ru")
        self._populate_hf_model_history()
        hf_layout.addWidget(self.combo_hf_model, 0, 1, 1, 2)

        btn_validate_hf = QPushButton("Validate")
        btn_validate_hf.clicked.connect(self._validate_hf_model)
        hf_layout.addWidget(btn_validate_hf, 0, 3)

        # Model info display
        hf_layout.addWidget(QLabel("Model Info:"), 1, 0, Qt.AlignmentFlag.AlignTop)
        self.txt_model_info = QTextEdit()
        self.txt_model_info.setReadOnly(True)
        self.txt_model_info.setPlaceholderText("Model information will appear here...")
        self.txt_model_info.setMaximumHeight(100)
        hf_layout.addWidget(self.txt_model_info, 1, 1, 1, 3)

        self.model_tabs.addTab(hf_tab, "HuggingFace")

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

        # Background normalization
        self.chk_normalize = QCheckBox("Normalize Background")
        self.chk_normalize.setChecked(False)
        self.chk_normalize.stateChanged.connect(self._on_normalize_changed)
        device_settings_layout.addWidget(self.chk_normalize)
        device_settings_layout.addStretch()

        settings_layout.addLayout(device_settings_layout)

        # Inference parameters row
        inference_layout = QHBoxLayout()

        # Beam search
        inference_layout.addWidget(QLabel("Beam Search:"))
        self.spin_beams = QSpinBox()
        self.spin_beams.setRange(1, 10)
        self.spin_beams.setValue(4)
        self.spin_beams.valueChanged.connect(self._on_beams_changed)
        inference_layout.addWidget(self.spin_beams)
        inference_layout.addSpacing(20)

        # Max length
        inference_layout.addWidget(QLabel("Max Length:"))
        self.spin_max_length = QSpinBox()
        self.spin_max_length.setRange(64, 256)
        self.spin_max_length.setValue(128)
        self.spin_max_length.valueChanged.connect(self._on_max_length_changed)
        inference_layout.addWidget(self.spin_max_length)
        inference_layout.addStretch()

        settings_layout.addLayout(inference_layout)

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

        # Set default font to 12pt for better readability
        default_font = QFont()
        default_font.setPointSize(12)
        self.text_editor.setFont(default_font)

        layout.addWidget(self.text_editor)

        # Character count
        self.lbl_char_count = QLabel("Characters: 0 | Words: 0")
        self.text_editor.textChanged.connect(self._update_char_count)
        layout.addWidget(self.lbl_char_count)

        # Statistics panel
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout()

        stats_layout.addWidget(QLabel("Lines Processed:"), 0, 0)
        self.lbl_lines_processed = QLabel("0")
        stats_layout.addWidget(self.lbl_lines_processed, 0, 1)

        stats_layout.addWidget(QLabel("Avg Confidence:"), 0, 2)
        self.lbl_avg_confidence = QLabel("N/A")
        stats_layout.addWidget(self.lbl_avg_confidence, 0, 3)

        stats_layout.addWidget(QLabel("Processing Time:"), 1, 0)
        self.lbl_processing_time = QLabel("N/A")
        stats_layout.addWidget(self.lbl_processing_time, 1, 1)

        stats_layout.addWidget(QLabel("Segmentation:"), 1, 2)
        self.lbl_seg_method = QLabel("N/A")
        stats_layout.addWidget(self.lbl_seg_method, 1, 3)

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
            is_qwen3 = QWEN3_AVAILABLE and (self.model_tabs.currentIndex() == self.model_tabs.count() - 1)

            if is_qwen3:
                # Qwen3 mode: Enable process button directly (no segmentation needed)
                self.btn_process.setEnabled(True)
            else:
                # TrOCR mode: Enable segment button
                self.btn_segment.setEnabled(True)

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
        """Handle model tab changes - show/hide segmentation controls for Qwen3."""
        # Check if Qwen3 tab (it's the last tab if available)
        is_qwen3 = QWEN3_AVAILABLE and (index == self.model_tabs.count() - 1)

        # Hide/show entire segmentation group in Qwen3 mode (no segmentation needed)
        if hasattr(self, 'seg_group'):
            self.seg_group.setVisible(not is_qwen3)

        # Update button text based on mode
        if is_qwen3:
            self.btn_process.setText("Transcribe Page")
            # If an image is already loaded, enable process button
            if self.current_image_path is not None:
                self.btn_process.setEnabled(True)
        else:
            self.btn_process.setText("Process All Lines")
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

            # Transcribe entire page
            result = self.qwen3.transcribe_page(
                page_image,
                prompt=prompt,
                max_new_tokens=max_tokens,
                return_confidence=estimate_confidence
            )

            # Display result
            self.text_editor.setPlainText(result.text)

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

    def _process_all_lines(self):
        """Process all detected lines with OCR (or entire page with Qwen3)."""

        # Check if using Qwen3
        is_qwen3 = QWEN3_AVAILABLE and (self.model_tabs.currentIndex() == self.model_tabs.count() - 1)

        if is_qwen3:
            # Use Qwen3 VLM (no line segmentation)
            self._process_with_qwen3()
            return

        # Standard line-based processing (TrOCR)
        if not self.segments:
            QMessageBox.warning(
                self,
                "No Lines Available",
                "No text lines have been detected yet.\n\n"
                "Please click 'Detect Lines' first to segment the image into text lines."
            )
            return

        # Determine which model source to use (Local or HuggingFace)
        is_hf_tab = (self.model_tabs.currentIndex() == 1)

        if is_hf_tab:
            # HuggingFace model
            model_id = self.combo_hf_model.currentText().strip()
            if not model_id:
                QMessageBox.warning(self, "Warning", "Please enter a HuggingFace model ID!")
                return
            model_path = model_id
            is_huggingface = True
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

        try:
            # Initialize OCR if needed (or if model changed)
            if self.ocr is None or self.ocr.model_path != model_path:
                self.status_bar.showMessage(f"Loading model on {self.device.upper()}...")
                self.ocr = TrOCRInference(
                    model_path,
                    device=self.device,
                    normalize_bg=self.normalize_bg,
                    is_huggingface=is_huggingface
                )

            # Start processing in background thread
            self.ocr_worker = OCRWorker(
                self.segments, self.ocr,
                self.num_beams, self.max_length
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

        format_layout.addWidget(radio_txt)
        format_layout.addWidget(radio_csv)
        format_layout.addWidget(radio_tsv)
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

        if dialog.exec() != QDialog.DialogResult.Accepted:
            return

        # Determine export format
        if radio_csv.isChecked():
            file_filter = "CSV Files (*.csv)"
            default_ext = ".csv"
            delimiter = ","
        elif radio_tsv.isChecked():
            file_filter = "TSV Files (*.tsv)"
            default_ext = ".tsv"
            delimiter = "\t"
        else:
            file_filter = "Text Files (*.txt)"
            default_ext = ".txt"
            delimiter = None

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
