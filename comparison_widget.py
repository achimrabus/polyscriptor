"""
Comparison Widget for Side-by-Side Transcription Comparison

Provides a collapsible panel for comparing HTR engine outputs or evaluating
against ground truth. Integrates into the main transcription GUI.

Features:
- Engine vs Engine comparison
- Engine vs Ground Truth evaluation
- Line-by-line navigation
- Color-coded diff visualization
- CSV export of metrics

Author: Claude Code
Date: 2025-11-05
"""

import csv
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
from PIL import Image

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QComboBox, QFileDialog, QRadioButton,
    QButtonGroup, QMessageBox, QStackedWidget, QCheckBox, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QColor, QTextCharFormat, QFont

from transcription_metrics import (
    ComparisonMode,
    TranscriptionMetrics,
    LineMetrics,
)
from htr_engine_base import HTREngine, TranscriptionResult, get_global_registry


# Worker thread for running comparison engine transcription
class ComparisonWorker(QThread):
    """Background worker for transcribing with comparison engine."""

    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(list)  # List of TranscriptionResult
    error = pyqtSignal(str)

    def __init__(self, engine: HTREngine, line_images: List[np.ndarray]):
        super().__init__()
        self.engine = engine
        self.line_images = line_images

    def run(self):
        """Transcribe all lines with comparison engine."""
        results = []
        try:
            for i, img in enumerate(self.line_images):
                # Engines have different input expectations
                # Pass the image in its original format (numpy array)
                # Engines will handle conversion internally if needed
                result = self.engine.transcribe_line(img)
                results.append(result)

                # Emit progress
                self.progress.emit(i + 1, len(self.line_images))

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class ComparisonTextEdit(QTextEdit):
    """
    Text edit widget with colored diff highlighting.

    Colors:
    - Green: Matching characters
    - Red: Substitutions
    - Blue background: Insertions (hypothesis only)
    - Yellow background: Deletions (reference only)
    """

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Courier New", 11))
        self.setMinimumHeight(100)

    def display_with_diff(self, text: str, metrics: Optional[LineMetrics], is_reference: bool):
        """
        Display text with colored diff highlighting.

        Args:
            text: Text to display
            metrics: LineMetrics with diff operations (None for plain display)
            is_reference: True for left panel (reference), False for right (hypothesis)
        """
        self.clear()

        if not metrics:
            # Plain display without diff
            self.setPlainText(text)
            return

        cursor = self.textCursor()
        cursor.beginEditBlock()

        # Build position-to-operation mapping
        if is_reference:
            # For reference: track by ref_pos
            op_map = {}
            for op in metrics.diff_ops:
                if op.operation == 'equal':
                    op_map[op.ref_pos] = op
                elif op.operation == 'replace':
                    op_map[op.ref_pos] = op
                elif op.operation == 'delete':
                    op_map[op.ref_pos] = op
                # Insertions don't appear in reference
        else:
            # For hypothesis: track by hyp_pos
            op_map = {}
            for op in metrics.diff_ops:
                if op.operation == 'equal':
                    op_map[op.hyp_pos] = op
                elif op.operation == 'replace':
                    op_map[op.hyp_pos] = op
                elif op.operation == 'insert':
                    op_map[op.hyp_pos] = op
                # Deletions don't appear in hypothesis

        # Render text with formatting
        for i, char in enumerate(text):
            fmt = QTextCharFormat()
            fmt.setFont(QFont("Courier New", 11))

            if i in op_map:
                op = op_map[i]

                if op.operation == 'equal':
                    # Green for matches
                    fmt.setForeground(QColor(0, 128, 0))

                elif op.operation == 'replace':
                    # Red bold for substitutions
                    fmt.setForeground(QColor(200, 0, 0))
                    fmt.setFontWeight(QFont.Weight.Bold)

                elif op.operation == 'delete' and is_reference:
                    # Yellow background for deletions (reference only)
                    fmt.setBackground(QColor(255, 255, 150))
                    fmt.setForeground(QColor(100, 100, 0))

                elif op.operation == 'insert' and not is_reference:
                    # Blue background for insertions (hypothesis only)
                    fmt.setBackground(QColor(173, 216, 230))
                    fmt.setForeground(QColor(0, 0, 150))

            cursor.insertText(char, fmt)

        cursor.endEditBlock()


class ComparisonWidget(QWidget):
    """
    Collapsible comparison panel for transcription comparison.

    Replaces statistics panel when comparison mode is active.
    Allows comparing two engines or evaluating against ground truth.
    """

    comparison_closed = pyqtSignal()  # Emitted when user closes comparison
    status_message = pyqtSignal(str)  # For status bar messages

    def __init__(
        self,
        base_engine: HTREngine,
        line_segments: List,
        line_images: List[np.ndarray],
        parent=None
    ):
        super().__init__(parent)
        # Store main GUI reference separately — parentWidget() changes after
        # addWidget() re-parents us to the QStackedWidget
        self._main_gui = parent
        self.base_engine = base_engine
        self.line_segments = line_segments
        self.line_images = line_images
        self.comparison_engine: Optional[HTREngine] = None
        self.ground_truth: Optional[List[str]] = None
        self.current_line_idx = 0
        self.base_transcriptions: List[str] = []
        self.comparison_transcriptions: List[str] = []
        self.comparison_worker: Optional[ComparisonWorker] = None
        self._sync_scroll: bool = True   # scroll panels together by default

        self.setup_ui()

    def setup_ui(self):
        """Build the comparison UI — compact toolbar + full-height text panels."""
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Compact toolbar (single row) ──────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        # Mode radio buttons
        self.mode_button_group = QButtonGroup()
        self.engine_mode_radio = QRadioButton("Engine vs Engine")
        self.gt_mode_radio = QRadioButton("Engine vs GT")
        self.engine_mode_radio.setChecked(True)
        self.engine_mode_radio.toggled.connect(self.on_mode_changed)
        self.mode_button_group.addButton(self.engine_mode_radio)
        self.mode_button_group.addButton(self.gt_mode_radio)
        toolbar.addWidget(self.engine_mode_radio)
        toolbar.addWidget(self.gt_mode_radio)

        sep = QLabel("│")
        sep.setStyleSheet("color: #bbb; padding: 0 2px;")
        toolbar.addWidget(sep)

        # Stacked control area — switches with mode radio
        self.mode_stack = QStackedWidget()
        self.mode_stack.setMaximumHeight(32)

        # Page 0 — engine controls
        engine_page = QWidget()
        ep = QHBoxLayout(engine_page)
        ep.setContentsMargins(0, 0, 0, 0)
        ep.setSpacing(4)
        self.engine_combo = QComboBox()
        self._populate_engines()
        ep.addWidget(self.engine_combo)
        self.load_engine_btn = QPushButton("Load & Transcribe")
        self.load_engine_btn.clicked.connect(self.load_and_transcribe_engine)
        ep.addWidget(self.load_engine_btn)
        self.unload_engine_btn = QPushButton("Unload")
        self.unload_engine_btn.clicked.connect(self.unload_comparison_engine)
        self.unload_engine_btn.setEnabled(False)
        ep.addWidget(self.unload_engine_btn)
        ep.addStretch()
        self.mode_stack.addWidget(engine_page)

        # Page 1 — ground truth controls
        gt_page = QWidget()
        gp = QHBoxLayout(gt_page)
        gp.setContentsMargins(0, 0, 0, 0)
        gp.setSpacing(4)
        self.load_gt_btn = QPushButton("Load TXT…")
        self.load_gt_btn.clicked.connect(self.load_ground_truth)
        gp.addWidget(self.load_gt_btn)
        self.clear_gt_btn = QPushButton("Clear")
        self.clear_gt_btn.clicked.connect(self.clear_ground_truth)
        self.clear_gt_btn.setEnabled(False)
        gp.addWidget(self.clear_gt_btn)
        self.gt_file_label = QLabel("No file loaded")
        self.gt_file_label.setStyleSheet("color: gray; font-size: 9pt;")
        gp.addWidget(self.gt_file_label, 1)
        self.mode_stack.addWidget(gt_page)

        toolbar.addWidget(self.mode_stack, 1)

        # Close button (compact ✕)
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedWidth(28)
        self.close_btn.setToolTip("Close comparison")
        self.close_btn.clicked.connect(self.close_comparison)
        toolbar.addWidget(self.close_btn)

        layout.addLayout(toolbar)

        # ── Side-by-side text panels (stretch = all remaining space) ─────────
        panels = QHBoxLayout()
        panels.setSpacing(6)

        # Left panel (base / reference)
        left_col = QVBoxLayout()
        left_col.setSpacing(2)
        self.left_label = QLabel(f"{self.base_engine.get_name()} (Base)")
        self.left_label.setStyleSheet("font-weight: bold;")
        self.left_text = ComparisonTextEdit()
        self.left_metrics_label = QLabel("Length: -")
        self.left_metrics_label.setStyleSheet("color: #666; font-size: 9pt;")
        left_col.addWidget(self.left_label)
        left_col.addWidget(self.left_text, 1)
        left_col.addWidget(self.left_metrics_label)

        # Right panel (comparison / hypothesis)
        right_col = QVBoxLayout()
        right_col.setSpacing(2)
        self.right_label = QLabel("Load comparison engine or ground truth")
        self.right_label.setStyleSheet("font-weight: bold;")
        self.right_text = ComparisonTextEdit()
        self.right_metrics_label = QLabel(
            self._placeholder_metrics_text(ComparisonMode.ENGINE_COMPARISON)
        )
        self.right_metrics_label.setStyleSheet("color: #666; font-size: 9pt;")
        right_col.addWidget(self.right_label)
        right_col.addWidget(self.right_text, 1)
        right_col.addWidget(self.right_metrics_label)

        panels.addLayout(left_col, 1)
        panels.addLayout(right_col, 1)
        layout.addLayout(panels, 1)

        # ── Bottom row: navigation + sync + legend + export ──────────────────
        bottom = QHBoxLayout()
        bottom.setSpacing(6)

        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self.show_previous_line)
        bottom.addWidget(self.prev_btn)

        self.line_label = QLabel("Line 0 of 0")
        self.line_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom.addWidget(self.line_label)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.show_next_line)
        bottom.addWidget(self.next_btn)

        bottom.addSpacing(8)

        # Scroll sync toggle
        self.sync_scroll_cb = QCheckBox("Sync scroll")
        self.sync_scroll_cb.setChecked(True)
        self.sync_scroll_cb.setToolTip("Keep both panels scrolled to the same position")
        self.sync_scroll_cb.toggled.connect(self._on_sync_toggle)
        bottom.addWidget(self.sync_scroll_cb)

        bottom.addSpacing(8)

        # Diff legend (inline, compact)
        for color, bg, label_text in [
            ("green", None, "Match"),
            ("red", None, "Subst."),
            (None, "#ADD8E6", "Ins."),
            (None, "#FFFF96", "Del."),
        ]:
            style = "font-size: 9pt; padding: 1px 4px; border-radius: 2px;"
            if color:
                style += f" color: {color};"
            if bg:
                style += f" background: {bg};"
            lbl = QLabel(f"■ {label_text}")
            lbl.setStyleSheet(style)
            bottom.addWidget(lbl)

        bottom.addStretch()

        # Export dropdown: HTML colored diff / CSV metrics table
        export_menu = QMenu(self)
        export_menu.addAction("Export HTML (colored diff)…", self.export_html)
        export_menu.addAction("Export CSV (metrics table)…", self.export_csv)
        self.export_btn = QPushButton("📊 Export")
        self.export_btn.setMenu(export_menu)
        bottom.addWidget(self.export_btn)

        layout.addLayout(bottom)

        # Connect scrollbars for sync (both text edits must exist before this)
        self.left_text.verticalScrollBar().valueChanged.connect(self._on_left_scroll)
        self.right_text.verticalScrollBar().valueChanged.connect(self._on_right_scroll)

        self.setLayout(layout)

    def _populate_engines(self):
        """Populate engine dropdown with available engines."""
        registry = get_global_registry()
        available = registry.get_available_engines()

        base_name = self.base_engine.get_name()
        for engine in available:
            if engine.get_name() != base_name:  # Don't include current engine
                self.engine_combo.addItem(engine.get_name())

        if self.engine_combo.count() == 0:
            self.engine_combo.addItem("No other engines available")
            self.load_engine_btn.setEnabled(False)

    def on_mode_changed(self):
        """Handle mode change between Engine vs Engine and Engine vs GT."""
        self.mode_stack.setCurrentIndex(0 if self.engine_mode_radio.isChecked() else 1)
        self.update_display()

    def _active_comparison_mode(self) -> ComparisonMode:
        """Return the semantic mode that matches the currently active data."""
        if self.gt_mode_radio.isChecked() and self.ground_truth:
            return ComparisonMode.GROUND_TRUTH
        return ComparisonMode.ENGINE_COMPARISON

    def _display_labels(self, mode: Optional[ComparisonMode] = None):
        return TranscriptionMetrics.get_display_labels(mode or self._active_comparison_mode())

    def _placeholder_metrics_text(self, mode: Optional[ComparisonMode] = None) -> str:
        labels = self._display_labels(mode)
        return f"{labels.char_rate}: - | {labels.word_rate}: - | {labels.match_rate}: -"

    def _format_display_metrics(self, metrics, mode: ComparisonMode) -> str:
        labels = self._display_labels(mode)
        return (
            f"{labels.char_rate}: {metrics.char_rate:.2f}% | "
            f"{labels.word_rate}: {metrics.word_rate:.2f}% | "
            f"{labels.match_rate}: {metrics.match_percent:.2f}%"
        )

    def _metric_color(self, value: float, mode: ComparisonMode) -> str:
        green_cutoff, amber_cutoff = self._display_labels(mode).color_thresholds
        if value < green_cutoff:
            return "#007700"
        if value < amber_cutoff:
            return "#cc7700"
        return "#cc0000"

    def _refresh_line_images_from_parent(self):
        """Re-read line segments and images from parent GUI (handles late layout analysis)."""
        parent = self._main_gui
        if parent is None:
            return
        # Try to get current line segments from the main GUI
        segments = getattr(parent, 'line_segments', None)
        current_image = getattr(parent, 'current_image', None)
        if not segments:
            return  # No segments available yet — keep whatever we have
        line_images = []
        for seg in segments:
            if getattr(seg, 'image', None) is not None:
                line_images.append(np.array(seg.image))
            elif current_image:
                x1, y1, x2, y2 = seg.bbox
                img_np = np.array(current_image)
                line_images.append(img_np[y1:y2, x1:x2])
        if line_images:
            self.line_segments = segments
            self.line_images = line_images

    def load_and_transcribe_engine(self):
        """Load comparison engine and transcribe all lines."""
        engine_name = self.engine_combo.currentText()

        if engine_name == "No other engines available":
            return

        registry = get_global_registry()

        try:
            # Get engine instance by name
            self.comparison_engine = registry.get_engine_by_name(engine_name)

            if not self.comparison_engine:
                raise Exception(f"Engine '{engine_name}' not found")

            # Require model to be pre-loaded in the main panel.
            # This ensures the correct model + config is used (e.g. the right
            # CRNN-CTC checkpoint). The user loads the model via the HTR Engine
            # section first, then switches to comparison mode.
            if not self.comparison_engine.is_model_loaded():
                QMessageBox.warning(
                    self, "Model Not Loaded",
                    f"Please load a model for '{engine_name}' in the main HTR Engine "
                    f"panel first (select the engine, configure the model path, and "
                    f"click 'Load Model'), then return to comparison mode."
                )
                self.status_message.emit(f"{engine_name}: no model loaded — load it in the main panel first")
                return

            self.status_message.emit(f"Using already-loaded {engine_name} model")

            # Refresh line images in case layout analysis was run after comparison
            # mode was activated (captures the latest segments)
            self._refresh_line_images_from_parent()

            if not self.line_images:
                QMessageBox.warning(
                    self, "No Line Images",
                    "No line segments found. Please run layout analysis first."
                )
                return

            # Start transcription in background
            self.status_message.emit(f"Transcribing {len(self.line_images)} lines with {engine_name}...")
            self.comparison_worker = ComparisonWorker(
                self.comparison_engine,
                self.line_images
            )
            self.comparison_worker.progress.connect(self.on_transcription_progress)
            self.comparison_worker.finished.connect(self.on_transcription_finished)
            self.comparison_worker.error.connect(self.on_transcription_error)

            # Disable buttons during transcription
            self.load_engine_btn.setEnabled(False)
            self.engine_combo.setEnabled(False)

            self.comparison_worker.start()

        except Exception as e:
            QMessageBox.warning(self, "Error",
                               f"Failed to load {engine_name}: {str(e)}")
            self.status_message.emit("Error loading comparison engine")

    def on_transcription_progress(self, current: int, total: int):
        """Handle transcription progress update."""
        self.status_message.emit(f"Transcribing: {current}/{total} lines...")

    def on_transcription_finished(self, results: List[TranscriptionResult]):
        """Handle transcription completion."""
        raw = [r.text for r in results]
        base_count = len(self.base_transcriptions)

        # Normalize count mismatches between page-level and line-level engines
        if base_count == 1 and len(raw) > 1:
            # Page-based base (e.g. Qwen) vs line-based comparison (e.g. CRNN-CTC):
            # join all comparison lines for a page-level diff
            self.comparison_transcriptions = ["\n".join(raw)]
        elif base_count > 1 and len(raw) == 1:
            # Line-based base vs page-based comparison: split comparison by newlines
            split = raw[0].split("\n")
            while len(split) < base_count:
                split.append("")
            self.comparison_transcriptions = split[:base_count]
        else:
            self.comparison_transcriptions = raw

        self.right_label.setText(f"{self.engine_combo.currentText()} (Comparison)")

        # Re-enable buttons
        self.load_engine_btn.setEnabled(True)
        self.engine_combo.setEnabled(True)
        self.unload_engine_btn.setEnabled(True)

        # Update display
        self.update_display()
        self.status_message.emit(f"Comparison ready! {len(results)} lines transcribed.")

        QMessageBox.information(
            self,
            "Success",
            f"Transcribed {len(results)} lines with {self.engine_combo.currentText()}"
        )

    def on_transcription_error(self, error_msg: str):
        """Handle transcription error."""
        self.load_engine_btn.setEnabled(True)
        self.engine_combo.setEnabled(True)
        self.status_message.emit("Transcription failed")

        QMessageBox.warning(self, "Transcription Error", f"Error: {error_msg}")

    def unload_comparison_engine(self):
        """Unload comparison engine to free memory."""
        if self.comparison_engine:
            self.comparison_engine.unload_model()
            self.comparison_engine = None
            self.comparison_transcriptions = []

            self.right_label.setText("Load comparison engine or ground truth")
            self.right_text.clear()
            self.right_metrics_label.setText(self._placeholder_metrics_text())

            self.unload_engine_btn.setEnabled(False)
            self.status_message.emit("Comparison engine unloaded")

    def load_ground_truth(self):
        """Load ground truth from plain TXT file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Ground Truth", "", "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.rstrip('\n\r') for line in f]

                if len(lines) != len(self.line_segments):
                    QMessageBox.warning(
                        self, "Warning",
                        f"Ground truth has {len(lines)} lines but document has "
                        f"{len(self.line_segments)} lines.\n\n"
                        f"Comparison may be inaccurate. Please ensure ground truth "
                        f"matches line segmentation order."
                    )

                self.ground_truth = lines
                self.gt_file_label.setText(f"✓ {Path(file_path).name}")
                self.clear_gt_btn.setEnabled(True)

                # Update display
                self.update_display()
                self.status_message.emit(f"Ground truth loaded: {len(lines)} lines")

            except Exception as e:
                QMessageBox.warning(self, "Error",
                                   f"Failed to load ground truth: {str(e)}")

    def clear_ground_truth(self):
        """Clear loaded ground truth."""
        self.ground_truth = None
        self.gt_file_label.setText("No file loaded")
        self.clear_gt_btn.setEnabled(False)
        self.update_display()
        self.status_message.emit("Ground truth cleared")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Remove blank lines and join content lines with a single space.

        Joining with space (not \\n) keeps line-level comparison metrics on flat
        text, avoiding inflation from misaligned newline positions between
        page-level and line-level engines. QTextEdit word-wraps the flat text
        for display automatically.
        """
        return " ".join(line for line in text.splitlines() if line.strip())

    def set_base_transcriptions(self, transcriptions: List[str]):
        """Set transcriptions from base engine."""
        self.base_transcriptions = transcriptions
        self.current_line_idx = 0
        self.update_display()

    def show_previous_line(self):
        """Navigate to previous line."""
        if self.current_line_idx > 0:
            self.current_line_idx -= 1
            self.update_display()

    def show_next_line(self):
        """Navigate to next line."""
        if self.current_line_idx < len(self.base_transcriptions) - 1:
            self.current_line_idx += 1
            self.update_display()

    def update_display(self):
        """Update the comparison display for current line."""
        if not self.base_transcriptions:
            return

        # Use transcription count as authoritative total (handles page-vs-line mismatch)
        total_lines = len(self.base_transcriptions)
        self.current_line_idx = min(self.current_line_idx, total_lines - 1)
        self.line_label.setText(f"Line {self.current_line_idx + 1} of {total_lines}")
        self.prev_btn.setEnabled(self.current_line_idx > 0)
        self.next_btn.setEnabled(self.current_line_idx < total_lines - 1)

        # Get base transcription (safe index), strip blank lines for display
        base_text = self._normalize_text(self.base_transcriptions[self.current_line_idx])

        # Determine reference and hypothesis based on mode
        if self.gt_mode_radio.isChecked() and self.ground_truth:
            # Engine vs Ground Truth mode
            if self.current_line_idx < len(self.ground_truth):
                reference = self.ground_truth[self.current_line_idx]
                hypothesis = base_text
                self.left_label.setText("Ground Truth (Reference)")
                self.right_label.setText(f"{self.base_engine.get_name()} (Hypothesis)")
            else:
                # No GT for this line
                self.left_text.setPlainText("(No ground truth for this line)")
                self.right_text.setPlainText(base_text)
                self.left_metrics_label.setText("")
                self.right_metrics_label.setText("")
                return

        elif self.comparison_transcriptions:
            # Engine vs Engine mode (safe index with fallback)
            reference = base_text
            hyp_idx = min(self.current_line_idx, len(self.comparison_transcriptions) - 1)
            hypothesis = self._normalize_text(self.comparison_transcriptions[hyp_idx])
            self.left_label.setText(f"{self.base_engine.get_name()} (Base)")
            self.right_label.setText(f"{self.engine_combo.currentText()} (Comparison)")
        else:
            # No comparison available - show plain text
            self.left_text.setPlainText(base_text)
            self.right_text.setPlainText("Load comparison engine or ground truth")
            self.left_metrics_label.setText(f"Length: {len(base_text)} chars")
            self.right_metrics_label.setText("")
            return

        # Calculate metrics
        mode = self._active_comparison_mode()
        metrics = TranscriptionMetrics.compare_lines(reference, hypothesis)
        display_metrics = TranscriptionMetrics.get_display_metrics(metrics, mode)

        # Display texts with diff highlighting
        self.left_text.display_with_diff(reference, metrics, is_reference=True)
        self.right_text.display_with_diff(hypothesis, metrics, is_reference=False)

        # Update metrics labels
        self.left_metrics_label.setText(f"Length: {len(reference)} chars")
        self.right_metrics_label.setText(self._format_display_metrics(display_metrics, mode))

    def export_csv(self):
        """Export comparison metrics with GT-aware / GT-free terminology."""
        data = self._get_comparison_data()
        if not data:
            QMessageBox.warning(self, "Error",
                                "Load comparison engine or ground truth first!")
            return
        references, hypotheses, ref_label, hyp_label = data
        mode = self._active_comparison_mode()
        labels = self._display_labels(mode)

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Comparison CSV", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"

        try:
            import datetime
            line_count = min(len(references), len(hypotheses))
            all_metrics = [
                TranscriptionMetrics.compare_lines(
                    references[i] if i < len(references) else "",
                    hypotheses[i] if i < len(hypotheses) else ""
                )
                for i in range(line_count)
            ]
            display_metrics = [
                TranscriptionMetrics.get_display_metrics(metric, mode)
                for metric in all_metrics
            ]
            summary = TranscriptionMetrics.calculate_summary_metrics(
                references[:line_count],
                hypotheses[:line_count],
                mode,
            )

            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Metadata block
                writer.writerow(["# Generated", datetime.datetime.now().strftime('%Y-%m-%d %H:%M')])
                writer.writerow(["# Reference", ref_label])
                writer.writerow(["# Hypothesis", hyp_label])
                writer.writerow(["# Lines", summary.line_count])
                writer.writerow([f"# {labels.macro_char_rate} (%)", f"{summary.macro_char_rate:.4f}"])
                writer.writerow([
                    f"# {labels.micro_char_rate} (%)",
                    f"{summary.micro_char_rate:.4f}",
                    labels.micro_char_rate_note,
                ])
                writer.writerow([f"# {labels.macro_word_rate} (%)", f"{summary.macro_word_rate:.4f}"])
                writer.writerow([f"# Avg {labels.match_rate} (%)", f"{summary.avg_match_percent:.4f}"])
                writer.writerow([])
                # Per-line data
                writer.writerow([
                    "Line", ref_label, hyp_label,
                    "Ref_Chars", "Hyp_Chars", "Edit_Distance",
                    labels.char_rate_column, labels.word_rate_column, f"{labels.match_rate} (%)"
                ])
                for i, (raw_metrics, display) in enumerate(zip(all_metrics, display_metrics)):
                    ref = references[i] if i < len(references) else ""
                    hyp = hypotheses[i] if i < len(hypotheses) else ""
                    writer.writerow([
                        i + 1, ref, hyp,
                        len(ref), len(hyp), raw_metrics.edit_distance,
                        f"{display.char_rate:.4f}",
                        f"{display.word_rate:.4f}",
                        f"{display.match_percent:.4f}",
                    ])

            self.status_message.emit(f"CSV exported: {Path(file_path).name}")
            QMessageBox.information(self, "Export Complete", f"CSV saved:\n{file_path}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export CSV: {e}")

    def close_comparison(self):
        """Signal parent to close comparison (parent handles all cleanup)."""
        if self.comparison_worker and self.comparison_worker.isRunning():
            self.comparison_worker.terminate()
            self.comparison_worker.wait()
        self.comparison_closed.emit()

    # ------------------------------------------------------------------
    # Scroll sync
    # ------------------------------------------------------------------

    def _on_sync_toggle(self, checked: bool):
        self._sync_scroll = checked

    def _on_left_scroll(self, value: int):
        if not self._sync_scroll:
            return
        sb = self.right_text.verticalScrollBar()
        sb.blockSignals(True)
        sb.setValue(value)
        sb.blockSignals(False)

    def _on_right_scroll(self, value: int):
        if not self._sync_scroll:
            return
        sb = self.left_text.verticalScrollBar()
        sb.blockSignals(True)
        sb.setValue(value)
        sb.blockSignals(False)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _get_comparison_data(self):
        """Return (references, hypotheses, ref_label, hyp_label) or None if not ready."""
        if self.gt_mode_radio.isChecked() and self.ground_truth:
            return (self.ground_truth, self.base_transcriptions,
                    "Ground Truth", self.base_engine.get_name())
        if self.comparison_transcriptions:
            return (self.base_transcriptions, self.comparison_transcriptions,
                    self.base_engine.get_name(), self.engine_combo.currentText())
        return None

    def _diff_to_html(self, text: str, metrics, is_reference: bool) -> str:
        """Render text as an HTML fragment with inline diff highlighting."""
        import html as _html
        if not metrics:
            return _html.escape(text)

        if is_reference:
            op_map = {op.ref_pos: op for op in metrics.diff_ops
                      if op.operation in ('equal', 'replace', 'delete')}
        else:
            op_map = {op.hyp_pos: op for op in metrics.diff_ops
                      if op.operation in ('equal', 'replace', 'insert')}

        parts = []
        for i, char in enumerate(text):
            esc = _html.escape(char)
            op = op_map.get(i)
            if op is None:
                parts.append(esc)
            elif op.operation == 'equal':
                parts.append(f'<span style="color:#007700">{esc}</span>')
            elif op.operation == 'replace':
                parts.append(f'<span style="color:#cc0000;font-weight:bold">{esc}</span>')
            elif op.operation == 'delete' and is_reference:
                parts.append(f'<span style="background:#FFFF96;color:#666">{esc}</span>')
            elif op.operation == 'insert' and not is_reference:
                parts.append(f'<span style="background:#ADD8E6;color:#000066">{esc}</span>')
            else:
                parts.append(esc)
        return ''.join(parts)

    def export_html(self):
        """Export a self-contained HTML report with GT-aware / GT-free terminology."""
        import datetime
        import html as _html

        data = self._get_comparison_data()
        if not data:
            QMessageBox.warning(self, "Error",
                                "Load comparison engine or ground truth first!")
            return
        references, hypotheses, ref_label, hyp_label = data
        mode = self._active_comparison_mode()
        labels = self._display_labels(mode)

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Comparison HTML", "", "HTML Files (*.html);;All Files (*)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".html"):
            file_path += ".html"

        try:
            line_count = min(len(references), len(hypotheses))
            all_metrics = [
                TranscriptionMetrics.compare_lines(
                    references[i] if i < len(references) else "",
                    hypotheses[i] if i < len(hypotheses) else ""
                )
                for i in range(line_count)
            ]
            display_metrics = [
                TranscriptionMetrics.get_display_metrics(metric, mode)
                for metric in all_metrics
            ]
            summary = TranscriptionMetrics.calculate_summary_metrics(
                references[:line_count],
                hypotheses[:line_count],
                mode,
            )

            # Build per-line HTML blocks
            line_blocks = []
            for i, (raw_metrics, display) in enumerate(zip(all_metrics, display_metrics)):
                ref = references[i] if i < len(references) else ""
                hyp = hypotheses[i] if i < len(hypotheses) else ""
                left_html = self._diff_to_html(ref, raw_metrics, is_reference=True)
                right_html = self._diff_to_html(hyp, raw_metrics, is_reference=False)
                color = self._metric_color(display.char_rate, mode)
                line_blocks.append(f"""  <div class="line-block">
    <div class="line-header">
      <span>Line {i + 1}</span>
      <span style="color:{color};font-weight:bold">{_html.escape(labels.char_rate)} {display.char_rate:.1f}%&nbsp;&nbsp;{_html.escape(labels.word_rate)} {display.word_rate:.1f}%&nbsp;&nbsp;{_html.escape(labels.match_rate)} {display.match_percent:.1f}%&nbsp;&nbsp;ED {display.edit_distance}</span>
    </div>
    <div class="line-body">
      <div class="panel"><div class="panel-label">{_html.escape(ref_label)}</div>{left_html}</div>
      <div class="panel"><div class="panel-label">{_html.escape(hyp_label)}</div>{right_html}</div>
    </div>
  </div>""")

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            macro_color = self._metric_color(summary.macro_char_rate, mode)
            micro_color = self._metric_color(summary.micro_char_rate, mode)

            html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Comparison: {_html.escape(ref_label)} vs {_html.escape(hyp_label)}</title>
<style>
  body {{ font-family: sans-serif; max-width: 1500px; margin: 0 auto; padding: 20px; color: #222; }}
  h1 {{ font-size: 1.3em; color: #333; margin-bottom: 4px; }}
  .meta {{ color: #888; font-size: 0.82em; margin-bottom: 16px; }}
  .summary {{ background: #f7f7f7; border: 1px solid #ddd; border-radius: 6px;
              padding: 14px 18px; margin-bottom: 18px; }}
  .summary table {{ border-collapse: collapse; font-size: 0.88em; }}
  .summary td, .summary th {{ padding: 5px 16px; border: 1px solid #ccc; text-align: right; }}
  .summary th {{ background: #eee; text-align: left; font-weight: normal; color: #555; }}
  .summary .val {{ font-weight: bold; font-size: 1.05em; }}
  .note {{ font-size: 0.76em; color: #999; margin-top: 8px; }}
  .legend {{ display: flex; gap: 12px; margin-bottom: 14px; font-size: 0.83em; flex-wrap: wrap; }}
  .legend span {{ padding: 2px 8px; border-radius: 3px; }}
  .line-block {{ margin: 4px 0; border: 1px solid #ddd; border-radius: 4px; overflow: hidden; }}
  .line-header {{ background: #f2f2f2; padding: 3px 12px; font-size: 0.8em; color: #555;
                  display: flex; justify-content: space-between; border-bottom: 1px solid #ddd; }}
  .line-body {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: #ccc; }}
  .panel {{ background: white; padding: 6px 12px; font-family: "Courier New", monospace;
            font-size: 0.9em; line-height: 1.6; word-break: break-all; min-height: 28px; }}
  .panel-label {{ font-family: sans-serif; font-size: 0.72em; color: #aaa; margin-bottom: 2px; }}
</style>
</head>
<body>
<h1>Transcription Comparison: {_html.escape(ref_label)} vs {_html.escape(hyp_label)}</h1>
<p class="meta">Generated {timestamp} &nbsp;·&nbsp; {summary.line_count} lines &nbsp;·&nbsp; {summary.total_char_units} {_html.escape(labels.char_unit_label)}</p>

<div class="summary">
  <table>
    <tr>
      <th>Metric</th>
      <th>Value</th>
      <th>Notes</th>
    </tr>
    <tr>
      <th>{_html.escape(labels.macro_char_rate)}</th>
      <td class="val" style="color:{macro_color}">{summary.macro_char_rate:.2f}%</td>
      <td style="color:#888;font-size:0.9em">{_html.escape(labels.macro_char_rate_note)}</td>
    </tr>
    <tr>
      <th>{_html.escape(labels.micro_char_rate)}</th>
      <td class="val" style="color:{micro_color}">{summary.micro_char_rate:.2f}%</td>
      <td style="color:#888;font-size:0.9em">{_html.escape(labels.micro_char_rate_note)}</td>
    </tr>
    <tr>
      <th>{_html.escape(labels.macro_word_rate)}</th>
      <td class="val">{summary.macro_word_rate:.2f}%</td>
      <td style="color:#888;font-size:0.9em">{_html.escape(labels.macro_word_rate_note)}</td>
    </tr>
    <tr>
      <th>Avg {_html.escape(labels.match_rate)}</th>
      <td class="val">{summary.avg_match_percent:.2f}%</td>
      <td style="color:#888;font-size:0.9em">mean of per-line match rates</td>
    </tr>
    <tr>
      <th>Total edit distance</th>
      <td class="val">{summary.total_edit_distance}</td>
      <td style="color:#888;font-size:0.9em">Levenshtein operations across all lines</td>
    </tr>
  </table>
</div>

<div class="legend">
  <span style="color:#007700">■ Match</span>
  <span style="color:#cc0000;font-weight:bold">■ Substitution</span>
  <span style="background:#ADD8E6;padding:2px 8px">■ Insertion (hypothesis only)</span>
  <span style="background:#FFFF96;padding:2px 8px">■ Deletion (reference only)</span>
</div>

{''.join(line_blocks)}
</body>
</html>"""

            Path(file_path).write_text(html_doc, encoding='utf-8')
            self.status_message.emit(f"HTML report saved: {Path(file_path).name}")
            QMessageBox.information(self, "Export Complete", f"HTML report saved:\n{file_path}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export HTML: {e}")
