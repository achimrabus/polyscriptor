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
    QPushButton, QComboBox, QGroupBox, QFileDialog, QRadioButton,
    QButtonGroup, QMessageBox, QProgressDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QColor, QTextCharFormat, QFont

from transcription_metrics import TranscriptionMetrics, LineMetrics
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
        self.base_engine = base_engine
        self.line_segments = line_segments
        self.line_images = line_images
        self.comparison_engine: Optional[HTREngine] = None
        self.ground_truth: Optional[List[str]] = None
        self.current_line_idx = 0
        self.base_transcriptions: List[str] = []
        self.comparison_transcriptions: List[str] = []
        self.comparison_worker: Optional[ComparisonWorker] = None

        self.setup_ui()

    def setup_ui(self):
        """Build the comparison UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Header with mode selection
        header_layout = QHBoxLayout()
        header_label = QLabel("<b>Transcription Comparison</b>")
        header_label.setStyleSheet("font-size: 14px;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        # Close button
        self.close_btn = QPushButton("✕ Close")
        self.close_btn.clicked.connect(self.close_comparison)
        self.close_btn.setMaximumWidth(100)
        header_layout.addWidget(self.close_btn)

        layout.addLayout(header_layout)

        # Mode selection
        mode_group = QGroupBox("Comparison Mode")
        mode_layout = QVBoxLayout()

        self.mode_button_group = QButtonGroup()
        self.engine_mode_radio = QRadioButton("Engine vs Engine")
        self.gt_mode_radio = QRadioButton("Engine vs Ground Truth")
        self.engine_mode_radio.setChecked(True)
        self.engine_mode_radio.toggled.connect(self.on_mode_changed)

        self.mode_button_group.addButton(self.engine_mode_radio)
        self.mode_button_group.addButton(self.gt_mode_radio)

        mode_layout.addWidget(self.engine_mode_radio)
        mode_layout.addWidget(self.gt_mode_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Engine selection (for Engine vs Engine mode)
        self.engine_group = QGroupBox("Comparison Engine")
        engine_layout = QVBoxLayout()

        engine_select_layout = QHBoxLayout()
        engine_select_layout.addWidget(QLabel("Engine:"))
        self.engine_combo = QComboBox()
        self._populate_engines()
        engine_select_layout.addWidget(self.engine_combo, 1)
        engine_layout.addLayout(engine_select_layout)

        engine_button_layout = QHBoxLayout()
        self.load_engine_btn = QPushButton("Load & Transcribe")
        self.load_engine_btn.clicked.connect(self.load_and_transcribe_engine)
        engine_button_layout.addWidget(self.load_engine_btn)

        self.unload_engine_btn = QPushButton("Unload")
        self.unload_engine_btn.clicked.connect(self.unload_comparison_engine)
        self.unload_engine_btn.setEnabled(False)
        engine_button_layout.addWidget(self.unload_engine_btn)

        engine_layout.addLayout(engine_button_layout)
        self.engine_group.setLayout(engine_layout)
        layout.addWidget(self.engine_group)

        # Ground truth loading (for Engine vs GT mode)
        self.gt_group = QGroupBox("Ground Truth")
        gt_layout = QVBoxLayout()

        self.gt_file_label = QLabel("No file loaded")
        self.gt_file_label.setWordWrap(True)
        gt_layout.addWidget(self.gt_file_label)

        gt_button_layout = QHBoxLayout()
        self.load_gt_btn = QPushButton("Load TXT File...")
        self.load_gt_btn.clicked.connect(self.load_ground_truth)
        gt_button_layout.addWidget(self.load_gt_btn)

        self.clear_gt_btn = QPushButton("Clear")
        self.clear_gt_btn.clicked.connect(self.clear_ground_truth)
        self.clear_gt_btn.setEnabled(False)
        gt_button_layout.addWidget(self.clear_gt_btn)

        gt_layout.addLayout(gt_button_layout)
        self.gt_group.setLayout(gt_layout)
        self.gt_group.hide()  # Hidden initially
        layout.addWidget(self.gt_group)

        # Side-by-side comparison view
        comparison_layout = QHBoxLayout()

        # Left panel (base/reference)
        left_panel = QVBoxLayout()
        self.left_label = QLabel(f"{self.base_engine.get_name()} (Base)")
        self.left_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.left_text = ComparisonTextEdit()
        self.left_metrics_label = QLabel("CER: - | WER: - | Match: -")
        self.left_metrics_label.setStyleSheet("color: #666; font-size: 10px;")
        left_panel.addWidget(self.left_label)
        left_panel.addWidget(self.left_text)
        left_panel.addWidget(self.left_metrics_label)

        # Right panel (comparison/hypothesis)
        right_panel = QVBoxLayout()
        self.right_label = QLabel("Load comparison engine or ground truth")
        self.right_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.right_text = ComparisonTextEdit()
        self.right_metrics_label = QLabel("CER: - | WER: - | Match: -")
        self.right_metrics_label.setStyleSheet("color: #666; font-size: 10px;")
        right_panel.addWidget(self.right_label)
        right_panel.addWidget(self.right_text)
        right_panel.addWidget(self.right_metrics_label)

        comparison_layout.addLayout(left_panel)
        comparison_layout.addLayout(right_panel)
        layout.addLayout(comparison_layout)

        # Navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self.show_previous_line)
        nav_layout.addWidget(self.prev_btn)

        self.line_label = QLabel("Line 0 of 0")
        self.line_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.line_label, 1)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.show_next_line)
        nav_layout.addWidget(self.next_btn)

        layout.addLayout(nav_layout)

        # Export button
        export_layout = QHBoxLayout()
        self.export_csv_btn = QPushButton("📊 Export to CSV")
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_csv_btn.setMinimumHeight(35)
        export_layout.addWidget(self.export_csv_btn)
        layout.addLayout(export_layout)

        # Color legend
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("<span style='color: green;'>■</span> Match"))
        legend_layout.addWidget(QLabel("<span style='color: red;'>■</span> Substitution"))
        legend_layout.addWidget(QLabel("<span style='background-color: #ADD8E6;'>■</span> Insertion"))
        legend_layout.addWidget(QLabel("<span style='background-color: #FFFF96;'>■</span> Deletion"))
        legend_layout.addStretch()
        layout.addLayout(legend_layout)

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
        if self.engine_mode_radio.isChecked():
            # Show engine selection, hide GT
            self.engine_group.show()
            self.gt_group.hide()
        else:
            # Show GT selection, hide engine
            self.engine_group.hide()
            self.gt_group.show()

        # Update display
        self.update_display()

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

            # Check if model is already loaded (use existing model if available)
            if self.comparison_engine.is_model_loaded():
                self.status_message.emit(f"Using already-loaded {engine_name} model")
            else:
                # Get default config and load model
                config = self.comparison_engine.get_config()

                # Load model
                self.status_message.emit(f"Loading {engine_name} model...")
                if not self.comparison_engine.load_model(config):
                    raise Exception("Failed to load model")

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
        self.comparison_transcriptions = [r.text for r in results]
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
            self.right_metrics_label.setText("CER: - | WER: - | Match: -")

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
        if self.current_line_idx < len(self.line_segments) - 1:
            self.current_line_idx += 1
            self.update_display()

    def update_display(self):
        """Update the comparison display for current line."""
        if not self.base_transcriptions:
            return

        # Update line label and navigation buttons
        total_lines = len(self.line_segments)
        self.line_label.setText(f"Line {self.current_line_idx + 1} of {total_lines}")
        self.prev_btn.setEnabled(self.current_line_idx > 0)
        self.next_btn.setEnabled(self.current_line_idx < total_lines - 1)

        # Get base transcription
        base_text = self.base_transcriptions[self.current_line_idx]

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
            # Engine vs Engine mode
            reference = base_text
            hypothesis = self.comparison_transcriptions[self.current_line_idx]
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
        metrics = TranscriptionMetrics.compare_lines(reference, hypothesis)

        # Display texts with diff highlighting
        self.left_text.display_with_diff(reference, metrics, is_reference=True)
        self.right_text.display_with_diff(hypothesis, metrics, is_reference=False)

        # Update metrics labels
        self.left_metrics_label.setText(f"Length: {len(reference)} chars")
        self.right_metrics_label.setText(
            f"CER: {metrics.cer:.2f}% | WER: {metrics.wer:.2f}% | "
            f"Match: {metrics.match_percent:.2f}%"
        )

    def export_csv(self):
        """Export comparison results to CSV."""
        if not self.base_transcriptions:
            QMessageBox.warning(self, "Error", "No transcriptions to export!")
            return

        # Determine what we're comparing
        if self.gt_mode_radio.isChecked() and self.ground_truth:
            references = self.ground_truth
            hypotheses = self.base_transcriptions
            ref_label = "Ground Truth"
            hyp_label = self.base_engine.get_name()
        elif self.comparison_transcriptions:
            references = self.base_transcriptions
            hypotheses = self.comparison_transcriptions
            ref_label = self.base_engine.get_name()
            hyp_label = self.engine_combo.currentText()
        else:
            QMessageBox.warning(self, "Error",
                               "Load comparison engine or ground truth first!")
            return

        # Get file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Comparison CSV", "", "CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow([
                    'Line', ref_label, hyp_label, 'CER (%)', 'WER (%)',
                    'Match (%)', 'Edit Distance'
                ])

                # Calculate overall metrics
                total_cer = 0.0
                total_wer = 0.0
                total_match = 0.0
                line_count = min(len(references), len(hypotheses))

                # Per-line metrics
                for i in range(line_count):
                    ref = references[i] if i < len(references) else ""
                    hyp = hypotheses[i] if i < len(hypotheses) else ""

                    metrics = TranscriptionMetrics.compare_lines(ref, hyp)

                    writer.writerow([
                        i + 1,
                        ref,
                        hyp,
                        f"{metrics.cer:.2f}",
                        f"{metrics.wer:.2f}",
                        f"{metrics.match_percent:.2f}",
                        metrics.edit_distance
                    ])

                    total_cer += metrics.cer
                    total_wer += metrics.wer
                    total_match += metrics.match_percent

                # Overall averages
                writer.writerow([])
                writer.writerow([
                    'OVERALL',
                    f'{line_count} lines',
                    '',
                    f"{total_cer / line_count:.2f}",
                    f"{total_wer / line_count:.2f}",
                    f"{total_match / line_count:.2f}",
                    ''
                ])

            self.status_message.emit(f"Exported to {Path(file_path).name}")
            QMessageBox.information(self, "Success",
                                   f"Comparison exported to:\n{file_path}")

        except Exception as e:
            QMessageBox.warning(self, "Error",
                               f"Failed to export CSV: {str(e)}")

    def close_comparison(self):
        """Close comparison panel and return to normal view."""
        # Clean up
        if self.comparison_worker and self.comparison_worker.isRunning():
            self.comparison_worker.terminate()
            self.comparison_worker.wait()

        self.unload_comparison_engine()
        self.comparison_closed.emit()
