from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import pyqtSignal, QObject, QThread
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLineEdit, QPushButton,
    QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QSpinBox, QCheckBox,
    QPlainTextEdit, QProgressBar, QMessageBox
)

# Local import from the batch module
try:
    from .pagexml_batch_segmenter import run_batch
except Exception:
    # Fallback for running as script without package context
    from pagexml_batch_segmenter import run_batch


@dataclass
class BatchParams:
    input_dir: str
    output_dir: str
    overlays_dir: Optional[str]
    device: str
    mode: str
    neural_model_path: str
    qc_csv_path: Optional[str]
    max_columns: int
    min_line_height: int
    deskew: bool


class BatchWorker(QObject):
    progress = pyqtSignal(int, int)                  # current, total
    file_done = pyqtSignal(str, int, int)            # name, regions, lines
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, params: BatchParams):
        super().__init__()
        self.params = params
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            def _log(msg: str):
                if not self._stop_event.is_set():
                    try:
                        self.log.emit(msg)
                    except Exception:
                        pass  # Ignore signal emission errors during shutdown

            def _on_progress(cur: int, total: int):
                if not self._stop_event.is_set():
                    try:
                        self.progress.emit(cur, total)
                    except Exception:
                        pass  # Ignore signal emission errors during shutdown

            def _on_file(name: str, regions: int, lines: int):
                if not self._stop_event.is_set():
                    try:
                        self.file_done.emit(name, regions, lines)
                    except Exception:
                        pass  # Ignore signal emission errors during shutdown

            run_batch(
                input_dir=self.params.input_dir,
                output_dir=self.params.output_dir,
                overlays_dir=self.params.overlays_dir,
                device=self.params.device,
                mode=self.params.mode,
                neural_model_path=self.params.neural_model_path,
                qc_csv_path=self.params.qc_csv_path,
                max_columns=self.params.max_columns,
                min_line_height=self.params.min_line_height,
                deskew=self.params.deskew,
                log=_log,
                on_progress=_on_progress,
                on_file=_on_file,
                stop_event=self._stop_event,
            )
        except Exception as e:
            if not self._stop_event.is_set():
                import traceback
                error_msg = f"Batch processing error: {e}\n{traceback.format_exc()}"
                try:
                    self.error.emit(error_msg)
                except Exception:
                    print(error_msg)  # Fallback to print if signal fails
        finally:
            try:
                self.finished.emit()
            except Exception as finish_err:
                print(f"Error emitting finished signal: {finish_err}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PAGE XML Batch Segmenter")
        self.resize(900, 680)

        self.thread: Optional[QThread] = None
        self.worker: Optional[BatchWorker] = None

        # Widgets
        self.input_edit = QLineEdit()
        self.input_browse = QPushButton("Browse…")
        self.output_edit = QLineEdit()
        self.output_browse = QPushButton("Browse…")

        self.overlays_check = QCheckBox("Generate overlays")
        self.overlays_edit = QLineEdit()
        self.overlays_browse = QPushButton("Browse…")
        self.overlays_edit.setEnabled(False)
        self.overlays_browse.setEnabled(False)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["classical", "neural", "auto"])
        self.mode_combo.setCurrentText("auto")

        # Set default path to the actual blla.mlmodel location
        default_model_path = os.path.join(os.path.dirname(__file__), "blla.mlmodel")
        self.neural_model_edit = QLineEdit(default_model_path)
        self.neural_model_browse = QPushButton("Browse…")

        self.qc_csv_check = QCheckBox("Export QC metrics")
        self.qc_csv_edit = QLineEdit()
        self.qc_csv_browse = QPushButton("Browse…")
        self.qc_csv_edit.setEnabled(False)
        self.qc_csv_browse.setEnabled(False)

        self.maxcols_spin = QSpinBox()
        self.maxcols_spin.setRange(1, 8)
        self.maxcols_spin.setValue(4)

        self.minheight_spin = QSpinBox()
        self.minheight_spin.setRange(1, 500)
        self.minheight_spin.setValue(8)

        self.deskew_check = QCheckBox("Deskew (light)")

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        self.open_output_btn = QPushButton("Open Output Folder")
        self.view_log_btn = QPushButton("Clear Log")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)

        # Layouts
        path_row_1 = QHBoxLayout()
        path_row_1.addWidget(QLabel("Input:"))
        path_row_1.addWidget(self.input_edit)
        path_row_1.addWidget(self.input_browse)

        path_row_2 = QHBoxLayout()
        path_row_2.addWidget(QLabel("Output XML:"))
        path_row_2.addWidget(self.output_edit)
        path_row_2.addWidget(self.output_browse)

        overlays_row = QHBoxLayout()
        overlays_row.addWidget(self.overlays_check)
        overlays_row.addWidget(self.overlays_edit)
        overlays_row.addWidget(self.overlays_browse)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Segmentation mode:"))
        mode_row.addWidget(self.mode_combo)
        mode_row.addSpacing(12)
        mode_row.addWidget(QLabel("Neural model:"))
        mode_row.addWidget(self.neural_model_edit)
        mode_row.addWidget(self.neural_model_browse)
        mode_row.addStretch(1)

        qc_row = QHBoxLayout()
        qc_row.addWidget(self.qc_csv_check)
        qc_row.addWidget(self.qc_csv_edit)
        qc_row.addWidget(self.qc_csv_browse)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Device:"))
        params_row.addWidget(self.device_combo)
        params_row.addSpacing(12)
        params_row.addWidget(QLabel("Max columns:"))
        params_row.addWidget(self.maxcols_spin)
        params_row.addSpacing(12)
        params_row.addWidget(QLabel("Min line height (px):"))
        params_row.addWidget(self.minheight_spin)
        params_row.addSpacing(12)
        params_row.addWidget(self.deskew_check)
        params_row.addStretch(1)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.start_btn)
        buttons_row.addWidget(self.stop_btn)
        buttons_row.addStretch(1)
        buttons_row.addWidget(self.open_output_btn)
        buttons_row.addWidget(self.view_log_btn)

        central = QWidget()
        v = QVBoxLayout(central)
        v.addLayout(path_row_1)
        v.addLayout(path_row_2)
        v.addLayout(overlays_row)
        v.addLayout(qc_row)
        v.addLayout(mode_row)
        v.addLayout(params_row)
        v.addWidget(self.progress)
        v.addWidget(self.log_view)
        v.addLayout(buttons_row)
        self.setCentralWidget(central)

        # Signals
        self.input_browse.clicked.connect(self._pick_input)
        self.output_browse.clicked.connect(self._pick_output)
        self.overlays_browse.clicked.connect(self._pick_overlays)
        self.overlays_check.toggled.connect(self._toggle_overlays)
        self.neural_model_browse.clicked.connect(self._pick_neural_model)
        self.qc_csv_browse.clicked.connect(self._pick_qc_csv)
        self.qc_csv_check.toggled.connect(self._toggle_qc_csv)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.open_output_btn.clicked.connect(self._open_output)
        self.view_log_btn.clicked.connect(self._clear_log)

        # Defaults
        self.overlays_check.setChecked(True)
        self.qc_csv_check.setChecked(False)
        self._on_mode_changed(self.mode_combo.currentText())
        self._append_log("Ready.")

    # UI helpers
    def _toggle_overlays(self, checked: bool):
        self.overlays_edit.setEnabled(checked)
        self.overlays_browse.setEnabled(checked)
        # Provide a sensible default overlays folder if enabled and empty
        if checked and not self.overlays_edit.text().strip():
            out_dir = self.output_edit.text().strip()
            if out_dir:
                self.overlays_edit.setText(os.path.join(out_dir, "overlays"))

    def _pick_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select input folder")
        if d:
            self.input_edit.setText(d)
            # Auto-suggest output folder if empty
            if not self.output_edit.text().strip():
                suggested = os.path.join(d, "pagexml_xml")
                self.output_edit.setText(suggested)
            # If overlays enabled and empty, suggest overlays inside output
            if self.overlays_check.isChecked() and not self.overlays_edit.text().strip():
                out_dir = self.output_edit.text().strip()
                if out_dir:
                    self.overlays_edit.setText(os.path.join(out_dir, "overlays"))

    def _pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder for PAGE XML")
        if d:
            self.output_edit.setText(d)

    def _pick_overlays(self):
        d = QFileDialog.getExistingDirectory(self, "Select overlays folder")
        if d:
            self.overlays_edit.setText(d)

    def _pick_neural_model(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select neural model", "", "Model files (*.mlmodel *.pt *.pth);;All files (*)"
        )
        if f:
            self.neural_model_edit.setText(f)

    def _pick_qc_csv(self):
        out_dir = self.output_edit.text().strip()
        default_dir = out_dir if out_dir else ""
        f, _ = QFileDialog.getSaveFileName(
            self, "Save QC metrics CSV", default_dir, "CSV files (*.csv);;All files (*)"
        )
        if f:
            self.qc_csv_edit.setText(f)

    def _toggle_qc_csv(self, checked: bool):
        self.qc_csv_edit.setEnabled(checked)
        self.qc_csv_browse.setEnabled(checked)
        # Auto-suggest QC CSV path if enabled and empty
        if checked and not self.qc_csv_edit.text().strip():
            out_dir = self.output_edit.text().strip()
            if out_dir:
                self.qc_csv_edit.setText(os.path.join(out_dir, "qc_metrics.csv"))

    def _on_mode_changed(self, mode: str):
        # Enable/disable neural model input based on mode
        neural_needed = mode in ["neural", "auto"]
        self.neural_model_edit.setEnabled(neural_needed)
        self.neural_model_browse.setEnabled(neural_needed)
        # Max columns only relevant for classical or auto (fallback)
        classical_params = mode in ["classical", "auto"]
        self.maxcols_spin.setEnabled(classical_params)

    def _append_log(self, msg: str):
        self.log_view.appendPlainText(msg)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _clear_log(self):
        self.log_view.clear()

    def _open_output(self):
        out_dir = self.output_edit.text().strip()
        if not out_dir:
            QMessageBox.information(self, "Output", "No output folder set.")
            return
        if not os.path.isdir(out_dir):
            QMessageBox.warning(self, "Output", f"Folder not found: {out_dir}")
            return
        # Best-effort open folder
        try:
            if sys.platform.startswith('linux'):
                os.system(f'xdg-open "{out_dir}"')
            elif sys.platform == 'darwin':
                os.system(f'open "{out_dir}"')
            elif sys.platform == 'win32':
                os.startfile(out_dir)  # type: ignore[attr-defined]
        except Exception as e:
            QMessageBox.warning(self, "Open output", str(e))

    def _collect_params(self) -> Optional[BatchParams]:
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        overlays_dir = self.overlays_edit.text().strip() if self.overlays_check.isChecked() else None
        qc_csv_path = self.qc_csv_edit.text().strip() if self.qc_csv_check.isChecked() else None

        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Input", "Please select a valid input folder.")
            return None
        if not output_dir:
            QMessageBox.warning(self, "Output", "Please select an output folder.")
            return None

        device = self.device_combo.currentText()
        mode = self.mode_combo.currentText()
        neural_model_path = self.neural_model_edit.text().strip()
        max_columns = self.maxcols_spin.value()
        min_line_height = self.minheight_spin.value()
        deskew = self.deskew_check.isChecked()

        # Validate neural model path if neural/auto mode
        if mode in ["neural", "auto"]:
            if not neural_model_path:
                QMessageBox.warning(self, "Neural Model", "Please specify a neural model path for neural/auto mode.")
                return None

        return BatchParams(
            input_dir=input_dir,
            output_dir=output_dir,
            overlays_dir=overlays_dir,
            device=device,
            mode=mode,
            neural_model_path=neural_model_path,
            qc_csv_path=qc_csv_path,
            max_columns=max_columns,
            min_line_height=min_line_height,
            deskew=deskew,
        )

    def _start(self):
        params = self._collect_params()
        if not params:
            return

        # Log chosen paths for clarity
        self._append_log(f"Input dir: {params.input_dir}")
        self._append_log(f"Output XML dir: {params.output_dir}")
        if params.overlays_dir:
            self._append_log(f"Overlays dir: {params.overlays_dir}")
        if params.qc_csv_path:
            self._append_log(f"QC metrics CSV: {params.qc_csv_path}")
        self._append_log(f"Segmentation mode: {params.mode}")

        # Ensure dirs
        os.makedirs(params.output_dir, exist_ok=True)
        if params.overlays_dir:
            os.makedirs(params.overlays_dir, exist_ok=True)

        # Prepare UI
        self._append_log("Starting batch…")
        self.progress.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # Worker in a thread
        self.thread = QThread()
        self.worker = BatchWorker(params)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

        # Wire signals
        self.worker.progress.connect(self._on_progress)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.log.connect(self._append_log)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        # When thread truly finishes, clear our references
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _stop(self):
        if self.worker:
            self._append_log("Stop requested…")
            self.worker.stop()
            # Do not block UI; worker will finish after current page. We wait briefly for thread cleanup.
            if self.thread and self.thread.isRunning():
                # Let the worker hit its stop check; we avoid force termination.
                pass

    # Slots for worker
    def _on_progress(self, cur: int, total: int):
        # Initialize range on first update
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(cur)

    def _on_file_done(self, name: str, regions: int, lines: int):
        # Additional UI reactions could be added here
        pass

    def _on_error(self, msg: str):
        self._append_log(f"[ERROR] {msg}")

    def _on_finished(self):
        """Called when worker's run() completes. Thread may still be running."""
        self._append_log("Finished.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # Don't clear references here - wait for thread.finished signal
    
    def _on_thread_finished(self):
        """Called when thread truly finishes. Safe to clear references now."""
        self.worker = None
        self.thread = None

    def closeEvent(self, event):
        """Ensure background thread is cleanly stopped before window closes to avoid 'QThread destroyed' crash."""
        if self.worker and self.thread and self.thread.isRunning():
            self._append_log("[INFO] Waiting for batch to complete before closing...")
            try:
                # Signal worker to stop
                self.worker.stop()
                
                # Disconnect signals to prevent crashes during cleanup
                try:
                    self.worker.progress.disconnect()
                    self.worker.file_done.disconnect()
                    self.worker.log.disconnect()
                    self.worker.error.disconnect()
                    self.worker.finished.disconnect()
                except Exception:
                    pass  # Signals may already be disconnected or worker deleted
                
                # Wait longer for thread to finish naturally (10 seconds)
                if not self.thread.wait(10000):
                    self._append_log("[WARN] Thread didn't finish, forcing quit...")
                    # Thread didn't finish, force quit
                    self.thread.quit()
                    if not self.thread.wait(3000):
                        self._append_log("[WARN] Thread still running, terminating...")
                        # Still running, terminate as last resort
                        self.thread.terminate()
                        self.thread.wait(2000)
            except Exception as e:
                print(f"Error during cleanup: {e}")
        
        # Re-enable app quit on last window close
        QApplication.instance().setQuitOnLastWindowClosed(True)
        event.accept()


def exception_hook(exctype, value, tb):
    """Global exception handler to prevent silent crashes"""
    import traceback
    print("Uncaught exception:", file=sys.stderr)
    traceback.print_exception(exctype, value, tb)
    # Don't exit on exceptions, just log them


def main():
    # Install global exception hook
    sys.excepthook = exception_hook
    
    app = QApplication(sys.argv)
    
    # Prevent app from quitting when last window closes if thread is still running
    app.setQuitOnLastWindowClosed(False)
    
    win = MainWindow()
    win.show()
    
    # When window actually closes, allow app to quit
    win.destroyed.connect(app.quit)
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
