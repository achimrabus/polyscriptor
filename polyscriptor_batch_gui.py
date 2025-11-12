#!/usr/bin/env python3
"""
Polyscriptor Batch - Minimal GUI Launcher for Batch HTR Processing

A lightweight Qt6 GUI that builds and executes batch_processing.py commands.
Design philosophy: CLI wrapper, not reimplementation.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
    QFileDialog, QSpinBox, QDoubleSpinBox, QMessageBox, QTextEdit
)
from PyQt6.QtCore import Qt, QProcess, pyqtSignal
from PyQt6.QtGui import QFont


# Available engines with their configuration needs
ENGINES = {
    "PyLaia": {
        "needs_model_path": True,
        "needs_model_id": False,
        "default_segmentation": "kraken",
        "supports_beams": False,
    },
    "TrOCR": {
        "needs_model_path": True,
        "needs_model_id": True,
        "default_segmentation": "kraken",
        "supports_beams": True,
    },
    "Qwen3-VL": {
        "needs_model_path": False,
        "needs_model_id": True,
        "default_segmentation": "none",
        "supports_beams": False,
        "warning": "VERY SLOW: ~1-2 min/page. Use only for small batches!",
    },
    "Party": {
        "needs_model_path": True,
        "needs_model_id": False,
        "default_segmentation": "none",
        "supports_beams": False,
        "requires_pagexml": True,
    },
    "Kraken": {
        "needs_model_path": True,
        "needs_model_id": False,
        "default_segmentation": "none",
        "supports_beams": False,
    },
    "Churro": {
        "needs_model_path": True,
        "needs_model_id": True,
        "default_segmentation": "kraken",
        "supports_beams": False,
    },
}

# Built-in presets
BUILTIN_PRESETS = {
    "Church Slavonic (PyLaia + Kraken)": {
        "engine": "PyLaia",
        "model_path": "models/pylaia_church_slavonic_20251103_162857/best_model.pt",
        "segmentation_method": "kraken",
        "use_pagexml": True,
        "device": "cuda:0",
        "output_formats": ["txt"],
    },
    "Ukrainian (PyLaia + Kraken)": {
        "engine": "PyLaia",
        "model_path": "models/pylaia_ukrainian_pagexml_20251101_182736/best_model.pt",
        "segmentation_method": "kraken",
        "use_pagexml": True,
        "device": "cuda:0",
        "output_formats": ["txt"],
    },
    "Glagolitic (PyLaia + Kraken)": {
        "engine": "PyLaia",
        "model_path": "models/pylaia_glagolitic_with_spaces_20251102_182103/best_model.pt",
        "segmentation_method": "kraken",
        "use_pagexml": True,
        "device": "cuda:0",
        "output_formats": ["txt"],
    },
    "Russian (TrOCR HF)": {
        "engine": "TrOCR",
        "model_id": "kazars24/trocr-base-handwritten-ru",
        "segmentation_method": "kraken",
        "use_pagexml": True,
        "device": "cuda:0",
        "output_formats": ["txt"],
        "num_beams": 4,
    },
}


class PolyscriptorBatchGUI(QMainWindow):
    """Minimal GUI launcher for batch HTR processing."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polyscriptor Batch - HTR Batch Processing")
        self.resize(800, 900)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Input/Output section
        input_group = self._create_input_output_group()
        layout.addWidget(input_group)

        # Engine configuration
        engine_group = self._create_engine_group()
        layout.addWidget(engine_group)

        # Segmentation options
        seg_group = self._create_segmentation_group()
        layout.addWidget(seg_group)

        # Output options
        output_group = self._create_output_options_group()
        layout.addWidget(output_group)

        # Presets
        preset_group = self._create_preset_group()
        layout.addWidget(preset_group)

        # Action buttons
        button_layout = self._create_action_buttons()
        layout.addLayout(button_layout)

        # Command preview
        self.command_preview = QTextEdit()
        self.command_preview.setMaximumHeight(100)
        self.command_preview.setReadOnly(True)
        self.command_preview.setFont(QFont("Monospace", 9))
        layout.addWidget(QLabel("Command Preview:"))
        layout.addWidget(self.command_preview)

        # Update command preview on any change
        self._connect_update_signals()
        self._update_command_preview()

    def _create_input_output_group(self) -> QGroupBox:
        """Create input/output folder selection group."""
        group = QGroupBox("Input/Output")
        layout = QVBoxLayout()

        # Input folder
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Folder:"))
        self.input_folder_edit = QLineEdit("HTR_Images/")
        input_layout.addWidget(self.input_folder_edit)
        self.input_browse_btn = QPushButton("Browse...")
        self.input_browse_btn.clicked.connect(self._browse_input_folder)
        input_layout.addWidget(self.input_browse_btn)
        layout.addLayout(input_layout)

        # Output folder
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.output_folder_edit = QLineEdit("output")
        output_layout.addWidget(self.output_folder_edit)
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self._browse_output_folder)
        output_layout.addWidget(self.output_browse_btn)
        layout.addLayout(output_layout)

        group.setLayout(layout)
        return group

    def _create_engine_group(self) -> QGroupBox:
        """Create engine configuration group."""
        group = QGroupBox("Engine Configuration")
        layout = QVBoxLayout()

        # Engine selection
        engine_layout = QHBoxLayout()
        engine_layout.addWidget(QLabel("Engine:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(list(ENGINES.keys()))
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        engine_layout.addWidget(self.engine_combo)
        layout.addLayout(engine_layout)

        # Model path (local file)
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(QLabel("Model Path:"))
        self.model_path_edit = QLineEdit()
        model_path_layout.addWidget(self.model_path_edit)
        self.model_browse_btn = QPushButton("Browse...")
        self.model_browse_btn.clicked.connect(self._browse_model_path)
        model_path_layout.addWidget(self.model_browse_btn)
        layout.addLayout(model_path_layout)

        # Model ID (HuggingFace)
        model_id_layout = QHBoxLayout()
        model_id_layout.addWidget(QLabel("Model ID (HF):"))
        self.model_id_edit = QLineEdit()
        self.model_id_edit.setPlaceholderText("username/model-name")
        model_id_layout.addWidget(self.model_id_edit)
        layout.addLayout(model_id_layout)

        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda:0", "cuda:1", "cpu"])
        device_layout.addWidget(self.device_combo)
        layout.addLayout(device_layout)

        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        self.batch_spin.setToolTip("Leave at default for auto-optimization")
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addWidget(QLabel("(auto-optimized if default)"))
        batch_layout.addStretch()
        layout.addLayout(batch_layout)

        # Num beams (for TrOCR)
        beams_layout = QHBoxLayout()
        beams_layout.addWidget(QLabel("Num Beams:"))
        self.num_beams_spin = QSpinBox()
        self.num_beams_spin.setRange(1, 10)
        self.num_beams_spin.setValue(1)
        self.num_beams_spin.setToolTip("Beam search width (1=greedy, 4=quality, slower)")
        beams_layout.addWidget(self.num_beams_spin)
        beams_layout.addStretch()
        layout.addLayout(beams_layout)
        self.beams_layout_widget = QWidget()
        self.beams_layout_widget.setLayout(beams_layout)
        self.beams_layout_widget.setVisible(False)  # Hidden by default

        group.setLayout(layout)
        return group

    def _create_segmentation_group(self) -> QGroupBox:
        """Create segmentation options group."""
        group = QGroupBox("Segmentation")
        layout = QVBoxLayout()

        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.seg_method_combo = QComboBox()
        self.seg_method_combo.addItems(["kraken", "hpp", "none"])
        self.seg_method_combo.setToolTip(
            "kraken: Neural segmentation (best)\n"
            "hpp: Horizontal projection (fast)\n"
            "none: Pre-segmented line images"
        )
        method_layout.addWidget(self.seg_method_combo)
        layout.addLayout(method_layout)

        # PAGE XML checkbox
        self.pagexml_check = QCheckBox("Use PAGE XML (auto-detect from page/ folder)")
        self.pagexml_check.setChecked(True)
        self.pagexml_check.setToolTip("Auto-detect and use PAGE XML for segmentation")
        layout.addWidget(self.pagexml_check)

        # Sensitivity slider (for HPP/Kraken)
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(QLabel("Sensitivity:"))
        self.sensitivity_spin = QDoubleSpinBox()
        self.sensitivity_spin.setRange(0.01, 1.0)
        self.sensitivity_spin.setValue(0.1)
        self.sensitivity_spin.setSingleStep(0.01)
        self.sensitivity_spin.setDecimals(2)
        self.sensitivity_spin.setToolTip("Segmentation sensitivity (lower = fewer lines)")
        sens_layout.addWidget(self.sensitivity_spin)
        sens_layout.addStretch()
        layout.addLayout(sens_layout)

        group.setLayout(layout)
        return group

    def _create_output_options_group(self) -> QGroupBox:
        """Create output options group."""
        group = QGroupBox("Output Options")
        layout = QVBoxLayout()

        # Output formats
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Formats:"))
        self.txt_check = QCheckBox("TXT")
        self.txt_check.setChecked(True)
        format_layout.addWidget(self.txt_check)
        self.csv_check = QCheckBox("CSV")
        format_layout.addWidget(self.csv_check)
        self.pagexml_out_check = QCheckBox("PAGE XML")
        format_layout.addWidget(self.pagexml_out_check)
        format_layout.addStretch()
        layout.addLayout(format_layout)

        # Flags
        self.resume_check = QCheckBox("Resume (skip already processed images)")
        self.resume_check.setChecked(False)
        layout.addWidget(self.resume_check)

        self.verbose_check = QCheckBox("Verbose logging")
        self.verbose_check.setChecked(False)
        layout.addWidget(self.verbose_check)

        group.setLayout(layout)
        return group

    def _create_preset_group(self) -> QGroupBox:
        """Create preset management group."""
        group = QGroupBox("Presets")
        layout = QHBoxLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("-- Custom --")
        self.preset_combo.addItems(list(BUILTIN_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        layout.addWidget(self.preset_combo)

        self.save_preset_btn = QPushButton("Save")
        self.save_preset_btn.clicked.connect(self._save_preset)
        layout.addWidget(self.save_preset_btn)

        self.load_preset_btn = QPushButton("Load")
        self.load_preset_btn.clicked.connect(self._load_preset_file)
        layout.addWidget(self.load_preset_btn)

        layout.addStretch()

        group.setLayout(layout)
        return group

    def _create_action_buttons(self) -> QHBoxLayout:
        """Create action buttons (Dry Run, Start)."""
        layout = QHBoxLayout()

        self.dry_run_btn = QPushButton("Dry Run (Test First)")
        self.dry_run_btn.clicked.connect(self._run_dry_run)
        self.dry_run_btn.setToolTip("Test configuration with first image")
        layout.addWidget(self.dry_run_btn)

        self.start_btn = QPushButton("Start Batch Processing")
        self.start_btn.clicked.connect(self._start_batch)
        self.start_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(self.start_btn)

        return layout

    def _browse_input_folder(self):
        """Browse for input folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder_edit.setText(folder)

    def _browse_output_folder(self):
        """Browse for output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_edit.setText(folder)

    def _browse_model_path(self):
        """Browse for model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "Model Files (*.pt *.pth *.safetensors *.mlmodel);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def _on_engine_changed(self, engine_name: str):
        """Handle engine selection change."""
        if engine_name not in ENGINES:
            return

        config = ENGINES[engine_name]

        # Show/hide model path/ID based on engine
        self.model_path_edit.setEnabled(config.get("needs_model_path", True))
        self.model_browse_btn.setEnabled(config.get("needs_model_path", True))
        self.model_id_edit.setEnabled(config.get("needs_model_id", False))

        # Show/hide num_beams for TrOCR
        self.beams_layout_widget.setVisible(config.get("supports_beams", False))

        # Set default segmentation method
        default_seg = config.get("default_segmentation", "kraken")
        idx = self.seg_method_combo.findText(default_seg)
        if idx >= 0:
            self.seg_method_combo.setCurrentIndex(idx)

        # Show warning for slow engines
        if "warning" in config:
            QMessageBox.warning(self, "Engine Warning", config["warning"])

        self._update_command_preview()

    def _on_preset_changed(self, preset_name: str):
        """Load preset configuration."""
        if preset_name == "-- Custom --":
            return

        if preset_name in BUILTIN_PRESETS:
            self._load_preset_dict(BUILTIN_PRESETS[preset_name])

    def _load_preset_dict(self, preset: Dict[str, Any]):
        """Load configuration from preset dictionary."""
        # Engine
        if "engine" in preset:
            idx = self.engine_combo.findText(preset["engine"])
            if idx >= 0:
                self.engine_combo.setCurrentIndex(idx)

        # Model
        if "model_path" in preset:
            self.model_path_edit.setText(preset["model_path"])
        if "model_id" in preset:
            self.model_id_edit.setText(preset["model_id"])

        # Device
        if "device" in preset:
            idx = self.device_combo.findText(preset["device"])
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)

        # Segmentation
        if "segmentation_method" in preset:
            idx = self.seg_method_combo.findText(preset["segmentation_method"])
            if idx >= 0:
                self.seg_method_combo.setCurrentIndex(idx)

        if "use_pagexml" in preset:
            self.pagexml_check.setChecked(preset["use_pagexml"])

        # Output formats
        if "output_formats" in preset:
            formats = preset["output_formats"]
            self.txt_check.setChecked("txt" in formats)
            self.csv_check.setChecked("csv" in formats)
            self.pagexml_out_check.setChecked("pagexml" in formats)

        # Num beams
        if "num_beams" in preset:
            self.num_beams_spin.setValue(preset["num_beams"])

        self._update_command_preview()

    def _save_preset(self):
        """Save current configuration as preset."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name:
            return

        preset = self._get_current_config()
        preset_file = Path.home() / ".config" / "polyscriptor" / "presets.json"
        preset_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing presets
        presets = {}
        if preset_file.exists():
            try:
                with open(preset_file, 'r') as f:
                    presets = json.load(f)
            except:
                pass

        # Add new preset
        presets[name] = preset

        # Save
        with open(preset_file, 'w') as f:
            json.dump(presets, f, indent=2)

        QMessageBox.information(self, "Preset Saved", f"Preset '{name}' saved successfully!")

        # Update combo box
        if self.preset_combo.findText(name) < 0:
            self.preset_combo.addItem(name)

    def _load_preset_file(self):
        """Load preset from file."""
        preset_file = Path.home() / ".config" / "polyscriptor" / "presets.json"
        if not preset_file.exists():
            QMessageBox.warning(self, "No Presets", "No saved presets found.")
            return

        try:
            with open(preset_file, 'r') as f:
                presets = json.load(f)

            if not presets:
                QMessageBox.warning(self, "No Presets", "No saved presets found.")
                return

            from PyQt6.QtWidgets import QInputDialog
            name, ok = QInputDialog.getItem(
                self, "Load Preset", "Select preset:", list(presets.keys()), 0, False
            )

            if ok and name:
                self._load_preset_dict(presets[name])

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load presets: {e}")

    def _get_current_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        config = {
            "input_folder": self.input_folder_edit.text(),
            "output_folder": self.output_folder_edit.text(),
            "engine": self.engine_combo.currentText(),
            "device": self.device_combo.currentText(),
            "segmentation_method": self.seg_method_combo.currentText(),
            "use_pagexml": self.pagexml_check.isChecked(),
        }

        # Model
        if self.model_path_edit.text():
            config["model_path"] = self.model_path_edit.text()
        if self.model_id_edit.text():
            config["model_id"] = self.model_id_edit.text()

        # Output formats
        formats = []
        if self.txt_check.isChecked():
            formats.append("txt")
        if self.csv_check.isChecked():
            formats.append("csv")
        if self.pagexml_out_check.isChecked():
            formats.append("pagexml")
        config["output_formats"] = formats

        # Flags
        config["resume"] = self.resume_check.isChecked()
        config["verbose"] = self.verbose_check.isChecked()

        # Engine-specific
        if self.num_beams_spin.value() > 1:
            config["num_beams"] = self.num_beams_spin.value()

        if self.sensitivity_spin.value() != 0.1:
            config["sensitivity"] = self.sensitivity_spin.value()

        return config

    def _build_command(self, dry_run: bool = False) -> List[str]:
        """Build batch_processing.py command from current configuration."""
        config = self._get_current_config()

        cmd = ["python", "batch_processing.py"]

        # Required args
        cmd += ["--input-folder", config["input_folder"]]
        cmd += ["--engine", config["engine"]]

        # Output folder
        if config["output_folder"]:
            cmd += ["--output-folder", config["output_folder"]]

        # Model
        if config.get("model_path"):
            cmd += ["--model-path", config["model_path"]]
        elif config.get("model_id"):
            cmd += ["--model-id", config["model_id"]]

        # Device
        cmd += ["--device", config["device"]]

        # Segmentation
        if config["segmentation_method"] != "none":
            cmd += ["--segmentation-method", config["segmentation_method"]]

        # PAGE XML
        if config["use_pagexml"]:
            cmd += ["--use-pagexml"]
        else:
            cmd += ["--no-pagexml"]

        # Output formats
        for fmt in config["output_formats"]:
            cmd += ["--output-format", fmt]

        # Flags
        if config["resume"]:
            cmd += ["--resume"]
        if config["verbose"]:
            cmd += ["--verbose"]
        if dry_run:
            cmd += ["--dry-run"]

        # Engine-specific
        if config.get("num_beams"):
            cmd += ["--num-beams", str(config["num_beams"])]
        if config.get("sensitivity"):
            cmd += ["--segmentation-sensitivity", str(config["sensitivity"])]

        return cmd

    def _update_command_preview(self):
        """Update command preview text."""
        cmd = self._build_command()
        cmd_str = " ".join(cmd)
        self.command_preview.setPlainText(cmd_str)

    def _connect_update_signals(self):
        """Connect all widgets to update command preview."""
        # Text edits
        self.input_folder_edit.textChanged.connect(self._update_command_preview)
        self.output_folder_edit.textChanged.connect(self._update_command_preview)
        self.model_path_edit.textChanged.connect(self._update_command_preview)
        self.model_id_edit.textChanged.connect(self._update_command_preview)

        # Combos
        self.engine_combo.currentTextChanged.connect(self._update_command_preview)
        self.device_combo.currentTextChanged.connect(self._update_command_preview)
        self.seg_method_combo.currentTextChanged.connect(self._update_command_preview)

        # Checkboxes
        self.pagexml_check.stateChanged.connect(self._update_command_preview)
        self.txt_check.stateChanged.connect(self._update_command_preview)
        self.csv_check.stateChanged.connect(self._update_command_preview)
        self.pagexml_out_check.stateChanged.connect(self._update_command_preview)
        self.resume_check.stateChanged.connect(self._update_command_preview)
        self.verbose_check.stateChanged.connect(self._update_command_preview)

        # Spinners
        self.batch_spin.valueChanged.connect(self._update_command_preview)
        self.num_beams_spin.valueChanged.connect(self._update_command_preview)
        self.sensitivity_spin.valueChanged.connect(self._update_command_preview)

    def _validate_config(self) -> Optional[str]:
        """Validate current configuration. Returns error message or None."""
        config = self._get_current_config()

        # Check input folder exists
        if not Path(config["input_folder"]).exists():
            return f"Input folder does not exist: {config['input_folder']}"

        # Check model is specified
        if not config.get("model_path") and not config.get("model_id"):
            return "Please specify a model path or HuggingFace model ID"

        # Check model path exists (if specified)
        if config.get("model_path"):
            if not Path(config["model_path"]).exists():
                return f"Model file does not exist: {config['model_path']}"

        # Check at least one output format
        if not config["output_formats"]:
            return "Please select at least one output format"

        return None

    def _run_dry_run(self):
        """Execute dry run."""
        error = self._validate_config()
        if error:
            QMessageBox.critical(self, "Configuration Error", error)
            return

        cmd = self._build_command(dry_run=True)
        self._execute_command(cmd, "Dry Run")

    def _start_batch(self):
        """Start batch processing."""
        error = self._validate_config()
        if error:
            QMessageBox.critical(self, "Configuration Error", error)
            return

        # Confirmation dialog
        reply = QMessageBox.question(
            self, "Start Batch Processing",
            "Start batch processing with current configuration?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            cmd = self._build_command(dry_run=False)
            self._execute_command(cmd, "Batch Processing")

    def _execute_command(self, cmd: List[str], title: str):
        """Execute command and show output in a dialog."""
        cmd_str = " ".join(cmd)

        # Create execution dialog
        dialog = QWidget()
        dialog.setWindowTitle(title)
        dialog.setMinimumSize(800, 600)
        layout = QVBoxLayout(dialog)

        # Command display
        layout.addWidget(QLabel("Command:"))
        cmd_display = QTextEdit()
        cmd_display.setPlainText(cmd_str)
        cmd_display.setMaximumHeight(80)
        cmd_display.setReadOnly(True)
        cmd_display.setFont(QFont("Monospace", 9))
        layout.addWidget(cmd_display)

        # Output display
        layout.addWidget(QLabel("Output:"))
        output_display = QTextEdit()
        output_display.setFont(QFont("Monospace", 9))
        output_display.setReadOnly(True)
        layout.addWidget(output_display)

        # Buttons
        button_layout = QHBoxLayout()
        copy_btn = QPushButton("Copy Command")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(cmd_str))
        button_layout.addWidget(copy_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.show()

        # Execute command using QProcess
        process = QProcess()

        def append_output():
            output = process.readAllStandardOutput().data().decode('utf-8', errors='ignore')
            output_display.append(output)

        def append_error():
            error = process.readAllStandardError().data().decode('utf-8', errors='ignore')
            output_display.append(f"<span style='color: red;'>{error}</span>")

        def process_finished(exit_code, exit_status):
            if exit_code == 0:
                output_display.append("\n<b>✓ Process completed successfully!</b>")
            else:
                output_display.append(f"\n<b>❌ Process failed with exit code {exit_code}</b>")

        process.readyReadStandardOutput.connect(append_output)
        process.readyReadStandardError.connect(append_error)
        process.finished.connect(process_finished)

        # Start process
        output_display.append(f"<b>Starting {title}...</b>\n")
        process.start(cmd[0], cmd[1:])

        # Keep dialog and process alive
        dialog.process = process


def main():
    app = QApplication(sys.argv)
    window = PolyscriptorBatchGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
