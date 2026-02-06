#!/usr/bin/env python3
"""
Polyscriptor Batch - Minimal GUI Launcher for Batch HTR Processing

A lightweight Qt6 GUI that builds and executes batch_processing.py commands.
Design philosophy: CLI wrapper, not reimplementation.
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in the project root
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, will fall back to environment variables

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
        "needs_model_path": True,  # Local model path OR HF model ID
        "needs_model_id": True,    # Both available for flexibility
        "needs_adapter": True,     # For local LoRA adapters (with HF base)
        "supports_line_mode": True,  # For line-trained models
        "default_segmentation": "none",
        "supports_beams": False,
        "warning": "VERY SLOW: ~1-2 min/page. Use only for small batches!",
        "model_path_tooltip": "Local Qwen model folder (use this OR Model ID, not both)",
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
    "OpenWebUI": {
        "needs_model_path": False,
        "needs_model_id": True,  # Model name from server
        "needs_api_key": True,   # Requires API key
        "default_segmentation": "none",  # VLM processes full pages
        "supports_beams": False,
        "warning": "API-based: Requires API key from openwebui.uni-freiburg.de",
    },
    "LightOnOCR": {
        "needs_model_path": False,
        "needs_model_id": True,  # HuggingFace model ID
        "default_segmentation": "kraken",  # LINE-LEVEL model
        "supports_beams": False,
        "has_lighton_options": True,  # Enable LightOnOCR-specific controls
        "warning": "Requires transformers from git: pip install git+https://github.com/huggingface/transformers.git",
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
    "Church Slavonic (Qwen3-VL Pages)": {
        "engine": "Qwen3-VL",
        "model_id": "wjbmattingly/Qwen3-VL-8B-old-church-slavonic",
        "segmentation_method": "none",
        "use_pagexml": False,
        "device": "cuda:0",
        "output_formats": ["txt"],
    },
    "Ukrainian (Qwen3-VL + LoRA Adapter)": {
        "engine": "Qwen3-VL",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "adapter": "models/Qwen3-VL-8B-ukrainian/final_model",
        "segmentation_method": "none",
        "use_pagexml": False,
        "device": "cuda:0",
        "output_formats": ["txt"],
    },
    "Glagolitic (Qwen3-VL + LoRA Adapter)": {
        "engine": "Qwen3-VL",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "adapter": "models/Qwen3-VL-8B-glagolitic/final_model",
        "segmentation_method": "none",
        "use_pagexml": False,
        "device": "cuda:0",
        "output_formats": ["txt"],
    },
    "Church Slavonic (Qwen3-VL Lines + PAGE XML)": {
        "engine": "Qwen3-VL",
        "model_id": "wjbmattingly/Qwen3-VL-8B-old-church-slavonic-line-3-epochs",
        "segmentation_method": "kraken",
        "use_pagexml": True,
        "line_mode": True,  # Force line segmentation for line-trained model
        "device": "cuda:0",
        "output_formats": ["txt"],
    },
    "German Shorthand (LightOnOCR)": {
        "engine": "LightOnOCR",
        "model_id": "wjbmattingly/LightOnOCR-2-1B-german-shorthand-line",
        "segmentation_method": "kraken",
        "use_pagexml": True,
        "device": "cuda:0",
        "output_formats": ["txt"],
        "longest_edge": 700,
        "max_new_tokens": 256,
    },
    "Multilingual (LightOnOCR Base)": {
        "engine": "LightOnOCR",
        "model_id": "lightonai/LightOnOCR-2-1B-base",
        "segmentation_method": "kraken",
        "use_pagexml": True,
        "device": "cuda:0",
        "output_formats": ["txt"],
        "longest_edge": 700,
        "max_new_tokens": 256,
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

        # Adapter path (for Qwen3 LoRA models)
        adapter_layout = QHBoxLayout()
        adapter_layout.addWidget(QLabel("Adapter Path:"))
        self.adapter_edit = QLineEdit()
        self.adapter_edit.setPlaceholderText("Optional: path to LoRA adapter folder")
        adapter_layout.addWidget(self.adapter_edit)
        self.adapter_browse_btn = QPushButton("Browse...")
        self.adapter_browse_btn.clicked.connect(self._browse_adapter_path)
        adapter_layout.addWidget(self.adapter_browse_btn)
        layout.addLayout(adapter_layout)
        # Initially hidden, shown only for Qwen3-VL
        self.adapter_edit.setVisible(False)
        self.adapter_browse_btn.setVisible(False)
        # Store the label for visibility toggle
        self.adapter_label = adapter_layout.itemAt(0).widget()
        self.adapter_label.setVisible(False)

        # Line mode checkbox (for line-trained Qwen3 models)
        self.line_mode_check = QCheckBox("Line Mode (for line-trained models)")
        self.line_mode_check.setToolTip(
            "Enable for Qwen3 models trained on line images.\n"
            "Forces segmentation so each line is processed separately.\n"
            "Leave unchecked for page-trained models that output line breaks."
        )
        self.line_mode_check.setVisible(False)  # Hidden by default, shown for Qwen3-VL
        layout.addWidget(self.line_mode_check)

        # API Key (for OpenWebUI)
        api_key_layout = QHBoxLayout()
        self.api_key_label = QLabel("API Key:")
        api_key_layout.addWidget(self.api_key_label)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("OpenWebUI API key (or set OPENWEBUI_API_KEY env var)")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_layout.addWidget(self.api_key_edit)
        self.api_key_show_check = QCheckBox("Show")
        self.api_key_show_check.toggled.connect(
            lambda checked: self.api_key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        api_key_layout.addWidget(self.api_key_show_check)
        layout.addLayout(api_key_layout)
        # Initially hidden, shown only for OpenWebUI
        self.api_key_label.setVisible(False)
        self.api_key_edit.setVisible(False)
        self.api_key_show_check.setVisible(False)

        # OpenWebUI Model selection (dropdown with refresh)
        openwebui_model_layout = QHBoxLayout()
        self.openwebui_model_label = QLabel("Model:")
        openwebui_model_layout.addWidget(self.openwebui_model_label)
        self.openwebui_model_combo = QComboBox()
        self.openwebui_model_combo.setMinimumWidth(250)
        self.openwebui_model_combo.addItem("Click 'Refresh' to load models")
        openwebui_model_layout.addWidget(self.openwebui_model_combo)
        self.openwebui_refresh_btn = QPushButton("Refresh")
        self.openwebui_refresh_btn.setToolTip("Fetch available models from OpenWebUI server")
        self.openwebui_refresh_btn.clicked.connect(self._refresh_openwebui_models)
        openwebui_model_layout.addWidget(self.openwebui_refresh_btn)
        layout.addLayout(openwebui_model_layout)
        # Initially hidden, shown only for OpenWebUI
        self.openwebui_model_label.setVisible(False)
        self.openwebui_model_combo.setVisible(False)
        self.openwebui_refresh_btn.setVisible(False)

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

        # LightOnOCR-specific controls
        self.lighton_group = QGroupBox("LightOnOCR Settings")
        lighton_layout = QVBoxLayout()

        # Longest edge
        edge_layout = QHBoxLayout()
        edge_layout.addWidget(QLabel("Longest Edge:"))
        self.longest_edge_spin = QSpinBox()
        self.longest_edge_spin.setRange(512, 1024)
        self.longest_edge_spin.setValue(700)
        self.longest_edge_spin.setSingleStep(50)
        self.longest_edge_spin.setToolTip(
            "Image resize target (512-1024, default 700)\n"
            "Larger = better quality but slower and more VRAM"
        )
        edge_layout.addWidget(self.longest_edge_spin)
        edge_layout.addWidget(QLabel("px"))
        edge_layout.addStretch()
        lighton_layout.addLayout(edge_layout)

        # Max new tokens
        tokens_layout = QHBoxLayout()
        tokens_layout.addWidget(QLabel("Max Tokens:"))
        self.max_new_tokens_spin = QSpinBox()
        self.max_new_tokens_spin.setRange(64, 512)
        self.max_new_tokens_spin.setValue(256)
        self.max_new_tokens_spin.setSingleStep(32)
        self.max_new_tokens_spin.setToolTip("Maximum output length (default 256)")
        tokens_layout.addWidget(self.max_new_tokens_spin)
        tokens_layout.addStretch()
        lighton_layout.addLayout(tokens_layout)

        # Custom prompt (optional)
        prompt_layout = QVBoxLayout()
        self.lighton_prompt_label = QLabel("Custom Prompt (optional):")
        prompt_layout.addWidget(self.lighton_prompt_label)
        self.lighton_prompt_edit = QLineEdit()
        self.lighton_prompt_edit.setPlaceholderText("e.g., 'Transcribe the German shorthand text'")
        self.lighton_prompt_edit.setToolTip("Leave empty for default OCR prompt")
        prompt_layout.addWidget(self.lighton_prompt_edit)
        lighton_layout.addLayout(prompt_layout)

        self.lighton_group.setLayout(lighton_layout)
        self.lighton_group.setVisible(False)  # Hidden by default
        layout.addWidget(self.lighton_group)

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

    def _browse_adapter_path(self):
        """Browse for adapter directory (Qwen3 LoRA)."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Adapter Directory", "models"
        )
        if folder:
            self.adapter_edit.setText(folder)

    def _refresh_openwebui_models(self):
        """Fetch available models from OpenWebUI API."""
        api_key = self.api_key_edit.text().strip()

        if not api_key:
            self.openwebui_model_combo.clear()
            self.openwebui_model_combo.addItem("Enter API key first")
            return

        try:
            from openai import OpenAI

            # Create temporary client to fetch models
            client = OpenAI(
                base_url="https://openwebui.uni-freiburg.de/api",
                api_key=api_key
            )

            # Fetch models
            models = client.models.list()

            available_models = []
            for model in models.data:
                available_models.append(model.id)

            # Update combo box
            self.openwebui_model_combo.clear()
            if available_models:
                self.openwebui_model_combo.addItems(sorted(available_models))
                print(f"[OpenWebUI] Loaded {len(available_models)} models")
            else:
                self.openwebui_model_combo.addItem("No models found")

            self._update_command_preview()

        except ImportError:
            self.openwebui_model_combo.clear()
            self.openwebui_model_combo.addItem("Error: openai package not installed")
        except Exception as e:
            print(f"Error fetching models: {e}")
            self.openwebui_model_combo.clear()
            self.openwebui_model_combo.addItem(f"Error: {str(e)[:40]}")

    def _on_engine_changed(self, engine_name: str):
        """Handle engine selection change."""
        if engine_name not in ENGINES:
            return

        config = ENGINES[engine_name]

        # Show/hide model path/ID based on engine
        self.model_path_edit.setEnabled(config.get("needs_model_path", True))
        self.model_browse_btn.setEnabled(config.get("needs_model_path", True))
        self.model_id_edit.setEnabled(config.get("needs_model_id", False))

        # Update tooltip for model path (Qwen3 has special instructions)
        if "model_path_tooltip" in config:
            self.model_path_edit.setToolTip(config["model_path_tooltip"])
        else:
            self.model_path_edit.setToolTip("")

        # Show/hide adapter path for Qwen3-VL
        needs_adapter = config.get("needs_adapter", False)
        self.adapter_label.setVisible(needs_adapter)
        self.adapter_edit.setVisible(needs_adapter)
        self.adapter_browse_btn.setVisible(needs_adapter)

        # Show/hide line mode checkbox for Qwen3-VL
        supports_line_mode = config.get("supports_line_mode", False)
        self.line_mode_check.setVisible(supports_line_mode)
        if not supports_line_mode:
            self.line_mode_check.setChecked(False)  # Reset when switching away

        # Show/hide API key and model dropdown for OpenWebUI
        needs_api_key = config.get("needs_api_key", False)
        is_openwebui = (engine_name == "OpenWebUI")
        self.api_key_label.setVisible(needs_api_key)
        self.api_key_edit.setVisible(needs_api_key)
        self.api_key_show_check.setVisible(needs_api_key)
        self.openwebui_model_label.setVisible(is_openwebui)
        self.openwebui_model_combo.setVisible(is_openwebui)
        self.openwebui_refresh_btn.setVisible(is_openwebui)

        # Auto-populate API key from environment if empty
        if needs_api_key and not self.api_key_edit.text():
            env_key = os.environ.get("OPENWEBUI_API_KEY", "")
            if env_key:
                self.api_key_edit.setText(env_key)
                # Auto-refresh models if API key is available
                if is_openwebui:
                    self._refresh_openwebui_models()

        # Show/hide num_beams for TrOCR
        self.beams_layout_widget.setVisible(config.get("supports_beams", False))

        # Show/hide LightOnOCR-specific controls
        has_lighton = config.get("has_lighton_options", False)
        self.lighton_group.setVisible(has_lighton)

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
        else:
            self.model_path_edit.clear()
        if "model_id" in preset:
            self.model_id_edit.setText(preset["model_id"])
        else:
            self.model_id_edit.clear()
        if "adapter" in preset:
            self.adapter_edit.setText(preset["adapter"])
        else:
            self.adapter_edit.clear()

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

        # Line mode
        if "line_mode" in preset:
            self.line_mode_check.setChecked(preset["line_mode"])
        else:
            self.line_mode_check.setChecked(False)

        # LightOnOCR-specific
        if "longest_edge" in preset:
            self.longest_edge_spin.setValue(preset["longest_edge"])
        else:
            self.longest_edge_spin.setValue(700)  # Reset to default
        if "max_new_tokens" in preset:
            self.max_new_tokens_spin.setValue(preset["max_new_tokens"])
        else:
            self.max_new_tokens_spin.setValue(256)  # Reset to default
        if "lighton_prompt" in preset:
            self.lighton_prompt_edit.setText(preset["lighton_prompt"])
        else:
            self.lighton_prompt_edit.clear()

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
        # For OpenWebUI, use the model dropdown instead of model_id_edit
        if config["engine"] == "OpenWebUI":
            model_text = self.openwebui_model_combo.currentText()
            # Only use if it's a valid model (not placeholder text)
            if model_text and not model_text.startswith(("Click", "Enter", "Error", "No models")):
                config["model_id"] = model_text
        elif self.model_id_edit.text():
            config["model_id"] = self.model_id_edit.text()
        if self.adapter_edit.text():
            config["adapter"] = self.adapter_edit.text()

        # API key (for OpenWebUI)
        if self.api_key_edit.text():
            config["api_key"] = self.api_key_edit.text()

        # Line mode (for line-trained Qwen3 models)
        if self.line_mode_check.isChecked():
            config["line_mode"] = True

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

        # LightOnOCR-specific
        if config["engine"] == "LightOnOCR":
            config["longest_edge"] = self.longest_edge_spin.value()
            config["max_new_tokens"] = self.max_new_tokens_spin.value()
            prompt = self.lighton_prompt_edit.text().strip()
            if prompt:
                config["lighton_prompt"] = prompt

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

        # Adapter (for Qwen3 LoRA)
        if config.get("adapter"):
            cmd += ["--adapter", config["adapter"]]

        # API key (for OpenWebUI)
        if config.get("api_key"):
            cmd += ["--api-key", config["api_key"]]

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
        if config.get("line_mode"):
            cmd += ["--line-mode"]

        # LightOnOCR-specific
        if config.get("engine") == "LightOnOCR":
            if config.get("longest_edge"):
                cmd += ["--longest-edge", str(config["longest_edge"])]
            if config.get("max_new_tokens"):
                cmd += ["--max-new-tokens", str(config["max_new_tokens"])]
            if config.get("lighton_prompt"):
                cmd += ["--prompt", config["lighton_prompt"]]

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
        self.adapter_edit.textChanged.connect(self._update_command_preview)
        self.api_key_edit.textChanged.connect(self._update_command_preview)

        # Combos
        self.engine_combo.currentTextChanged.connect(self._update_command_preview)
        self.device_combo.currentTextChanged.connect(self._update_command_preview)
        self.seg_method_combo.currentTextChanged.connect(self._update_command_preview)
        self.openwebui_model_combo.currentTextChanged.connect(self._update_command_preview)

        # Checkboxes
        self.pagexml_check.stateChanged.connect(self._update_command_preview)
        self.txt_check.stateChanged.connect(self._update_command_preview)
        self.csv_check.stateChanged.connect(self._update_command_preview)
        self.pagexml_out_check.stateChanged.connect(self._update_command_preview)
        self.resume_check.stateChanged.connect(self._update_command_preview)
        self.verbose_check.stateChanged.connect(self._update_command_preview)
        self.line_mode_check.stateChanged.connect(self._update_command_preview)

        # Spinners
        self.batch_spin.valueChanged.connect(self._update_command_preview)
        self.num_beams_spin.valueChanged.connect(self._update_command_preview)
        self.sensitivity_spin.valueChanged.connect(self._update_command_preview)

        # LightOnOCR-specific
        self.longest_edge_spin.valueChanged.connect(self._update_command_preview)
        self.max_new_tokens_spin.valueChanged.connect(self._update_command_preview)
        self.lighton_prompt_edit.textChanged.connect(self._update_command_preview)

    def _validate_config(self) -> Optional[str]:
        """Validate current configuration. Returns error message or None."""
        config = self._get_current_config()

        # Check input folder exists
        if not Path(config["input_folder"]).exists():
            return f"Input folder does not exist: {config['input_folder']}"

        # Check model is specified
        if not config.get("model_path") and not config.get("model_id"):
            return "Please specify a model path or HuggingFace model ID"

        # Qwen3-VL: model_path and model_id are mutually exclusive
        # model_path = fully local model
        # model_id + optional adapter = HuggingFace base + LoRA
        if config["engine"] == "Qwen3-VL":
            if config.get("model_path") and config.get("model_id"):
                return ("Qwen3-VL: Use EITHER Model Path (fully local model) OR "
                        "Model ID (HuggingFace base + optional Adapter), not both.\n\n"
                        "For local model: fill Model Path, clear Model ID\n"
                        "For HF + adapter: fill Model ID, optionally Adapter Path")

            # Detect if user accidentally put adapter path in model_path
            model_path = config.get("model_path", "")
            if model_path:
                model_path_lower = model_path.lower()
                # Check for adapter indicators
                if "adapter" in model_path_lower or model_path_lower.endswith(".safetensors"):
                    return ("Qwen3-VL: Model Path appears to be an adapter, not a base model.\n\n"
                            "For LoRA adapters, use:\n"
                            "  - Model ID: Qwen/Qwen3-VL-8B-Instruct (or other HF base)\n"
                            "  - Adapter Path: your adapter folder\n\n"
                            "Clear Model Path and set Model ID + Adapter Path instead.")

            # Validate Qwen3-VL model ID includes "-VL-"
            model_id = config.get("model_id", "")
            if model_id and "qwen" in model_id.lower() and "-vl" not in model_id.lower():
                return (f"Qwen3-VL: Model ID '{model_id}' appears to be a text-only Qwen model.\n\n"
                        "For Qwen3-VL (vision-language), use:\n"
                        "  - Qwen/Qwen3-VL-8B-Instruct (base)\n"
                        "  - Or a finetuned VL model like wjbmattingly/Qwen3-VL-8B-...\n\n"
                        "The model name must contain '-VL-' for vision-language support.")

            # Check adapter path is a folder, not a file
            adapter = config.get("adapter", "")
            if adapter and adapter.endswith(".safetensors"):
                return ("Qwen3-VL: Adapter Path should be a folder, not a .safetensors file.\n\n"
                        "Use the adapter folder, e.g.:\n"
                        "  models/Qwen3-VL-8B-glagolitic/final_model\n\n"
                        "Not the safetensors file inside it.")

        # OpenWebUI: requires API key and model_id
        if config["engine"] == "OpenWebUI":
            if not config.get("api_key"):
                return ("OpenWebUI: API key is required.\n\n"
                        "Get your API key from https://openwebui.uni-freiburg.de\n"
                        "Or set the OPENWEBUI_API_KEY environment variable.")
            if not config.get("model_id"):
                return ("OpenWebUI: Please select a model.\n\n"
                        "Click 'Refresh' to load available models from the server.")

        # Check model path exists (if specified)
        if config.get("model_path"):
            if not Path(config["model_path"]).exists():
                return f"Model file does not exist: {config['model_path']}"

        # Check adapter path exists (if specified)
        if config.get("adapter"):
            adapter_path = Path(config["adapter"])
            if not adapter_path.exists():
                return f"Adapter folder does not exist: {config['adapter']}"
            if not adapter_path.is_dir():
                return f"Adapter path must be a folder, not a file: {config['adapter']}"

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
        from PyQt6.QtWidgets import QDialog

        cmd_str = " ".join(cmd)

        # Create execution dialog as child of main window
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumSize(800, 600)
        layout = QVBoxLayout()

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

        # Execute command using QProcess (as child of dialog)
        process = QProcess(dialog)

        def append_output():
            # Check widget exists AND is visible (not deleted)
            try:
                if output_display and output_display.isVisible():
                    output = process.readAllStandardOutput().data().decode('utf-8', errors='ignore')
                    if output:
                        output_display.append(output)
            except RuntimeError:
                # Widget was deleted - ignore
                pass

        def append_error():
            # Check widget exists AND is visible (not deleted)
            try:
                if output_display and output_display.isVisible():
                    error = process.readAllStandardError().data().decode('utf-8', errors='ignore')
                    if error:
                        output_display.append(f"<span style='color: red;'>{error}</span>")
            except RuntimeError:
                # Widget was deleted - ignore
                pass

        def process_finished(exit_code, exit_status):
            # Check widget exists AND is visible (not deleted)
            try:
                if output_display and output_display.isVisible():
                    if exit_code == 0:
                        output_display.append("\n<b>✓ Process completed successfully!</b>")
                    else:
                        output_display.append(f"\n<b>❌ Process failed with exit code {exit_code}</b>")
            except RuntimeError:
                # Widget was deleted - ignore
                pass

        process.readyReadStandardOutput.connect(append_output)
        process.readyReadStandardError.connect(append_error)
        process.finished.connect(process_finished)

        # Start process
        output_display.append(f"<b>Starting {title}...</b>\n")
        process.start(cmd[0], cmd[1:])

        # Show dialog (modal)
        dialog.exec()


def main():
    app = QApplication(sys.argv)
    window = PolyscriptorBatchGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
