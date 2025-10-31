"""
Commercial API Engine Plugin

Wraps commercial HTR APIs (OpenAI, Gemini, Claude) as a unified plugin.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from htr_engine_base import HTREngine, TranscriptionResult

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in the project root (parent of engines/)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[CommercialAPIEngine] Loaded environment variables from {env_path}")
except ImportError:
    print("[CommercialAPIEngine] Warning: python-dotenv not installed. API keys will not be loaded from .env file.")
    print("Install with: pip install python-dotenv")

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QPushButton, QCheckBox, QLineEdit, QGroupBox, QTextEdit
    )
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

try:
    from inference_commercial_api import (
        OpenAIInference, GeminiInference, ClaudeInference,
        check_api_availability,
        OPENAI_MODELS, GEMINI_MODELS, CLAUDE_MODELS
    )
    COMMERCIAL_API_AVAILABLE = True
    API_AVAILABILITY = check_api_availability()
except ImportError:
    COMMERCIAL_API_AVAILABLE = False
    API_AVAILABILITY = {"openai": False, "gemini": False, "claude": False}
    OPENAI_MODELS = []
    GEMINI_MODELS = []
    CLAUDE_MODELS = []


class CommercialAPIEngine(HTREngine):
    """Commercial API HTR engine plugin."""

    def __init__(self):
        self.model: Optional[object] = None  # Can be OpenAI, Gemini, or Claude
        self._config_widget: Optional[QWidget] = None
        self._current_provider: Optional[str] = None

        # Widget references
        self._provider_combo: Optional[QComboBox] = None
        self._model_combo: Optional[QComboBox] = None
        self._api_key_edit: Optional[QLineEdit] = None
        self._show_key_check: Optional[QCheckBox] = None
        self._prompt_edit: Optional[QTextEdit] = None

    def get_name(self) -> str:
        return "Commercial APIs"

    def get_description(self) -> str:
        return "OpenAI GPT-4V, Google Gemini, Anthropic Claude vision APIs"

    def is_available(self) -> bool:
        return COMMERCIAL_API_AVAILABLE and PYQT_AVAILABLE and any(API_AVAILABILITY.values())

    def get_unavailable_reason(self) -> str:
        if not COMMERCIAL_API_AVAILABLE:
            return "Commercial API support not available. Install with: pip install openai google-generativeai anthropic"
        if not PYQT_AVAILABLE:
            return "PyQt6 not installed. Install with: pip install PyQt6"
        if not any(API_AVAILABILITY.values()):
            return "No API libraries installed. Install at least one: openai, google-generativeai, or anthropic"
        return ""

    def get_config_widget(self) -> QWidget:
        """Create Commercial API configuration panel."""
        if self._config_widget is not None:
            return self._config_widget

        widget = QWidget()
        layout = QVBoxLayout()

        # Provider selection
        provider_group = QGroupBox("API Provider")
        provider_layout = QVBoxLayout()

        self._provider_combo = QComboBox()
        available_providers = []
        if API_AVAILABILITY.get("openai", False):
            available_providers.append("OpenAI")
        if API_AVAILABILITY.get("gemini", False):
            available_providers.append("Gemini")
        if API_AVAILABILITY.get("claude", False):
            available_providers.append("Claude")

        if not available_providers:
            available_providers = ["No APIs available"]

        self._provider_combo.addItems(available_providers)
        self._provider_combo.currentTextChanged.connect(self._on_provider_changed)
        provider_layout.addWidget(self._provider_combo)

        provider_group.setLayout(provider_layout)
        layout.addWidget(provider_group)

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()

        self._model_combo = QComboBox()
        model_layout.addWidget(self._model_combo)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

            # API key
        key_group = QGroupBox("API Key")
        key_layout = QVBoxLayout()

        key_input_layout = QHBoxLayout()
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("Enter your API key")

        key_input_layout.addWidget(self._api_key_edit)


        self._show_key_check = QCheckBox("Show")
        self._show_key_check.toggled.connect(self._toggle_key_visibility)
        key_input_layout.addWidget(self._show_key_check)
        key_layout.addLayout(key_input_layout)

        key_hint = QLabel("API keys are stored locally in .trocr_gui/")
        key_hint.setStyleSheet("color: gray; font-size: 9pt;")
        key_layout.addWidget(key_hint)

        key_group.setLayout(key_layout)
        layout.addWidget(key_group)

        # Prompt section (custom prompt)
        prompt_group = QGroupBox("Custom Prompt (Optional)")
        prompt_layout = QVBoxLayout()

        self._prompt_edit = QTextEdit()
        self._prompt_edit.setPlaceholderText("Enter custom transcription prompt...")
        self._prompt_edit.setMaximumHeight(100)
        prompt_layout.addWidget(self._prompt_edit)

        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        layout.addStretch()
        widget.setLayout(layout)

        self._config_widget = widget

        # Initialize model list based on default provider
        self._on_provider_changed(self._provider_combo.currentText())

        return widget

    def _get_api_key_file(self) -> 'Path':
        """Get path to API key storage file."""
        from pathlib import Path
        storage_dir = Path.home() / ".trocr_gui"
        storage_dir.mkdir(exist_ok=True)
        return storage_dir / "api_keys.json"

    def _load_saved_api_key(self):
        """Load saved API key for current provider."""
        try:
            import json
            key_file = self._get_api_key_file()

            if key_file.exists():
                with open(key_file, "r") as f:
                    keys = json.load(f)

                provider = self._provider_combo.currentText().lower()
                if provider in keys:
                    self._api_key_edit.setText(keys[provider])
        except Exception as e:
            print(f"Warning: Could not load saved API key: {e}")

    def _save_api_key(self):
        """Save API key for current provider."""
        try:
            import json
            key_file = self._get_api_key_file()

            # Load existing keys
            keys = {}
            if key_file.exists():
                with open(key_file, "r") as f:
                    keys = json.load(f)

            # Update key for current provider
            provider = self._provider_combo.currentText().lower()
            api_key = self._api_key_edit.text().strip()

            if api_key:
                keys[provider] = api_key

                with open(key_file, "w") as f:
                    json.dump(keys, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save API key: {e}")

    def _on_provider_changed(self, provider: str):
        """Update model list when provider changes and load API key from environment."""
        if self._model_combo is None:
            return

        self._model_combo.clear()

        if provider == "OpenAI":
            self._model_combo.addItems(OPENAI_MODELS)
        elif provider == "Gemini":
            self._model_combo.addItems(GEMINI_MODELS)
        elif provider == "Claude":
            self._model_combo.addItems(CLAUDE_MODELS)
        else:
            self._model_combo.addItem("No models available")

        # Auto-load API key from environment variables
        if self._api_key_edit is not None:
            env_key = self._get_api_key_from_env(provider)
            if env_key:
                self._api_key_edit.setText(env_key)
                print(f"[CommercialAPIEngine] Loaded {provider} API key from environment")

    def _get_api_key_from_env(self, provider: str) -> Optional[str]:
        """Get API key from environment variables based on provider."""
        env_var_map = {
            "OpenAI": "OPENAI_API_KEY",
            "Gemini": "GOOGLE_API_KEY",
            "Claude": "ANTHROPIC_API_KEY"
        }

        env_var = env_var_map.get(provider)
        if env_var:
            return os.getenv(env_var, "")

    def _toggle_key_visibility(self, checked: bool):
        """Toggle API key visibility."""
        if checked:
            self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)

    def get_config(self) -> Dict[str, Any]:
        """Extract configuration from widget controls."""
        if self._config_widget is None:
            return {}

        prompt_text = self._prompt_edit.toPlainText().strip()

        return {
            "provider": self._provider_combo.currentText(),
            "model": self._model_combo.currentText(),
            "api_key": self._api_key_edit.text().strip(),
            "custom_prompt": prompt_text if prompt_text else None,
        }

    def set_config(self, config: Dict[str, Any]):
        """Restore configuration to widget controls."""
        if self._config_widget is None:
            return

        provider = config.get("provider", "")
        idx = self._provider_combo.findText(provider)
        if idx >= 0:
            self._provider_combo.setCurrentIndex(idx)

        model = config.get("model", "")
        idx = self._model_combo.findText(model)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)

        self._api_key_edit.setText(config.get("api_key", ""))

        custom_prompt = config.get("custom_prompt", "")
        if custom_prompt:
            self._prompt_edit.setPlainText(custom_prompt)

    def load_model(self, config: Dict[str, Any]) -> bool:
        """Load (initialize) API client."""
        try:
            provider = config.get("provider", "")
            model_name = config.get("model", "")
            api_key = config.get("api_key", "")

            if not api_key:
                print("Error: No API key provided")
                return False

            # Unload previous model
            self.unload_model()

            # Initialize appropriate client
            if provider == "OpenAI":
                self.model = OpenAIInference(api_key=api_key, model=model_name)
                self._current_provider = "openai"
            elif provider == "Gemini":
                self.model = GeminiInference(api_key=api_key, model=model_name)
                self._current_provider = "gemini"
            elif provider == "Claude":
                self.model = ClaudeInference(api_key=api_key, model=model_name)
                self._current_provider = "claude"
            else:
                return False

            return True

        except Exception as e:
            print(f"Error initializing API client: {e}")
            self.model = None
            self._current_provider = None
            return False

    def unload_model(self):
        """Unload (clear) API client."""
        if self.model is not None:
            del self.model
            self.model = None
            self._current_provider = None

    def is_model_loaded(self) -> bool:
        """Check if API client is initialized."""
        return self.model is not None

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """Transcribe a line image with commercial API."""
        if self.model is None:
            return TranscriptionResult(text="[API client not initialized]", confidence=0.0)

        if config is None:
            config = self.get_config()

        custom_prompt = config.get("custom_prompt")

        try:
            # Convert numpy array to PIL Image
            from PIL import Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # All API clients have transcribe() method
            # It returns a string directly, not a dict
            text = self.model.transcribe(pil_image, prompt=custom_prompt)

            return TranscriptionResult(
                text=text if text else "",
                confidence=1.0,  # API models don't provide confidence
                metadata={
                    "provider": self._current_provider,
                    "model": config.get("model", "")
                }
            )

        except Exception as e:
            print(f"Error in API transcription: {e}")
            import traceback
            traceback.print_exc()
            return TranscriptionResult(text=f"[API Error: {e}]", confidence=0.0)

    def get_capabilities(self) -> Dict[str, bool]:
        """Commercial API capabilities."""
        return {
            "batch_processing": False,  # APIs typically process one at a time
            "confidence_scores": False,  # Most don't provide confidence
            "beam_search": False,  # Internal to API
            "language_model": True,  # All are language models
            "preprocessing": True,  # APIs handle preprocessing
        }

    def requires_line_segmentation(self) -> bool:
        """Commercial APIs can process full pages without segmentation."""
        return False
