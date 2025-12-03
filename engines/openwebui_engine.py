"""
OpenWebUI Engine Plugin

Wraps the OpenWebUI API (OpenAI-compatible) from uni-freiburg.de as an HTR engine.
Supports multiple models available on the OpenWebUI platform.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import os
import numpy as np
from PIL import Image
import io
import base64

from htr_engine_base import HTREngine, TranscriptionResult

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QPushButton, QCheckBox, QLineEdit, QGroupBox, QTextEdit,
        QSpinBox
    )
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class OpenWebUIEngine(HTREngine):
    """OpenWebUI API HTR engine plugin (OpenAI-compatible)."""

    def __init__(self):
        self.client: Optional[OpenAI] = None
        self._config_widget: Optional[QWidget] = None
        self._available_models: List[str] = []

        # Store config from load_model for batch processing
        self._loaded_config: Dict[str, Any] = {}

        # Widget references
        self._model_combo: Optional[QComboBox] = None
        self._api_key_edit: Optional[QLineEdit] = None
        self._show_key_check: Optional[QCheckBox] = None
        self._prompt_edit: Optional[QTextEdit] = None
        self._temperature_spin: Optional[QSpinBox] = None
        self._max_tokens_spin: Optional[QSpinBox] = None
        self._refresh_models_btn: Optional[QPushButton] = None

        # Default API configuration
        self.base_url = "https://openwebui.uni-freiburg.de/api"
        
        # Load environment variables from .env file (only once when instantiated)
        self._load_env_variables()

    def _load_env_variables(self):
        """Load environment variables from .env file if available."""
        try:
            from dotenv import load_dotenv
            # Look for .env in the project root (parent of engines/)
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
        except ImportError:
            # Silently skip if python-dotenv is not installed
            # Environment variables can still be set via OS
            pass

        # Load environment variables from .env file (if available)
        self._load_env_file()

    def _load_env_file(self):
        """Load environment variables from project root's .env file.
        
        Looks for .env in the project root directory (parent of engines/).
        Silently skips loading if python-dotenv is not installed or if .env doesn't exist.
        
        Environment variables loaded (if present):
            - OPENWEBUI_API_KEY: Used as fallback when API key not in config
        
        If .env loading fails or is skipped, the engine will still work if API keys
        are provided through other means (config, OS environment variables).
        """
        if not DOTENV_AVAILABLE:
            return
            
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    def get_name(self) -> str:
        return "OpenWebUI"

    def get_description(self) -> str:
        return "OpenWebUI API from openwebui.uni-freiburg.de (OpenAI-compatible, multiple models)"

    def is_available(self) -> bool:
        return OPENAI_AVAILABLE and PYQT_AVAILABLE

    def get_unavailable_reason(self) -> str:
        if not OPENAI_AVAILABLE:
            return "OpenAI library not installed. Install with: pip install openai"
        if not PYQT_AVAILABLE:
            return "PyQt6 not installed. Install with: pip install PyQt6"
        return ""

    def get_config_widget(self) -> QWidget:
        """Create OpenWebUI configuration panel."""
        if self._config_widget is not None:
            return self._config_widget

        widget = QWidget()
        layout = QVBoxLayout()

        # API Key section
        key_group = QGroupBox("API Key")
        key_layout = QVBoxLayout()

        key_input_layout = QHBoxLayout()
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("Enter your OpenWebUI API key")
        key_input_layout.addWidget(self._api_key_edit)

        self._show_key_check = QCheckBox("Show")
        self._show_key_check.toggled.connect(self._toggle_key_visibility)
        key_input_layout.addWidget(self._show_key_check)
        key_layout.addLayout(key_input_layout)

        key_hint = QLabel("Get your API key from https://openwebui.uni-freiburg.de")
        key_hint.setStyleSheet("color: gray; font-size: 9pt;")
        key_layout.addWidget(key_hint)

        key_group.setLayout(key_layout)
        layout.addWidget(key_group)

        # Model selection with refresh button
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()

        model_select_layout = QHBoxLayout()
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(300)
        model_select_layout.addWidget(self._model_combo)

        self._refresh_models_btn = QPushButton("Refresh Models")
        self._refresh_models_btn.clicked.connect(self._refresh_models)
        model_select_layout.addWidget(self._refresh_models_btn)

        model_layout.addLayout(model_select_layout)

        model_hint = QLabel("Click 'Refresh Models' to load available models from the server")
        model_hint.setStyleSheet("color: gray; font-size: 9pt;")
        model_layout.addWidget(model_hint)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Generation parameters
        params_group = QGroupBox("Generation Parameters")
        params_layout = QVBoxLayout()

        # Temperature
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self._temperature_spin = QSpinBox()
        self._temperature_spin.setRange(0, 100)
        self._temperature_spin.setValue(10)  # 0.1
        self._temperature_spin.setSuffix(" (×0.01)")
        temp_layout.addWidget(self._temperature_spin)
        temp_layout.addStretch()
        params_layout.addLayout(temp_layout)

        # Max tokens
        tokens_layout = QHBoxLayout()
        tokens_layout.addWidget(QLabel("Max Tokens:"))
        self._max_tokens_spin = QSpinBox()
        self._max_tokens_spin.setRange(100, 4096)
        self._max_tokens_spin.setValue(500)
        tokens_layout.addWidget(self._max_tokens_spin)
        tokens_layout.addStretch()
        params_layout.addLayout(tokens_layout)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Custom prompt section
        prompt_group = QGroupBox("Custom Prompt (Optional)")
        prompt_layout = QVBoxLayout()

        self._prompt_edit = QTextEdit()
        self._prompt_edit.setPlaceholderText(
            "Enter custom transcription prompt...\n\n"
            "Default prompt:\n"
            "Transcribe the text in this historical manuscript line image. "
            "Return only the transcribed text without any explanation or formatting."
        )
        self._prompt_edit.setMaximumHeight(120)
        prompt_layout.addWidget(self._prompt_edit)

        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        layout.addStretch()
        widget.setLayout(layout)

        self._config_widget = widget

        # Try to load saved API key
        self._load_saved_api_key()

        return widget

    def _toggle_key_visibility(self, checked: bool):
        """Toggle API key visibility."""
        if checked:
            self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)

    def _get_api_key_file(self) -> 'Path':
        """Get path to API key storage file."""
        from pathlib import Path
        storage_dir = Path.home() / ".trocr_gui"
        storage_dir.mkdir(exist_ok=True)
        return storage_dir / "api_keys.json"

    def _load_saved_api_key(self):
        """Load saved API key."""
        try:
            import json
            key_file = self._get_api_key_file()

            if key_file.exists():
                with open(key_file, "r") as f:
                    keys = json.load(f)

                if "openwebui" in keys:
                    self._api_key_edit.setText(keys["openwebui"])
        except Exception as e:
            print(f"Warning: Could not load saved API key: {e}")

    def _save_api_key(self):
        """Save API key."""
        try:
            import json
            key_file = self._get_api_key_file()

            # Load existing keys
            keys = {}
            if key_file.exists():
                with open(key_file, "r") as f:
                    keys = json.load(f)

            # Update key for OpenWebUI
            api_key = self._api_key_edit.text().strip()

            if api_key:
                keys["openwebui"] = api_key

                with open(key_file, "w") as f:
                    json.dump(keys, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save API key: {e}")

    def _refresh_models(self):
        """Fetch available models from OpenWebUI API."""
        api_key = self._api_key_edit.text().strip()

        if not api_key:
            self._model_combo.clear()
            self._model_combo.addItem("Please enter API key first")
            return

        try:
            # Create temporary client to fetch models
            client = OpenAI(
                base_url=self.base_url,
                api_key=api_key
            )

            # Fetch models
            models = client.models.list()

            self._available_models = []
            for model in models.data:
                self._available_models.append(model.id)

            # Update combo box
            self._model_combo.clear()
            if self._available_models:
                self._model_combo.addItems(sorted(self._available_models))
                print(f"[OpenWebUI] Loaded {len(self._available_models)} models")
            else:
                self._model_combo.addItem("No models found")

        except Exception as e:
            print(f"Error fetching models: {e}")
            self._model_combo.clear()
            self._model_combo.addItem(f"Error: {str(e)[:50]}")

    def get_config(self) -> Dict[str, Any]:
        """Extract configuration from widget controls."""
        if self._config_widget is None:
            return {}

        prompt_text = self._prompt_edit.toPlainText().strip()

        return {
            "api_key": self._api_key_edit.text().strip(),
            "model": self._model_combo.currentText(),
            "temperature": self._temperature_spin.value() / 100.0,
            "max_tokens": self._max_tokens_spin.value(),
            "custom_prompt": prompt_text if prompt_text else None,
        }

    def set_config(self, config: Dict[str, Any]):
        """Restore configuration to widget controls."""
        if self._config_widget is None:
            return

        self._api_key_edit.setText(config.get("api_key", ""))

        model = config.get("model", "")
        idx = self._model_combo.findText(model)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)

        temp = int(config.get("temperature", 0.1) * 100)
        self._temperature_spin.setValue(temp)

        self._max_tokens_spin.setValue(config.get("max_tokens", 500))

        custom_prompt = config.get("custom_prompt", "")
        if custom_prompt:
            self._prompt_edit.setPlainText(custom_prompt)

    def load_model(self, config: Dict[str, Any]) -> bool:
        """Initialize OpenWebUI client."""
        try:
            api_key = config.get("api_key", "")

            # Fall back to environment variable if no API key provided
            if not api_key:
                api_key = os.environ.get("OPENWEBUI_API_KEY", "")

            if not api_key:
                print("Error: No API key provided. Set via config or OPENWEBUI_API_KEY env var.")
                return False

            # Store config for batch processing (model, temperature, etc.)
            self._loaded_config = config.copy()

            # Save API key for future use
            if self._api_key_edit and self._api_key_edit.text().strip():
                self._save_api_key()

            # Initialize client
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=api_key
            )

            model = config.get("model", config.get("model_id", "unknown"))
            print(f"[OpenWebUI] Client initialized with base URL: {self.base_url}, model: {model}")
            return True

        except Exception as e:
            print(f"Error initializing OpenWebUI client: {e}")
            self.client = None
            return False

    def unload_model(self):
        """Unload OpenWebUI client."""
        if self.client is not None:
            self.client = None
        self._loaded_config = {}

    def is_model_loaded(self) -> bool:
        """Check if client is initialized."""
        return self.client is not None

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """Transcribe a line image with OpenWebUI API."""
        if self.client is None:
            return TranscriptionResult(text="[OpenWebUI client not initialized]", confidence=0.0)

        if config is None:
            # First try loaded config (from batch processing), then GUI config
            if self._loaded_config:
                config = self._loaded_config
            else:
                config = self.get_config()

        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Encode image to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Prepare prompt
            custom_prompt = config.get("custom_prompt")
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = (
                    "Transcribe the text in this historical manuscript line image. "
                    "Return only the transcribed text without any explanation or formatting."
                )

            # Get model and parameters
            model = config.get("model", "gpt-4-vision-preview")
            temperature = config.get("temperature", 0.1)
            max_tokens = config.get("max_tokens", 500)

            # Call OpenWebUI API (OpenAI-compatible)
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract transcription
            text = response.choices[0].message.content.strip()

            # Extract usage info
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            return TranscriptionResult(
                text=text,
                confidence=1.0,  # OpenWebUI doesn't provide confidence
                metadata={
                    "provider": "openwebui",
                    "model": model,
                    "usage": usage
                }
            )

        except Exception as e:
            print(f"Error in OpenWebUI transcription: {e}")
            import traceback
            traceback.print_exc()
            return TranscriptionResult(text=f"[OpenWebUI Error: {e}]", confidence=0.0)

    def get_capabilities(self) -> Dict[str, bool]:
        """OpenWebUI capabilities."""
        return {
            "batch_processing": False,
            "confidence_scores": False,
            "beam_search": False,
            "language_model": True,
            "preprocessing": True,
        }

    def requires_line_segmentation(self) -> bool:
        """OpenWebUI VLMs can process full pages directly without segmentation."""
        return False  # VLMs process full page images
