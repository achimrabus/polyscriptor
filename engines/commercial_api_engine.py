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
        OPENAI_MODELS, GEMINI_MODELS, CLAUDE_MODELS,
        fetch_openai_models, fetch_gemini_models
    )
    COMMERCIAL_API_AVAILABLE = True
    API_AVAILABILITY = check_api_availability()
except ImportError:
    COMMERCIAL_API_AVAILABLE = False
    API_AVAILABILITY = {"openai": False, "gemini": False, "claude": False}
    OPENAI_MODELS = []
    GEMINI_MODELS = []
    CLAUDE_MODELS = []
    fetch_openai_models = lambda api_key=None: []
    fetch_gemini_models = lambda api_key=None: []


class CommercialAPIEngine(HTREngine):
    """Commercial API HTR engine plugin."""

    def __init__(self):
        # Instance attributes (avoid type annotations here for broader runtime compatibility in some environments)
        self.model = None  # Can be OpenAI, Gemini, or Claude
        self._config_widget = None
        self._current_provider = None

        # Widget references
        self._provider_combo = None
        self._model_combo = None
        self._custom_model_edit = None
        self._use_custom_model_check = None
        self._refresh_models_btn = None
        self._api_key_edit = None
        self._show_key_check = None
        self._prompt_edit = None
        self._thinking_combo = None
        self._temperature_edit = None
        self._max_tokens_edit = None
        self._early_exit_check = None
        self._auto_continue_check = None
        self._max_continuations_edit = None

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

    def get_config_widget(self):
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

        # Dropdown for standard models
        model_dropdown_layout = QHBoxLayout()
        self._model_combo = QComboBox()
        model_dropdown_layout.addWidget(self._model_combo)

        # Refresh models button
        self._refresh_models_btn = QPushButton("🔄 Refresh")
        self._refresh_models_btn.setToolTip("Fetch latest models from API")
        self._refresh_models_btn.setMaximumWidth(80)
        self._refresh_models_btn.clicked.connect(self._on_refresh_models)
        model_dropdown_layout.addWidget(self._refresh_models_btn)

        model_layout.addLayout(model_dropdown_layout)

        # Custom model ID checkbox and field
        custom_model_layout = QHBoxLayout()
        self._use_custom_model_check = QCheckBox("Use custom model ID:")
        self._use_custom_model_check.toggled.connect(self._on_custom_model_toggled)
        custom_model_layout.addWidget(self._use_custom_model_check)

        self._custom_model_edit = QLineEdit()
        self._custom_model_edit.setPlaceholderText("e.g., gpt-4.5, o1-preview-2024-12-17")
        self._custom_model_edit.setEnabled(False)  # Disabled by default
        custom_model_layout.addWidget(self._custom_model_edit)

        model_layout.addLayout(custom_model_layout)

        model_hint = QLabel("💡 Use custom model ID for bleeding-edge models not in the dropdown")
        model_hint.setStyleSheet("color: gray; font-size: 8pt;")
        model_hint.setWordWrap(True)
        model_layout.addWidget(model_hint)

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

        # Prompt & Sampling section
        prompt_group = QGroupBox("Prompt & Sampling (Optional)")
        prompt_layout = QVBoxLayout()

        self._prompt_edit = QTextEdit()
        self._prompt_edit.setPlaceholderText("Enter custom transcription prompt...")
        self._prompt_edit.setMaximumHeight(100)
        prompt_layout.addWidget(self._prompt_edit)

        # Temperature control
        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Temperature:"))
        self._temperature_edit = QLineEdit()
        self._temperature_edit.setPlaceholderText("1.0 (default)")
        self._temperature_edit.setToolTip(
            "Sampling temperature (web default ~1.0).\n"
            "Use 0-0.3 for deterministic; >1 can increase variability."
        )
        self._temperature_edit.setMaximumWidth(90)
        temp_row.addWidget(self._temperature_edit)
        temp_row.addStretch()
        prompt_layout.addLayout(temp_row)

        # Max output tokens control
        tokens_row = QHBoxLayout()
        tokens_row.addWidget(QLabel("Max output tokens:"))
        self._max_tokens_edit = QLineEdit()
        self._max_tokens_edit.setPlaceholderText("4096 preview / 2048 default")
        self._max_tokens_edit.setToolTip(
            "Upper limit on generated tokens. Lowering may force earlier output.\n"
            "Raising (e.g. 8192) may help high reasoning but risks long 'thinking'."
        )
        self._max_tokens_edit.setMaximumWidth(130)
        tokens_row.addWidget(self._max_tokens_edit)
        tokens_row.addStretch()
        prompt_layout.addLayout(tokens_row)

        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        # Thinking Mode section (for Gemini models)
        thinking_group = QGroupBox("Thinking Mode (Gemini only)")
        thinking_layout = QVBoxLayout()
        
    # (Removed warning banner recommending alternative models; preview model retained for Church Slavonic use)
        
        thinking_row = QHBoxLayout()
        thinking_row.addWidget(QLabel("Reasoning:"))
        self._thinking_combo = QComboBox()
        self._thinking_combo.addItems(["Auto (Low for preview)", "Low (Fast)", "High (More reasoning)"])
        self._thinking_combo.setToolTip(
            "Low: Fast, direct output\n"
            "High: Slower, uses more tokens for reasoning\n"
            "Auto: Uses Low for preview models to avoid token waste"
        )
        thinking_row.addWidget(self._thinking_combo)
        thinking_row.addStretch()
        thinking_layout.addLayout(thinking_row)
        
        thinking_group.setLayout(thinking_layout)
        layout.addWidget(thinking_group)

        # Advanced Gemini controls
        advanced_group = QGroupBox("Gemini Advanced")
        adv_layout = QVBoxLayout()

        # Row 1: Checkboxes
        adv_row1 = QHBoxLayout()
        self._early_exit_check = QCheckBox("Early exit on first chunk")
        self._early_exit_check.setChecked(True)
        self._early_exit_check.setToolTip("If checked, streaming returns after first non-empty text chunk. Uncheck to collect full stream.")
        adv_row1.addWidget(self._early_exit_check)
        
        self._auto_continue_check = QCheckBox("Auto continuation")
        self._auto_continue_check.setChecked(False)  # Default: off for speed
        self._auto_continue_check.setToolTip("If checked, performs additional continuation calls to capture missed trailing text.")
        adv_row1.addWidget(self._auto_continue_check)
        adv_row1.addStretch()
        adv_layout.addLayout(adv_row1)

        # Row 2: Continuation settings (symmetrical grid)
        adv_row2 = QHBoxLayout()
        adv_row2.addWidget(QLabel("Max passes:"))
        self._max_continuations_edit = QLineEdit()
        self._max_continuations_edit.setText("2")  # Default value
        self._max_continuations_edit.setToolTip("Maximum number of continuation attempts (2-3 recommended)")
        self._max_continuations_edit.setFixedWidth(60)
        adv_row2.addWidget(self._max_continuations_edit)
        
        adv_row2.addSpacing(20)
        
        adv_row2.addWidget(QLabel("Min new chars:"))
        self._min_new_chars_edit = QLineEdit()
        self._min_new_chars_edit.setText("50")  # Default value
        self._min_new_chars_edit.setToolTip("Minimum number of new characters required to accept a continuation chunk.")
        self._min_new_chars_edit.setFixedWidth(60)
        adv_row2.addWidget(self._min_new_chars_edit)
        adv_row2.addStretch()
        adv_layout.addLayout(adv_row2)

        # Row 3: Token & fallback settings (symmetrical grid)
        adv_row3 = QHBoxLayout()
        adv_row3.addWidget(QLabel("Low-mode tokens:"))
        self._low_initial_tokens_edit = QLineEdit()
        self._low_initial_tokens_edit.setText("6144")  # Default value
        self._low_initial_tokens_edit.setToolTip("Initial max_output_tokens for LOW thinking before fallback escalation (4096-8192).")
        self._low_initial_tokens_edit.setFixedWidth(60)
        adv_row3.addWidget(self._low_initial_tokens_edit)
        
        adv_row3.addSpacing(20)
        
        adv_row3.addWidget(QLabel("Fallback %:"))
        self._reasoning_fallback_edit = QLineEdit()
        self._reasoning_fallback_edit.setText("0.6")  # Default value
        self._reasoning_fallback_edit.setToolTip("Fraction of token budget consumed internally (no output) that triggers early fallback (0.5-0.8).")
        self._reasoning_fallback_edit.setFixedWidth(60)
        adv_row3.addWidget(self._reasoning_fallback_edit)

        adv_row3.addSpacing(20)
        adv_row3.addWidget(QLabel("Fallback cap:"))
        self._fallback_cap_edit = QLineEdit()
        self._fallback_cap_edit.setText("8192")  # Default configurable cap
        self._fallback_cap_edit.setToolTip("Maximum tokens for fallback attempt. Increase for page-wise recognition (e.g. 12288 or 16384).")
        self._fallback_cap_edit.setFixedWidth(70)
        adv_row3.addWidget(self._fallback_cap_edit)
        adv_row3.addStretch()
        adv_layout.addLayout(adv_row3)

        advanced_group.setLayout(adv_layout)
        layout.addWidget(advanced_group)

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

    def _on_custom_model_toggled(self, checked: bool):
        """Enable/disable custom model field."""
        self._custom_model_edit.setEnabled(checked)
        self._model_combo.setEnabled(not checked)

    def _on_refresh_models(self):
        """Refresh model list from API dynamically."""
        if self._model_combo is None or self._api_key_edit is None:
            return

        provider = self._provider_combo.currentText()
        api_key = self._api_key_edit.text().strip()

        if not api_key:
            print(f"[CommercialAPIEngine] Cannot refresh models: No API key provided")
            return

        print(f"[CommercialAPIEngine] Refreshing {provider} models from API...")

        # Save current selection
        current_model = self._model_combo.currentText()

        # Fetch models dynamically
        if provider == "OpenAI":
            models = fetch_openai_models(api_key)
        elif provider == "Gemini":
            models = fetch_gemini_models(api_key)
        else:
            print(f"[CommercialAPIEngine] Dynamic refresh not supported for {provider}")
            return

        # Update dropdown
        self._model_combo.clear()
        self._model_combo.addItems(models)

        # Restore selection if possible
        idx = self._model_combo.findText(current_model)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)

        print(f"[CommercialAPIEngine] Refreshed {len(models)} models for {provider}")

    def get_config(self) -> Dict[str, Any]:
        """Extract configuration from widget controls."""
        if self._config_widget is None:
            return {}

        prompt_text = self._prompt_edit.toPlainText().strip()

        # Use custom model if checkbox is enabled, otherwise use dropdown
        if self._use_custom_model_check.isChecked():
            model = self._custom_model_edit.text().strip()
        else:
            model = self._model_combo.currentText()

        return {
            "provider": self._provider_combo.currentText(),
            "model": model,
            "api_key": self._api_key_edit.text().strip(),
            "custom_prompt": prompt_text if prompt_text else None,
            "use_custom_model": self._use_custom_model_check.isChecked(),
            "custom_model_id": self._custom_model_edit.text().strip(),
        }

    def set_config(self, config: Dict[str, Any]):
        """Restore configuration to widget controls."""
        if self._config_widget is None:
            return

        provider = config.get("provider", "")
        idx = self._provider_combo.findText(provider)
        if idx >= 0:
            self._provider_combo.setCurrentIndex(idx)

        # Restore custom model checkbox and field
        use_custom = config.get("use_custom_model", False)
        self._use_custom_model_check.setChecked(use_custom)

        if use_custom:
            custom_model_id = config.get("custom_model_id", "")
            self._custom_model_edit.setText(custom_model_id)
        else:
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
            # Enable retry logic for Gemini to handle content blocking
            if self._current_provider == "gemini":
                # Get thinking mode setting
                thinking_mode = None
                temperature = None
                if self._thinking_combo is not None:
                    thinking_text = self._thinking_combo.currentText()
                    if "Low" in thinking_text:
                        thinking_mode = "low"
                        fast_direct = True  # low mode: request immediate output
                    elif "High" in thinking_text:
                        thinking_mode = "high"
                    # else: Auto = None (default)
                if self._temperature_edit is not None:
                    t_text = self._temperature_edit.text().strip()
                    if t_text:
                        try:
                            temperature = float(t_text)
                        except ValueError:
                            temperature = None
                max_tokens = None
                if self._max_tokens_edit is not None:
                    mt_text = self._max_tokens_edit.text().strip()
                    if mt_text:
                        try:
                            max_tokens = int(mt_text)
                        except ValueError:
                            max_tokens = None
                fast_direct_early_exit = True
                if self._early_exit_check is not None and not self._early_exit_check.isChecked():
                    fast_direct_early_exit = False
                # Extract continuation settings
                auto_continue = False
                max_auto_continuations = 2  # Default
                if self._auto_continue_check is not None and self._auto_continue_check.isChecked():
                    auto_continue = True
                    if self._max_continuations_edit is not None:
                        mc_text = self._max_continuations_edit.text().strip()
                        if mc_text:
                            try:
                                max_auto_continuations = int(mc_text)
                            except ValueError:
                                pass  # Keep default of 2
                
                # Extract continuation settings with defaults
                continuation_min_new_chars = 50
                if hasattr(self, '_min_new_chars_edit') and self._min_new_chars_edit is not None:
                    mnc_text = self._min_new_chars_edit.text().strip()
                    if mnc_text:
                        try:
                            continuation_min_new_chars = int(mnc_text)
                        except ValueError:
                            pass  # Keep default
                
                reasoning_fallback_threshold = 0.6
                if hasattr(self, '_reasoning_fallback_edit') and self._reasoning_fallback_edit is not None:
                    rft_text = self._reasoning_fallback_edit.text().strip()
                    if rft_text:
                        try:
                            reasoning_fallback_threshold = float(rft_text)
                        except ValueError:
                            pass  # Keep default

                fallback_cap = 8192
                if hasattr(self, '_fallback_cap_edit') and self._fallback_cap_edit is not None:
                    fc_text = self._fallback_cap_edit.text().strip()
                    if fc_text:
                        try:
                            fallback_cap = int(fc_text)
                        except ValueError:
                            pass  # Keep default if invalid value
                
                # Override max_tokens for LOW thinking mode if specified
                if thinking_mode == 'low' and hasattr(self, '_low_initial_tokens_edit') and self._low_initial_tokens_edit is not None:
                    lit_text = self._low_initial_tokens_edit.text().strip()
                    if lit_text:
                        try:
                            lit_val = int(lit_text)
                            if lit_val > 0:
                                max_tokens = lit_val
                                print(f"🔧 LOW thinking mode: overriding max_output_tokens to {max_tokens}")
                        except ValueError:
                            pass  # Keep existing max_tokens
                
                # Debug: show final token budget
                final_max_tokens = max_tokens if max_tokens is not None else 2048
                print(f"📊 Final settings: thinking_mode={thinking_mode}, max_output_tokens={final_max_tokens}, temp={temperature if temperature is not None else 1.0}")
                
                text = self.model.transcribe(
                    pil_image, 
                    prompt=custom_prompt,
                    temperature=temperature if temperature is not None else 1.0,
                    max_output_tokens=max_tokens if max_tokens is not None else 2048,
                    auto_retry_on_block=True,
                    safety_relax=True,
                    verbose_block_logging=True,
                    thinking_mode=thinking_mode,
                    fast_direct=fast_direct if 'fast_direct' in locals() else False,
                    fast_direct_early_exit=fast_direct_early_exit,
                    auto_continue=auto_continue,
                    max_auto_continuations=max_auto_continuations,
                    continuation_min_new_chars=continuation_min_new_chars,
                    reasoning_fallback_threshold=reasoning_fallback_threshold,
                    fallback_max_output_tokens=fallback_cap,
                    record_stats_csv="gemini_runs.csv"
                )
            else:
                temperature = None
                if self._temperature_edit is not None:
                    t_text = self._temperature_edit.text().strip()
                    if t_text:
                        try:
                            temperature = float(t_text)
                        except ValueError:
                            temperature = None
                max_tokens = None
                if self._max_tokens_edit is not None:
                    mt_text = self._max_tokens_edit.text().strip()
                    if mt_text:
                        try:
                            max_tokens = int(mt_text)
                        except ValueError:
                            max_tokens = None
                text = self.model.transcribe(
                    pil_image,
                    prompt=custom_prompt,
                    temperature=temperature if temperature is not None else 1.0,
                    max_output_tokens=max_tokens if max_tokens is not None else 2048,
                )

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
