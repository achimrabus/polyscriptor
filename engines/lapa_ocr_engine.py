"""
LapaOCR Engine Plugin

Optional engine for Ukrainian OCR using:
- Base model: lapa-llm/lapa-v0.1.2-instruct
- LoRA adapter: VmF0x/lapa-ocr-lora

This engine is line-level/crop-level and therefore requires segmentation for pages.
"""

from typing import Dict, Any, Optional, List

import numpy as np

from htr_engine_base import HTREngine, TranscriptionResult

try:
    from PyQt6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QComboBox,
        QSpinBox,
        QTextEdit,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object


LAPA_AVAILABLE = False
LAPA_MISSING_DEPS: List[str] = []

try:
    import torch
except ImportError:
    torch = None
    LAPA_MISSING_DEPS.append("torch")

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
    LAPA_MISSING_DEPS.append("peft")

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:
    AutoModelForImageTextToText = None
    AutoProcessor = None
    LAPA_MISSING_DEPS.append("transformers")

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    BNB_AVAILABLE = False

if not LAPA_MISSING_DEPS:
    LAPA_AVAILABLE = True


class LapaOCREngine(HTREngine):
    """Lapa OCR LoRA engine for Ukrainian text crops."""

    DEFAULT_BASE_MODEL = "lapa-llm/lapa-v0.1.2-instruct"
    DEFAULT_ADAPTER = "VmF0x/lapa-ocr-lora"
    DEFAULT_PROMPT = "Transcribe Ukrainian text literally. Output only the text, no preamble."
    DEFAULT_MAX_NEW_TOKENS = 128

    def __init__(self):
        self.model = None
        self.processor = None
        self._config_widget: Optional[QWidget] = None

        self._base_model_edit: Optional[QLineEdit] = None
        self._adapter_edit: Optional[QLineEdit] = None
        self._quant_combo: Optional[QComboBox] = None
        self._attn_combo: Optional[QComboBox] = None
        self._max_tokens_spin: Optional[QSpinBox] = None
        self._prompt_edit: Optional[QTextEdit] = None
        self._device_combo: Optional[QComboBox] = None

    def get_name(self) -> str:
        return "LapaOCR"

    def get_aliases(self) -> List[str]:
        return ["lapa-ocr", "lapa-ocr-lora"]

    def get_description(self) -> str:
        return "Lapa v0.1.2 + lapa-ocr-lora (Ukrainian line/crop OCR)"

    def is_available(self) -> bool:
        return LAPA_AVAILABLE

    def get_unavailable_reason(self) -> str:
        if not LAPA_MISSING_DEPS:
            return ""
        deps = ", ".join(LAPA_MISSING_DEPS)
        return (
            f"Missing dependencies: {deps}. "
            "Install: pip install -U torch transformers peft"
        )

    def get_config_widget(self) -> QWidget:
        if self._config_widget is not None:
            return self._config_widget

        widget = QWidget()
        layout = QVBoxLayout()

        base_layout = QHBoxLayout()
        base_layout.addWidget(QLabel("Base Model:"))
        self._base_model_edit = QLineEdit(self.DEFAULT_BASE_MODEL)
        base_layout.addWidget(self._base_model_edit)
        layout.addLayout(base_layout)

        adapter_layout = QHBoxLayout()
        adapter_layout.addWidget(QLabel("Adapter:"))
        self._adapter_edit = QLineEdit(self.DEFAULT_ADAPTER)
        adapter_layout.addWidget(self._adapter_edit)
        layout.addLayout(adapter_layout)

        quant_layout = QHBoxLayout()
        quant_layout.addWidget(QLabel("Quantization:"))
        self._quant_combo = QComboBox()
        self._quant_combo.addItems(["none", "8bit", "4bit"])
        quant_layout.addWidget(self._quant_combo)
        layout.addLayout(quant_layout)

        attn_layout = QHBoxLayout()
        attn_layout.addWidget(QLabel("Attention:"))
        self._attn_combo = QComboBox()
        self._attn_combo.addItems(["auto", "sdpa", "eager"])
        attn_layout.addWidget(self._attn_combo)
        layout.addLayout(attn_layout)

        tokens_layout = QHBoxLayout()
        tokens_layout.addWidget(QLabel("Max New Tokens:"))
        self._max_tokens_spin = QSpinBox()
        self._max_tokens_spin.setRange(64, 1024)
        self._max_tokens_spin.setSingleStep(32)
        self._max_tokens_spin.setValue(self.DEFAULT_MAX_NEW_TOKENS)
        tokens_layout.addWidget(self._max_tokens_spin)
        layout.addLayout(tokens_layout)

        prompt_label = QLabel("Prompt:")
        layout.addWidget(prompt_label)
        self._prompt_edit = QTextEdit()
        self._prompt_edit.setPlainText(self.DEFAULT_PROMPT)
        self._prompt_edit.setMaximumHeight(90)
        layout.addWidget(self._prompt_edit)

        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cuda:0", "cuda:1", "cpu"])
        device_layout.addWidget(self._device_combo)
        layout.addLayout(device_layout)

        layout.addStretch()
        widget.setLayout(layout)
        self._config_widget = widget
        return widget

    def get_config(self) -> Dict[str, Any]:
        if self._config_widget is None:
            return {}

        return {
            "base_model": self._base_model_edit.text().strip() or self.DEFAULT_BASE_MODEL,
            "adapter": self._adapter_edit.text().strip() or self.DEFAULT_ADAPTER,
            "quantization": self._quant_combo.currentText(),
            "attn_implementation": self._attn_combo.currentText(),
            "max_new_tokens": self._max_tokens_spin.value(),
            "prompt": self._prompt_edit.toPlainText().strip() or self.DEFAULT_PROMPT,
            "device": self._device_combo.currentText(),
        }

    def set_config(self, config: Dict[str, Any]):
        if self._config_widget is None:
            return

        self._base_model_edit.setText(config.get("base_model", self.DEFAULT_BASE_MODEL))
        self._adapter_edit.setText(config.get("adapter", self.DEFAULT_ADAPTER))

        quant = str(config.get("quantization", "none"))
        q_idx = self._quant_combo.findText(quant)
        if q_idx >= 0:
            self._quant_combo.setCurrentIndex(q_idx)

        attn = str(config.get("attn_implementation", "auto"))
        a_idx = self._attn_combo.findText(attn)
        if a_idx >= 0:
            self._attn_combo.setCurrentIndex(a_idx)

        self._max_tokens_spin.setValue(int(config.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS)))
        self._prompt_edit.setPlainText(config.get("prompt", self.DEFAULT_PROMPT))

        device = str(config.get("device", "auto"))
        d_idx = self._device_combo.findText(device)
        if d_idx >= 0:
            self._device_combo.setCurrentIndex(d_idx)

    def load_model(self, config: Dict[str, Any]) -> bool:
        if not LAPA_AVAILABLE:
            print(self.get_unavailable_reason())
            return False

        base = None
        try:
            if self.model is not None:
                self.unload_model()

            base_model = (
                config.get("base_model")
                or config.get("model_id")
                or self.DEFAULT_BASE_MODEL
            )
            adapter = config.get("adapter") or self.DEFAULT_ADAPTER
            quantization = str(config.get("quantization", "none")).lower()
            requested_device = str(config.get("device", "auto"))

            attn_impl = config.get("attn_implementation", "auto")
            if attn_impl == "auto":
                attn_impl = "eager" if quantization in ("4bit", "8bit") else "sdpa"

            model_kwargs: Dict[str, Any] = {
                "attn_implementation": attn_impl,
            }

            if quantization in ("4bit", "8bit"):
                if requested_device == "cpu":
                    print("LapaOCR quantized mode requires CUDA. Set device to auto/cuda.")
                    return False
                if not BNB_AVAILABLE:
                    print("bitsandbytes is required for 4bit/8bit quantization.")
                    return False

                if quantization == "4bit":
                    bnb_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        llm_int8_skip_modules=[
                            "vision_tower",
                            "multi_modal_projector",
                            "lm_head",
                            "embed_tokens",
                        ],
                    )
                else:
                    bnb_cfg = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_skip_modules=[
                            "vision_tower",
                            "multi_modal_projector",
                            "lm_head",
                            "embed_tokens",
                        ],
                    )

                model_kwargs["quantization_config"] = bnb_cfg
                model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["device_map"] = "auto"
            else:
                if requested_device == "auto" or (requested_device != "cpu" and torch.cuda.is_available()):
                    model_kwargs["torch_dtype"] = torch.bfloat16
                    model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["torch_dtype"] = torch.float32

            print(f"Loading Lapa base model: {base_model}")
            base = AutoModelForImageTextToText.from_pretrained(base_model, **model_kwargs)

            if quantization == "none" and "device_map" not in model_kwargs and requested_device not in ("auto", "cpu"):
                base = base.to(requested_device)

            print(f"Loading Lapa adapter: {adapter}")
            self.model = PeftModel.from_pretrained(base, adapter).eval()
            self.processor = AutoProcessor.from_pretrained(base_model)

            # Some load paths lose generation tokens; restore safe defaults.
            try:
                self.model.generation_config.eos_token_id = [1, 106]
                self.model.generation_config.pad_token_id = 0
            except Exception:
                pass

            return True

        except Exception as e:
            print(f"Error loading LapaOCR model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.processor = None
            return False

        finally:
            # A failed from_pretrained() can leave partially-materialized
            # tensors pinned on the GPU for as long as the exception's
            # traceback is alive; `e` is cleared by Python before this
            # `finally` runs, so this is the earliest point where dropping
            # `base` and clearing the CUDA cache actually frees that memory.
            base = None
            if torch is not None and torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()

    def unload_model(self):
        if self.model is None:
            return

        try:
            if torch is not None and torch.cuda.is_available():
                try:
                    self.model.cpu()
                except Exception:
                    pass
                torch.cuda.empty_cache()
        except Exception:
            pass

        self.model = None
        self.processor = None

    def is_model_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def requires_line_segmentation(self) -> bool:
        # Model card targets text crops/regions, not full-page layout parsing.
        return True

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        if not self.is_model_loaded():
            return TranscriptionResult(text="[Model not loaded]", confidence=0.0)

        try:
            from PIL import Image

            cfg = config or {}
            prompt = (
                cfg.get("prompt")
                or cfg.get("custom_prompt")
                or self.DEFAULT_PROMPT
            )
            max_new_tokens = int(cfg.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS))
            max_time_raw = cfg.get("max_time_s", cfg.get("max_time"))
            max_time_s = None
            if max_time_raw is not None:
                try:
                    parsed = float(max_time_raw)
                    if parsed > 0:
                        max_time_s = parsed
                except (TypeError, ValueError):
                    max_time_s = None

            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )

            try:
                target_device = self.model.device
            except Exception:
                target_device = next(self.model.parameters()).device
            inputs = inputs.to(target_device)

            eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)
            pad_token_id = getattr(self.model.generation_config, "pad_token_id", None)

            with torch.inference_mode():
                generate_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "num_beams": 1,
                    "eos_token_id": eos_token_id,
                    "pad_token_id": pad_token_id,
                }
                if max_time_s is not None:
                    generate_kwargs["max_time"] = max_time_s

                generated = self.model.generate(
                    **inputs,
                    **generate_kwargs,
                )

            prefix_len = inputs["input_ids"].shape[1]
            text = self.processor.batch_decode(
                generated[:, prefix_len:],
                skip_special_tokens=True,
            )[0].strip()

            return TranscriptionResult(
                text=text,
                confidence=1.0,
                metadata={
                    "model": "LapaOCR",
                    "base_model": cfg.get("base_model", self.DEFAULT_BASE_MODEL),
                    "adapter": cfg.get("adapter", self.DEFAULT_ADAPTER),
                },
            )

        except Exception as e:
            return TranscriptionResult(text=f"[Error: {e}]", confidence=0.0)

    def transcribe_lines(self, images: List[np.ndarray], config: Optional[Dict[str, Any]] = None) -> List[TranscriptionResult]:
        return [self.transcribe_line(img, config) for img in images]

    def supports_batch(self) -> bool:
        return False

    def get_capabilities(self) -> Dict[str, bool]:
        return {
            "batch_processing": False,
            "confidence_scores": False,
            "beam_search": False,
            "language_model": True,
            "preprocessing": True,
        }
