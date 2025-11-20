"""
Commercial VLM/LLM API inference for manuscript transcription.

Supports:
- OpenAI GPT-4 Vision / GPT-4o
- Google Gemini Pro Vision / Gemini Flash
- Anthropic Claude 3 (Opus, Sonnet, Haiku)

Usage:
    # OpenAI
    api = OpenAIInference(api_key="sk-...")
    text = api.transcribe(image)

    # Gemini
    api = GeminiInference(api_key="...")
    text = api.transcribe(image)

    # Claude
    api = ClaudeInference(api_key="sk-ant-...")
    text = api.transcribe(image)
"""

import base64
import io
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image

# API clients (install with: pip install openai google-generativeai anthropic)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


class BaseAPIInference(ABC):
    """Base class for commercial API inference."""

    def __init__(self, api_key: str, default_prompt: Optional[str] = None):
        """
        Initialize API client.

        Args:
            api_key: API key for the service
            default_prompt: Default transcription prompt
        """
        self.api_key = api_key
        self.default_prompt = default_prompt or self._get_default_prompt()

    @abstractmethod
    def _get_default_prompt(self) -> str:
        """Get default transcription prompt."""
        pass

    @abstractmethod
    def transcribe(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe a manuscript line image.

        Args:
            image: PIL Image
            prompt: Custom prompt (uses default if None)
            **kwargs: Provider-specific parameters

        Returns:
            Transcribed text
        """
        pass

    @staticmethod
    def encode_image_base64(image: Image.Image, format: str = "PNG") -> str:
        """
        Encode PIL Image to base64 string.

        Args:
            image: PIL Image
            format: Image format (PNG, JPEG, etc.)

        Returns:
            Base64-encoded image string
        """
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def resize_image_if_needed(
        image: Image.Image,
        max_dimension: int = 2048
    ) -> Image.Image:
        """
        Resize image if larger than max dimension while preserving aspect ratio.

        Args:
            image: PIL Image
            max_dimension: Maximum width or height

        Returns:
            Resized image (or original if already small enough)
        """
        width, height = image.size

        if width <= max_dimension and height <= max_dimension:
            return image

        # Calculate new size preserving aspect ratio
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


class OpenAIInference(BaseAPIInference):
    """OpenAI GPT-4 Vision / GPT-4o inference."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",  # gpt-4o, gpt-4-vision-preview, gpt-4-turbo
        default_prompt: Optional[str] = None
    ):
        """
        Initialize OpenAI inference.

        Args:
            api_key: OpenAI API key
            model: Model name
            default_prompt: Default transcription prompt
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")

        super().__init__(api_key, default_prompt)
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def _get_default_prompt(self) -> str:
        return (
            "Transcribe all handwritten text in this manuscript image. "
            "Preserve the original language (Cyrillic, Latin, etc.) and layout. "
            "Output only the transcribed text without any additional commentary."
        )

    def transcribe(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_tokens: int = 500,
    temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Transcribe with OpenAI GPT-4 Vision.

        Args:
            image: PIL Image
            prompt: Custom prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (web default ~1.0). Lower (0-0.3) = deterministic; higher = more variation.
            **kwargs: Additional OpenAI parameters

        Returns:
            Transcribed text
        """
        prompt = prompt or self.default_prompt

        # Resize if needed (GPT-4V supports up to 2048x2048)
        image = self.resize_image_if_needed(image, max_dimension=2048)

        # Encode image
        base64_image = self.encode_image_base64(image, format="PNG")

        # API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        return response.choices[0].message.content.strip()


class GeminiInference(BaseAPIInference):
    """Google Gemini Pro Vision / Flash inference."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",  # gemini-2.0-flash, gemini-1.5-pro-002
        default_prompt: Optional[str] = None
    ):
        """
        Initialize Gemini inference.

        Args:
            api_key: Google API key
            model: Model name
            default_prompt: Default transcription prompt
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library not installed. "
                              "Install with: pip install google-generativeai")

        super().__init__(api_key, default_prompt)
        self.model_name = model

        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

        # Detect availability of safety classes (version-dependent)
        try:
            from google.generativeai.types import SafetySetting, HarmCategory, HarmBlockThreshold  # type: ignore
            self._safety_classes_available = True
        except Exception:
            # Newer versions expose only dict helpers
            self._safety_classes_available = False

    def _get_default_prompt(self) -> str:
        return (
            "Transcribe all handwritten text in this manuscript image. "
            "Preserve the original language (Cyrillic, Latin, etc.) and layout. "
            "Output only the transcribed text without any additional commentary."
        )
    
    def _maybe_continue(
        self,
        current_text: str,
        original_prompt: str,
        image: Image.Image,
        generation_config,
        safety_settings,
        auto_continue: bool,
        max_auto_continuations: int,
        continuation_min_new_chars: int,
        verbose_block_logging: bool,
    ) -> str:
        """Optionally perform continuation calls to extend transcription.

        Heuristic: if auto_continue is enabled, we ask for continuation until no new text
        is added or we hit max_auto_continuations. We guard against the model re-sending
        previous text by diffing appended length.
        """
        if not auto_continue:
            return current_text

        accumulated = current_text
        last_len = len(accumulated)
        for pass_idx in range(1, max_auto_continuations + 1):
            continuation_prompt = (
                f"{original_prompt}\n\nPartial transcription so far (DO NOT repeat it):\n"  # original base
                f"{accumulated}\n\nContinue transcribing remaining, previously UNTRANSCRIBED text. "
                "Output ONLY the new continuation without repeating prior characters."  # instruction
            )
            try:
                cont_resp = self.model.generate_content([
                    continuation_prompt,
                    image,
                ], generation_config=generation_config, safety_settings=safety_settings)
            except Exception as e:
                if verbose_block_logging:
                    print(f"❌ Continuation attempt {pass_idx} failed: {e}")
                break

            new_chunk = ""
            if hasattr(cont_resp, 'candidates') and cont_resp.candidates:
                cand = cont_resp.candidates[0]
                if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                    parts_text = [p.text for p in cand.content.parts if hasattr(p, 'text') and p.text]
                    new_chunk = ''.join(parts_text).strip()

            if not new_chunk:
                if verbose_block_logging:
                    print(f"ℹ️ Continuation attempt {pass_idx} produced no new text; stopping.")
                break

            # Remove any accidental repetition by trimming existing prefix
            if accumulated and new_chunk.startswith(accumulated[:200]):  # crude repetition guard
                # Attempt to find overlap
                overlap_pos = new_chunk.find(accumulated[-50:])
                if overlap_pos > 0:
                    new_chunk = new_chunk[overlap_pos + len(accumulated[-50:]):]

            # Only append if sufficiently new
            delta = len(new_chunk)
            if delta < continuation_min_new_chars:
                if verbose_block_logging:
                    print(f"ℹ️ Continuation attempt {pass_idx} yielded only {delta} chars (<{continuation_min_new_chars}); stopping.")
                break

            accumulated += ("\n" if not accumulated.endswith('\n') else "") + new_chunk
            new_total = len(accumulated)
            if verbose_block_logging:
                print(f"➕ Continuation {pass_idx} appended {delta} chars (total {new_total})")

            # If growth is minimal relative to previous length, stop
            if new_total - last_len < continuation_min_new_chars:
                if verbose_block_logging:
                    print("ℹ️ Growth below threshold after append; stopping continuation loop.")
                break
            last_len = new_total

        return accumulated

    def transcribe(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        auto_retry_on_block: bool = True,
        safety_relax: bool = True,
        verbose_block_logging: bool = True,
        thinking_mode: Optional[str] = None,
        fast_direct: bool = False,
        fast_direct_early_exit: bool = True,
        auto_continue: bool = False,
        max_auto_continuations: int = 2,
        continuation_min_new_chars: int = 50,
        reasoning_fallback_threshold: float = 0.6,
        record_stats_csv: Optional[str] = "gemini_runs.csv",
        **kwargs
    ) -> str:
        """
        Transcribe with Google Gemini.

        Args:
            image: PIL Image
            prompt: Custom prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_output_tokens: Maximum tokens to generate
            thinking_mode: Reasoning mode - "low", "high", or None (default: None for preview models uses low)
            **kwargs: Additional Gemini parameters

        Returns:
            Transcribed text
        """
        prompt = prompt or self.default_prompt

        # Fast direct mode augments prompt to discourage internal reasoning
        if fast_direct:
            prompt = (
                prompt
                + "\n\nReturn the transcription immediately without extended internal reasoning. "
                  "Do NOT spend tokens thinking; output only the raw transcribed text now."
            )
            if verbose_block_logging:
                print("⚡ Fast-direct mode enabled: prompting for immediate output")

        # Resize if needed (Gemini supports up to 3072x3072)
        image = self.resize_image_if_needed(image, max_dimension=3072)

        # Prepare generation config (remove unsupported response_modalities)
        gen_config_params = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }

        is_preview_model = any(x in self.model_name.lower() for x in ['preview', 'exp', 'experimental'])

        # Simulate thinking modes via token/temperature adjustments (API version lacks explicit reasoning switch)
        if thinking_mode:
            mode_str = thinking_mode.lower()
            if mode_str == "low":
                if verbose_block_logging:
                    print("🧠 Using LOW thinking mode (direct decoding)")
                # Keep deterministic low-temp unless user overrides
                gen_config_params["temperature"] = temperature
            elif mode_str == "high":
                if verbose_block_logging:
                    print("🧠 Using HIGH thinking mode (more tokens & slight exploration)")
                # Increase token budget and mild temperature for more exploration
                if max_output_tokens < 8192:
                    gen_config_params["max_output_tokens"] = 8192
                if temperature < 0.15:
                    gen_config_params["temperature"] = 0.15
        elif is_preview_model:
            # Default to LOW style for preview to avoid wasted internal reasoning tokens
            if verbose_block_logging:
                print("🧠 Defaulting to LOW thinking mode for preview model (simulated)")

        # Merge any additional kwargs after adjustments
        gen_config_params.update(kwargs)

        # Generate
        generation_config = genai.GenerationConfig(**gen_config_params)

        # For preview/experimental models, use relaxed safety from the start and higher token limit
        initial_safety = None
        
        if safety_relax and is_preview_model:
            if verbose_block_logging:
                print(f"🔓 Using relaxed safety settings for preview model: {self.model_name}")
            from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
            initial_safety = [
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            ]
            
            # Preview models may use tokens for "thinking" - increase limit significantly
            if max_output_tokens < 4096:
                if verbose_block_logging:
                    print(f"   Increasing max_output_tokens from {max_output_tokens} to 4096 for preview model")
                max_output_tokens = 4096
            elif verbose_block_logging:
                print(f"   Using max_output_tokens={max_output_tokens} (from config)")

        # Attempt 1: generation (optionally streaming for fast_direct)
        response = None
        collected_stream_text: list[str] = []
        if fast_direct:
            try:
                stream = self.model.generate_content(
                    [prompt, image],
                    generation_config=generation_config,
                    safety_settings=initial_safety,
                    stream=True,
                )
                reasoning_fallback_triggered = False
                first_usage_meta = None
                for event in stream:
                    # Token usage introspection (if available)
                    if verbose_block_logging and hasattr(event, 'usage_metadata'):
                        meta = event.usage_metadata
                        try:
                            prompt_tok = getattr(meta,'prompt_token_count',None)
                            cand_tok = getattr(meta,'candidates_token_count',None)
                            total_tok = getattr(meta,'total_token_count',None)
                            print(f"[tokens] prompt={prompt_tok} candidates={cand_tok} total={total_tok}")
                            if first_usage_meta is None:
                                first_usage_meta = (prompt_tok, cand_tok, total_tok)
                            # Early reasoning fallback: if no emitted text yet and internal reasoning exceeded threshold
                            if not collected_stream_text and prompt_tok is not None and total_tok is not None:
                                internal_tok = max(0, (total_tok or 0) - (prompt_tok or 0) - (cand_tok or 0))
                                budget = getattr(generation_config, 'max_output_tokens', max_output_tokens)
                                if budget and internal_tok >= reasoning_fallback_threshold * budget:
                                    if verbose_block_logging:
                                        pct = internal_tok / budget
                                        print(f"⏱️ Early reasoning fallback triggered: internal={internal_tok} ({pct:.0%} of budget) with no output; aborting stream.")
                                    reasoning_fallback_triggered = True
                                    break
                        except Exception:
                            pass
                    elif verbose_block_logging and hasattr(event, 'candidates') and event.candidates:
                        # Approximate progress by count of events
                        print(f"[stream] event candidates={len(event.candidates)} parts={[len(getattr(c.content,'parts',[])) for c in event.candidates if hasattr(c,'content')]}")
                    if hasattr(event, 'candidates') and event.candidates:
                        for cand in event.candidates:
                            if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                                for part in cand.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        collected_stream_text.append(part.text)
                                        # Early exit once first non-empty aggregated text if enabled
                                        if fast_direct_early_exit and ''.join(collected_stream_text).strip():
                                            result = ''.join(collected_stream_text).strip()
                                            if verbose_block_logging:
                                                print(f"✅ Early streamed output ({len(result)} chars) [early-exit]")
                                            if record_stats_csv:
                                                try:
                                                    from datetime import datetime
                                                    with open(record_stats_csv,'a') as f:
                                                        pt, ct, tt = first_usage_meta if first_usage_meta else (None,None,None)
                                                        internal_tok = (tt - pt - ct) if (pt is not None and tt is not None and ct is not None) else None
                                                        f.write(f"{datetime.utcnow().isoformat()},{self.model_name},{thinking_mode or 'default'},stream_early_exit,{pt},{ct},{tt},{internal_tok},{len(result)}\n")
                                                except Exception as e:
                                                    if verbose_block_logging:
                                                        print(f"⚠️ Stats logging failed: {e}")
                                            return self._maybe_continue(result, prompt, image, generation_config, initial_safety, auto_continue, max_auto_continuations, continuation_min_new_chars, verbose_block_logging)
                # If we reach here, streaming produced no immediate text or was aborted; fall back to non-stream call
                if verbose_block_logging:
                    if reasoning_fallback_triggered:
                        print("⚠️ Streaming aborted due to excessive internal reasoning; switching to standard generation.")
                    elif collected_stream_text:
                        print(f"ℹ️ Streaming completed. Collected {len(collected_stream_text)} fragments (total chars {len(''.join(collected_stream_text))}).")
                    else:
                        print("⚠️ Streaming produced no early text; falling back to standard generation")
                if collected_stream_text and not fast_direct_early_exit:
                    full_stream_text = ''.join(collected_stream_text).strip()
                    if full_stream_text:
                        if verbose_block_logging:
                            print(f"✅ Stream finished ({len(full_stream_text)} chars) without early exit")
                        if record_stats_csv:
                            try:
                                from datetime import datetime
                                with open(record_stats_csv,'a') as f:
                                    pt, ct, tt = first_usage_meta if first_usage_meta else (None,None,None)
                                    internal_tok = (tt - pt - ct) if (pt is not None and tt is not None and ct is not None) else None
                                    f.write(f"{datetime.utcnow().isoformat()},{self.model_name},{thinking_mode or 'default'},stream_full,{pt},{ct},{tt},{internal_tok},{len(full_stream_text)}\n")
                            except Exception as e:
                                if verbose_block_logging:
                                    print(f"⚠️ Stats logging failed: {e}")
                        return self._maybe_continue(full_stream_text, prompt, image, generation_config, initial_safety, auto_continue, max_auto_continuations, continuation_min_new_chars, verbose_block_logging)
            except Exception as e:
                if verbose_block_logging:
                    print(f"⚠️ Streaming mode failed: {type(e).__name__}: {e}; reverting to standard generation")

        # Standard (non-stream) generation path
        try:
            response = self.model.generate_content(
                [prompt, image],
                generation_config=generation_config,
                safety_settings=initial_safety,
            )
            if verbose_block_logging and hasattr(response, 'candidates') and response.candidates:
                finish_reason = getattr(response.candidates[0], 'finish_reason', None)
                if finish_reason and finish_reason != 'STOP':
                    print(f"⚠️  Initial attempt finish_reason: {finish_reason}")
        except Exception as e:
            if verbose_block_logging:
                print(f"⚠️  Initial attempt raised exception: {type(e).__name__}: {e}")
            if auto_retry_on_block and safety_relax:
                response = None
            else:
                raise

        # Handle response with proper error checking
        # Special case: if finish_reason is MAX_TOKENS (2), check if we have valid content
        if response is not None and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason == 2:  # MAX_TOKENS
                if verbose_block_logging:
                    print(f"⚠️  Hit MAX_TOKENS limit (finish_reason=2)")
                
                # Check if we actually got any output parts
                has_output = False
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    try:
                        text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                        if text_parts:
                            result = ''.join(text_parts).strip()
                            if result:
                                if verbose_block_logging:
                                    print(f"✓ Extracted partial response ({len(result)} chars)")
                                return self._maybe_continue(result, prompt, image, generation_config, initial_safety, auto_continue, max_auto_continuations, continuation_min_new_chars, verbose_block_logging)
                            has_output = True
                    except Exception as e:
                        if verbose_block_logging:
                            print(f"   Error extracting parts: {e}")
                
                # No output generated - model consumed all tokens for "thinking"
                if not has_output:
                    if verbose_block_logging:
                        print(f"⚠️  No output parts generated - model used all tokens for internal processing")
                        print(f"   Attempting automatic fallback with HIGH thinking mode and expanded token budget...")

                    # Automatic fallback attempt: escalate thinking mode and token budget
                    # Cap fallback at 8192 to avoid excessive costs (was: max(8192, max_output_tokens * 2))
                    try:
                        fallback_tokens = 8192
                        if verbose_block_logging:
                            print(f"   Fallback max_output_tokens={fallback_tokens} (capped for cost control)")
                        fallback_config = genai.GenerationConfig(
                            temperature=generation_config.temperature if hasattr(generation_config, 'temperature') else 1.0,
                            max_output_tokens=fallback_tokens,
                        )
                        fallback_response = self.model.generate_content(
                            [prompt, image],
                            generation_config=fallback_config,
                            safety_settings=initial_safety
                        )
                        if hasattr(fallback_response, 'candidates') and fallback_response.candidates:
                            fb_candidate = fallback_response.candidates[0]
                            fb_parts = []
                            if hasattr(fb_candidate, 'content') and hasattr(fb_candidate.content, 'parts'):
                                fb_parts = [part.text for part in fb_candidate.content.parts if hasattr(part, 'text')]
                            if fb_parts:
                                fb_text = ''.join(fb_parts).strip()
                                if fb_text:
                                    if verbose_block_logging:
                                        print(f"✅ Fallback succeeded ({len(fb_text)} chars)")
                                    if record_stats_csv:
                                        try:
                                            from datetime import datetime
                                            with open(record_stats_csv,'a') as f:
                                                f.write(f"{datetime.utcnow().isoformat()},{self.model_name},{thinking_mode or 'default'},fallback_success,,,,,{len(fb_text)}\n")
                                        except Exception as e:
                                            if verbose_block_logging:
                                                print(f"⚠️ Stats logging failed: {e}")
                                    return self._maybe_continue(fb_text, prompt, image, generation_config, initial_safety, auto_continue, max_auto_continuations, continuation_min_new_chars, verbose_block_logging)
                            if verbose_block_logging:
                                print("❌ Fallback also produced no text parts")
                    except Exception as fb_e:
                        if verbose_block_logging:
                            print(f"❌ Fallback attempt failed: {fb_e}")

                    if verbose_block_logging:
                        print(f"   Giving up after fallback. Recommend switching to stable model (e.g., gemini-2.0-flash) or lowering temperature.")
                    raise ValueError(
                        f"Model '{self.model_name}' produced no text after primary + fallback attempts (token budgets {max_output_tokens} & {fallback_tokens}). Try a stable model or different settings."
                    )
        
        if response is None or not response.parts:
            # If blocked, collect detailed diagnostics
            block_details = []
            prompt_feedback = getattr(response, 'prompt_feedback', None) if response else None

            if prompt_feedback:
                # Newer Gemini responses include safety ratings inside prompt_feedback
                ratings = getattr(prompt_feedback, 'safety_ratings', [])
                if ratings and verbose_block_logging:
                    for r in ratings:
                        cat = getattr(r, 'category', 'UNKNOWN_CATEGORY')
                        prob = getattr(r, 'probability', 'UNKNOWN_PROB')
                        blk = getattr(r, 'blocked', False)
                        block_details.append(f"{cat} prob={prob} blocked={blk}")
                block_msg = f"Content generation blocked. Feedback: {prompt_feedback}. "
            else:
                block_msg = "Content generation blocked (no prompt_feedback available). "

            # Auto-retry strategy: relax safety thresholds if requested
            if auto_retry_on_block and safety_relax:
                if verbose_block_logging:
                    model_name = getattr(self.model, '_model_name', 'unknown')
                    print(f"⚠️  Content blocked on model '{model_name}'")
                    print("   Attempting retry with BLOCK_NONE (all safety filters disabled)...")
                try:
                    if self._safety_classes_available:
                        from google.generativeai.types import SafetySetting, HarmCategory, HarmBlockThreshold  # type: ignore
                        relaxed_safety = [
                            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
                            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
                            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
                            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
                        ]
                    else:
                        # Fallback: use enum objects inside dicts (supported by 0.8.x)
                        from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
                        relaxed_safety = [
                            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
                            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                        ]
                    retry_prompt = (
                        prompt + "\n\nIMPORTANT: The image contains historical handwritten text for transcription only. "
                        "It does not contain harmful, personal, or sensitive content. Provide a literal transcription." 
                    )
                    retry_config = genai.GenerationConfig(
                        temperature=0.0,  # force deterministic on retry
                        max_output_tokens=max_output_tokens,
                        **{k: v for k, v in kwargs.items() if k not in ['safety_settings']}
                    )
                    retry_response = self.model.generate_content(
                        [retry_prompt, image],
                        generation_config=retry_config,
                        safety_settings=relaxed_safety
                    )
                    
                    # Debug: Show finish reason if available
                    if verbose_block_logging and hasattr(retry_response, 'candidates') and retry_response.candidates:
                        finish_reason = getattr(retry_response.candidates[0], 'finish_reason', None)
                        print(f"   Retry finish_reason: {finish_reason}")
                    
                    if retry_response.parts:
                        try:
                            result = retry_response.text.strip()
                            if verbose_block_logging:
                                print("✓ Retry successful with relaxed safety settings!")
                            return result
                        except Exception as text_e:
                            if verbose_block_logging:
                                print(f"   Warning: Had parts but couldn't extract text: {text_e}")
                            pass  # Fall through to error handling below
                    # If still blocked, append retry diagnostics
                    if verbose_block_logging:
                        print("❌ Retry also blocked - no response parts generated")
                    if hasattr(retry_response, 'prompt_feedback') and verbose_block_logging:
                        pf = retry_response.prompt_feedback
                        ratings2 = getattr(pf, 'safety_ratings', [])
                        for r in ratings2:
                            cat = getattr(r, 'category', 'UNKNOWN_CATEGORY')
                            prob = getattr(r, 'probability', 'UNKNOWN_PROB')
                            blk = getattr(r, 'blocked', False)
                            block_details.append(f"(retry) {cat} prob={prob} blocked={blk}")
                except Exception as retry_e:
                    if verbose_block_logging:
                        print(f"❌ Retry exception: {retry_e}")
                    block_details.append(f"Retry attempt failed: {retry_e}")

            detail_str = " | ".join(block_details) if block_details else "(no detailed safety ratings)"
            raise ValueError(block_msg + detail_str)

        # Extract text from response
        try:
            result_text = response.text.strip()
            if record_stats_csv:
                try:
                    from datetime import datetime
                    with open(record_stats_csv,'a') as f:
                        f.write(f"{datetime.utcnow().isoformat()},{self.model_name},{thinking_mode or 'default'},final_success,,,,,{len(result_text)}\n")
                except Exception as e:
                    if verbose_block_logging:
                        print(f"⚠️ Stats logging failed: {e}")
            return result_text
        except ValueError as e:
            # Response might be blocked or incomplete
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    raise ValueError(f"Content generation issue: {candidate.finish_reason}. "
                                   "This might be due to safety filters or content policy violations.")
            raise ValueError(f"Failed to extract text from response: {e}")


class ClaudeInference(BaseAPIInference):
    """Anthropic Claude 3 inference (Opus, Sonnet, Haiku)."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",  # claude-3-5-sonnet-20241022, claude-3-opus-20240229, claude-3-haiku-20240307
        default_prompt: Optional[str] = None
    ):
        """
        Initialize Claude inference.

        Args:
            api_key: Anthropic API key
            model: Model name
            default_prompt: Default transcription prompt
        """
        if not CLAUDE_AVAILABLE:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")

        super().__init__(api_key, default_prompt)
        self.model = model
        self.client = Anthropic(api_key=api_key)

    def _get_default_prompt(self) -> str:
        return (
            "Transcribe all handwritten text in this manuscript image. "
            "Preserve the original language (Cyrillic, Latin, etc.) and layout. "
            "Output only the transcribed text without any additional commentary."
        )

    def transcribe(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Transcribe with Anthropic Claude.

        Args:
            image: PIL Image
            prompt: Custom prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            **kwargs: Additional Claude parameters

        Returns:
            Transcribed text
        """
        prompt = prompt or self.default_prompt

        # Resize if needed (Claude supports up to 1568px on longest side)
        image = self.resize_image_if_needed(image, max_dimension=1568)

        # Encode image
        base64_image = self.encode_image_base64(image, format="PNG")

        # API call
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            **kwargs
        )

        return response.content[0].text.strip()


# Model availability checks
def check_api_availability() -> Dict[str, bool]:
    """Check which API libraries are installed."""
    return {
        "openai": OPENAI_AVAILABLE,
        "gemini": GEMINI_AVAILABLE,
        "claude": CLAUDE_AVAILABLE,
    }


# Fallback API model lists (used if dynamic fetching fails)
OPENAI_MODELS_FALLBACK = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-2024-11-20",
    "chatgpt-4o-latest",
    "gpt-4-turbo",
    "gpt-4-vision-preview",
    "o1-preview",
    "o1-mini",
]

GEMINI_MODELS_FALLBACK = [
    # Free tier models (generally available)
    "gemini-1.5-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash-exp",
    # Paid/preview models (may require upgrade)
    "gemini-1.5-pro",
    "gemini-1.5-pro-002",
    "gemini-1.5-pro-exp-0827",
    # Experimental (may not be available to all users)
    "gemini-exp-1206",
    "gemini-exp-1121",
    # Gemini 3 preview models (latest, may have restrictions)
    "gemini-3-pro-preview",
]

CLAUDE_MODELS_FALLBACK = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


def fetch_openai_models(api_key: str = None) -> list:
    """
    Dynamically fetch available OpenAI models from API.

    Args:
        api_key: OpenAI API key (uses env var if not provided)

    Returns:
        List of vision-capable model IDs, or fallback list if fetch fails
    """
    if not OPENAI_AVAILABLE:
        return OPENAI_MODELS_FALLBACK

    try:
        import os
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return OPENAI_MODELS_FALLBACK

        client = OpenAI(api_key=api_key)
        models = client.models.list()

        # Filter for vision-capable models (GPT-4 family + o1)
        vision_models = []
        for model in models.data:
            model_id = model.id
            # Include GPT-4 vision models and o1 models
            if any(prefix in model_id for prefix in [
                "gpt-4o", "gpt-4-turbo", "gpt-4-vision",
                "chatgpt-4o", "o1-", "gpt-4.5"  # Include potential GPT-4.5
            ]):
                vision_models.append(model_id)

        # Sort with newest/best models first
        vision_models.sort(reverse=True)

        # Return dynamic list if we found models, otherwise fallback
        return vision_models if vision_models else OPENAI_MODELS_FALLBACK

    except Exception as e:
        print(f"[OpenAI] Could not fetch models dynamically: {e}")
        print(f"[OpenAI] Using fallback model list")
        return OPENAI_MODELS_FALLBACK


def fetch_gemini_models(api_key: str = None) -> list:
    """
    Dynamically fetch available Gemini models from API.

    Args:
        api_key: Google API key (uses env var if not provided)

    Returns:
        List of Gemini model IDs, or fallback list if fetch fails
    """
    if not GEMINI_AVAILABLE:
        return GEMINI_MODELS_FALLBACK

    try:
        import os
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return GEMINI_MODELS_FALLBACK

        genai.configure(api_key=api_key)

        # List all available models
        available_models = []
        for model in genai.list_models():
            # Filter for vision-capable models (have 'generateContent' method)
            if 'generateContent' in model.supported_generation_methods:
                # Extract short model name (e.g., "models/gemini-1.5-pro" -> "gemini-1.5-pro")
                model_id = model.name.replace("models/", "")
                available_models.append(model_id)

        # Sort with newest models first
        available_models.sort(reverse=True)

        # Return dynamic list if we found models, otherwise fallback
        return available_models if available_models else GEMINI_MODELS_FALLBACK

    except Exception as e:
        print(f"[Gemini] Could not fetch models dynamically: {e}")
        print(f"[Gemini] Using fallback model list")
        return GEMINI_MODELS_FALLBACK


# Initialize model lists (will be updated when API keys are provided)
OPENAI_MODELS = OPENAI_MODELS_FALLBACK.copy()
GEMINI_MODELS = GEMINI_MODELS_FALLBACK.copy()
CLAUDE_MODELS = CLAUDE_MODELS_FALLBACK.copy()  # Claude doesn't have dynamic listing API yet


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 4:
        print("Usage: python inference_commercial_api.py <provider> <api_key> <image_path>")
        print("Providers: openai, gemini, claude")
        sys.exit(1)

    provider = sys.argv[1].lower()
    api_key = sys.argv[2]
    image_path = sys.argv[3]

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Initialize appropriate inference client
    if provider == "openai":
        api = OpenAIInference(api_key)
    elif provider == "gemini":
        api = GeminiInference(api_key)
    elif provider == "claude":
        api = ClaudeInference(api_key)
    else:
        print(f"Unknown provider: {provider}")
        sys.exit(1)

    # Transcribe
    print(f"Transcribing with {provider}...")
    text = api.transcribe(image)
    print(f"\nResult: {text}")
