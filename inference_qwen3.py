"""
Qwen3 VLM Inference for Whole-Page OCR
No line segmentation needed - processes entire page images directly
"""

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from PIL import Image
import torch
from typing import Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PageTranscription:
    """Result from Qwen3 VLM page transcription."""
    text: str
    confidence: Optional[float] = None
    processing_time: Optional[float] = None


class Qwen3VLMInference:
    """
    Qwen3 VLM inference for whole-page OCR.

    Key differences from TrOCR:
    - No line segmentation needed
    - Processes entire page images
    - Can handle complex layouts
    - Supports multiple finetuned adapters
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-VL-8B-Instruct",
        adapter_model: Optional[str] = None,
        device: str = "auto",
        torch_dtype = torch.float16,
        max_memory: Optional[dict] = None,
        max_image_size: int = 1536
    ):
        """
        Initialize Qwen3 VLM inference.

        Args:
            base_model: Base Qwen3 VL model from HuggingFace
            adapter_model: Optional LoRA/PEFT adapter for finetuning
            device: Device placement ("auto", "cuda", "cpu")
            torch_dtype: Model precision (float16 recommended)
            max_memory: Memory limits per GPU (e.g., {0: "20GB", 1: "20GB"})
            max_image_size: Max dimension for image resizing
        """
        self.base_model = base_model
        self.adapter_model = adapter_model
        self.max_image_size = max_image_size
        self.device = device

        print(f"Loading Qwen3 VLM: {base_model}")
        if adapter_model:
            print(f"  with adapter: {adapter_model}")

        # Auto-detect GPU configuration
        if device == "auto" and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"  Detected {num_gpus} GPU(s)")

            if max_memory is None and num_gpus > 1:
                # Auto-configure memory for multi-GPU
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                per_gpu_memory = f"{int(gpu_memory * 0.9)}GB"  # Use 90% of available
                max_memory = {i: per_gpu_memory for i in range(num_gpus)}
                print(f"  Auto-configured: {per_gpu_memory} per GPU")

        # Load base model
        # IMPORTANT: Force single GPU to avoid multi-GPU generation hanging issue
        # Multi-GPU device_map="auto" causes generation to hang on Linux (inter-GPU communication issue)
        # Single GPU works perfectly (<4s inference time)
        effective_device_map = device
        if device == "auto":
            # Auto-detect: prefer single GPU if available
            if torch.cuda.is_available():
                effective_device_map = "cuda:0"
                print(f"  Using single GPU (cuda:0) to avoid multi-GPU hanging issue")
            else:
                effective_device_map = "cpu"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=effective_device_map,
            max_memory=max_memory,
            trust_remote_code=True
        )

        # Load adapter if provided
        if adapter_model:
            print(f"  Loading adapter: {adapter_model}")
            self.model = PeftModel.from_pretrained(self.model, adapter_model)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True
        )

        # Print device map
        if hasattr(self.model, 'hf_device_map'):
            print("\nModel device distribution:")
            device_summary = {}
            for layer, device_id in self.model.hf_device_map.items():
                device_summary[device_id] = device_summary.get(device_id, 0) + 1

            for device_id, count in sorted(device_summary.items()):
                print(f"  Device {device_id}: {count} layers")

        # Print GPU memory
        if torch.cuda.is_available():
            print("\nGPU Memory:")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.total_memory / 1e9:.2f} GB total")

        self.model.eval()
        print("Qwen3 VLM loaded successfully!\n")

    def cleanup(self):
        """
        Properly cleanup model from memory to prevent CUDA memory fragmentation.

        Call this before reloading a new model to avoid OOM errors.
        """
        print("Cleaning up Qwen3 VLM from memory...")

        # Move model to CPU first to free GPU memory
        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model.cpu()
            except Exception as e:
                print(f"Warning: Error moving model to CPU: {e}")

        # Delete model and processor references
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache on all GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        print("✓ Cleanup complete")

    def __del__(self):
        """Destructor to ensure cleanup when object is deleted."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destructor

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for Qwen3 VLM.

        Args:
            image: Input PIL Image

        Returns:
            Preprocessed image
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize if too large
        if max(image.size) > self.max_image_size:
            original_size = image.size
            image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
            print(f"Resized image: {original_size} → {image.size}")

        return image

    def transcribe_page(
        self,
        page_image: Image.Image,
        prompt: str = "Transcribe the text shown in this image.",
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        num_beams: int = 1,
        temperature: Optional[float] = None,
        return_confidence: bool = False
    ) -> PageTranscription:
        """
        Transcribe an entire page image.

        Args:
            page_image: PIL Image of the full page
            prompt: Instruction prompt for the model
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling (vs greedy)
            num_beams: Number of beams for beam search
            temperature: Sampling temperature (if do_sample=True)
            return_confidence: If True, estimate confidence scores

        Returns:
            PageTranscription with full page text and optional confidence
        """
        import time
        start_time = time.time()

        # Preprocess image
        image = self.preprocess_image(page_image)

        # Prepare message in chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        # Move to correct device
        if hasattr(self.model, 'device'):
            device = self.model.device
        elif hasattr(self.model, 'hf_device_map'):
            # Get first device from device map
            device = next(iter(set(self.model.hf_device_map.values())))
            if isinstance(device, int):
                device = torch.device(f"cuda:{device}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Clear GPU cache before generation
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

        # Generate transcription
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "repetition_penalty": 1.2,  # Penalize repetition (1.0 = no penalty, >1.0 = discourage)
            "no_repeat_ngram_size": 3,  # Prevent repeating 3-grams
            "early_stopping": True,  # Stop when EOS is generated
        }

        if do_sample and temperature is not None:
            generation_kwargs["temperature"] = temperature

        # Add output_scores for confidence estimation
        if return_confidence:
            generation_kwargs["output_scores"] = True
            generation_kwargs["return_dict_in_generate"] = True

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        # Extract generated IDs
        if return_confidence:
            generated_ids = outputs.sequences
        else:
            generated_ids = outputs

        # Decode output (remove input tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]

        transcription = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Calculate confidence if requested
        confidence = None
        if return_confidence and hasattr(outputs, 'scores') and outputs.scores:
            # Calculate average token probability
            import torch.nn.functional as F
            token_probs = []

            for step_idx, score_tensor in enumerate(outputs.scores):
                # Get probabilities for this generation step
                probs = F.softmax(score_tensor, dim=-1)

                # Get the actual generated token at this step
                if step_idx < len(generated_ids_trimmed[0]):
                    token_id = generated_ids_trimmed[0][step_idx]
                    token_prob = probs[0, token_id].item()
                    token_probs.append(token_prob)

            # Average confidence across all tokens
            if token_probs:
                confidence = sum(token_probs) / len(token_probs)

        processing_time = time.time() - start_time

        # Clean up GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

        return PageTranscription(
            text=transcription,
            confidence=confidence,
            processing_time=processing_time
        )

    def transcribe_batch(
        self,
        page_images: List[Image.Image],
        prompt: str = "Transcribe the text shown in this image.",
        **kwargs
    ) -> List[PageTranscription]:
        """
        Transcribe multiple pages.

        Args:
            page_images: List of PIL Images
            prompt: Instruction prompt
            **kwargs: Additional arguments for transcribe_page

        Returns:
            List of PageTranscription results
        """
        results = []
        for idx, image in enumerate(page_images):
            print(f"Processing page {idx+1}/{len(page_images)}...")
            result = self.transcribe_page(image, prompt=prompt, **kwargs)
            results.append(result)

        return results

    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {}

        usage = {}
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9

                usage[f"gpu_{i}"] = {
                    "allocated": f"{allocated:.2f} GB",
                    "reserved": f"{reserved:.2f} GB",
                    "total": f"{total:.2f} GB",
                    "utilization": f"{(allocated/total)*100:.1f}%"
                }

        return usage


# Available Qwen3 VLM Models
QWEN3_MODELS = {
    # Base models (no finetuning)
    "qwen3-vl-2b": {
        "base": "Qwen/Qwen3-VL-2B-Instruct",
        "adapter": None,
        "description": "Smallest Qwen3 VLM (2B params)",
        "vram": "4-6 GB",
        "speed": "Fast"
    },
    "qwen3-vl-8b": {
        "base": "Qwen/Qwen3-VL-8B-Instruct",
        "adapter": None,
        "description": "Medium Qwen3 VLM (8B params)",
        "vram": "12-16 GB",
        "speed": "Medium"
    },

    # Finetuned models (with adapters)
    "qwen3-vl-8b-old-church-slavonic": {
        "base": "Qwen/Qwen3-VL-8B-Instruct",
        "adapter": "wjbmattingly/Qwen3-VL-8B-old-church-slavonic",
        "description": "Finetuned for Old Church Slavonic handwriting",
        "vram": "12-16 GB",
        "speed": "Medium"
    },

    # Add more finetuned models here as they become available
    "qwen3-vl-8b-ukrainian": {
        "base": "Qwen/Qwen3-VL-8B-Instruct",
        "adapter": "./models/Qwen3-VL-8B-ukrainian/final_model",
        "description": "Finetuned for Ukrainian manuscripts (locally trained)",
        "vram": "12-16 GB",
        "speed": "Medium"
    },
}


def list_available_models():
    """Print available Qwen3 VLM models."""
    print("\nAvailable Qwen3 VLM Models:")
    print("=" * 80)
    for model_id, info in QWEN3_MODELS.items():
        print(f"\n{model_id}:")
        print(f"  Base: {info['base']}")
        if info['adapter']:
            print(f"  Adapter: {info['adapter']}")
        print(f"  Description: {info['description']}")
        print(f"  VRAM: {info['vram']}")
        print(f"  Speed: {info['speed']}")
    print("=" * 80)


# Example usage
if __name__ == "__main__":
    # List available models
    list_available_models()

    # Load model (with Old Church Slavonic adapter as example)
    model_config = QWEN3_MODELS["qwen3-vl-8b-old-church-slavonic"]

    vlm = Qwen3VLMInference(
        base_model=model_config["base"],
        adapter_model=model_config["adapter"],
        device="auto",
        max_image_size=1536
    )

    # Load test image
    test_image = Image.open("test_page.png")

    # Transcribe with different prompts
    prompts = [
        "Transcribe the text shown in this image.",
        "Transcribe all text from this historical document.",
        "Extract all handwritten text from this page, preserving the original language and formatting.",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 80)

        result = vlm.transcribe_page(
            test_image,
            prompt=prompt,
            max_new_tokens=2048
        )

        print(f"Transcription:\n{result.text}")
        print(f"\nProcessing time: {result.processing_time:.2f}s")

    # Show memory usage
    print("\nGPU Memory Usage:")
    for gpu, stats in vlm.get_memory_usage().items():
        print(f"  {gpu}: {stats}")
