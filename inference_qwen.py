"""
Inference with fine-tuned Qwen3-VL models for Ukrainian manuscript transcription.

Usage:
    # Single image
    python inference_qwen.py --image line.jpg --checkpoint ./results-ukrainian-qwen/final_model

    # Batch inference
    python inference_qwen.py --image_dir ./test_images --checkpoint ./results-ukrainian-qwen/final_model
"""

import argparse
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class Qwen3VLInference:
    """Qwen3-VL inference wrapper for manuscript transcription."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
    ):
        """
        Initialize Qwen3-VL inference.

        Args:
            checkpoint_path: Path to fine-tuned model checkpoint
            device: Device to run inference on
            torch_dtype: Data type for model weights
        """
        self.device = device
        self.checkpoint_path = checkpoint_path

        print(f"Loading Qwen3-VL model from {checkpoint_path}...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)

        # Load model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )

        if device != "cuda":
            self.model = self.model.to(device)

        self.model.eval()

        print(f"Model loaded successfully on {device}")

    def transcribe(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,  # Greedy decoding for deterministic output
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """
        Transcribe a line image.

        Args:
            image: PIL Image of manuscript line
            prompt: Custom prompt (default: Russian transcription prompt)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Nucleus sampling threshold
            do_sample: Enable sampling (False = greedy decoding)

        Returns:
            Transcribed text
        """
        # Default prompt in Russian (can be customized)
        if prompt is None:
            prompt = "Транскрибуйте текст на этом изображении."
            # Alternatives:
            # Ukrainian: "Транскрибуйте текст на цьому зображенні."
            # English: "Transcribe the text in this image."

        # Build chat messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]
            }
        ]

        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0].strip()

    def transcribe_batch(
        self,
        images: list[Image.Image],
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        batch_size: int = 4,
    ) -> list[str]:
        """
        Transcribe a batch of images.

        Args:
            images: List of PIL Images
            prompt: Custom prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            batch_size: Batch size for inference

        Returns:
            List of transcribed texts
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            for image in batch:
                text = self.transcribe(
                    image,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                results.append(text)

        return results


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL inference for manuscript transcription")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned Qwen3-VL checkpoint")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to single image")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Path to directory of images")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for transcriptions (default: stdout)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom transcription prompt")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")

    args = parser.parse_args()

    # Validate inputs
    if args.image is None and args.image_dir is None:
        parser.error("Must provide either --image or --image_dir")

    # Initialize inference
    inference = Qwen3VLInference(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # Single image
    if args.image:
        print(f"\nTranscribing {args.image}...")
        image = Image.open(args.image).convert("RGB")
        text = inference.transcribe(
            image,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )

        print(f"\nResult: {text}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text + "\n")

    # Batch inference
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

        if not image_files:
            print(f"No images found in {image_dir}")
            return

        print(f"\nTranscribing {len(image_files)} images...")

        # Load images
        images = []
        for img_path in image_files:
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")

        # Transcribe
        texts = inference.transcribe_batch(
            images,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size
        )

        # Print results
        for img_path, text in zip(image_files, texts):
            print(f"{img_path.name}: {text}")

        # Save to file
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for img_path, text in zip(image_files, texts):
                    f.write(f"{img_path.name}\t{text}\n")

            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
