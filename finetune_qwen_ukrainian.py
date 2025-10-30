"""
Fine-tune Qwen3-VL on Ukrainian/Cyrillic manuscripts for handwriting recognition.

This script adapts the Qwen3-VL-8B model to Ukrainian manuscript transcription
using LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning.

Hardware: Optimized for 2x RTX 4090 (48GB total VRAM)
Training time: ~2-3 hours for 5 epochs on 19K line images

Usage:
    # Single GPU
    python finetune_qwen_ukrainian.py --data_root ./data/ukrainian_train_aspect_ratio --epochs 5

    # Multi-GPU with torchrun (recommended for 2x4090)
    torchrun --nproc_per_node=2 finetune_qwen_ukrainian.py \
        --data_root ./data/ukrainian_train_aspect_ratio \
        --train_csv train.csv \
        --val_csv val.csv \
        --epochs 5

    # Resume from checkpoint
    python finetune_qwen_ukrainian.py --data_root ./data/ukrainian_train_aspect_ratio --resume ./models/Qwen3-VL-8B-ukrainian/checkpoint-1000
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import torch
from PIL import Image
import pandas as pd

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info


def load_ukrainian_dataset(data_dir: str, csv_filename: str) -> Dataset:
    """
    Load Ukrainian manuscript dataset from transkribus_parser.py output.

    Expected structure:
        data_dir/
            train.csv  (or val.csv)
            line_images/
                image_001.jpg
                image_002.jpg
                ...

    CSV format: image_path,text (no header)
    """
    data_path = Path(data_dir)
    csv_file = data_path / csv_filename

    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_file}")

    # Load CSV without header (transkribus_parser.py format)
    df = pd.read_csv(csv_file, header=None, names=["image_path", "text"])

    # Filter out empty text
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]

    print(f"Loaded {len(df)} samples from {csv_file}")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    # Add image loading function
    def load_image(example):
        # image_path is relative from CSV (e.g., "line_images\image.png")
        image_full_path = data_path / example["image_path"]
        if not image_full_path.exists():
            # Return None for missing images - will be filtered later
            return {"image": None, "text": example["text"]}

        try:
            example["image"] = Image.open(image_full_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to load {image_full_path}: {e}")
            example["image"] = None

        return example

    # Load images in parallel - optimized for 16 core / 32 thread CPU
    # Use 20 workers (1.25x physical cores) for optimal I/O + CPU performance
    import multiprocessing
    num_workers = max(1, int(multiprocessing.cpu_count() * 0.625))  # 32 * 0.625 = 20
    dataset = dataset.map(load_image, num_proc=num_workers)

    # Filter out missing images
    dataset = dataset.filter(lambda x: x["image"] is not None)

    print(f"Successfully loaded {len(dataset)} samples with valid images")

    return dataset


def create_collate_fn(processor):
    """Create custom data collator for Ukrainian manuscript vision-language data."""

    def collate_fn(examples):
        # Build messages for Qwen3-VL chat format
        messages_list = []
        for example in examples:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": example["image"]  # PIL Image
                        },
                        {
                            "type": "text",
                            "text": "Транскрибуйте текст на этом изображении."
                                    # "Transcribe the text in this image." in Russian
                                    # You can also use Ukrainian: "Транскрибуйте текст на цьому зображенні."
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": example["text"]
                        }
                    ]
                }
            ]
            messages_list.append(messages)

        # Process with Qwen3-VL processor
        texts = processor.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=False
        )

        # Process images and text together
        image_inputs, video_inputs = process_vision_info(messages_list)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Set labels for causal LM training
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs

    return collate_fn


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL-8B on Ukrainian manuscripts")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset directory (output from transkribus_parser.py)")
    parser.add_argument("--train_csv", type=str, default="train.csv",
                        help="Training CSV filename (default: train.csv)")
    parser.add_argument("--val_csv", type=str, default="val.csv",
                        help="Validation CSV filename (default: val.csv)")
    parser.add_argument("--output_dir", type=str, default="./models/Qwen3-VL-8B-ukrainian",
                        help="Output directory for checkpoints")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Base Qwen3-VL model (default: Qwen3-VL-8B-Instruct)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size (default: 4 for 8B model on 4090)")
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                        help="Gradient accumulation steps (default: 8, effective batch=64)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Early stopping patience in evaluations (default: 3)")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.01,
                        help="Early stopping threshold (default: 0.01)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to HuggingFace Hub after training")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="HuggingFace Hub model ID (if pushing)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save training config
    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print("Qwen3-VL-8B Fine-tuning for Ukrainian Manuscripts")
    print("=" * 80)
    print(f"Base model: {args.base_model}")
    print(f"Data directory: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} (per device)")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation * torch.cuda.device_count()}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Early stopping: patience={args.early_stopping_patience}, threshold={args.early_stopping_threshold}")
    print("=" * 80)

    # Load dataset
    print("\nLoading training dataset...")
    train_dataset = load_ukrainian_dataset(args.data_root, args.train_csv)

    print("\nLoading validation dataset...")
    try:
        val_dataset = load_ukrainian_dataset(args.data_root, args.val_csv)
        eval_strategy = "steps"
        eval_steps = 100  # More frequent evaluation for early stopping
    except FileNotFoundError:
        print("No validation set found, training without evaluation")
        val_dataset = None
        eval_strategy = "no"
        eval_steps = None

    # Configure training arguments (optimized for 2x RTX 4090)
    training_args = SFTConfig(
        output_dir=args.output_dir,

        # Training duration
        num_train_epochs=args.epochs,
        max_steps=-1,  # Use epochs instead

        # Batch size and gradient accumulation
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,

        # Memory optimization
        gradient_checkpointing=True,  # Essential for large VLM
        fp16=True,  # Mixed precision (use bf16 if available)
        # bf16=True,  # Uncomment if your GPU supports bfloat16 (better for VLM)

        # Learning rate and optimization
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=500,
        lr_scheduler_type="cosine",

        # Evaluation (use eval_strategy for SFTConfig, not evaluation_strategy)
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,

        # Early stopping
        load_best_model_at_end=(val_dataset is not None),
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,  # Lower eval_loss is better

        # Logging and checkpointing
        logging_steps=50,
        save_steps=100,  # More frequent saves for early stopping
        save_strategy="steps",
        save_total_limit=5,  # Keep 5 best checkpoints for early stopping

        # HuggingFace Hub
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,

        # Qwen3-VL specific
        remove_unused_columns=False,  # Keep all columns for custom collate_fn

        # DataLoader optimization - use multiple workers for data loading during training
        # Note: This is separate from dataset.map() num_proc (which loads images during preprocessing)
        # During training, use fewer workers to avoid CPU bottleneck (GPU is the bottleneck)
        # Windows multiprocessing issue: set to 0 to avoid pickle errors with local functions
        dataloader_num_workers=0,  # Must be 0 on Windows due to multiprocessing limitations

        # Multi-GPU (DDP will be used automatically if CUDA_VISIBLE_DEVICES is set)
        # For 2x 4090: torchrun --nproc_per_node=2 finetune_qwen_ukrainian.py ...
    )

    # Load processor
    print(f"\nLoading processor from {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model)

    # Load model
    print(f"\nLoading vision-language model from {args.base_model}...")
    # Check if using DDP (torchrun sets LOCAL_RANK)
    is_ddp = "LOCAL_RANK" in os.environ

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,  # Use fp16 for memory efficiency
        device_map=None if is_ddp else "auto",  # Let DDP handle device placement in multi-GPU
    )

    # Configure LoRA
    print(f"\nConfiguring LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # Create data collator
    collate_fn = create_collate_fn(processor)

    # Initialize trainer with early stopping
    print("\nInitializing trainer...")
    callbacks = []
    if val_dataset is not None:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
        callbacks.append(early_stopping)
        print(f"Early stopping enabled: patience={args.early_stopping_patience}, threshold={args.early_stopping_threshold}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    # Resume from checkpoint if specified
    resume_from_checkpoint = args.resume if args.resume else None

    # Train!
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    print("\nSaving final model...")
    final_model_path = Path(args.output_dir) / "final_model"
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)

    print(f"\nTraining complete! Model saved to {final_model_path}")

    # Push to Hub if requested
    if args.push_to_hub and args.hub_model_id:
        print(f"\nPushing model to HuggingFace Hub: {args.hub_model_id}...")
        trainer.push_to_hub()

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
