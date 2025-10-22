"""
Optimized TrOCR training script with significant performance improvements.

Key optimizations:
1. Cached image preprocessing (10-50x faster data loading)
2. Larger batch sizes with gradient accumulation
3. DataLoader with multiple workers
4. Mixed precision training (FP16)
5. Optimized evaluation strategy
6. Better augmentation pipeline

Usage:
    python optimized_training.py --config config.yaml
"""

import os
import gc
import torch
import evaluate
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import yaml
from dataclasses import dataclass, asdict
from tqdm.auto import tqdm

from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    AutoTokenizer,
    TrainerCallback
)


@dataclass
class OptimizedTrainingConfig:
    """Training configuration with optimized defaults."""

    # Model
    model_name: str = "kazars24/trocr-base-handwritten-ru"
    max_length: int = 64

    # Data
    data_root: str = "./data"
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    cache_images: bool = True  # Cache preprocessed images in memory
    num_workers: int = 4  # DataLoader workers

    # Training
    output_dir: str = "./output"
    batch_size: int = 16  # Increased from 4!
    gradient_accumulation_steps: int = 4  # Effective batch size: 16*4=64
    epochs: int = 10
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    optim: str = "adamw_torch"  # Optimizer: adamw_torch, adafactor, etc.

    # Optimization
    fp16: bool = True
    gradient_checkpointing: bool = False  # Disabled for speed
    dataloader_num_workers: int = 4

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 50

    # Generation (for eval)
    predict_with_generate: bool = True
    generation_max_length: int = 64
    generation_num_beams: int = 1  # Beam=1 for faster eval (greedy)

    # Augmentation
    use_augmentation: bool = True
    aug_rotation_degrees: int = 2
    aug_brightness: float = 0.3
    aug_contrast: float = 0.3

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str):
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


class OptimizedOCRDataset(Dataset):
    """
    Optimized dataset with image caching and efficient preprocessing.

    Major improvements:
    - Caches preprocessed images in memory
    - Applies augmentations during training only
    - Efficient batch processing
    """

    def __init__(
        self,
        data_root: str,
        csv_path: str,
        processor: TrOCRProcessor,
        max_length: int = 64,
        is_train: bool = True,
        use_augmentation: bool = True,
        cache_images: bool = True,
        config: Optional[OptimizedTrainingConfig] = None
    ):
        self.data_root = Path(data_root)
        self.processor = processor
        self.max_length = max_length
        self.is_train = is_train
        self.use_augmentation = use_augmentation and is_train
        self.cache_images = cache_images
        self.config = config

        # Load CSV
        self.df = pd.read_csv(
            csv_path,
            names=['image_path', 'text'],
            encoding='utf-8'
        )

        print(f"Loaded {len(self.df)} samples from {csv_path}")

        # GUARDRAIL: Filter out samples with missing image files
        # This handles cases where line images were manually removed
        valid_indices = []
        missing_count = 0

        print("Validating image files...")
        for idx in range(len(self.df)):
            image_path = self.data_root / self.df.iloc[idx]['image_path']
            if image_path.exists():
                valid_indices.append(idx)
            else:
                missing_count += 1
                # Suppress individual file warnings to avoid Unicode errors
                # if missing_count <= 10:  # Show first 10 missing files
                #     print(f"  Warning: Missing image file: {image_path}")

        if missing_count > 0:
            print(f"\n[WARNING] Found {missing_count} missing image files - filtering them out")
            if missing_count > 10:
                print(f"  (showing first 10, {missing_count - 10} more not shown)")
            self.df = self.df.iloc[valid_indices].reset_index(drop=True)
            print(f"[OK] Dataset size after filtering: {len(self.df)} samples")
        else:
            print(f"[OK] All {len(self.df)} image files found")

        # Image cache
        self.image_cache = {}

        # Setup augmentation transforms
        if self.use_augmentation and config:
            self.aug_transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=config.aug_brightness,
                    contrast=config.aug_contrast
                ),
                transforms.RandomRotation(
                    degrees=(-config.aug_rotation_degrees, config.aug_rotation_degrees),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=255
                ),
            ])
        else:
            self.aug_transform = None

        # Pre-cache images if requested
        if self.cache_images:
            print("Pre-caching images...")
            self._cache_all_images()

    def _cache_all_images(self):
        """Pre-load and cache all images."""
        for idx in tqdm(range(len(self.df)), desc="Caching images"):
            image_path = self.data_root / self.df.iloc[idx]['image_path']
            try:
                # Load and convert to RGB
                image = Image.open(image_path).convert('RGB')
                self.image_cache[idx] = image
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                # Use blank image as fallback
                self.image_cache[idx] = Image.new('RGB', (100, 32), color='white')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a single training sample with robust error handling.
        Returns a fallback sample if anything goes wrong.
        """
        try:
            # Get text
            text = str(self.df.iloc[idx]['text'])

            # Validate text is not empty
            if not text or text.strip() == '' or text.lower() == 'nan':
                raise ValueError(f"Empty or invalid text at index {idx}")

            # Get image (from cache or load)
            if self.cache_images and idx in self.image_cache:
                image = self.image_cache[idx].copy()  # Copy to avoid modifying cache
            else:
                image_path = self.data_root / self.df.iloc[idx]['image_path']
                try:
                    image = Image.open(image_path).convert('RGB')
                    # Validate image dimensions
                    if image.width < 10 or image.height < 10:
                        raise ValueError(f"Image too small: {image.size}")
                except Exception as e:
                    print(f"Warning: Error loading {image_path}: {e} - using fallback")
                    image = Image.new('RGB', (384, 32), color='white')

            # Apply augmentation if training
            if self.use_augmentation and self.aug_transform:
                try:
                    image = self.aug_transform(image)
                except Exception as e:
                    print(f"Warning: Augmentation failed at index {idx}: {e}")
                    # Continue with non-augmented image

            # Process image with TrOCR processor
            pixel_values = self.processor(
                image,
                return_tensors='pt'
            ).pixel_values.squeeze()

            # Tokenize text
            labels = self.processor.tokenizer(
                text,
                padding='max_length',
                max_length=self.max_length,
                truncation=True
            ).input_ids

            # Replace padding token id with -100 (ignored by loss)
            labels = [
                label if label != self.processor.tokenizer.pad_token_id else -100
                for label in labels
            ]

            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(labels)
            }

        except Exception as e:
            # Gracefully handle any error by returning a fallback sample
            print(f"Warning: Skipping problematic sample at index {idx}: {e}")

            # Create a fallback sample with blank image and minimal text
            fallback_image = Image.new('RGB', (384, 32), color='white')
            pixel_values = self.processor(
                fallback_image,
                return_tensors='pt'
            ).pixel_values.squeeze()

            # Use a simple text for fallback
            fallback_text = "."
            labels = self.processor.tokenizer(
                fallback_text,
                padding='max_length',
                max_length=self.max_length,
                truncation=True
            ).input_ids

            labels = [
                label if label != self.processor.tokenizer.pad_token_id else -100
                for label in labels
            ]

            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(labels)
            }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_memory():
    """Clear GPU and CPU memory to prevent OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")


class MemoryMonitorCallback(TrainerCallback):
    """Callback to monitor and clear GPU memory during training."""

    def __init__(self, clear_every_n_steps: int = 500):
        self.clear_every_n_steps = clear_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        """Clear memory periodically during training."""
        if state.global_step % self.clear_every_n_steps == 0:
            clear_memory()

    def on_evaluate(self, args, state, control, **kwargs):
        """Clear memory after evaluation."""
        clear_memory()


def compute_metrics(processor, cer_metric):
    """Compute CER metric."""
    def _compute(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Decode predictions
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # Decode labels
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        # Compute CER
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    return _compute


def train(config: OptimizedTrainingConfig):
    """Train TrOCR model with optimized settings."""
    set_seed(config.seed)

    # Clear memory before starting
    clear_memory()

    # Force gloo backend for Windows DDP (NCCL not available on Windows)
    if os.name == 'nt':  # Windows
        os.environ.setdefault('TORCH_DISTRIBUTED_BACKEND', 'gloo')

    # Detect if we're in a distributed run
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in [-1, 0]

    if is_main_process:
        print("\nBaseline GPU memory:")
        print_gpu_memory()
    
    if is_main_process:
        print("=" * 80)
        print(f"Starting optimized TrOCR training")
        print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        if torch.cuda.is_available():
            print(f"Available GPUs: {torch.cuda.device_count()}")
            print(f"Local rank: {local_rank}")
        print("=" * 80)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    if is_main_process:
        config_path = output_dir / "training_config.yaml"
        config.to_yaml(str(config_path))
        print(f"\nConfiguration saved to: {config_path}")
    
    # Load processor and tokenizer
    if is_main_process:
        print("\nLoading TrOCR processor and tokenizer...")
    processor = TrOCRProcessor.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    processor.tokenizer = tokenizer

    # Load CER metric
    if is_main_process:
        print("Loading CER metric...")
    cer_metric = evaluate.load("cer")
    
    # Create datasets
    if is_main_process:
        print("\nCreating datasets...")
    train_dataset = OptimizedOCRDataset(
        data_root=config.data_root,
        csv_path=os.path.join(config.data_root, config.train_csv),
        processor=processor,
        max_length=config.max_length,
        is_train=True,
        use_augmentation=config.use_augmentation,
        cache_images=config.cache_images,
        config=config
    )

    val_dataset = OptimizedOCRDataset(
        data_root=config.data_root,
        csv_path=os.path.join(config.data_root, config.val_csv),
        processor=processor,
        max_length=config.max_length,
        is_train=False,
        use_augmentation=False,
        cache_images=config.cache_images,
        config=config
    )

    if is_main_process:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print("\nGPU memory after dataset creation:")
        print_gpu_memory()
    
    # Load model - DO NOT call .to(device) — let Trainer handle device placement
    if is_main_process:
        print(f"\nLoading model from {config.model_name}...")
    model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
    
    # Set generation config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = config.generation_max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = config.generation_num_beams

    if is_main_process:
        print("\nGPU memory after model loading:")
        print_gpu_memory()

    # Compute effective batch size
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1
    
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps * num_gpus
    
    if is_main_process:
        print(f"\nTraining configuration:")
        print(f"  Per-device batch size: {config.batch_size}")
        print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
        print(f"  Number of GPUs: {num_gpus}")
        print(f"  Effective batch size: {effective_batch_size}")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,

        # Batch and accumulation
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # Optimization
        optim=config.optim,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.epochs,

        # Mixed precision
        fp16=config.fp16,

        # Gradient checkpointing (disabled for speed)
        gradient_checkpointing=config.gradient_checkpointing,

        # DataLoader
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,

        # Evaluation
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        predict_with_generate=config.predict_with_generate,

        # Logging and saving
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,

        # Other
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to=["tensorboard"],
    )

    # Initialize trainer
    if is_main_process:
        print("\nInitializing trainer...")

    # Add memory monitoring callback
    memory_callback = MemoryMonitorCallback(clear_every_n_steps=500)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics(processor, cer_metric),
        callbacks=[memory_callback]
    )

    # Train
    if is_main_process:
        print("\nStarting training...")
        print("GPU memory before training:")
        print_gpu_memory()

    try:
        train_result = trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if is_main_process:
                print("\n" + "=" * 60)
                print("ERROR: Out of memory detected!")
                print("=" * 60)
                print_gpu_memory()
                print("\nSuggestions:")
                print("  1. Reduce batch_size in config (currently: {})".format(config.batch_size))
                print("  2. Increase gradient_accumulation_steps")
                print("  3. Enable gradient_checkpointing")
                print("  4. Reduce max_length")
            clear_memory()
        raise

    # Save final model
    if is_main_process:
        print("\nSaving final model...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)

    # Final evaluation
    if is_main_process:
        print("\nFinal evaluation...")
    metrics = trainer.evaluate()

    if is_main_process:
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"Final CER: {metrics['eval_cer']:.4f}")
        print(f"Model saved to: {config.output_dir}")

    return trainer, metrics


def main():
    parser = argparse.ArgumentParser(description="Optimized TrOCR training")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Override data root directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory'
    )

    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        config = OptimizedTrainingConfig.from_yaml(args.config)
    else:
        config = OptimizedTrainingConfig()

    # Override with command line args
    if args.data_root:
        config.data_root = args.data_root
    if args.output_dir:
        config.output_dir = args.output_dir

    # Train
    train(config)


if __name__ == '__main__':
    main()
