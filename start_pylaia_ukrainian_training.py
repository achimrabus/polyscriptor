#!/usr/bin/env python3
"""
Start PyLaia CRNN training for Ukrainian PAGE XML dataset.

Dataset:
- Training: 21,944 line images from Ukrainian manuscripts
- Validation: 814 line images
- Height: 128px (aspect ratio preserved)
- Vocabulary: ~70 Cyrillic characters

Usage:
    python start_pylaia_ukrainian_training.py

Output:
    models/pylaia_ukrainian_pagexml_<timestamp>/
    - best_model.pt (lowest validation CER)
    - checkpoint_epoch_*.pt (periodic checkpoints)
    - training_history.json
    - model_config.json
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import logging
from datetime import datetime
import sys

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from train_pylaia import (
    PyLaiaDataset,
    CRNN,
    PyLaiaTrainer,
    collate_fn
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Configuration
    config = {
        # Data
        "train_dir": "data/pylaia_ukrainian_pagexml_train",
        "val_dir": "data/pylaia_ukrainian_pagexml_val",
        "img_height": 128,

        # Model architecture (matching CRNN class signature)
        "cnn_filters": [12, 24, 48, 48],  # Transkribus default
        "cnn_poolsize": [2, 2, 0, 2],     # Transkribus default
        "rnn_hidden": 256,
        "rnn_layers": 3,
        "dropout": 0.5,

        # Training hyperparameters (Transkribus optimized)
        "batch_size": 16,  # Adjust based on GPU memory
        "num_epochs": 250,
        "learning_rate": 0.0003,  # Transkribus default
        "weight_decay": 1e-5,
        "early_stopping_patience": 15,

        # Data augmentation
        "augment_train": True,

        # Hardware (single GPU to avoid multi-GPU errors)
        "num_workers": 4,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu"
    }

    logger.info("="*80)
    logger.info("PyLaia Ukrainian PAGE XML Training")
    logger.info("="*80)
    logger.info(f"Device: {config['device']}")
    logger.info(f"Training data: {config['train_dir']}")
    logger.info(f"Validation data: {config['val_dir']}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"models/pylaia_ukrainian_pagexml_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Load datasets
    logger.info("\nLoading datasets...")
    train_dataset = PyLaiaDataset(
        data_dir=config["train_dir"],
        img_height=config["img_height"],
        augment=config["augment_train"]
    )

    val_dataset = PyLaiaDataset(
        data_dir=config["val_dir"],
        img_height=config["img_height"],
        augment=False  # No augmentation for validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True
    )

    vocab_size = len(train_dataset.symbols) + 1  # +1 for CTC blank
    logger.info(f"Vocabulary size: {vocab_size} (including CTC blank)")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Batches per epoch: {len(train_loader)}")

    # Create model
    logger.info("\nInitializing model...")
    model = CRNN(
        img_height=config["img_height"],
        num_classes=vocab_size,
        cnn_filters=config["cnn_filters"],
        cnn_poolsize=config["cnn_poolsize"],
        rnn_hidden=config["rnn_hidden"],
        rnn_layers=config["rnn_layers"],
        dropout=config["dropout"]
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create trainer (single GPU only to avoid errors)
    device = torch.device(config["device"])
    use_multi_gpu = False  # Force single GPU

    trainer = PyLaiaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=str(output_dir),
        idx2char=train_dataset.idx2char,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        max_epochs=config["num_epochs"],
        early_stopping_patience=config["early_stopping_patience"],
        use_multi_gpu=use_multi_gpu
    )

    # Start training
    logger.info("\n" + "="*80)
    logger.info("Starting training...")
    logger.info("="*80)

    trainer.train()

    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Best validation CER: {trainer.best_val_cer:.4f} ({trainer.best_val_cer*100:.2f}%)")
    logger.info(f"Models saved to: {output_dir}")
    logger.info(f"- best_model.pt (CER: {trainer.best_val_cer:.4f})")
    logger.info(f"- checkpoint_epoch_*.pt (periodic checkpoints)")
    logger.info(f"- training_history.json")


if __name__ == "__main__":
    main()
