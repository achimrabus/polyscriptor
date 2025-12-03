#!/usr/bin/env python3
"""
Start PyLaia CRNN training for Ukrainian V2c dataset (EXIF + CASE-SENSITIVITY FIXED, GPU 0).

Dataset V2c Improvements over V2b:
- EXIF rotation bug fixed (ImageOps.exif_transpose at line 232)
- Case-sensitivity bug fixed (.JPG vs .jpg on Linux at line 344)
- 24,706 training lines (+2,762 from V2b's 21,944)
- 970 validation lines (+156 from V2b's 814)
- 2,772 Лист (printed) training lines (was 0 in V2b)
- 156 Лист validation lines (was 0 in V2b)
- Total: 25,676 lines (+12.8% improvement over V2b)
- Vocabulary: ~187 symbols (Ukrainian Cyrillic + diacritics)

Critical Bug Fixes:
1. EXIF Bug: V2b data extracted BEFORE EXIF fix (Oct 31), fix added Nov 21
   - Impact: 99 Лист files with EXIF tag 8 (270° rotation) had out-of-bounds coordinates
   - Resolution: Re-extracted with ImageOps.exif_transpose() applied
   
2. Case-Sensitivity Bug: transkribus_parser only checked lowercase extensions
   - Impact: All 99 Лист files with .JPG extension were skipped on Linux
   - Resolution: Extended image_extensions to include uppercase variants

Expected CER: 
- Overall: ~6% (V2b achieved 6.11% on handwritten only)
- Лист files (printed): <5% (V2b had ~90%+ CER due to missing training data)
- Handwritten: ~6-8% (similar to V2b)

Usage:
    # Convert data first (if not done):
    python3 convert_ukrainian_v2c_to_pylaia.py
    
    # Start training with default paths:
    python3 train_pylaia_ukrainian_v2c.py
    
    # Or specify custom directories:
    python3 train_pylaia_ukrainian_v2c.py --data-dir /path/to/data --output-dir /path/to/models
    
    # Or with nohup (recommended for long training):
    nohup python3 train_pylaia_ukrainian_v2c.py > training_ukrainian_v2c.log 2>&1 &
    tail -f training_ukrainian_v2c.log

Output:
    models/pylaia_ukrainian_v2c_<timestamp>/
    - best_model.pt (lowest validation CER)
    - checkpoint_epoch_*.pt (periodic checkpoints)
    - training_history.json
    - model_config.json
    - symbols.txt
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import logging
from datetime import datetime
import shutil
import sys
import argparse

# Import from train_pylaia.py
from train_pylaia import (
    PyLaiaDataset,
    CRNN,
    PyLaiaTrainer,
    collate_fn
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training_ukrainian_v2c.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def check_data_directory(data_dir_path):
    """Verify that the data directory exists and has been converted to PyLaia format."""
    data_dir = Path(data_dir_path)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Please run convert_ukrainian_v2c_to_pylaia.py first:")
        logger.error("  python3 convert_ukrainian_v2c_to_pylaia.py")
        sys.exit(1)
    
    required_files = ['lines.txt', 'lines_val.txt', 'syms.txt', 'dataset_info.json']
    missing_files = []
    
    for filename in required_files:
        if not (data_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        logger.error(f"Missing required files in {data_dir}:")
        for filename in missing_files:
            logger.error(f"  - {filename}")
        logger.error("\nPlease run convert_ukrainian_v2c_to_pylaia.py first:")
        logger.error("  python3 convert_ukrainian_v2c_to_pylaia.py")
        sys.exit(1)
    
    return data_dir

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train PyLaia CRNN model on Ukrainian V2c dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python3 train_pylaia_ukrainian_v2c.py

  # Specify custom data directory
  python3 train_pylaia_ukrainian_v2c.py --data-dir /path/to/data

  # Specify custom output directory
  python3 train_pylaia_ukrainian_v2c.py --output-dir /path/to/models

  # With nohup (recommended for long training)
  nohup python3 train_pylaia_ukrainian_v2c.py > training.log 2>&1 &
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/pylaia_ukrainian_v2c_combined',
        help='Path to the PyLaia format data directory (default: data/pylaia_ukrainian_v2c_combined)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Path to output directory for model checkpoints. If not specified, uses models/pylaia_ukrainian_v2c_<timestamp>'
    )
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Check data directory (validates existence and required files)
    data_dir = check_data_directory(args.data_dir)
    
    # Load dataset info
    with open(data_dir / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    logger.info("\n" + "="*70)
    logger.info("UKRAINIAN V2c DATASET INFO")
    logger.info("="*70)
    logger.info(f"Training lines:   {dataset_info['num_train']:,}")
    logger.info(f"Validation lines: {dataset_info['num_val']:,}")
    logger.info(f"Total lines:      {dataset_info['total_lines']:,}")
    logger.info(f"Vocabulary size:  {dataset_info['vocab_size']}")
    logger.info(f"EXIF corrected:   {dataset_info['exif_corrected']}")
    logger.info(f"Includes Лист:    {dataset_info['includes_list_files']}")
    logger.info("="*70 + "\n")
    
    # Configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine output directory
    if args.output_dir:
        output_dir_path = args.output_dir
    else:
        output_dir_path = f'models/pylaia_ukrainian_v2c_{timestamp}'
    
    # Ensure parent directory exists for output directory
    output_parent = Path(output_dir_path).parent
    if output_parent != Path('.'):
        output_parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured parent directory exists: {output_parent}")

    config = {
        'data_dir': str(data_dir),
        'output_dir': output_dir_path,
        'img_height': 128,
        'batch_size': 64,  # Optimized for NVIDIA L40S 46GB (was 32)
        'num_workers': 8,  # Optimized for AMD EPYC 9124
        'cnn_filters': [12, 24, 48, 48],  # Transkribus architecture
        'cnn_poolsize': [2, 2, 0, 2],
        'rnn_hidden': 256,
        'rnn_layers': 3,
        'dropout': 0.5,
        'learning_rate': 0.0003,
        'weight_decay': 0.0,
        'max_epochs': 250,
        'early_stopping': 15,  # Same as V2b
        'augment': True,
        'use_multi_gpu': False  # Single GPU (cuda:1)
    }

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {config_path}")

    # Copy vocabulary file to model directory
    vocab_src = Path(config['data_dir']) / "syms.txt"
    vocab_dst = output_dir / "symbols.txt"
    shutil.copy(vocab_src, vocab_dst)
    logger.info(f"Vocabulary copied to {vocab_dst}")

    # Set device to GPU 1 (user preference: avoid GPU 0)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(1)
        gpu_memory = torch.cuda.get_device_properties(1).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("CUDA not available! Training will be very slow on CPU.")
        logger.warning("Consider running on a machine with GPU support.")

    # Load datasets
    logger.info("\nLoading datasets...")
    train_dataset = PyLaiaDataset(
        data_dir=str(data_dir),
        list_file="lines.txt",
        symbols_file="syms.txt",
        img_height=config['img_height'],
        augment=config['augment']
    )

    val_dataset = PyLaiaDataset(
        data_dir=str(data_dir),
        list_file="lines_val.txt",
        symbols_file="syms.txt",
        img_height=config['img_height'],
        augment=False  # No augmentation for validation
    )

    logger.info(f"Training samples: {len(train_dataset):,}")
    logger.info(f"Validation samples: {len(val_dataset):,}")
    logger.info(f"Vocabulary size: {len(train_dataset.symbols)} symbols")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create model
    logger.info("\nInitializing CRNN model...")
    model = CRNN(
        img_height=config['img_height'],
        num_channels=1,  # Grayscale
        num_classes=len(train_dataset.symbols) + 1,  # +1 for CTC blank
        cnn_filters=config['cnn_filters'],
        cnn_poolsize=config['cnn_poolsize'],
        rnn_hidden=config['rnn_hidden'],
        rnn_layers=config['rnn_layers'],
        dropout=config['dropout']
    )

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Create trainer
    trainer = PyLaiaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        idx2char=train_dataset.idx2char,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        max_epochs=config['max_epochs'],
        early_stopping_patience=config['early_stopping'],
        use_multi_gpu=config['use_multi_gpu']
    )

    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING UKRAINIAN V2c PYLAIA TRAINING (EXIF + CASE-SENSITIVITY FIXED)")
    logger.info("="*70)
    logger.info(f"Dataset: {dataset_info['num_train']:,} train + {dataset_info['num_val']:,} val")
    logger.info(f"")
    logger.info(f"Improvements over V2b:")
    logger.info(f"  ✅ EXIF rotation bug fixed (ImageOps.exif_transpose)")
    logger.info(f"  ✅ Case-sensitivity bug fixed (.JPG vs .jpg on Linux)")
    logger.info(f"  ✅ +2,762 training lines (+12.6% more data)")
    logger.info(f"  ✅ +156 validation lines (+19.2% more data)")
    logger.info(f"  ✅ 2,772 Лист (printed) training lines (was 0 in V2b)")
    logger.info(f"  ✅ 156 Лист validation lines (was 0 in V2b)")
    logger.info(f"")
    logger.info(f"V2b CER on Лист files: ~90%+ (catastrophic failure)")
    logger.info(f"Expected V2c CER on Лист: <5% (printed text, well-trained)")
    logger.info(f"Expected V2c overall CER: ~6% (similar to V2b on handwritten)")
    logger.info(f"")
    logger.info(f"Architecture: {config['cnn_filters']} CNN + {config['rnn_layers']}x{config['rnn_hidden']} LSTM")
    logger.info(f"Batch size: {config['batch_size']} (optimized for L40S 46GB)")
    logger.info(f"Max epochs: {config['max_epochs']}")
    logger.info(f"Early stopping patience: {config['early_stopping']} epochs")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*70 + "\n")

    try:
        trainer.train()
        
        logger.info("\n" + "="*70)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Best model saved to: {output_dir / 'best_model.pt'}")
        logger.info(f"Vocabulary saved to: {output_dir / 'symbols.txt'}")
        logger.info(f"Training history: {output_dir / 'training_history.json'}")
        logger.info(f"")
        logger.info(f"Next steps:")
        logger.info(f"1. Evaluate on Лист validation files (expected <5% CER)")
        logger.info(f"2. Compare V2b vs V2c CER on Лист 021, 041, 061, 081, 101")
        logger.info(f"3. If CER is good, deploy V2c as production model")
        logger.info("="*70 + "\n")
        
    except KeyboardInterrupt:
        logger.warning("\n\nTraining interrupted by user (Ctrl+C)")
        logger.info(f"Checkpoint saved to: {output_dir}")
        logger.info("You can resume training using the checkpoint.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\nTraining failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == '__main__':
    main()
