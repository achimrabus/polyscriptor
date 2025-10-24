"""
Train PyLaia CRNN model for Ukrainian dataset.

Based on Transkribus PyLaia advanced parameters.

Usage:
    python train_pylaia_ukrainian.py
"""

import sys
sys.path.append('.')
from train_pylaia import *

def main():
    # Configuration for Ukrainian dataset - optimized for single RTX 4090
    config = {
        'train_dir': 'data/pylaia_ukrainian_train',
        'val_dir': 'data/pylaia_ukrainian_val',
        'output_dir': 'models/pylaia_ukrainian',
        'img_height': 128,
        'batch_size': 32,  # Optimized for single RTX 4090
        'num_workers': 8,
        'cnn_filters': [12, 24, 48, 48],
        'cnn_poolsize': [2, 2, 0, 2],
        'rnn_hidden': 256,
        'rnn_layers': 3,
        'dropout': 0.5,
        'learning_rate': 0.0003,
        'weight_decay': 0.0,
        'max_epochs': 100,
        'early_stopping': 20,
        'augment': True,
        'use_multi_gpu': False  # Disable multi-GPU
    }
    
    # Set device to first GPU only
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {num_gpus}")
        logger.info(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
        
        # Show GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {gpu_mem:.1f} GB")
    else:
        logger.error("No GPU found! GPU is required for training.")
        return
    
    logger.info("\nTraining configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create datasets
    logger.info("\nLoading datasets...")
    train_dataset = PyLaiaDataset(
        data_dir=config['train_dir'],
        img_height=config['img_height'],
        augment=config['augment']
    )
    
    val_dataset = PyLaiaDataset(
        data_dir=config['val_dir'],
        img_height=config['img_height'],
        augment=False
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Vocabulary size: {len(train_dataset.symbols)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    logger.info(f"Train batches per epoch: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    num_classes = len(train_dataset.symbols) + 1  # +1 for CTC blank
    logger.info(f"\nCreating CRNN model...")
    logger.info(f"Number of classes: {num_classes}")
    
    model = CRNN(
        img_height=config['img_height'],
        num_channels=1,
        num_classes=num_classes,
        cnn_filters=config['cnn_filters'],
        cnn_poolsize=config['cnn_poolsize'],
        rnn_hidden=config['rnn_hidden'],
        rnn_layers=config['rnn_layers'],
        dropout=config['dropout']
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {num_params:,}")
    
    # Save vocabulary and config
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_path = output_dir / "symbols.txt"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for symbol in train_dataset.symbols:
            f.write(f"{symbol}\n")
    logger.info(f"Vocabulary saved to {vocab_path}")
    
    config_path = output_dir / "model_config.json"
    with open(config_path, 'w') as f:
        # Remove non-serializable items
        config_to_save = {k: v for k, v in config.items() if k != 'use_multi_gpu'}
        json.dump(config_to_save, f, indent=2)
    logger.info(f"Model config saved to {config_path}")
    
    # Create trainer and train
    logger.info("\nInitializing trainer...")
    trainer = PyLaiaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=config['output_dir'],
        idx2char=train_dataset.idx2char,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        max_epochs=config['max_epochs'],
        early_stopping_patience=config['early_stopping'],
        use_multi_gpu=False  # Disabled
    )
    
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    logger.info(f"Training on: {len(train_dataset)} samples")
    logger.info(f"Validating on: {len(val_dataset)} samples")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Batches per epoch: {len(train_loader)}")
    logger.info(f"Estimated time per epoch: ~10-15 minutes")
    logger.info("="*60 + "\n")
    
    trainer.train()


if __name__ == '__main__':
    main()