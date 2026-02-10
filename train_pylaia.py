"""
Train PyLaia CRNN model for Efendiev dataset with optimized hyperparameters.

Based on Transkribus PyLaia advanced parameters.

Usage:
    python train_pylaia.py
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import json
import logging
from jiwer import cer as compute_cer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PyLaiaDataset(Dataset):
    """PyLaia dataset loader with Transkribus-like preprocessing."""
    
    def __init__(
        self,
        data_dir: str,
        list_file: str = "lines.txt",
        symbols_file: str = "symbols.txt",
        img_height: int = 128,
        augment: bool = False
    ):
        """
        Args:
            data_dir: Directory containing images/, gt/, lines.txt, symbols.txt
            list_file: Name of file containing list of sample IDs
            symbols_file: Name of vocabulary file
            img_height: Target image height (128 from Transkribus)
            augment: Apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.img_height = img_height
        self.augment = augment

        # Load list of samples (new format: "image_path.png text")
        list_path = self.data_dir / list_file
        self.samples = []  # List of (image_path, text) tuples
        with open(list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # CRITICAL: Filenames can contain spaces! Split on ".png " not first space
                # Example: "line_images/0210_apo_2023-06-20 11_09_01_line.png кꙋскѝ жꙋючѝ"
                if '.png' in line:
                    # img_path, text = line.split('.png', 1)
                    # img_path = img_path + '.png'  # Add back the extension
                    img_path, text = line.split(',', 1)
                    img_path = img_path.strip()
                    text = text.strip()
                    self.samples.append((img_path, text))
                else:
                    logger.warning(f"Skipping malformed line (no '.png '): {line[:100]}")
        
        # Load vocabulary (handle both list and KALDI formats)
        symbols_path = self.data_dir / symbols_file
        with open(symbols_path, 'r', encoding='utf-8') as f:
            # CRITICAL: Use rstrip('\n\r') not strip() to preserve TAB and other whitespace symbols
            symbols_raw = [line.rstrip('\n\r') for line in f if line.rstrip('\n\r')]

        # Auto-detect format: KALDI format has "symbol index" pairs
        if symbols_raw and ' ' in symbols_raw[0]:
            parts = symbols_raw[0].split()
            if len(parts) == 2 and parts[1].isdigit():
                # KALDI format: "symbol index"
                self.symbols = [line.split()[0] for line in symbols_raw if line.split()]
                logger.info(f"Detected KALDI format vocabulary")
            else:
                # List format (one symbol per line)
                self.symbols = symbols_raw
        else:
            # List format (one symbol per line)
            self.symbols = symbols_raw

        # Remove <ctc> token if present (CTC blank is handled separately as index 0)
        if self.symbols and self.symbols[0] == '<ctc>':
            self.symbols = self.symbols[1:]
            logger.info(f"Removed <ctc> token from vocabulary (using index 0 for CTC blank)")

        # Create char-to-index mapping (0 reserved for CTC blank)
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.symbols)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.idx2char[0] = ''  # CTC blank

        # Map <SPACE> or <space> to actual space (handle both uppercase and lowercase)
        if '<SPACE>' in self.char2idx:
            space_idx = self.char2idx['<SPACE>']
            self.idx2char[space_idx] = ' '
        elif '<space>' in self.char2idx:
            space_idx = self.char2idx['<space>']
            self.idx2char[space_idx] = ' '

        logger.info(f"Loaded {len(self.samples)} samples from {list_path}")
        logger.info(f"Vocabulary size: {len(self.symbols)} characters")
        
        # Transkribus-like preprocessing: Deslope, Deslant, Stretch, Enhance
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=2),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale
            ])
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel_path, text = self.samples[idx]

        # Load image (relative to data_dir)
        img_path = self.data_dir / img_rel_path
        image = Image.open(img_path).convert('L')  # Grayscale (matches Church Slavonic 2.89% CER)

        # Normalize height while preserving aspect ratio
        width, height = image.size
        if height > 0:
            new_width = int(width * self.img_height / height)
            # Enforce max width from Transkribus (10000)
            new_width = min(new_width, 10000)
        else:
            new_width = width

        image = image.resize((new_width, self.img_height), Image.Resampling.LANCZOS)

        # Apply transforms
        image = self.transform(image)
        
        # Convert text to indices
        target = []
        for char in text:
            if char == ' ':
                # Try both <SPACE> and <space> to handle different vocab formats
                space_idx = self.char2idx.get('<SPACE>')
                if space_idx is None:
                    space_idx = self.char2idx.get('<space>', 0)
                target.append(space_idx)
            else:
                target.append(self.char2idx.get(char, 0))

        return image, torch.LongTensor(target), text, img_rel_path


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    images, targets, texts, sample_ids = zip(*batch)
    
    # Get image widths
    widths = [img.shape[2] for img in images]
    max_width = max(widths)
    
    # Pad images to same width
    batch_images = []
    for img in images:
        _, h, w = img.shape
        if w < max_width:
            padding = torch.zeros(1, h, max_width - w)
            img = torch.cat([img, padding], dim=2)
        batch_images.append(img)
    
    batch_images = torch.stack(batch_images)
    
    # Get target lengths
    target_lengths = torch.LongTensor([len(t) for t in targets])
    
    # Concatenate targets
    targets_concat = torch.cat(targets)
    
    # Get input lengths (width before passing through model)
    input_lengths = torch.LongTensor(widths)
    
    return batch_images, targets_concat, input_lengths, target_lengths, texts, sample_ids


class CRNN(nn.Module):
    """
    CRNN architecture based on Transkribus PyLaia parameters.
    
    From image:
    - CNN: kernel_size=3, dilation=1, num_features=[12,24,48,48], poolsize=[2,2,0,2]
    - RNN: LSTM, 3 layers, 256 units, dropout=0.5
    - Batch size: 24
    """
    
    def __init__(
        self,
        img_height: int = 128,
        num_channels: int = 1,
        num_classes: int = 100,
        cnn_filters: List[int] = [12, 24, 48, 48],
        cnn_poolsize: List[int] = [2, 2, 0, 2],
        rnn_hidden: int = 256,
        rnn_layers: int = 3,
        dropout: float = 0.5
    ):
        super(CRNN, self).__init__()
        
        self.img_height = img_height
        self.num_classes = num_classes
        self.cnn_poolsize = cnn_poolsize
        
        # CNN layers with Transkribus parameters
        cnn_layers = []
        in_channels = num_channels
        
        for i, out_channels in enumerate(cnn_filters):
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            # Apply pooling if specified
            if cnn_poolsize[i] > 0:
                cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate feature height after CNN
        num_pools = sum(1 for p in cnn_poolsize if p > 0)
        cnn_output_height = img_height // (2 ** num_pools)
        rnn_input_size = cnn_filters[-1] * cnn_output_height
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0,
            bidirectional=True,
            batch_first=False
        )
        
        # Linear dropout
        self.lin_dropout = nn.Dropout(dropout)
        
        # Output projection
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)
        
        logger.info(f"CRNN model initialized (Transkribus config):")
        logger.info(f"  CNN filters: {cnn_filters}, poolsize: {cnn_poolsize}")
        logger.info(f"  RNN hidden: {rnn_hidden}, layers: {rnn_layers}")
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Output classes: {num_classes}")
        logger.info(f"  CNN output height: {cnn_output_height}, RNN input size: {rnn_input_size}")
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width]
        
        Returns:
            log_probs: [width, batch, num_classes]
        """
        # CNN
        conv = self.cnn(x)
        
        # Reshape for RNN
        batch, channels, height, width = conv.size()
        conv = conv.permute(3, 0, 1, 2)  # [width, batch, channels, height]
        conv = conv.reshape(width, batch, channels * height)
        
        # RNN
        rnn_out, _ = self.rnn(conv)
        
        # Linear dropout
        rnn_out = self.lin_dropout(rnn_out)
        
        # Output projection
        output = self.fc(rnn_out)
        
        # Log softmax for CTC
        log_probs = nn.functional.log_softmax(output, dim=2)
        
        return log_probs


class PyLaiaTrainer:
    """Trainer for PyLaia CRNN model with CER calculation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str,
        idx2char: dict,
        learning_rate: float = 0.0003,
        weight_decay: float = 0.0,
        max_epochs: int = 100,
        early_stopping_patience: int = 20,
        use_multi_gpu: bool = False  # NEW: Multi-GPU flag
    ):
        self.device = device
        self.use_multi_gpu = use_multi_gpu
        
        # NEW: Wrap model with DataParallel for multi-GPU
        if use_multi_gpu and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.idx2char = idx2char
        
        # CTC Loss
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
        # Optimizer
        self.optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.best_val_cer = float('inf')
        self.epochs_without_improvement = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_cer': [],  # NEW: Track training CER to detect overfitting
            'val_loss': [],
            'val_cer': [],
            'learning_rate': []
        }
    
    def decode_predictions(self, log_probs: torch.Tensor) -> List[str]:
        """Decode CTC predictions to text."""
        # log_probs: [width, batch, num_classes]
        predictions = []
        
        _, preds = log_probs.max(2)  # [width, batch]
        preds = preds.transpose(1, 0).contiguous()  # [batch, width]
        
        for pred in preds:
            # CTC greedy decoding: remove blanks and consecutive duplicates
            chars = []
            prev_char = None
            for idx in pred.tolist():
                if idx == 0:  # blank
                    prev_char = None
                    continue
                if idx == prev_char:
                    continue
                chars.append(self.idx2char.get(idx, ''))
                prev_char = idx
            
            text = ''.join(chars)
            predictions.append(text)
        
        return predictions
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, targets, input_lengths, target_lengths, _, _) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            # Forward pass
            log_probs = self.model(images)
            
            # Use actual output sequence length from model
            batch_size = images.size(0)
            seq_len = log_probs.size(0)
            actual_input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)
            
            # CTC loss
            loss = self.criterion(log_probs, targets, actual_input_lengths, target_lengths)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model and calculate CER."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_refs = []
        
        with torch.no_grad():
            for images, targets, input_lengths, target_lengths, texts, _ in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                log_probs = self.model(images)
                
                # Use actual output sequence length
                batch_size = images.size(0)
                seq_len = log_probs.size(0)
                actual_input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)
                
                loss = self.criterion(log_probs, targets, actual_input_lengths, target_lengths)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Decode predictions for CER calculation
                preds = self.decode_predictions(log_probs)
                all_preds.extend(preds)
                all_refs.extend(texts)
        
        avg_loss = total_loss / num_batches
        
        # Calculate CER
        cer = compute_cer(all_refs, all_preds)

        return avg_loss, cer

    def calculate_train_cer(self):
        """Calculate CER on training set (without augmentation, for overfitting detection)."""
        self.model.eval()
        all_preds = []
        all_refs = []

        # Sample a subset of training data for speed (e.g., 10% or max 1000 samples)
        num_samples = min(1000, len(self.train_loader.dataset) // 10)
        indices = torch.randperm(len(self.train_loader.dataset))[:num_samples]

        # Temporarily disable augmentation
        original_augment = self.train_loader.dataset.augment if hasattr(self.train_loader.dataset, 'augment') else False
        if hasattr(self.train_loader.dataset, 'augment'):
            self.train_loader.dataset.augment = False

        with torch.no_grad():
            for idx in tqdm(indices, desc="Train CER", leave=False):
                # Get sample: dataset returns (image, target, text, img_path)
                img, target, text, img_path = self.train_loader.dataset[idx.item()]

                # Run inference
                img = img.unsqueeze(0).to(self.device)
                log_probs = self.model(img)

                # Decode
                pred = self.decode_predictions(log_probs)[0]
                all_preds.append(pred)
                all_refs.append(text)

        # Restore augmentation
        if hasattr(self.train_loader.dataset, 'augment'):
            self.train_loader.dataset.augment = original_augment

        # Calculate CER
        cer = compute_cer(all_refs, all_preds)
        return cer

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.max_epochs}")
        logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        logger.info(f"Early stopping patience: {self.early_stopping_patience}")
        
        for epoch in range(1, self.max_epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{self.max_epochs}")
            logger.info(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()

            # Calculate training CER (on subset, without augmentation)
            train_cer = self.calculate_train_cer()

            # Validate
            val_loss, val_cer = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            logger.info(f"Train loss: {train_loss:.4f}")
            logger.info(f"Train CER:  {train_cer*100:.2f}%")
            logger.info(f"Val loss:   {val_loss:.4f}")
            logger.info(f"Val CER:    {val_cer*100:.2f}%")
            logger.info(f"LR:         {current_lr:.6f}")

            # Log overfitting indicator
            if train_cer < val_cer:
                gap = (val_cer - train_cer) * 100
                logger.info(f"⚠️  Overfitting gap: {gap:.2f}% (Val CER > Train CER)")

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_cer'].append(train_cer)
            self.history['val_loss'].append(val_loss)
            self.history['val_cer'].append(val_cer)
            self.history['learning_rate'].append(current_lr)
            
            # Save checkpoint
            is_best = val_cer < self.best_val_cer
            if is_best:
                self.best_val_cer = val_cer
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_cer, is_best=True)
                logger.info(f"✓ New best model! CER: {val_cer*100:.2f}%")
            else:
                self.epochs_without_improvement += 1
                logger.info(f"No improvement for {self.epochs_without_improvement} epochs")
            
            # Regular checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_cer, is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        logger.info("\n" + "="*60)
        logger.info("Training complete!")
        logger.info(f"Best validation CER: {self.best_val_cer*100:.2f}%")
        logger.info("="*60)
        
        # Save final history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
    
    def save_checkpoint(self, epoch: int, cer: float, is_best: bool = False):
        """Save model checkpoint."""
        # NEW: Handle DataParallel wrapper when saving
        model_state = self.model.module.state_dict() if self.use_multi_gpu and torch.cuda.device_count() > 1 else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_cer': self.best_val_cer,
            'current_cer': cer,
            'history': self.history,
            'idx2char': self.idx2char,  # CRITICAL: Save character mapping for inference
            'char2idx': {char: idx for idx, char in self.idx2char.items()}  # Also save reverse mapping
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train PyLaia CRNN model')
    parser.add_argument('--train_dir', type=str, default='data/pylaia_efendiev_train', help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='data/pylaia_efendiev_val', help='Validation data directory')
    parser.add_argument('--output_dir', type=str, default='models/pylaia_efendiev', help='Output directory for models')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    args = parser.parse_args()

    # Configuration based on Transkribus and command-line arguments
    config = {
        'train_dir': args.train_dir,
        'val_dir': args.val_dir,
        'output_dir': args.output_dir,
        'img_height': 128,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'cnn_filters': [12, 24, 48, 48],
        'cnn_poolsize': [2, 2, 0, 2],
        'rnn_hidden': 256,
        'rnn_layers': 3,
        'dropout': 0.5,
        'learning_rate': args.learning_rate,
        'weight_decay': 0.0,
        'max_epochs': args.epochs,
        'early_stopping': 20,
        'augment': True,
        'use_multi_gpu': False  # TEMPORARY: Disable multi-GPU for debugging
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # NEW: Check GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPU(s):")
        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        if num_gpus > 1 and config['use_multi_gpu']:
            logger.info(f"Multi-GPU training enabled with DataParallel")
            # Increase batch size for multi-GPU
            config['batch_size'] = config['batch_size'] * num_gpus
            logger.info(f"Adjusted batch size to {config['batch_size']} for {num_gpus} GPUs")
    else:
        logger.warning("No GPU found, training on CPU")
        config['use_multi_gpu'] = False
    
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
    
    # Create data loaders
    # IMPORTANT: num_workers=0 to avoid multiprocessing deadlock
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Single-process loading to avoid deadlock
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,  # Single-process loading to avoid deadlock
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
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
        json.dump(config, f, indent=2)
    logger.info(f"Model config saved to {config_path}")
    
    # Create trainer and train
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
        use_multi_gpu=config['use_multi_gpu']  # NEW: Pass multi-GPU flag
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
# python train_pylaia.py --train_dir ./train --val_dir ./val --output_dir ./models/my_model --batch_size 4 --epochs 20
# train & val each contain images, symbols.txt and lines.txt (in format: imagename.png,text)
# data_dir: Directory containing images/, gt/, lines.txt, symbols.txt
# why do we need gt/?