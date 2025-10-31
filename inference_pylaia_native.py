"""
Native PyLaia Inference (No WSL)

This module provides inference for PyLaia CRNN models trained with train_pylaia.py.
It loads the PyTorch checkpoint directly and runs inference natively on Linux.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image
import torchvision.transforms as transforms
import logging
import json

logger = logging.getLogger(__name__)


class CRNN(nn.Module):
    """
    CRNN architecture (same as train_pylaia.py).
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

        # CNN layers
        cnn_layers = []
        in_channels = num_channels

        for i, out_channels in enumerate(cnn_filters):
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])

            if cnn_poolsize[i] > 0:
                cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate RNN input size
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

        self.lin_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

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
        rnn_out = self.lin_dropout(rnn_out)

        # Output projection
        output = self.fc(rnn_out)

        # Log softmax for CTC
        log_probs = torch.nn.functional.log_softmax(output, dim=2)

        return log_probs


class PyLaiaInference:
    """
    Native PyLaia inference (no WSL dependency).
    Loads PyTorch checkpoint directly and runs inference on Linux.
    """

    def __init__(self, checkpoint_path: str, syms_path: str = None):
        """
        Initialize PyLaia inference.

        Args:
            checkpoint_path: Path to .ckpt checkpoint file
            syms_path: Path to symbols file. If None, will look in data directory.
        """
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Find symbols file
        if syms_path is None:
            # Look in data/pylaia_glagolitic/syms.txt
            syms_path = Path("data/pylaia_glagolitic/syms.txt")

        self.syms_path = Path(syms_path)
        if not self.syms_path.exists():
            raise FileNotFoundError(f"Symbols file not found: {syms_path}")

        # Load symbols
        with open(self.syms_path, 'r', encoding='utf-8') as f:
            self.symbols = [line.strip() for line in f]

        # Create char-to-index mapping (0 reserved for CTC blank)
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.symbols)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.idx2char[0] = ''  # CTC blank

        # Map <SPACE> to actual space
        if '<SPACE>' in self.char2idx:
            space_idx = self.char2idx['<SPACE>']
            self.idx2char[space_idx] = ' '

        # Load checkpoint
        logger.info(f"Loading PyLaia checkpoint: {checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

        # Extract model state dict from checkpoint
        # train_pylaia.py saves checkpoints with 'model_state_dict' key
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

        # Infer number of classes from checkpoint (fc.weight shape is [num_classes, rnn_hidden*2])
        fc_weight_shape = state_dict['fc.weight'].shape
        num_classes = fc_weight_shape[0]

        logger.info(f"Inferred {num_classes} output classes from checkpoint")
        logger.info(f"Vocabulary has {len(self.symbols)} symbols (+ 1 blank = {len(self.symbols)+1} expected)")

        # Initialize model
        self.model = CRNN(
            img_height=128,
            num_channels=1,
            num_classes=num_classes,
            cnn_filters=[12, 24, 48, 48],
            cnn_poolsize=[2, 2, 0, 2],
            rnn_hidden=256,
            rnn_layers=3,
            dropout=0.5
        )

        # Load weights
        self.model.load_state_dict(state_dict, strict=True)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing (same as training)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        logger.info(f"Loaded PyLaia model with {num_classes} output classes")
        logger.info(f"Using device: {self.device}")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image: PIL Image (RGB or grayscale)

        Returns:
            Preprocessed tensor [1, 1, height, width]
        """
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')

        # Resize to target height (128) while preserving aspect ratio
        target_height = 128
        aspect_ratio = image.width / image.height
        new_width = int(target_height * aspect_ratio)
        image = image.resize((new_width, target_height), Image.LANCZOS)

        # Apply transforms
        img_tensor = self.transform(image)  # [1, height, width]
        img_tensor = img_tensor.unsqueeze(0)  # [1, 1, height, width]

        return img_tensor

    def decode_ctc(self, log_probs: torch.Tensor) -> Tuple[str, float]:
        """
        Decode CTC output using greedy decoding.

        Args:
            log_probs: [seq_len, 1, num_classes]

        Returns:
            Tuple of (decoded_text, confidence)
        """
        # Get most likely class at each time step
        probs = torch.exp(log_probs)
        _, pred_indices = torch.max(probs, dim=2)  # [seq_len, 1]
        pred_indices = pred_indices.squeeze(1).cpu().numpy()  # [seq_len]

        # CTC greedy decoding: remove consecutive duplicates and blanks
        decoded_chars = []
        prev_idx = -1
        confidences = []

        for t, idx in enumerate(pred_indices):
            if idx != 0 and idx != prev_idx:  # Not blank and not duplicate
                char = self.idx2char.get(idx, '')
                if char:
                    decoded_chars.append(char)
                    # Get confidence for this character
                    char_conf = probs[t, 0, idx].item()
                    confidences.append(char_conf)
            prev_idx = idx

        # Join characters
        text = ''.join(decoded_chars)

        # Average confidence
        confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return text, confidence

    def transcribe(self, image: Image.Image) -> Tuple[str, float]:
        """
        Transcribe a single line image.

        Args:
            image: PIL Image of text line

        Returns:
            Tuple of (transcription_text, confidence_score)
        """
        try:
            # Preprocess
            img_tensor = self.preprocess_image(image).to(self.device)

            # Forward pass
            with torch.no_grad():
                log_probs = self.model(img_tensor)  # [width, 1, num_classes]

            # Decode
            text, confidence = self.decode_ctc(log_probs)

            return text, confidence

        except Exception as e:
            logger.error(f"Error during PyLaia inference: {e}")
            import traceback
            traceback.print_exc()
            return "", 0.0


# Model registry (updated for trained models)
PYLAIA_MODELS = {
    "Glagolitic (best)": {
        "checkpoint": "models/pylaia_glagolitic_single_gpu/best_model.pt",
        "syms": "models/pylaia_glagolitic_single_gpu/symbols.txt",
        "description": "PyLaia model - best checkpoint from Glagolitic training"
    }
}
