"""
PyLaia inference script with line segmentation.

Takes a page image (JPEG/PNG) and outputs transcribed text.

Usage:
    python infer_pylaia.py --image page.jpg --output output.txt
    python infer_pylaia.py --image page.jpg --model models/pylaia_efendiev/best_model.pt
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import json
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from train_pylaia import CRNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LineSegmenter:
    """
    Improved line segmentation using horizontal projection profile.
    
    Handles both dark-on-light and light-on-dark images.
    """
    
    def __init__(
        self,
        min_line_height: int = 15,
        max_line_height: int = 150,
        min_gap: int = 3,
        adaptive_threshold: bool = True
    ):
        """
        Args:
            min_line_height: Minimum height of a text line in pixels
            max_line_height: Maximum height of a text line in pixels
            min_gap: Minimum gap between lines in pixels
            adaptive_threshold: Use adaptive thresholding
        """
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height
        self.min_gap = min_gap
        self.adaptive_threshold = adaptive_threshold
    
    def segment_lines(self, image_path: str, debug: bool = False) -> List[Tuple[np.ndarray, int, int]]:
        """
        Segment page image into text lines.
        
        Args:
            image_path: Path to page image
            debug: Save debug images
        
        Returns:
            List of (line_image, y_start, y_end) tuples
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better binarization
        if self.adaptive_threshold:
            # Use adaptive thresholding with smaller block size for handwriting
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 15, 8
            )
        else:
            # Simple thresholding with Otsu
            mean_val = np.mean(gray)
            if mean_val > 127:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to connect text components horizontally
        # Longer horizontal kernel to connect characters in a line
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_horizontal)
        
        # Small vertical closing to connect broken strokes
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_vertical)
        
        if debug:
            debug_path = Path(image_path).with_name(Path(image_path).stem + '_binary.png')
            cv2.imwrite(str(debug_path), binary)
            logger.info(f"Debug binary saved to {debug_path}")
        
        # Calculate horizontal projection profile
        h_projection = np.sum(binary, axis=1)
        
        # Light smoothing to reduce noise but preserve line boundaries
        from scipy.ndimage import gaussian_filter1d
        h_projection_smooth = gaussian_filter1d(h_projection, sigma=1.0)
        
        if debug:
            # Save projection profile as image
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.plot(h_projection_smooth)
            plt.xlabel('Y position')
            plt.ylabel('Projection value')
            plt.title('Horizontal Projection Profile')
            plt.grid(True)
            debug_plot = Path(image_path).with_name(Path(image_path).stem + '_projection.png')
            plt.savefig(debug_plot)
            plt.close()
            logger.info(f"Debug projection saved to {debug_plot}")
        
        # Find threshold - use a lower percentile to catch more text
        threshold = np.percentile(h_projection_smooth, 3)
        
        # Alternative: use mean-based threshold
        mean_projection = np.mean(h_projection_smooth)
        std_projection = np.std(h_projection_smooth)
        threshold = max(threshold, mean_projection + 0.1 * std_projection)
        
        logger.info(f"Using threshold: {threshold:.1f} (mean: {mean_projection:.1f}, std: {std_projection:.1f})")
        
        # Find line boundaries
        lines = []
        in_line = False
        line_start = 0
        gap_counter = 0
        
        for i, projection in enumerate(h_projection_smooth):
            if projection > threshold:
                if not in_line:
                    # Start of new line
                    line_start = i
                    in_line = True
                gap_counter = 0
            else:
                if in_line:
                    gap_counter += 1
                    # End line only if gap is large enough
                    if gap_counter >= self.min_gap:
                        line_end = i - gap_counter
                        line_height = line_end - line_start
                        
                        # Filter by height
                        if self.min_line_height <= line_height <= self.max_line_height:
                            # Add padding
                            padding = 5
                            y_start = max(0, line_start - padding)
                            y_end = min(gray.shape[0], line_end + padding)
                            
                            # Extract line image
                            line_img = gray[y_start:y_end, :]
                            lines.append((line_img, y_start, y_end))
                            logger.debug(f"Found line: y={y_start}-{y_end}, height={line_height}")
                        else:
                            logger.debug(f"Rejected line: y={line_start}-{line_end}, height={line_height} (outside {self.min_line_height}-{self.max_line_height})")
                        
                        in_line = False
                        gap_counter = 0
        
        # Handle last line
        if in_line:
            line_end = len(h_projection_smooth)
            line_height = line_end - line_start
            if self.min_line_height <= line_height <= self.max_line_height:
                padding = 5
                y_start = max(0, line_start - padding)
                y_end = min(gray.shape[0], line_end + padding)
                line_img = gray[y_start:y_end, :]
                lines.append((line_img, y_start, y_end))
        
        logger.info(f"Segmented {len(lines)} lines from {image_path}")
        
        # If still too few lines, warn user
        if len(lines) < 5:
            logger.warning(f"Only found {len(lines)} lines. Try adjusting --min-gap or --min-line-height")
        
        return lines
    
    def visualize_segmentation(
        self,
        image_path: str,
        lines: List[Tuple[np.ndarray, int, int]],
        output_path: Optional[str] = None
    ):
        """Draw segmentation boundaries on image."""
        img = cv2.imread(str(image_path))
        
        for line_idx, (_, y_start, y_end) in enumerate(lines):
            # Draw horizontal lines
            cv2.line(img, (0, y_start), (img.shape[1], y_start), (0, 255, 0), 2)
            cv2.line(img, (0, y_end), (img.shape[1], y_end), (0, 255, 0), 2)
            
            # Draw rectangle
            cv2.rectangle(img, (0, y_start), (img.shape[1], y_end), (0, 255, 0), 1)
            
            # Add line number
            cv2.putText(img, f"L{line_idx+1}", (10, y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, img)
            logger.info(f"Segmentation visualization saved to {output_path}")
        
        return img


class PyLaiaInference:
    """PyLaia model inference."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            device: Device to run inference on
        """
        self.model_path = Path(model_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model config
        model_dir = self.model_path.parent
        config_path = model_dir / "model_config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load vocabulary
        vocab_path = model_dir / "symbols.txt"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            symbols = [line.strip() for line in f]
        
        # Create idx2char mapping (0 is CTC blank)
        self.idx2char = {0: ''}
        for idx, char in enumerate(symbols, start=1):
            self.idx2char[idx] = char
        
        # Map <SPACE> to actual space
        for idx, char in self.idx2char.items():
            if char == '<SPACE>':
                self.idx2char[idx] = ' '
        
        # Load model
        num_classes = len(symbols) + 1
        self.model = CRNN(
            img_height=self.config['img_height'],
            num_channels=1,
            num_classes=num_classes,
            cnn_filters=self.config['cnn_filters'],
            cnn_poolsize=self.config['cnn_poolsize'],
            rnn_hidden=self.config['rnn_hidden'],
            rnn_layers=self.config['rnn_layers'],
            dropout=0.0  # Disable dropout for inference
        )
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {self.model_path}")
        logger.info(f"Best CER: {checkpoint.get('best_val_cer', 'unknown')}")
        logger.info(f"Using device: {self.device}")
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def preprocess_line(self, line_img: np.ndarray) -> torch.Tensor:
        """
        Preprocess line image for model input.
        
        Args:
            line_img: Grayscale line image (numpy array)
        
        Returns:
            Preprocessed tensor [1, 1, height, width]
        """
        # Convert to PIL
        pil_img = Image.fromarray(line_img)
        
        # Resize to target height while preserving aspect ratio
        width, height = pil_img.size
        target_height = self.config['img_height']
        
        if height > 0:
            new_width = int(width * target_height / height)
        else:
            new_width = width
        
        pil_img = pil_img.resize((new_width, target_height), Image.Resampling.LANCZOS)
        
        # Apply transform
        img_tensor = self.transform(pil_img)
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)  # [1, 1, height, width]
        
        return img_tensor
    
    def decode_prediction(self, log_probs: torch.Tensor) -> str:
        """
        Decode CTC output to text.
        
        Args:
            log_probs: Model output [seq_len, 1, num_classes]
        
        Returns:
            Decoded text string
        """
        # Get best class for each timestep
        _, preds = log_probs.max(2)  # [seq_len, 1]
        preds = preds.squeeze(1)  # [seq_len]
        
        # CTC greedy decoding
        chars = []
        prev_char = None
        
        for idx in preds.tolist():
            if idx == 0:  # CTC blank
                prev_char = None
                continue
            if idx == prev_char:  # Duplicate
                continue
            
            chars.append(self.idx2char.get(idx, ''))
            prev_char = idx
        
        text = ''.join(chars)
        return text
    
    def transcribe_line(self, line_img: np.ndarray) -> str:
        """
        Transcribe a single line image.
        
        Args:
            line_img: Grayscale line image
        
        Returns:
            Transcribed text
        """
        # Preprocess
        img_tensor = self.preprocess_line(line_img)
        img_tensor = img_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            log_probs = self.model(img_tensor)
        
        # Decode
        text = self.decode_prediction(log_probs)
        
        return text
    
    def transcribe_page(
        self,
        image_path: str,
        segmenter: Optional[LineSegmenter] = None,
        visualize_segmentation: bool = False,
        debug_segmentation: bool = False
    ) -> List[str]:
        """
        Transcribe entire page image.
        
        Args:
            image_path: Path to page image
            segmenter: Line segmentation instance (creates default if None)
            visualize_segmentation: Save segmentation visualization
            debug_segmentation: Save debug images for segmentation
        
        Returns:
            List of transcribed lines
        """
        if segmenter is None:
            segmenter = LineSegmenter()
        
        # Segment lines
        lines = segmenter.segment_lines(image_path, debug=debug_segmentation)
        
        if visualize_segmentation:
            vis_path = Path(image_path).with_suffix('.segmentation.jpg')
            segmenter.visualize_segmentation(image_path, lines, str(vis_path))
        
        # Transcribe each line
        transcriptions = []
        for i, (line_img, y_start, y_end) in enumerate(lines, 1):
            text = self.transcribe_line(line_img)
            transcriptions.append(text)
            logger.info(f"Line {i}/{len(lines)} (y={y_start}-{y_end}): {text}")
        
        return transcriptions


def main():
    parser = argparse.ArgumentParser(description="PyLaia inference on page images")
    parser.add_argument('--image', type=str, required=True, help='Input page image (JPEG/PNG)')
    parser.add_argument('--output', type=str, help='Output text file (default: input.txt)')
    parser.add_argument('--model', type=str, default='models/pylaia_efendiev/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--visualize', action='store_true',
                        help='Save segmentation visualization')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug images (binary, projection profile)')
    parser.add_argument('--min-line-height', type=int, default=15,
                        help='Minimum line height in pixels')
    parser.add_argument('--max-line-height', type=int, default=150,
                        help='Maximum line height in pixels')
    parser.add_argument('--min-gap', type=int, default=3,
                        help='Minimum gap between lines')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        args.output = Path(args.image).with_suffix('.txt')
    
    # Initialize segmenter with adjusted parameters
    segmenter = LineSegmenter(
        min_line_height=args.min_line_height,
        max_line_height=args.max_line_height,
        min_gap=args.min_gap
    )
    
    # Initialize inference
    inference = PyLaiaInference(model_path=args.model)
    
    # Transcribe page
    logger.info(f"Processing {args.image}")
    transcriptions = inference.transcribe_page(
        args.image,
        segmenter=segmenter,
        visualize_segmentation=args.visualize,
        debug_segmentation=args.debug
    )
    
    # Save output
    with open(args.output, 'w', encoding='utf-8') as f:
        for line in transcriptions:
            f.write(line + '\n')
    
    logger.info(f"\nTranscription saved to {args.output}")
    logger.info(f"Total lines: {len(transcriptions)}")
    
    # Print transcription
    print("\n" + "="*60)
    print("TRANSCRIPTION")
    print("="*60)
    for i, line in enumerate(transcriptions, 1):
        print(f"{i:3d}: {line}")


if __name__ == '__main__':
    main()