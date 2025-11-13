"""
Whole-page OCR inference for Ukrainian handwritten text using TrOCR.

This script performs line segmentation and transcription on unsegmented page images.

Usage:
    # Basic usage with checkpoint
    python inference_page.py --image path/to/page.jpg --checkpoint models/ukrainian_model/checkpoint-3000

    # With custom settings
    python inference_page.py --image page.jpg --checkpoint checkpoint-3000 --num_beams 4 --output output.txt

    # With Transkribus PAGE XML (uses existing segmentation)
    python inference_page.py --image page.jpg --xml page.xml --checkpoint checkpoint-3000

Future: Can be extended with a GUI using tkinter or PyQt.
"""

import argparse
import torch
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import cv2

# Disable PIL DecompressionBomb protection for large manuscript images
Image.MAX_IMAGE_PIXELS = None

from transformers import VisionEncoderDecoderModel, TrOCRProcessor


@dataclass
class LineSegment:
    """Represents a segmented text line."""
    image: Image.Image
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    coords: Optional[List[Tuple[int, int]]] = None  # polygon coordinates if available
    text: Optional[str] = None  # transcription result
    confidence: Optional[float] = None  # average confidence score (0-1)
    char_confidences: Optional[List[float]] = None  # per-character confidence scores


def normalize_background(image: Image.Image) -> Image.Image:
    """
    Normalize background to light gray (similar to Efendiev dataset).

    CRITICAL for Ukrainian dataset: Models trained on data with background
    normalization MUST have normalization applied at inference time as well.

    Args:
        image: PIL Image with potentially aged/colored background

    Returns:
        PIL Image with normalized background
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)

    # Convert to LAB color space for better lighting normalization
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    # This normalizes lighting variations across the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_normalized = clahe.apply(l)

    # Merge back and convert to RGB
    lab_normalized = cv2.merge([l_normalized, a, b])
    rgb_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2RGB)

    # Convert to grayscale to remove color variations (aged paper tones)
    gray = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2GRAY)

    # Convert back to RGB with uniform background
    # This creates a light gray background similar to Efendiev dataset
    normalized_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(normalized_rgb)


class LineSegmenter:
    """Improved line segmentation using horizontal projection with multiple strategies."""

    def __init__(self, min_line_height: int = 15, min_gap: int = 5,
                 sensitivity: float = 0.02, use_morph: bool = True):
        """
        Initialize LineSegmenter.

        Args:
            min_line_height: Minimum height of a line in pixels (default: 15, lowered for tighter spacing)
            min_gap: Minimum gap between lines in pixels (default: 5, lowered for tight spacing)
            sensitivity: Threshold for detecting text (0.01-0.1, lower = more sensitive, default: 0.02)
            use_morph: Apply morphological operations to clean up detection (default: True)
        """
        self.min_line_height = min_line_height
        self.min_gap = min_gap
        self.sensitivity = sensitivity
        self.use_morph = use_morph

    def segment_lines(self, image: Image.Image, debug: bool = False) -> List[LineSegment]:
        """
        Segment page image into text lines using horizontal projection.

        Improved algorithm:
        1. Multiple binarization strategies (Otsu + Sauvola for different scripts)
        2. Morphological operations to connect broken text
        3. Lower sensitivity threshold for tight line spacing
        4. Smart gap detection based on local context

        Args:
            image: Input page image (PIL Image)
            debug: If True, visualize segmentation

        Returns:
            List of LineSegment objects
        """
        # Convert to grayscale
        gray = np.array(image.convert('L'))

        # Try multiple binarization strategies and combine
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(gray, sigma=1.0)

        # Strategy 1: Otsu's method (global threshold)
        threshold_otsu = self._otsu_threshold(blurred)
        binary_otsu = blurred < threshold_otsu

        # Strategy 2: Adaptive threshold (local threshold, better for varying contrast)
        binary_adaptive = self._adaptive_threshold(gray)

        # Combine both strategies (logical OR to catch text in both)
        binary = np.logical_or(binary_otsu, binary_adaptive)

        # Apply morphological closing to connect broken characters
        if self.use_morph:
            from scipy.ndimage import binary_closing
            # Horizontal structuring element to connect characters on same line
            struct = np.ones((3, 5))  # 3 pixels tall, 5 pixels wide
            binary = binary_closing(binary, structure=struct, iterations=2)

        # Horizontal projection (sum of black pixels per row)
        h_projection = binary.sum(axis=1)

        # Adaptive threshold based on image statistics
        # Use lower threshold for better sensitivity
        if h_projection.max() > 0:
            threshold = h_projection.max() * self.sensitivity
        else:
            # Fallback if no text detected
            threshold = 1

        is_text = h_projection > threshold

        # Apply median filter to smooth out noise in projection
        from scipy.ndimage import median_filter
        is_text_smoothed = median_filter(is_text.astype(float), size=3) > 0.5

        # Find continuous text regions with improved gap detection
        lines = []
        in_line = False
        start_y = 0
        gap_count = 0

        for y in range(len(is_text_smoothed)):
            if is_text_smoothed[y]:
                if not in_line:
                    # Start of new line
                    start_y = y
                    in_line = True
                    gap_count = 0
                else:
                    # Continue line, reset gap counter
                    gap_count = 0
            else:
                if in_line:
                    # Potential gap - count consecutive gap pixels
                    gap_count += 1
                    if gap_count >= self.min_gap:
                        # End of line (gap is large enough)
                        end_y = y - gap_count
                        if end_y - start_y >= self.min_line_height:
                            lines.append((start_y, end_y))
                        in_line = False
                        gap_count = 0

        # Don't forget last line if image ends with text
        if in_line and len(is_text_smoothed) - start_y >= self.min_line_height:
            lines.append((start_y, len(is_text_smoothed)))

        # Post-process: Merge lines that are too close (likely one line split incorrectly)
        merged_lines = self._merge_close_lines(lines, max_gap=self.min_gap * 2)

        # Create LineSegment objects
        segments = []
        width = image.width

        for y1, y2 in merged_lines:
            # Add padding (larger padding for better context)
            padding = 8
            y1_pad = max(0, y1 - padding)
            y2_pad = min(image.height, y2 + padding)

            # Crop line (full width for now, could be refined with vertical projection)
            bbox = (0, y1_pad, width, y2_pad)
            line_img = image.crop(bbox)

            segments.append(LineSegment(
                image=line_img,
                bbox=bbox
            ))

        if debug:
            self._visualize_segmentation(image, segments, h_projection)

        print(f"[LineSegmenter] Detected {len(segments)} lines (sensitivity={self.sensitivity}, min_height={self.min_line_height})")

        return segments

    @staticmethod
    def _adaptive_threshold(gray: np.ndarray, block_size: int = 35) -> np.ndarray:
        """
        Apply adaptive thresholding using a local window.
        Better for images with varying illumination or contrast.
        """
        # Use cv2 if available, otherwise fallback to simple method
        try:
            import cv2
            # Adaptive Gaussian thresholding
            binary = cv2.adaptiveThreshold(
                gray.astype(np.uint8),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                10
            )
            return binary > 0
        except:
            # Fallback: simple global threshold
            threshold = np.mean(gray) - np.std(gray) * 0.5
            return gray < threshold

    @staticmethod
    def _merge_close_lines(lines: List[Tuple[int, int]], max_gap: int = 10) -> List[Tuple[int, int]]:
        """Merge lines that are very close together (likely one line split incorrectly)."""
        if not lines:
            return lines

        merged = [lines[0]]
        for y1, y2 in lines[1:]:
            prev_y1, prev_y2 = merged[-1]
            gap = y1 - prev_y2

            if gap <= max_gap:
                # Merge with previous line
                merged[-1] = (prev_y1, y2)
            else:
                # Add as new line
                merged.append((y1, y2))

        return merged

    @staticmethod
    def _otsu_threshold(gray_array: np.ndarray) -> float:
        """Compute Otsu's threshold."""
        hist, bin_edges = np.histogram(gray_array, bins=256, range=(0, 256))
        hist = hist.astype(float)

        # Normalize
        hist /= hist.sum()

        # Cumulative sums
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        # Cumulative means
        mean1 = np.cumsum(hist * np.arange(256))
        mean2 = (np.cumsum((hist * np.arange(256))[::-1])[::-1])

        # Avoid division by zero
        weight1 = np.clip(weight1, 1e-10, 1)
        weight2 = np.clip(weight2, 1e-10, 1)

        # Between-class variance
        variance = weight1 * weight2 * ((mean1 / weight1) - (mean2 / weight2)) ** 2

        return np.argmax(variance)

    @staticmethod
    def _visualize_segmentation(image: Image.Image, segments: List[LineSegment],
                                h_projection: Optional[np.ndarray] = None):
        """Visualize line segmentation for debugging."""
        vis = image.copy()
        draw = ImageDraw.Draw(vis)

        for i, seg in enumerate(segments):
            x1, y1, x2, y2 = seg.bbox
            # Alternate colors for visibility
            color = 'red' if i % 2 == 0 else 'blue'
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 5, y1 + 5), f"Line {i+1}", fill=color)

        vis.show()

        # Optionally show projection profile
        if h_projection is not None:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.plot(h_projection)
            plt.title("Horizontal Projection Profile")
            plt.xlabel("Y Position")
            plt.ylabel("Text Density")
            plt.grid(True)
            plt.show()


class PageXMLSegmenter:
    """Segment using existing Transkribus PAGE XML annotations."""

    NS = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    def __init__(self, xml_path: str):
        self.xml_path = Path(xml_path)

    def segment_lines(self, image: Image.Image) -> List[LineSegment]:
        """Extract lines using PAGE XML coordinates with correct reading order."""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # Store regions with their reading order
        regions_with_order = []

        for region in root.findall('.//page:TextRegion', self.NS):
            # Extract region reading order from custom attribute
            region_order = self._extract_reading_order(region.get('custom', ''))

            # Get region Y coordinate as fallback (from first TextLine or Coords)
            region_y = self._get_region_y_position(region)

            # Store lines for this region with their reading order
            lines_with_order = []

            for text_line in region.findall('.//page:TextLine', self.NS):
                # Get coordinates
                coords_elem = text_line.find('page:Coords', self.NS)
                if coords_elem is None:
                    continue

                coords_str = coords_elem.get('points')
                if not coords_str:
                    continue

                # Parse coordinates
                coords = self._parse_coords(coords_str)
                x1, y1, x2, y2 = self._get_bounding_box(coords)

                # Crop line with padding
                padding = 5
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(image.width, x2 + padding)
                y2_pad = min(image.height, y2 + padding)

                bbox = (x1_pad, y1_pad, x2_pad, y2_pad)
                line_img = image.crop(bbox)

                segment = LineSegment(
                    image=line_img,
                    bbox=bbox,
                    coords=coords
                )

                # Extract line reading order from custom attribute
                line_order = self._extract_reading_order(text_line.get('custom', ''))

                # Use line reading order if available, otherwise Y coordinate
                sort_key = line_order if line_order is not None else y1
                lines_with_order.append((sort_key, segment))

            # Sort lines within this region
            lines_with_order.sort(key=lambda x: x[0])
            sorted_lines = [seg for _, seg in lines_with_order]

            # Use region reading order if available, otherwise region Y position
            region_sort_key = region_order if region_order is not None else region_y
            regions_with_order.append((region_sort_key, sorted_lines))

        # Sort regions by reading order (or Y position fallback)
        regions_with_order.sort(key=lambda x: x[0])

        # Flatten: concatenate all lines from all regions in order
        segments = []
        for _, region_lines in regions_with_order:
            segments.extend(region_lines)

        return segments

    @staticmethod
    def _extract_reading_order(custom_attr: str) -> Optional[int]:
        """Extract reading order index from custom attribute.

        Format: custom="readingOrder {index:5;}"
        Returns: 5 (or None if not found/parseable)
        """
        if not custom_attr or 'readingOrder' not in custom_attr:
            return None

        try:
            # Find "index:X;" pattern
            start = custom_attr.index('index:') + 6
            end = custom_attr.index(';', start)
            return int(custom_attr[start:end])
        except (ValueError, IndexError):
            return None

    def _get_region_y_position(self, region) -> int:
        """Get Y position of region for fallback sorting.

        Uses the Y coordinate of the region's Coords or first TextLine.
        """
        # Try region Coords first
        coords_elem = region.find('page:Coords', self.NS)
        if coords_elem is not None:
            coords_str = coords_elem.get('points')
            if coords_str:
                coords = self._parse_coords(coords_str)
                _, y1, _, _ = self._get_bounding_box(coords)
                return y1

        # Fallback: use first TextLine Y position
        text_line = region.find('.//page:TextLine', self.NS)
        if text_line is not None:
            coords_elem = text_line.find('page:Coords', self.NS)
            if coords_elem is not None:
                coords_str = coords_elem.get('points')
                if coords_str:
                    coords = self._parse_coords(coords_str)
                    _, y1, _, _ = self._get_bounding_box(coords)
                    return y1

        # Default fallback
        return 0

    @staticmethod
    def _parse_coords(coords_str: str) -> List[Tuple[int, int]]:
        """Parse coordinate string from PAGE XML."""
        points = coords_str.split()
        return [(int(p.split(',')[0]), int(p.split(',')[1])) for p in points]

    @staticmethod
    def _get_bounding_box(coords: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Get bounding box from polygon coordinates."""
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        return min(xs), min(ys), max(xs), max(ys)


class TrOCRInference:
    """TrOCR model inference."""

    def __init__(self, model_path: str, device: Optional[str] = None,
                 base_model: str = "kazars24/trocr-base-handwritten-ru",
                 normalize_bg: bool = False,
                 is_huggingface: bool = False):
        """
        Initialize TrOCR inference.

        Args:
            model_path: Path to local checkpoint or HuggingFace model ID
            device: 'cuda', 'cpu', or None for auto-detect
            base_model: Base model for processor (used with local checkpoints)
            normalize_bg: Apply background normalization
            is_huggingface: If True, load from HuggingFace Hub instead of local path
        """
        self.model_path = model_path
        self.base_model = base_model
        self.normalize_bg = normalize_bg
        self.is_huggingface = is_huggingface

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model from {'HuggingFace Hub' if is_huggingface else 'local checkpoint'}: {model_path}...")
        print(f"Using device: {self.device}")
        print(f"Background normalization: {'Enabled' if self.normalize_bg else 'Disabled'}")

        if is_huggingface:
            # Load both processor and model from HuggingFace Hub
            print(f"Downloading from HuggingFace Hub (if not cached): {model_path}")

            # Try to load processor from model first, fallback to base_model if it fails
            try:
                print(f"Attempting to load processor from {model_path}...")
                self.processor = TrOCRProcessor.from_pretrained(model_path)
            except Exception as e:
                print(f"Failed to load processor from model: {e}")
                print(f"Falling back to base model processor: {self.base_model}")
                self.processor = TrOCRProcessor.from_pretrained(self.base_model)

            self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
            # For backwards compatibility
            self.checkpoint_path = model_path
        else:
            # Load processor from base model, model from local checkpoint
            self.checkpoint_path = Path(model_path)

            # If model_path points to a specific file (e.g., model.safetensors),
            # use the parent directory for from_pretrained()
            if self.checkpoint_path.is_file():
                model_dir = self.checkpoint_path.parent
                print(f"Model path is a file, using directory: {model_dir}")
            else:
                model_dir = self.checkpoint_path

            print(f"Loading processor from base model: {self.base_model}")
            self.processor = TrOCRProcessor.from_pretrained(self.base_model)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")

    def transcribe_line(self, line_image: Image.Image, num_beams: int = 4,
                       max_length: int = 128, return_confidence: bool = False):
        """
        Transcribe a single line image.

        Args:
            line_image: PIL Image of text line
            num_beams: Number of beams for beam search (higher = better quality, slower)
            max_length: Maximum sequence length
            return_confidence: If True, return (text, confidence) tuple

        Returns:
            If return_confidence=False: Transcribed text string
            If return_confidence=True: Tuple of (text, confidence_score, char_confidences)
        """
        # Apply background normalization if enabled
        if self.normalize_bg:
            line_image = normalize_background(line_image)

        # Ensure image is in RGB mode (TrOCR requires 3 channels)
        if line_image.mode != 'RGB':
            line_image = line_image.convert('RGB')

        # Prepare image
        pixel_values = self.processor(
            images=line_image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        # Generate text with scores
        with torch.no_grad():
            if return_confidence:
                # Generate with output scores for confidence
                outputs = self.model.generate(
                    pixel_values,
                    num_beams=num_beams,
                    max_length=max_length,
                    early_stopping=True,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                generated_ids = outputs.sequences

                # Calculate confidence from scores
                # scores is a tuple of tensors, one per generation step
                # generated_ids shape: (batch_size, sequence_length)
                if hasattr(outputs, 'scores') and outputs.scores and len(outputs.scores) > 0:
                    import torch.nn.functional as F

                    # Get the actual generated tokens (excluding special tokens like BOS)
                    # generated_ids[0] is the first (and only) sequence in the batch
                    generated_tokens = generated_ids[0].cpu().numpy()

                    # scores is a tuple with one tensor per generation step
                    # Each tensor has shape (batch_size * num_beams, vocab_size)
                    token_confidences = []

                    for step_idx, score_tensor in enumerate(outputs.scores):
                        # Get probabilities for this generation step
                        # score_tensor shape: (num_beams, vocab_size) for batch_size=1
                        probs = F.softmax(score_tensor, dim=-1)

                        # The actual generated token at this step
                        # Skip BOS token (index 0), so generated token index is step_idx + 1
                        if step_idx + 1 < len(generated_tokens):
                            actual_token_id = generated_tokens[step_idx + 1]

                            # Get probability of the actual selected token (from best beam, index 0)
                            token_prob = probs[0, actual_token_id].item()
                            token_confidences.append(token_prob)

                    # Calculate average confidence
                    avg_confidence = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
                    char_confidences = token_confidences
                else:
                    avg_confidence = 0.0
                    char_confidences = []
            else:
                generated_ids = self.model.generate(
                    pixel_values,
                    num_beams=num_beams,
                    max_length=max_length,
                    early_stopping=True
                )
                avg_confidence = None
                char_confidences = None

        # Decode
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if return_confidence:
            return text, avg_confidence, char_confidences
        else:
            return text

    def transcribe_segments(self, segments: List[LineSegment],
                          num_beams: int = 4, max_length: int = 128,
                          show_progress: bool = True) -> List[LineSegment]:
        """
        Transcribe multiple line segments.

        Args:
            segments: List of LineSegment objects
            num_beams: Beam search parameter
            max_length: Max sequence length
            show_progress: Show progress bar

        Returns:
            Updated segments with text field filled
        """
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(segments, desc="Transcribing lines")
        else:
            iterator = segments

        for segment in iterator:
            segment.text = self.transcribe_line(
                segment.image,
                num_beams=num_beams,
                max_length=max_length
            )

        return segments


def main():
    parser = argparse.ArgumentParser(
        description="Whole-page OCR inference for Ukrainian handwritten text"
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input page image'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to TrOCR checkpoint directory'
    )
    parser.add_argument(
        '--xml',
        type=str,
        default=None,
        help='Optional: PAGE XML file for line segmentation (if not provided, automatic segmentation is used)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output text file (default: <image_name>_transcription.txt)'
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default=4,
        help='Number of beams for beam search (default: 4, higher=better quality but slower)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='Maximum sequence length (default: 128)'
    )
    parser.add_argument(
        '--min_line_height',
        type=int,
        default=20,
        help='Minimum line height for automatic segmentation (default: 20)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Visualize line segmentation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use for inference (default: auto-detect)'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='kazars24/trocr-base-handwritten-ru',
        help='Base model for processor (default: kazars24/trocr-base-handwritten-ru)'
    )
    parser.add_argument(
        '--normalize-background',
        action='store_true',
        help='Apply background normalization (REQUIRED if model was trained with --normalize-background)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TrOCR Whole-Page Inference")
    print("=" * 80)
    print(f"Input image:  {args.image}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Segmentation: {'PAGE XML' if args.xml else 'Automatic'}")
    print(f"Beam search:  {args.num_beams}")
    print("=" * 80)

    # Load image
    print("\nLoading image...")
    Image.MAX_IMAGE_PIXELS = None  # Allow large images
    image = Image.open(args.image).convert('RGB')
    print(f"Image size: {image.width}x{image.height}")

    # Segment lines
    print("\nSegmenting lines...")
    if args.xml:
        segmenter = PageXMLSegmenter(args.xml)
        segments = segmenter.segment_lines(image)
        print(f"Found {len(segments)} lines in PAGE XML")
    else:
        segmenter = LineSegmenter(
            min_line_height=args.min_line_height
        )
        segments = segmenter.segment_lines(image, debug=args.debug)
        print(f"Detected {len(segments)} lines")

    if not segments:
        print("ERROR: No lines detected!")
        return

    # Initialize TrOCR
    print("\nInitializing TrOCR model...")
    ocr = TrOCRInference(
        args.checkpoint,
        device=args.device,
        base_model=args.base_model,
        normalize_bg=args.normalize_background  # NEW: pass normalization flag
    )

    # Transcribe
    print(f"\nTranscribing {len(segments)} lines...")
    segments = ocr.transcribe_segments(
        segments,
        num_beams=args.num_beams,
        max_length=args.max_length
    )

    # Prepare output
    transcription = "\n".join(seg.text for seg in segments if seg.text)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        image_path = Path(args.image)
        output_path = image_path.parent / f"{image_path.stem}_transcription.txt"

    # Save
    print(f"\nSaving transcription to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(transcription)

    # Print results
    print("\n" + "=" * 80)
    print("TRANSCRIPTION RESULT")
    print("=" * 80)
    print(transcription)
    print("=" * 80)
    print(f"\nTranscription saved to: {output_path}")
    print(f"Total lines: {len(segments)}")
    print(f"Average confidence: N/A (not implemented yet)")


if __name__ == '__main__':
    main()
