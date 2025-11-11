#!/usr/bin/env python3
"""
Batch HTR Processing CLI

Process multiple manuscript images with various HTR engines (PyLaia, TrOCR, Churro, etc.)
Supports line segmentation, multiple output formats, and robust error handling.

Usage:
    python batch_processing.py \\
        --input-folder data/manuscripts/ \\
        --output-folder output/ \\
        --engine PyLaia \\
        --model-path models/pylaia_ukrainian/best_model.pt \\
        --verbose

Author: Polyscriptor Team
"""

import argparse
import sys
import json
import logging
import time
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

# HTR Engine imports
from htr_engine_base import HTREngine, TranscriptionResult, get_global_registry

# Segmentation imports
from inference_page import LineSegmenter, LineSegment, PageXMLSegmenter

# Check for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GPU processing disabled.")

try:
    from kraken_segmenter import KrakenLineSegmenter
    KRAKEN_AVAILABLE = True
except ImportError:
    KRAKEN_AVAILABLE = False


# Engine-specific recommendations (shared server - conservative defaults)
ENGINE_CONFIG = {
    'PyLaia': {
        'min_device': 'cuda',
        'default_batch_size': 32,  # Conservative for shared server
        'batch_size_range': (8, 64),
        'speed_estimate': 30,  # images per minute
        'warning': None
    },
    'TrOCR': {
        'min_device': 'cpu',
        'default_batch_size': 24,
        'batch_size_range': (8, 48),
        'speed_estimate': 20,
        'max_num_beams': 5,
        'warning': 'Beam search >5 causes significant slowdown'
    },
    'Churro': {
        'min_device': 'cpu',
        'default_batch_size': 16,
        'batch_size_range': (8, 32),
        'speed_estimate': 15,
        'warning': 'Slower than PyLaia/TrOCR but more accurate for complex layouts'
    },
    'Qwen3-VL': {
        'min_device': 'cuda',
        'default_batch_size': 4,
        'batch_size_range': (1, 8),
        'speed_estimate': 5,
        'warning': 'VERY SLOW: ~1-2 min/page. Use only for complex layouts or small batches!'
    },
    'Party': {
        'min_device': 'cuda',
        'default_batch_size': 12,
        'batch_size_range': (8, 24),
        'speed_estimate': 12,
        'warning': 'Batch-optimized. Works best with PAGE XML input.'
    },
    'Kraken': {
        'min_device': 'cpu',
        'default_batch_size': 16,
        'batch_size_range': (8, 32),
        'speed_estimate': 18,
        'warning': None
    }
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch HTR processing for manuscript images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process folder with PyLaia Ukrainian model
  %(prog)s --input-folder data/manuscripts/ \\
           --engine PyLaia \\
           --model-path models/pylaia_ukrainian/best_model.pt

  # Process with TrOCR and Kraken segmentation
  %(prog)s --input-folder images/ \\
           --engine TrOCR \\
           --model-id kazars24/trocr-base-handwritten-ru \\
           --segmentation-method kraken \\
           --output-format txt,csv,pagexml

  # Dry run (preview without processing)
  %(prog)s --input-folder pages/ --engine PyLaia \\
           --model-path models/best.pt --dry-run

Shared Server Notice:
  This script runs on a shared server. Please be mindful of resource usage.
  Use conservative batch sizes and avoid running multiple instances simultaneously.
        """
    )

    # Required
    parser.add_argument('--input-folder', type=Path, required=True,
                       help='Folder containing input images')
    parser.add_argument('--engine', type=str, required=True,
                       help='HTR engine (PyLaia, TrOCR, Churro, Qwen3-VL, Party, Kraken)')

    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--model-path', type=Path,
                            help='Path to local model checkpoint')
    model_group.add_argument('--model-id', type=str,
                            help='HuggingFace model ID')

    # Output
    parser.add_argument('--output-folder', type=Path, default=Path('./output'),
                       help='Output folder (default: ./output)')
    parser.add_argument('--output-format', type=str, default='txt',
                       help='Output formats: txt, csv, pagexml, json (comma-separated, default: txt)')

    # Segmentation
    parser.add_argument('--segmentation-method', type=str, default='hpp',
                       choices=['hpp', 'kraken', 'none'],
                       help='Line segmentation method (default: hpp)')
    parser.add_argument('--segmentation-sensitivity', type=float, default=0.05,
                       help='HPP sensitivity (0.01-0.1, default: 0.05)')
    parser.add_argument('--min-line-height', type=int, default=15,
                       help='Minimum line height in pixels (default: 15)')
    parser.add_argument('--min-gap', type=int, default=5,
                       help='Minimum gap between lines (default: 5)')

    # PAGE XML support
    parser.add_argument('--use-pagexml', action='store_true', default=True,
                       help='Auto-detect and use PAGE XML if available (default: True)')
    parser.add_argument('--no-pagexml', action='store_false', dest='use_pagexml',
                       help='Disable PAGE XML auto-detection')
    parser.add_argument('--xml-folder', type=Path,
                       help='Custom PAGE XML folder (default: check same folder and page/ subfolder)')
    parser.add_argument('--xml-suffix', type=str, default='.xml',
                       help='PAGE XML file suffix (default: .xml)')

    # Performance
    parser.add_argument('--batch-size', type=str, default='auto',
                       help='Batch size for engine (default: auto, or specify number)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device: cuda:0, cuda:1, cpu, auto (default: cuda:0)')

    # Behavior
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output with progress bars')
    parser.add_argument('--resume', action='store_true',
                       help='Skip already processed images')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview without processing (tests first image)')

    # Engine-specific (optional)
    parser.add_argument('--num-beams', type=int, default=1,
                       help='Beam search width (TrOCR, Churro, default: 1)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (Qwen3, default: 1.0)')
    parser.add_argument('--prompt', type=str,
                       help='Custom prompt (Qwen3)')
    parser.add_argument('--language', type=str,
                       help='Language code (Party: chu, rus, ukr)')

    # Safety flags
    parser.add_argument('--i-understand-this-is-slow', action='store_true',
                       help='Required flag for Qwen3 with >50 images')

    args = parser.parse_args()

    # Validation
    if not args.input_folder.exists():
        parser.error(f"Input folder not found: {args.input_folder}")

    if args.engine in ['PyLaia', 'TrOCR', 'Churro'] and not (args.model_path or args.model_id):
        parser.error(f"{args.engine} requires --model-path or --model-id")

    if args.segmentation_method == 'kraken' and not KRAKEN_AVAILABLE:
        parser.error("Kraken not installed. Install with: pip install kraken")

    # Parse output formats
    args.output_format = [fmt.strip() for fmt in args.output_format.split(',')]

    return args


def discover_images(input_folder: Path, verbose: bool = False) -> List[Path]:
    """Discover all image files in folder (recursive)."""
    extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    images = []

    for ext in extensions:
        images.extend(input_folder.rglob(f'*{ext}'))
        images.extend(input_folder.rglob(f'*{ext.upper()}'))

    images = sorted(set(images))  # Remove duplicates, sort

    if verbose:
        print(f"\n{'='*60}")
        print(f"Found {len(images)} images in {input_folder}")
        print(f"{'='*60}")
        for img in images[:10]:
            print(f"  - {img.relative_to(input_folder)}")
        if len(images) > 10:
            print(f"  ... and {len(images) - 10} more")
        print(f"{'='*60}\n")

    return images


def discover_images_with_xml(input_folder: Path, xml_folder: Optional[Path],
                              xml_suffix: str, verbose: bool = False) -> List[Tuple[Path, Optional[Path]]]:
    """
    Discover images and pair with PAGE XML files.

    Args:
        input_folder: Folder containing images
        xml_folder: Optional custom XML folder (None = auto-detect)
        xml_suffix: XML file extension (default: .xml)
        verbose: Print discovery details

    Returns:
        List of (image_path, xml_path) tuples. xml_path is None if not found.
    """
    images = discover_images(input_folder, verbose=False)

    paired = []
    xml_found_count = 0

    for img_path in images:
        xml_path = None

        # Search locations for XML file
        if xml_folder is not None:
            # Custom XML folder specified
            search_paths = [xml_folder / f"{img_path.stem}{xml_suffix}"]
        else:
            # Auto-detect: check same folder and page/ subfolder
            search_paths = [
                img_path.parent / f"{img_path.stem}{xml_suffix}",  # Same folder
                img_path.parent / 'page' / f"{img_path.stem}{xml_suffix}",  # page/ subfolder
            ]

        # Find first existing XML
        for search_path in search_paths:
            if search_path.exists():
                xml_path = search_path
                xml_found_count += 1
                break

        paired.append((img_path, xml_path))

    if verbose:
        print(f"\n{'='*60}")
        print(f"Found {len(images)} images in {input_folder}")
        print(f"  - {xml_found_count} with PAGE XML")
        print(f"  - {len(images) - xml_found_count} without PAGE XML (will use segmentation)")
        print(f"{'='*60}\n")

    return paired


def validate_pagexml(xml_path: Path, image_width: int, image_height: int, logger) -> bool:
    """
    Quick validation of PAGE XML file.

    Args:
        xml_path: Path to PAGE XML file
        image_width: Actual image width
        image_height: Actual image height
        logger: Logger for warnings

    Returns:
        True if valid, False if should fall back to segmentation
    """
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        NS = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

        # Check for Page element with dimensions
        page = root.find('.//page:Page', NS)
        if page is None:
            logger.warning(f"  ⚠️  PAGE XML has no Page element, falling back to segmentation")
            return False

        # Validate dimensions (quick check, ~0.1ms)
        xml_width = page.get('imageWidth')
        xml_height = page.get('imageHeight')

        if xml_width and xml_height:
            xml_w, xml_h = int(xml_width), int(xml_height)
            if abs(xml_w - image_width) > 10 or abs(xml_h - image_height) > 10:
                logger.warning(f"  ⚠️  PAGE XML dimensions mismatch (XML: {xml_w}x{xml_h}, actual: {image_width}x{image_height})")
                logger.warning(f"     Falling back to automatic segmentation")
                return False

        # Check for TextLines
        text_lines = root.findall('.//page:TextLine', NS)
        if len(text_lines) == 0:
            logger.warning(f"  ⚠️  PAGE XML has no TextLines, falling back to segmentation")
            return False

        return True

    except Exception as e:
        logger.warning(f"  ⚠️  PAGE XML parsing error: {e}, falling back to segmentation")
        return False


def select_device(args, logger) -> str:
    """Select GPU device with validation."""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available. Using CPU.")
        return 'cpu'

    if args.device == 'cpu':
        return 'cpu'

    if args.device == 'auto':
        # Find GPU with most free memory
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Using CPU.")
            return 'cpu'

        max_free = 0
        best_gpu = 0
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            free, total = torch.cuda.mem_get_info()
            if free > max_free:
                max_free = free
                best_gpu = i

        device = f'cuda:{best_gpu}'
        logger.info(f"Auto-selected {device} ({max_free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total)")
        return device

    # Validate explicit device
    if args.device.startswith('cuda:'):
        if not torch.cuda.is_available():
            logger.error("CUDA not available but GPU device specified")
            raise RuntimeError("CUDA not available")

        gpu_id = int(args.device.split(':')[1])
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(f"GPU {gpu_id} not found (have {torch.cuda.device_count()} GPUs)")

        # Show GPU info
        torch.cuda.set_device(gpu_id)
        free, total = torch.cuda.mem_get_info()
        logger.info(f"Using {args.device}: {torch.cuda.get_device_name(gpu_id)}")
        logger.info(f"  VRAM: {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total")

    return args.device


def determine_batch_size(args, engine_name: str, device: str, logger) -> int:
    """Determine optimal batch size."""
    if args.batch_size != 'auto':
        return int(args.batch_size)

    # Get engine config
    config = ENGINE_CONFIG.get(engine_name, {})
    default_batch = config.get('default_batch_size', 16)

    if device == 'cpu':
        # CPU: use smaller batches
        batch_size = max(4, default_batch // 2)
        logger.info(f"Auto batch size (CPU): {batch_size}")
        return batch_size

    # GPU: use default (already conservative for shared server)
    logger.info(f"Auto batch size: {default_batch} (shared server optimized)")
    return default_batch


def validate_engine_config(engine_name: str, config: dict, image_count: int, logger):
    """Validate configuration and warn about issues."""
    if engine_name not in ENGINE_CONFIG:
        return

    rec = ENGINE_CONFIG[engine_name]

    # Check device requirement
    if config.get('device') == 'cpu' and rec['min_device'] == 'cuda':
        raise ValueError(f"{engine_name} requires GPU. Use --device cuda:0 or cuda:1")

    # Check batch size
    batch_size = config.get('batch_size', 16)
    min_bs, max_bs = rec['batch_size_range']
    if batch_size < min_bs or batch_size > max_bs:
        logger.warning(f"⚠️  {engine_name} recommends batch_size {min_bs}-{max_bs} (got {batch_size})")

    # Check num_beams
    if 'max_num_beams' in rec:
        num_beams = config.get('num_beams', 1)
        if num_beams > rec['max_num_beams']:
            logger.warning(f"⚠️  {engine_name}: num_beams={num_beams} will be VERY slow. Recommend ≤{rec['max_num_beams']}")

    # Special handling for Qwen3
    if engine_name == 'Qwen3-VL' and image_count > 50:
        speed = rec['speed_estimate']
        estimated_hours = (image_count / speed) / 60

        logger.error(f"\n{'='*60}")
        logger.error(f"❌ QWEN3 LARGE BATCH WARNING")
        logger.error(f"{'='*60}")
        logger.error(f"Qwen3-VL is VERY SLOW: ~{60/speed:.0f}-{120/speed:.0f} minutes per page")
        logger.error(f"Processing {image_count} images will take approximately:")
        logger.error(f"  {estimated_hours:.1f}-{estimated_hours*2:.1f} HOURS")
        logger.error(f"\nConsider using:")
        logger.error(f"  - PyLaia: {(image_count/30)*60:.0f} seconds (~{image_count/30:.1f} min)")
        logger.error(f"  - TrOCR: {(image_count/20)*60:.0f} seconds (~{image_count/20:.1f} min)")
        logger.error(f"  - Churro: {(image_count/15)*60:.0f} seconds (~{image_count/15:.1f} min)")
        logger.error(f"\nIf you really want to use Qwen3 for {image_count} images,")
        logger.error(f"add: --i-understand-this-is-slow")
        logger.error(f"{'='*60}\n")

        if not config.get('force_slow', False):
            raise RuntimeError("Qwen3 blocked for large batch. Use --i-understand-this-is-slow to override.")

    # Print warning if exists
    if rec['warning']:
        logger.warning(f"ℹ️  {engine_name}: {rec['warning']}")


class BatchHTRProcessor:
    """Batch HTR processor for multiple images."""

    def __init__(self, args):
        self.args = args
        self.logger = self._setup_logging()
        self.engine = None
        self.segmenter = None
        self.results = []
        self.errors = []
        self._image_count = 0

        # PAGE XML statistics
        self.xml_used_count = 0  # Images processed with PAGE XML
        self.xml_failed_count = 0  # PAGE XML found but invalid/failed
        self.auto_seg_count = 0  # Images auto-segmented

    def _setup_logging(self) -> logging.Logger:
        """Setup logging to file and console."""
        logger = logging.getLogger('batch_htr')
        logger.setLevel(logging.DEBUG if self.args.verbose else logging.INFO)
        logger.handlers.clear()  # Clear existing handlers

        # File handler
        self.args.output_folder.mkdir(parents=True, exist_ok=True)
        log_file = self.args.output_folder / 'batch_processing.log'
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

        return logger

    def initialize(self):
        """Initialize engine and segmenter."""
        self.logger.info("Initializing batch processor...")

        # Create output folders
        (self.args.output_folder / 'transcriptions').mkdir(exist_ok=True)

        if 'pagexml' in self.args.output_format:
            (self.args.output_folder / 'page_xml').mkdir(exist_ok=True)

        # Select device
        device = select_device(self.args, self.logger)

        # Determine batch size
        batch_size = determine_batch_size(self.args, self.args.engine, device, self.logger)

        # Initialize engine
        self.logger.info(f"Loading {self.args.engine} engine...")
        registry = get_global_registry()
        self.engine = registry.get_engine_by_name(self.args.engine)

        if not self.engine:
            raise ValueError(f"Engine not found: {self.args.engine}")

        if not self.engine.is_available():
            raise RuntimeError(f"Engine unavailable: {self.engine.get_unavailable_reason()}")

        # Build config
        config = self._build_engine_config(device, batch_size)

        # Validate config (throws if Qwen3 with too many images)
        validate_engine_config(self.args.engine, config, len(self.results), self.logger)

        # Load model
        if not self.engine.load_model(config):
            raise RuntimeError(f"Failed to load model for {self.args.engine}")

        self.logger.info(f"✓ {self.args.engine} model loaded")

        # Initialize segmenter (if needed)
        if self.engine.requires_line_segmentation() and self.args.segmentation_method != 'none':
            self._initialize_segmenter()
        else:
            self.logger.info("ℹ️  No line segmentation required (page-based engine)")

    def _build_engine_config(self, device: str, batch_size: int) -> Dict[str, Any]:
        """Build engine configuration from CLI arguments."""
        config = {
            'device': device,
            'batch_size': batch_size,
            'force_slow': self.args.i_understand_this_is_slow,
        }

        # Model path
        if self.args.model_path:
            config['model_path'] = str(self.args.model_path)
        elif self.args.model_id:
            config['model_id'] = self.args.model_id

        # Engine-specific
        if self.args.num_beams > 1:
            config['num_beams'] = self.args.num_beams

        if self.args.temperature != 1.0:
            config['temperature'] = self.args.temperature

        if self.args.prompt:
            config['prompt'] = self.args.prompt

        if self.args.language:
            config['language'] = self.args.language

        return config

    def _initialize_segmenter(self):
        """Initialize line segmenter."""
        if self.args.segmentation_method == 'hpp':
            self.segmenter = LineSegmenter(
                min_line_height=self.args.min_line_height,
                min_gap=self.args.min_gap,
                sensitivity=self.args.segmentation_sensitivity
            )
            self.logger.info(f"✓ HPP segmenter initialized (sensitivity={self.args.segmentation_sensitivity})")

        elif self.args.segmentation_method == 'kraken':
            self.segmenter = KrakenLineSegmenter()
            self.logger.info("✓ Kraken segmenter initialized")

    def process_batch(self, image_xml_pairs: List[Tuple[Path, Optional[Path]]]):
        """Process batch of images with optional PAGE XML."""
        self.logger.info(f"\nProcessing {len(image_xml_pairs)} images...")
        self.logger.info("⚠️  Shared server: Please monitor resource usage")

        # Progress bar
        pbar = tqdm(image_xml_pairs, desc="Processing", unit="image",
                   disable=not self.args.verbose, ncols=80)

        for image_path, xml_path in pbar:
            try:
                result = self.process_image(image_path, xml_path)
                self.results.append(result)

                if self.args.verbose:
                    pbar.set_postfix({
                        'lines': result['line_count'],
                        'chars': result['char_count']
                    })

            except Exception as e:
                self.logger.error(f"Error processing {image_path.name}: {e}")
                self.errors.append({
                    'image': str(image_path),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

            # Periodic memory cleanup (every 50 images)
            if self._image_count % 50 == 0 and self._image_count > 0:
                self._check_memory_health()

        pbar.close()

    def process_image(self, image_path: Path, xml_path: Optional[Path] = None) -> Dict[str, Any]:
        """Process single image with optional PAGE XML."""
        self.logger.debug(f"Processing {image_path.name}...")

        # Check if already processed (resume mode)
        output_txt = self.args.output_folder / 'transcriptions' / f"{image_path.stem}.txt"
        if self.args.resume and output_txt.exists():
            self.logger.debug(f"Skipping {image_path.name} (already processed)")
            return self._load_cached_result(output_txt, image_path)

        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        # Segment lines (priority: PAGE XML > auto-segmentation)
        used_pagexml = False

        if self.engine.requires_line_segmentation():
            # Try PAGE XML first (if available and enabled)
            if xml_path is not None and self.args.use_pagexml:
                # Validate PAGE XML
                if validate_pagexml(xml_path, image.width, image.height, self.logger):
                    try:
                        self.logger.info(f"  Using PAGE XML: {xml_path.name}")
                        xml_segmenter = PageXMLSegmenter(str(xml_path))
                        lines = xml_segmenter.segment_lines(image)
                        self.logger.debug(f"  PAGE XML: {len(lines)} lines")
                        used_pagexml = True
                        self.xml_used_count += 1
                    except Exception as e:
                        self.logger.warning(f"  ⚠️  PAGE XML segmentation failed: {e}, falling back to automatic segmentation")
                        self.xml_failed_count += 1
                        xml_path = None  # Force fallback
                else:
                    self.xml_failed_count += 1
                    xml_path = None  # Force fallback

            # Fallback to automatic segmentation if PAGE XML not used
            if not used_pagexml:
                self.auto_seg_count += 1

                if self.segmenter is None:
                    # No segmentation (--segmentation-method none)
                    # Treat whole image as pre-segmented single line
                    lines = [LineSegment(
                        image=image,
                        bbox=(0, 0, image.width, image.height),
                        coords=None,
                        text=None,
                        confidence=None,
                        char_confidences=None
                    )]
                    self.logger.debug(f"  No segmentation: treating image as single line")
                else:
                    # Segment lines from full page
                    lines = self.segmenter.segment_lines(image)
                    self.logger.debug(f"  Segmented {len(lines)} lines")

                    # Normalize Kraken LineSegments to inference_page format
                    # Kraken: bbox=(x1,y1,x2,y2), baseline attribute
                    # inference_page: bbox=(x,y,w,h), coords attribute
                    if self.args.segmentation_method == 'kraken' and len(lines) > 0:
                        normalized_lines = []
                        for line in lines:
                            x1, y1, x2, y2 = line.bbox
                            normalized_lines.append(LineSegment(
                                image=line.image,
                                bbox=(x1, y1, x2-x1, y2-y1),  # Convert to (x, y, w, h)
                                coords=line.baseline if hasattr(line, 'baseline') else None,
                                text=None,
                                confidence=None,
                                char_confidences=None
                            ))
                        lines = normalized_lines

                    # Check for empty segmentation
                    if len(lines) == 0:
                        self.logger.warning(f"  ⚠️  Segmentation found 0 lines! Image may be too small or blank.")
                        self.logger.warning(f"     Try: --segmentation-method none (if pre-segmented lines)")
                        self.logger.warning(f"     Or: adjust --segmentation-sensitivity (current: {self.args.segmentation_sensitivity})")
        else:
            # Page-based engine: treat whole image as single "line"
            lines = [LineSegment(
                image=image,
                bbox=(0, 0, image.width, image.height),
                coords=None,
                text=None,
                confidence=None,
                char_confidences=None
            )]

        # Extract line images (filter out too-small lines for PyLaia)
        line_images = []
        filtered_lines = []
        # PyLaia CNN needs minimum ~64px after resize to 128px height
        # Original height * (128 / original_height) >= 64  → original >= 64
        # But accounting for pooling layers, set conservative threshold
        min_height_for_cnn = 40  # Conservative minimum to avoid CNN dimension errors

        for line in lines:
            x, y, w, h = line.bbox

            # Skip lines that are too small for CNN
            if h < min_height_for_cnn:
                self.logger.debug(f"  Skipping line with height {h}px (too small for CNN)")
                continue

            line_img = image_np[y:y+h, x:x+w]
            line_images.append(line_img)
            filtered_lines.append(line)

        # Update lines to only include filtered ones
        lines = filtered_lines

        # Transcribe lines
        if len(line_images) == 0:
            self.logger.warning(f"  ⚠️  No lines to transcribe for {image_path.name}")
            return {
                'image': str(image_path),
                'lines': 0,
                'status': 'skipped (no lines)'
            }

        self.logger.info(f"  Processing {len(line_images)} line(s)...")
        try:
            transcriptions = self.engine.transcribe_lines(line_images)
        except Exception as e:
            self.logger.error(f"  ❌ Transcription failed: {e}")
            # Return empty transcriptions for all lines
            from htr_engine_base import TranscriptionResult
            transcriptions = [TranscriptionResult(text="[ERROR]", confidence=0.0) for _ in line_images]

        # Update lines with transcriptions
        for line, result in zip(lines, transcriptions):
            line.text = result.text
            line.confidence = result.confidence if hasattr(result, 'confidence') else None

        # Write outputs
        result = self._write_outputs(image_path, lines, transcriptions)

        self._image_count += 1
        return result

    def _write_outputs(self, image_path: Path, lines: List[LineSegment],
                      transcriptions: List[TranscriptionResult]) -> Dict[str, Any]:
        """Write outputs in requested formats."""
        base_name = image_path.stem
        text_lines = [t.text for t in transcriptions]

        # TXT
        if 'txt' in self.args.output_format:
            txt_path = self.args.output_folder / 'transcriptions' / f"{base_name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(text_lines))

        # CSV (per-line with confidence)
        if 'csv' in self.args.output_format:
            csv_path = self.args.output_folder / 'transcriptions' / f"{base_name}.csv"
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write("line_number,text,confidence\n")
                for i, (line, trans) in enumerate(zip(lines, transcriptions)):
                    conf = trans.confidence if hasattr(trans, 'confidence') and trans.confidence is not None else ''
                    # Escape quotes
                    text = trans.text.replace('"', '""')
                    f.write(f'{i+1},"{text}",{conf}\n')

        # PAGE XML
        if 'pagexml' in self.args.output_format:
            xml_path = self.args.output_folder / 'page_xml' / f"{base_name}.xml"
            self._write_pagexml(xml_path, image_path, lines)

        # JSON (per-image)
        if 'json' in self.args.output_format:
            json_path = self.args.output_folder / 'transcriptions' / f"{base_name}.json"
            self._write_json(json_path, image_path, lines, transcriptions)

        # Return result summary
        return {
            'image': str(image_path.name),
            'line_count': len(lines),
            'char_count': sum(len(t.text) for t in transcriptions),
            'avg_confidence': self._calculate_avg_confidence(transcriptions),
            'timestamp': datetime.now().isoformat()
        }

    def _write_pagexml(self, output_path: Path, image_path: Path, lines: List[LineSegment]):
        """Write PAGE XML output."""
        from page_xml_exporter import PageXMLExporter

        exporter = PageXMLExporter()
        exporter.export(
            image_path=str(image_path),
            output_path=str(output_path),
            line_segments=lines
        )

    def _write_json(self, output_path: Path, image_path: Path,
                   lines: List[LineSegment], transcriptions: List[TranscriptionResult]):
        """Write JSON output."""
        data = {
            'image': str(image_path.name),
            'timestamp': datetime.now().isoformat(),
            'engine': self.args.engine,
            'lines': [
                {
                    'line_number': i + 1,
                    'text': t.text,
                    'confidence': t.confidence if hasattr(t, 'confidence') else None,
                    'bbox': list(line.bbox),
                    'char_count': len(t.text)
                }
                for i, (line, t) in enumerate(zip(lines, transcriptions))
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _calculate_avg_confidence(self, transcriptions: List[TranscriptionResult]) -> Optional[float]:
        """Calculate average confidence."""
        confidences = [t.confidence for t in transcriptions
                      if hasattr(t, 'confidence') and t.confidence is not None]
        if confidences:
            return sum(confidences) / len(confidences)
        return None

    def _load_cached_result(self, txt_path: Path, image_path: Path) -> Dict[str, Any]:
        """Load cached result (resume mode)."""
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        return {
            'image': image_path.name,
            'line_count': len(text.split('\n')),
            'char_count': len(text),
            'avg_confidence': None,
            'timestamp': 'cached'
        }

    def _check_memory_health(self):
        """Check memory usage and cleanup if needed."""
        if not TORCH_AVAILABLE:
            return

        # Garbage collection
        gc.collect()

        # CUDA cache cleanup
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()

        self.logger.debug(f"Memory cleanup performed (after {self._image_count} images)")

    def write_summary(self):
        """Write summary CSV and JSON."""
        self.logger.info("\nWriting summary files...")

        # CSV summary
        csv_path = self.args.output_folder / 'batch_results.csv'
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("image,line_count,char_count,avg_confidence,timestamp\n")
            for result in self.results:
                conf = result.get('avg_confidence', '')
                if conf is not None:
                    conf = f"{conf:.4f}"
                f.write(f"{result['image']},{result['line_count']},{result['char_count']},"
                       f"{conf},{result['timestamp']}\n")

        # JSON summary
        json_path = self.args.output_folder / 'batch_results.json'
        summary = {
            'metadata': {
                'engine': self.args.engine,
                'model': str(self.args.model_path or self.args.model_id or 'default'),
                'segmentation': self.args.segmentation_method,
                'total_images': len(self.results),
                'total_errors': len(self.errors),
                'timestamp': datetime.now().isoformat()
            },
            'results': self.results,
            'errors': self.errors
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"✓ Summary written to {csv_path}")
        self.logger.info(f"✓ JSON written to {json_path}")

        # Print statistics
        self._print_statistics()

        # Print failure summary if errors
        if self.errors:
            self._print_failure_summary()

    def _print_statistics(self):
        """Print processing statistics."""
        total_images = len(self.results)
        total_lines = sum(r['line_count'] for r in self.results)
        total_chars = sum(r['char_count'] for r in self.results)

        confidences = [r['avg_confidence'] for r in self.results
                      if r.get('avg_confidence') is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        self.logger.info("\n" + "="*60)
        self.logger.info("BATCH PROCESSING SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total images processed: {total_images}")

        # PAGE XML statistics (if enabled)
        if self.args.use_pagexml:
            self.logger.info(f"  - PAGE XML used: {self.xml_used_count}")
            if self.xml_failed_count > 0:
                self.logger.info(f"  - PAGE XML failed/invalid: {self.xml_failed_count}")
            self.logger.info(f"  - Auto-segmented: {self.auto_seg_count}")

        self.logger.info(f"Total lines transcribed: {total_lines}")
        self.logger.info(f"Total characters: {total_chars}")
        if avg_confidence:
            self.logger.info(f"Average confidence: {avg_confidence:.2%}")
        self.logger.info(f"Errors: {len(self.errors)}")
        self.logger.info("="*60)

    def _print_failure_summary(self):
        """Print detailed failure summary."""
        self.logger.error(f"\n{'='*60}")
        self.logger.error(f"❌ FAILURES: {len(self.errors)}/{len(self.results) + len(self.errors)} images")
        self.logger.error(f"{'='*60}")

        # Group errors by type
        error_types = defaultdict(int)
        for error in self.errors:
            error_msg = error['error']
            error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg[:50]
            error_types[error_type] += 1

        self.logger.error("\nError breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            self.logger.error(f"  {error_type}: {count} images")

        self.logger.error(f"\nSee {self.args.output_folder / 'batch_processing.log'} for details")

        # Write failures CSV
        failures_csv = self.args.output_folder / 'failures.csv'
        with open(failures_csv, 'w', encoding='utf-8') as f:
            f.write("image,error,timestamp\n")
            for error in self.errors:
                img = Path(error['image']).name
                err = error['error'].replace('"', '""')
                f.write(f'"{img}","{err}",{error["timestamp"]}\n')

        self.logger.error(f"Failed images written to: {failures_csv}")
        self.logger.error(f"{'='*60}\n")

    def cleanup(self):
        """Cleanup resources."""
        if self.engine:
            self.engine.unload_model()
            self.logger.info("✓ Model unloaded")


def dry_run_validation(processor: BatchHTRProcessor, image_xml_pairs: List[Tuple[Path, Optional[Path]]]) -> bool:
    """Validate setup with dry run + single image test."""
    logger = processor.logger

    logger.info("\n" + "="*60)
    logger.info("DRY RUN - VALIDATING SETUP")
    logger.info("="*60)

    # Show configuration
    logger.info(f"Engine: {processor.args.engine}")
    logger.info(f"Model: {processor.args.model_path or processor.args.model_id or 'default'}")
    logger.info(f"Device: {processor.args.device}")
    logger.info(f"Segmentation: {processor.args.segmentation_method}")
    logger.info(f"Output folder: {processor.args.output_folder}")
    logger.info(f"Output formats: {', '.join(processor.args.output_format)}")

    # Show image list (extract image paths from tuples)
    logger.info(f"\nFound {len(image_xml_pairs)} images:")
    for img, xml in image_xml_pairs[:10]:
        xml_marker = " (with PAGE XML)" if xml else ""
        logger.info(f"  - {img.name}{xml_marker}")
    if len(image_xml_pairs) > 10:
        logger.info(f"  ... and {len(image_xml_pairs) - 10} more")

    # Estimate time
    engine_config = ENGINE_CONFIG.get(processor.args.engine, {})
    images_per_min = engine_config.get('speed_estimate', 20)
    estimated_minutes = len(image_xml_pairs) / images_per_min
    logger.info(f"\nEstimated time: {estimated_minutes:.1f} minutes ({estimated_minutes/60:.1f} hours)")

    # Test with first image
    logger.info("\nTesting with first image...")
    try:
        test_image, test_xml = image_xml_pairs[0]
        start_time = time.time()
        result = processor.process_image(test_image, test_xml)
        elapsed = time.time() - start_time

        logger.info(f"✓ Test successful!")
        logger.info(f"  - Lines: {result['line_count']}")
        logger.info(f"  - Characters: {result['char_count']}")
        logger.info(f"  - Time: {elapsed:.2f}s")
        logger.info(f"  - Estimated throughput: {60/elapsed:.1f} images/minute")

        # Show sample output
        output_txt = processor.args.output_folder / 'transcriptions' / f"{test_image.stem}.txt"
        if output_txt.exists():
            with open(output_txt, 'r', encoding='utf-8') as f:
                sample = f.read()[:200]
            logger.info(f"\nSample output:\n{sample}...")

    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\n" + "="*60)
    logger.info("Dry run complete. Ready to process batch.")
    logger.info("="*60)

    return True


def main():
    """Main batch processing function."""
    args = parse_args()

    print("="*60)
    print("POLYSCRIPTOR - BATCH HTR PROCESSING")
    print("="*60)
    print(f"Engine: {args.engine}")
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Segmentation: {args.segmentation_method}")
    print(f"Output formats: {', '.join(args.output_format)}")
    print("="*60)
    print("⚠️  Running on shared server - please be mindful of resources")
    print("="*60)

    # Discover images (with PAGE XML if enabled)
    if args.use_pagexml:
        image_xml_pairs = discover_images_with_xml(
            args.input_folder,
            args.xml_folder,
            args.xml_suffix,
            verbose=args.verbose
        )
        image_paths = [img for img, xml in image_xml_pairs]
    else:
        image_paths = discover_images(args.input_folder, verbose=args.verbose)
        image_xml_pairs = [(img, None) for img in image_paths]

    if not image_paths:
        print(f"ERROR: No images found in {args.input_folder}")
        return 1

    print(f"\nFound {len(image_paths)} images to process")

    # Initialize processor
    try:
        processor = BatchHTRProcessor(args)
        processor.initialize()

        # Validate with pre-check (includes image count for Qwen3 check)
        # Re-run validation with actual image count
        device = select_device(args, processor.logger)
        batch_size = determine_batch_size(args, args.engine, device, processor.logger)
        config = processor._build_engine_config(device, batch_size)
        config['force_slow'] = args.i_understand_this_is_slow
        validate_engine_config(args.engine, config, len(image_paths), processor.logger)

        # Dry run
        if args.dry_run:
            print("\n[DRY RUN MODE]")
            success = dry_run_validation(processor, image_xml_pairs)
            processor.cleanup()
            return 0 if success else 1

        # Process batch
        processor.process_batch(image_xml_pairs)

        # Write summary
        processor.write_summary()

        # Cleanup
        processor.cleanup()

        print("\n✓ Batch processing complete!")
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
