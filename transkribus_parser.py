"""
Parse Transkribus PAGE XML exports to create training datasets for TrOCR.

Usage:
    python transkribus_parser.py --input_dir /path/to/transkribus/export --output_dir /path/to/output

Expected structure:
    input_dir/
        page_001.xml
        page_001.jpg
        page_002.xml
        page_002.jpg
        ...
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tqdm import tqdm
import json
import csv
import cv2  # type: ignore
from multiprocessing import Pool, cpu_count


class TranskribusParser:
    """Parse PAGE XML files from Transkribus exports."""

    # PAGE XML namespace
    NS = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    def __init__(self, input_dir: str, output_dir: str, min_line_width: int = 20,
                 use_polygon_mask: bool = False, normalize_background: bool = False,
                 preserve_aspect_ratio: bool = False, target_height: int = 128,
                 num_workers: int = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_line_width = min_line_width
        self.use_polygon_mask = use_polygon_mask  # CHANGED: default False (rectangles)
        self.normalize_background = normalize_background  # NEW: background normalization flag
        self.preserve_aspect_ratio = preserve_aspect_ratio  # NEW: aspect ratio preservation
        self.target_height = target_height  # NEW: target height for resizing (default 128px as per best practices)

        # Optimize worker count for 16 core / 32 thread CPU
        # Use 1.25x physical cores for mixed I/O + CPU workload
        # Assumes hyperthreading (cpu_count() returns logical cores)
        if num_workers is not None:
            self.num_workers = num_workers
        else:
            # Use 62.5% of logical cores (equivalent to 1.25x physical cores with 2-way hyperthreading)
            # For 16 cores / 32 threads: 32 * 0.625 = 20 workers
            # This avoids excessive context switching while maximizing throughput
            default_workers = max(1, int(cpu_count() * 0.625))
            self.num_workers = default_workers

        # Create output directories
        self.images_dir = self.output_dir / "line_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = []

    def parse_coords(self, coords_str: str) -> List[Tuple[int, int]]:
        """Parse coordinate string from PAGE XML."""
        points = coords_str.split()
        return [(int(p.split(',')[0]), int(p.split(',')[1])) for p in points]

    def get_bounding_box(self, coords: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Get bounding box (x1, y1, x2, y2) from polygon coordinates."""
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        return min(xs), min(ys), max(xs), max(ys)

    def normalize_background_image(self, image: Image.Image) -> Image.Image:
        """
        Normalize background to light gray (similar to Efendiev dataset).

        This addresses aged paper, color variation, and lighting inconsistencies
        by normalizing the background to a uniform light tone while preserving
        text contrast.

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

    def resize_with_aspect_ratio(self, image: Image.Image) -> Image.Image:
        """
        Resize image to target height while preserving aspect ratio, then pad to square.

        This addresses the critical issue where TrOCR's ViTImageProcessor brutally resizes
        images to 384x384, causing 10.6x width downsampling for Ukrainian lines (4077x357).

        Following best practices from TROCR_CER_REDUCTION_STEPS.md:
        "Height target 96-128 px for line HTR with ViT; keep aspect => pad to square at
        the very end for the encoder (e.g., 384×384) rather than brutal resize."

        Process:
        1. Resize to target height (default 128px) keeping aspect ratio
        2. Pad width with white background to make square (if needed)

        Example: 4077×357 → 128px height → 1467×128 → pad to square
        Characters go from ~80px to ~28px width instead of ~7px (4x improvement)

        Args:
            image: PIL Image (cropped line)

        Returns:
            PIL Image resized with preserved aspect ratio
        """
        width, height = image.size

        # Calculate new width to maintain aspect ratio
        aspect_ratio = width / height
        new_height = self.target_height
        new_width = int(new_height * aspect_ratio)

        # Resize maintaining aspect ratio
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # For very wide images, we may need to limit the width
        # TrOCR's ViT will eventually resize to 384x384, but we want to preserve
        # as much resolution as possible in the saved images
        # Training script will handle final padding/resizing

        return resized

    def crop_polygon(self, image: Image.Image, coords: List[Tuple[int, int]]) -> Image.Image:
        """Crop image to polygon shape with masking."""
        # Get bounding box
        x1, y1, x2, y2 = self.get_bounding_box(coords)

        # Add padding
        padding = 5
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(image.width, x2 + padding)
        y2_pad = min(image.height, y2 + padding)

        # Crop to bounding box first
        cropped = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))

        # Apply background normalization if enabled (before polygon masking)
        if self.normalize_background:
            cropped = self.normalize_background_image(cropped)

        if not self.use_polygon_mask:
            # Apply aspect ratio preservation if enabled (after cropping, before returning)
            if self.preserve_aspect_ratio:
                cropped = self.resize_with_aspect_ratio(cropped)
            return cropped
        
        # Create mask for polygon
        mask = Image.new('L', (x2_pad - x1_pad, y2_pad - y1_pad), 0)
        draw = ImageDraw.Draw(mask)
        
        # Adjust coordinates relative to cropped image
        adjusted_coords = [(x - x1_pad, y - y1_pad) for x, y in coords]
        draw.polygon(adjusted_coords, fill=255)
        
        # Apply mask (sets outside polygon to white)
        result = Image.new('RGB', cropped.size, (255, 255, 255))
        result.paste(cropped, mask=mask)

        # Apply aspect ratio preservation if enabled (after all processing)
        if self.preserve_aspect_ratio:
            result = self.resize_with_aspect_ratio(result)

        return result

    def extract_lines_from_page(self, xml_path: Path, image_path: Path) -> List[dict]:
        """Extract all text lines from a PAGE XML file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        lines_data = []

        # Open the full page image
        # Increase PIL size limit for high-resolution scans
        Image.MAX_IMAGE_PIXELS = None
        try:
            page_image = Image.open(image_path)

            # ⚠️ CRITICAL FIX (2025-11-21): Apply EXIF orientation
            # ============================================================
            # JPEG files may have EXIF orientation tags (6=90°CW, 8=90°CCW)
            # indicating they're stored rotated. Image.open() does NOT
            # auto-rotate, so PAGE XML coordinates (which assume correct
            # orientation) will be misaligned.
            #
            # Impact: In Prosta Mova V2/V3, 32% of training data had EXIF
            # rotation tags, resulting in vertical text in line images.
            # This caused training to plateau at 19% CER.
            #
            # Fix: ImageOps.exif_transpose() reads EXIF tag and rotates
            # the image to correct orientation before we apply PAGE XML
            # coordinates.
            #
            # DO NOT REMOVE THIS LINE - it prevents data quality bugs.
            # See: PREPROCESSING_CHECKLIST.md, INVESTIGATION_SUMMARY.md
            # ============================================================
            page_image = ImageOps.exif_transpose(page_image)

            page_image = page_image.convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return lines_data

        # Find all TextLine elements
        for region in root.findall('.//page:TextRegion', self.NS):
            region_id = region.get('id', 'unknown')

            for idx, text_line in enumerate(region.findall('.//page:TextLine', self.NS)):
                line_id = text_line.get('id', f'{region_id}_line_{idx}')

                # Get coordinates
                coords_elem = text_line.find('page:Coords', self.NS)
                if coords_elem is None:
                    continue

                coords_str = coords_elem.get('points')
                if not coords_str:
                    continue

                coords = self.parse_coords(coords_str)
                x1, y1, x2, y2 = self.get_bounding_box(coords)

                # Get text content
                text_equiv = text_line.find('page:TextEquiv/page:Unicode', self.NS)
                if text_equiv is None or not text_equiv.text:
                    continue

                text = text_equiv.text.strip()
                if not text:
                    continue

                # Filter out too small lines
                if (x2 - x1) < self.min_line_width:
                    continue

                try:
                    # CHANGED: Use polygon cropping instead of simple bbox crop
                    line_image = self.crop_polygon(page_image, coords)

                    # Save line image
                    page_name = image_path.stem
                    line_filename = f"{page_name}_{line_id}.png"
                    line_image_path = self.images_dir / line_filename
                    line_image.save(line_image_path)

                    lines_data.append({
                        'image_path': str(line_image_path.relative_to(self.output_dir)),
                        'text': text,
                        'page': page_name,
                        'line_id': line_id,
                        'bbox': (x1, y1, x2, y2),
                        'width': x2 - x1,
                        'height': y2 - y1
                    })

                except Exception as e:
                    # Fix Unicode encoding issue on Windows console
                    try:
                        print(f"Error cropping line {line_id} from {image_path.name}: {e}")
                    except:
                        print(f"Error cropping line (Unicode error in path)")
                    continue

        return lines_data

    def _process_single_page_wrapper(self, xml_path: Path) -> List[dict]:
        """
        Wrapper for multiprocessing - extracts config and calls static method.

        This is needed because multiprocessing can't pickle instance methods properly.
        """
        return TranskribusParser._process_single_page_static(
            xml_path=xml_path,
            output_dir=self.output_dir,
            images_dir=self.images_dir,
            min_line_width=self.min_line_width,
            use_polygon_mask=self.use_polygon_mask,
            normalize_background=self.normalize_background,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
            target_height=self.target_height
        )

    @staticmethod
    def _process_single_page_static(xml_path: Path, output_dir: Path, images_dir: Path,
                                     min_line_width: int, use_polygon_mask: bool,
                                     normalize_background: bool, preserve_aspect_ratio: bool,
                                     target_height: int) -> List[dict]:
        """
        Process a single page (static method for parallel processing).

        Returns list of line metadata dicts.
        """
        # Create a temporary parser instance for this worker
        temp_parser = TranskribusParser(
            input_dir=xml_path.parent if xml_path.parent.name != "page" else xml_path.parent.parent,
            output_dir=output_dir,
            min_line_width=min_line_width,
            use_polygon_mask=use_polygon_mask,
            normalize_background=normalize_background,
            preserve_aspect_ratio=preserve_aspect_ratio,
            target_height=target_height,
            num_workers=1  # Each worker processes sequentially
        )

        # Override images_dir to use the shared one
        temp_parser.images_dir = images_dir

        # Find corresponding image
        # Include both lowercase and uppercase extensions (Linux is case-sensitive!)
        image_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF', '.tiff', '.TIFF']
        image_path = None

        # First try in same directory as XML
        for ext in image_extensions:
            potential_path = xml_path.with_suffix(ext)
            if potential_path.exists():
                image_path = potential_path
                break

        # If not found and XML is in page/ subdirectory, look in parent directory
        if image_path is None and xml_path.parent.name == "page":
            for ext in image_extensions:
                potential_path = xml_path.parent.parent / (xml_path.stem + ext)
                if potential_path.exists():
                    image_path = potential_path
                    break

        if image_path is None:
            print(f"Warning: No image found for {xml_path.name}")
            return []

        # Extract lines
        return temp_parser.extract_lines_from_page(xml_path, image_path)

    def process_all(self) -> pd.DataFrame:
        """Process all PAGE XML files in the input directory using parallel processing."""
        # First check page/ subdirectory (common Transkribus export structure)
        page_dir = self.input_dir / "page"
        xml_files = []

        if page_dir.exists():
            xml_files = list(page_dir.glob("*.xml"))
            if xml_files:
                print(f"Found {len(xml_files)} XML files in page/ subdirectory")

        # If no XMLs in page/, check root directory
        if not xml_files:
            xml_files = list(self.input_dir.glob("*.xml"))
            # Filter out metadata/mets files
            xml_files = [f for f in xml_files if f.stem not in ['metadata', 'mets']]

        if not xml_files:
            print(f"No PAGE XML files found in {self.input_dir} or {self.input_dir}/page/")
            return pd.DataFrame()

        print(f"Found {len(xml_files)} XML files")
        print(f"Using {self.num_workers} CPU cores for parallel processing")

        # Process pages in parallel
        temp_csv_path = self.output_dir / "temp_lines.csv"
        csv_file = open(temp_csv_path, 'w', encoding='utf-8', newline='')
        csv_writer = csv.writer(csv_file)

        # Track statistics without storing all data
        total_lines = 0
        total_width = 0
        total_height = 0
        unique_pages = set()

        # Use multiprocessing Pool for parallel processing
        if self.num_workers > 1:
            with Pool(processes=self.num_workers) as pool:
                # Chunksize: Each worker processes multiple pages before returning
                # Reduces task distribution overhead for datasets with many pages
                # Use dynamic sizing: total_pages / (workers * 4) ensures good load balancing
                chunksize = max(1, len(xml_files) // (self.num_workers * 4))

                # Process pages in parallel with progress bar
                for lines in tqdm(
                    pool.imap(self._process_single_page_wrapper, xml_files, chunksize=chunksize),
                    total=len(xml_files),
                    desc="Processing pages"
                ):
                    # Write lines to CSV immediately and update statistics
                    for line in lines:
                        csv_writer.writerow([line['image_path'], line['text'], line['page'], line['width'], line['height']])
                        total_lines += 1
                        total_width += line['width']
                        total_height += line['height']
                        unique_pages.add(line['page'])
        else:
            # Single-threaded processing (fallback)
            for xml_path in tqdm(xml_files, desc="Processing pages"):
                lines = self._process_single_page_wrapper(xml_path)

                # Write lines to CSV immediately and update statistics
                for line in lines:
                    csv_writer.writerow([line['image_path'], line['text'], line['page'], line['width'], line['height']])
                    total_lines += 1
                    total_width += line['width']
                    total_height += line['height']
                    unique_pages.add(line['page'])

        csv_file.close()

        if total_lines == 0:
            print("No lines extracted!")
            temp_csv_path.unlink(missing_ok=True)
            return pd.DataFrame()

        # Read the temporary CSV back into a DataFrame
        df = pd.read_csv(temp_csv_path, names=['image_path', 'text', 'page', 'width', 'height'], encoding='utf-8')
        temp_csv_path.unlink()  # Delete temp file

        print(f"\nExtracted {total_lines} text lines from {len(unique_pages)} pages")
        print(f"Average line width: {total_width/total_lines:.1f}px")
        print(f"Average line height: {total_height/total_lines:.1f}px")

        return df

    def save_dataset(self, df: pd.DataFrame, train_ratio: float = 0.8):
        """Save dataset in TrOCR-compatible format."""
        if df.empty:
            return

        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split train/val
        split_idx = int(len(df) * train_ratio)
        train_df = df[:split_idx]
        val_df = df[split_idx:]

        # Save as CSV (compatible with current training script)
        train_csv_path = self.output_dir / "train.csv"
        val_csv_path = self.output_dir / "val.csv"

        # Save with only filename and text columns
        train_df[['image_path', 'text']].to_csv(
            train_csv_path, index=False, header=False, encoding='utf-8'
        )
        val_df[['image_path', 'text']].to_csv(
            val_csv_path, index=False, header=False, encoding='utf-8'
        )

        # Save full metadata as JSON
        metadata = {
            'num_train': len(train_df),
            'num_val': len(val_df),
            'total_lines': len(df),
            'avg_line_width': float(df['width'].mean()),
            'avg_line_height': float(df['height'].mean()),
            'pages_processed': df['page'].nunique(),
            'background_normalized': self.normalize_background,  # Record preprocessing
            'preserve_aspect_ratio': self.preserve_aspect_ratio,  # NEW: record aspect ratio setting
            'target_height': self.target_height if self.preserve_aspect_ratio else None  # NEW: record target height
        }

        metadata_path = self.output_dir / "dataset_info.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\nDataset saved:")
        print(f"  Train: {len(train_df)} lines -> {train_csv_path}")
        print(f"  Val:   {len(val_df)} lines -> {val_csv_path}")
        print(f"  Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse Transkribus PAGE XML exports for TrOCR training"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing PAGE XML and image files from Transkribus'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--min_line_width',
        type=int,
        default=20,
        help='Minimum line width in pixels (default: 20)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of training data (default: 0.8)'
    )
    parser.add_argument(
        '--use-polygon-mask',  # CHANGED: inverted flag logic
        action='store_true',
        help='Enable polygon masking (default: use simple bounding box)'
    )
    parser.add_argument(
        '--normalize-background',
        action='store_true',
        help='Normalize image backgrounds to light gray (recommended for aged/colored paper)'
    )
    parser.add_argument(
        '--preserve-aspect-ratio',
        action='store_true',
        help='Resize to target height while preserving aspect ratio (RECOMMENDED for TrOCR - prevents brutal downsampling)'
    )
    parser.add_argument(
        '--target-height',
        type=int,
        default=128,
        help='Target height in pixels for aspect-ratio-preserving resize (default: 128px, recommended range: 96-150)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help=f'Number of CPU cores for parallel processing (default: all {cpu_count()} cores)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Transkribus PAGE XML Parser for TrOCR")
    print("=" * 60)
    print(f"Input directory:         {args.input_dir}")
    print(f"Output directory:        {args.output_dir}")
    print(f"Polygon masking:         {'Enabled' if args.use_polygon_mask else 'Disabled (using rectangles)'}")
    print(f"Background normalize:    {'Enabled' if args.normalize_background else 'Disabled'}")
    print(f"Preserve aspect ratio:   {'Enabled' if args.preserve_aspect_ratio else 'Disabled'}")
    if args.preserve_aspect_ratio:
        print(f"  Target height:         {args.target_height}px")
    print("=" * 60)

    parser_obj = TranskribusParser(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_line_width=args.min_line_width,
        use_polygon_mask=args.use_polygon_mask,
        normalize_background=args.normalize_background,
        preserve_aspect_ratio=args.preserve_aspect_ratio,
        target_height=args.target_height,
        num_workers=args.num_workers
    )

    df = parser_obj.process_all()

    if not df.empty:
        parser_obj.save_dataset(df, train_ratio=args.train_ratio)
        print("\n[SUCCESS] Dataset creation complete!")
    else:
        print("\n[FAILED] Failed to create dataset")


if __name__ == '__main__':
    main()
