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
from PIL import Image
from tqdm import tqdm
import json


class TranskribusParser:
    """Parse PAGE XML files from Transkribus exports."""

    # PAGE XML namespace
    NS = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    def __init__(self, input_dir: str, output_dir: str, min_line_width: int = 20):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_line_width = min_line_width

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

    def extract_lines_from_page(self, xml_path: Path, image_path: Path) -> List[dict]:
        """Extract all text lines from a PAGE XML file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        lines_data = []

        # Open the full page image
        # Increase PIL size limit for high-resolution scans
        Image.MAX_IMAGE_PIXELS = None
        try:
            page_image = Image.open(image_path).convert('RGB')
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

                # Crop line image with padding
                padding = 5
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(page_image.width, x2 + padding)
                y2_pad = min(page_image.height, y2 + padding)

                try:
                    line_image = page_image.crop((x1_pad, y1_pad, x2_pad, y2_pad))

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
                    print(f"Error cropping line {line_id} from {image_path}: {e}")
                    continue

        return lines_data

    def process_all(self) -> pd.DataFrame:
        """Process all PAGE XML files in the input directory."""
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

        all_lines = []

        for xml_path in tqdm(xml_files, desc="Processing pages"):
            # Find corresponding image
            image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
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
                continue

            # Extract lines
            lines = self.extract_lines_from_page(xml_path, image_path)
            all_lines.extend(lines)

        # Create DataFrame
        df = pd.DataFrame(all_lines)

        if df.empty:
            print("No lines extracted!")
            return df

        print(f"\nExtracted {len(df)} text lines")
        print(f"Average line width: {df['width'].mean():.1f}px")
        print(f"Average line height: {df['height'].mean():.1f}px")

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
            'pages_processed': df['page'].nunique()
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

    args = parser.parse_args()

    print("=" * 60)
    print("Transkribus PAGE XML Parser for TrOCR")
    print("=" * 60)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    parser = TranskribusParser(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_line_width=args.min_line_width
    )

    df = parser.process_all()

    if not df.empty:
        parser.save_dataset(df, train_ratio=args.train_ratio)
        print("\n[SUCCESS] Dataset creation complete!")
    else:
        print("\n[FAILED] Failed to create dataset")


if __name__ == '__main__':
    main()
