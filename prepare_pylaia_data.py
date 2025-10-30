#!/usr/bin/env python3
"""
PyLaia Data Preparation Script

Converts Transkribus PAGE XML data to PyLaia Parquet format.

Input:
  - PAGE XML files with line coordinates and transcriptions
  - Full page images

Output:
  - Parquet file with extracted line images and texts
  - Character set file (charset.txt)

Usage:
  python prepare_pylaia_data.py \
    --input Ukrainian_Data/training_set \
    --output data/train.parquet \
    --charset data/charset.txt
"""

import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import io
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
from collections import Counter
import numpy as np

class PageXMLParser:
    """Parse Transkribus PAGE XML files."""

    # PAGE XML namespace
    NS = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    def __init__(self, xml_path: Path, image_path: Path):
        """
        Initialize parser.

        Args:
            xml_path: Path to PAGE XML file
            image_path: Path to corresponding page image
        """
        self.xml_path = xml_path
        self.image_path = image_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()

        # Load full page image
        self.page_image = Image.open(image_path).convert('RGB')

    def parse_coords(self, coords_str: str) -> List[Tuple[int, int]]:
        """
        Parse coordinate string from PAGE XML.

        Args:
            coords_str: String like "100,200 150,200 150,250 100,250"

        Returns:
            List of (x, y) tuples
        """
        points = []
        for point in coords_str.strip().split():
            x, y = map(int, point.split(','))
            points.append((x, y))
        return points

    def get_bounding_box(self, points: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """
        Get bounding box from polygon points.

        Args:
            points: List of (x, y) coordinates

        Returns:
            (min_x, min_y, max_x, max_y)
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))

    def extract_line_image(self, coords: str, padding: int = 5) -> Image.Image:
        """
        Extract line image from page using coordinates.

        Args:
            coords: Coordinate string from PAGE XML
            padding: Pixels to add around bounding box

        Returns:
            PIL Image of line region
        """
        points = self.parse_coords(coords)
        min_x, min_y, max_x, max_y = self.get_bounding_box(points)

        # Add padding
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(self.page_image.width, max_x + padding)
        max_y = min(self.page_image.height, max_y + padding)

        # Crop line region
        line_image = self.page_image.crop((min_x, min_y, max_x, max_y))

        return line_image

    def parse_lines(self) -> List[Tuple[Image.Image, str, str]]:
        """
        Parse all text lines from PAGE XML.

        Returns:
            List of (line_image, text, line_id) tuples
        """
        lines = []

        # Find all TextLine elements
        for text_line in self.root.findall('.//ns:TextLine', self.NS):
            line_id = text_line.get('id', 'unknown')

            # Get coordinates
            coords_elem = text_line.find('ns:Coords', self.NS)
            if coords_elem is None:
                print(f"  Warning: No coordinates for line {line_id}")
                continue

            coords_str = coords_elem.get('points', '')
            if not coords_str:
                print(f"  Warning: Empty coordinates for line {line_id}")
                continue

            # Get transcription
            text_equiv = text_line.find('ns:TextEquiv/ns:Unicode', self.NS)
            if text_equiv is None or not text_equiv.text:
                print(f"  Warning: No transcription for line {line_id}")
                continue

            text = text_equiv.text.strip()

            # Extract line image
            try:
                line_image = self.extract_line_image(coords_str)
                lines.append((line_image, text, line_id))
            except Exception as e:
                print(f"  Error extracting line {line_id}: {e}")
                continue

        return lines


def convert_to_parquet(
    input_dir: Path,
    output_file: Path,
    charset_file: Optional[Path] = None,
    max_samples: Optional[int] = None
):
    """
    Convert PAGE XML dataset to PyLaia Parquet format.

    Args:
        input_dir: Directory containing PAGE XML and images
        output_file: Output Parquet file path
        charset_file: Optional path to save character set
        max_samples: Optional maximum number of samples (for testing)
    """
    print(f"Converting PAGE XML to Parquet...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_file}")

    input_dir = Path(input_dir)
    page_dir = input_dir / "page"

    if not page_dir.exists():
        raise FileNotFoundError(f"PAGE XML directory not found: {page_dir}")

    # Find all PAGE XML files
    xml_files = sorted(page_dir.glob("*.xml"))
    print(f"\nFound {len(xml_files)} PAGE XML files")

    data = []
    all_chars = Counter()

    for i, xml_file in enumerate(xml_files):
        if max_samples and i >= max_samples:
            print(f"\nReached max samples limit ({max_samples})")
            break

        # Find corresponding image
        image_name = xml_file.stem + ".jpg"
        image_path = input_dir / image_name

        if not image_path.exists():
            print(f"  Warning: Image not found for {xml_file.name}: {image_path}")
            continue

        print(f"\n[{i+1}/{len(xml_files)}] Processing {xml_file.name}...")

        try:
            # Parse PAGE XML
            parser = PageXMLParser(xml_file, image_path)
            lines = parser.parse_lines()

            print(f"  Extracted {len(lines)} lines")

            # Convert each line to Parquet row
            for line_image, text, line_id in lines:
                # Convert image to bytes (JPEG format)
                img_buffer = io.BytesIO()
                line_image.save(img_buffer, format='JPEG', quality=95)
                img_bytes = img_buffer.getvalue()

                # Add to dataset
                data.append({
                    'image': img_bytes,
                    'text': text,
                    'id': f"{xml_file.stem}_{line_id}",
                    'page': xml_file.stem,
                    'line': line_id
                })

                # Count characters
                all_chars.update(text)

        except Exception as e:
            print(f"  Error processing {xml_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create DataFrame
    print(f"\n{'='*60}")
    print(f"Creating Parquet file...")
    df = pd.DataFrame(data)

    print(f"  Total samples: {len(df)}")
    print(f"  Total characters: {sum(all_chars.values())}")
    print(f"  Unique characters: {len(all_chars)}")

    # Save to Parquet
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, compression='snappy', index=False)
    print(f"  Saved: {output_file}")

    # Save character set
    if charset_file:
        charset = sorted(all_chars.keys())
        charset_file.parent.mkdir(parents=True, exist_ok=True)

        with open(charset_file, 'w', encoding='utf-8') as f:
            for char in charset:
                f.write(f"{char}\n")

        print(f"  Charset saved: {charset_file}")
        print(f"\nMost common characters:")
        for char, count in all_chars.most_common(20):
            # Display character safely (handle Unicode issues on Windows)
            if char in [' ', '\n', '\t']:
                display_char = repr(char)
            else:
                try:
                    # Try to print, but use repr if it fails
                    display_char = char
                    _ = char.encode('utf-8')
                except:
                    display_char = repr(char)

            try:
                print(f"    {display_char}: {count}")
            except UnicodeEncodeError:
                print(f"    [U+{ord(char):04X}]: {count}")

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"  Output: {output_file}")
    print(f"  Samples: {len(df)}")
    print(f"  Charset: {len(all_chars)} unique characters")

    return df, all_chars


def main():
    parser = argparse.ArgumentParser(
        description="Convert PAGE XML to PyLaia Parquet format"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory (containing page/ subfolder with XML files)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Parquet file path"
    )
    parser.add_argument(
        "--charset",
        type=Path,
        help="Output charset file path (optional)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of pages to process (for testing)"
    )

    args = parser.parse_args()

    try:
        convert_to_parquet(
            input_dir=args.input,
            output_file=args.output,
            charset_file=args.charset,
            max_samples=args.max_samples
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
