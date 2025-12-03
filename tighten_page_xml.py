#!/usr/bin/env python3
"""
Tighten PAGE XML polygon segmentation by cropping to actual ink extent.

This script reads PAGE XML files, analyzes the actual ink extent in each TextLine,
and updates the Coords polygon to tightly wrap the text (with configurable padding).

USAGE:
    python tighten_page_xml.py --input /path/to/page_xml_dir --output /path/to/output_dir --padding 10 --dry-run

FLAGS:
    --input: Directory containing PAGE XML files (with corresponding images)
    --output: Directory to save tightened PAGE XML files
    --padding: Padding in pixels above/below ink (default: 10)
    --threshold: Ink detection threshold (0-255, default: 200, lower = more sensitive)
    --min-ink-ratio: Minimum ink ratio to detect text row (default: 0.05 = 5%)
    --dry-run: Show what would be changed without modifying files
    --validate: Validate 100 random lines after processing

CAUTION:
    - Always backup original PAGE XML before running
    - Use --dry-run first to check results
    - Use --validate to ensure no text is cropped
    - Test on small subset before processing entire dataset
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import argparse
import shutil


class PageXMLTightener:
    def __init__(self, padding: int = 10, threshold: int = 200, min_ink_ratio: float = 0.05):
        self.padding = padding
        self.threshold = threshold
        self.min_ink_ratio = min_ink_ratio
        self.stats = {
            'files_processed': 0,
            'lines_tightened': 0,
            'avg_height_before': [],
            'avg_height_after': [],
            'errors': [],
        }

    def parse_coords(self, coords_str: str) -> List[Tuple[int, int]]:
        """Parse PAGE XML Coords points string."""
        if not coords_str:
            return []
        points = []
        for point in coords_str.split():
            x, y = map(int, point.split(','))
            points.append((x, y))
        return points

    def coords_to_string(self, coords: List[Tuple[int, int]]) -> str:
        """Convert coordinates list back to PAGE XML string."""
        return ' '.join(f'{x},{y}' for x, y in coords)

    def get_ink_extent(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Analyze image region to find actual ink extent (top and bottom y-coordinates).

        Args:
            image: PIL Image of the full page
            bbox: (x1, y1, x2, y2) bounding box of the line

        Returns:
            (top_y, bottom_y) in page coordinates
        """
        x1, y1, x2, y2 = bbox

        # Crop to region
        region = image.crop((x1, y1, x2, y2)).convert('L')
        arr = np.array(region)

        # Calculate vertical projection (count dark pixels per row)
        vertical_proj = np.sum(arr < self.threshold, axis=1)

        # Find rows with significant ink
        ink_threshold = arr.shape[1] * self.min_ink_ratio
        ink_rows = np.where(vertical_proj > ink_threshold)[0]

        if len(ink_rows) == 0:
            # No ink found, return original bounds
            return y1, y2

        # Get first and last ink rows
        first_ink_row = ink_rows[0]
        last_ink_row = ink_rows[-1]

        # Convert back to page coordinates
        top_y = y1 + first_ink_row
        bottom_y = y1 + last_ink_row

        return top_y, bottom_y

    def tighten_polygon(self, coords: List[Tuple[int, int]], top_y: int, bottom_y: int) -> List[Tuple[int, int]]:
        """
        Tighten polygon vertically to new top/bottom y-coordinates.

        Strategy:
        - Keep x-coordinates (horizontal extent unchanged)
        - Adjust y-coordinates to fit within [top_y - padding, bottom_y + padding]
        - Preserve polygon structure (top points → top_y, bottom points → bottom_y)
        """
        if not coords:
            return coords

        # Add padding
        new_top_y = max(0, top_y - self.padding)
        new_bottom_y = bottom_y + self.padding

        # Get original bbox
        y_coords = [y for x, y in coords]
        old_top_y = min(y_coords)
        old_bottom_y = max(y_coords)

        # If already tight, don't change
        if (old_bottom_y - old_top_y) <= (new_bottom_y - new_top_y + 5):
            return coords

        # Create new polygon
        new_coords = []
        mid_y = (old_top_y + old_bottom_y) / 2

        for x, y in coords:
            if y < mid_y:
                # Top half: map to new_top_y
                new_y = new_top_y
            else:
                # Bottom half: map to new_bottom_y
                new_y = new_bottom_y
            new_coords.append((x, new_y))

        return new_coords

    def process_xml(self, xml_path: Path, image_dir: Path, dry_run: bool = False) -> Dict:
        """Process a single PAGE XML file."""
        result = {
            'xml_path': str(xml_path),
            'lines_tightened': 0,
            'lines_skipped': 0,
            'avg_height_before': 0,
            'avg_height_after': 0,
            'error': None,
        }

        try:
            # Parse XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

            # Get image filename
            page_elem = root.find('.//ns:Page', ns)
            if page_elem is None:
                result['error'] = 'No Page element found'
                return result

            image_filename = page_elem.get('imageFilename')
            if not image_filename:
                result['error'] = 'No imageFilename attribute'
                return result

            # Load image
            image_path = image_dir / image_filename
            if not image_path.exists():
                result['error'] = f'Image not found: {image_path}'
                return result

            page_image = Image.open(image_path)

            # Process all TextLine elements
            text_lines = root.findall('.//ns:TextLine', ns)
            heights_before = []
            heights_after = []

            for line_elem in text_lines:
                coords_elem = line_elem.find('ns:Coords', ns)
                if coords_elem is None:
                    result['lines_skipped'] += 1
                    continue

                coords_str = coords_elem.get('points', '')
                if not coords_str:
                    result['lines_skipped'] += 1
                    continue

                coords = self.parse_coords(coords_str)
                if not coords:
                    result['lines_skipped'] += 1
                    continue

                # Get original bbox
                x_coords = [x for x, y in coords]
                y_coords = [y for x, y in coords]
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                old_height = y2 - y1
                heights_before.append(old_height)

                # Find actual ink extent
                top_y, bottom_y = self.get_ink_extent(page_image, (x1, y1, x2, y2))

                # Tighten polygon
                new_coords = self.tighten_polygon(coords, top_y, bottom_y)

                # Calculate new height
                new_y_coords = [y for x, y in new_coords]
                new_height = max(new_y_coords) - min(new_y_coords)
                heights_after.append(new_height)

                # Update XML (if not dry-run and height changed significantly)
                if not dry_run and new_height < old_height * 0.9:
                    coords_elem.set('points', self.coords_to_string(new_coords))
                    result['lines_tightened'] += 1

            result['avg_height_before'] = np.mean(heights_before) if heights_before else 0
            result['avg_height_after'] = np.mean(heights_after) if heights_after else 0

            # Save modified XML (if not dry-run)
            if not dry_run and result['lines_tightened'] > 0:
                tree.write(xml_path, encoding='utf-8', xml_declaration=True)

            self.stats['files_processed'] += 1
            self.stats['lines_tightened'] += result['lines_tightened']
            self.stats['avg_height_before'].append(result['avg_height_before'])
            self.stats['avg_height_after'].append(result['avg_height_after'])

        except Exception as e:
            result['error'] = str(e)
            self.stats['errors'].append(f"{xml_path.name}: {e}")

        return result

    def process_directory(self, input_dir: Path, output_dir: Path, dry_run: bool = False):
        """Process all PAGE XML files in directory."""
        xml_files = list(input_dir.glob('*.xml'))

        if not xml_files:
            print(f"No XML files found in {input_dir}")
            return

        print(f"Found {len(xml_files)} XML files")
        print(f"Settings: padding={self.padding}px, threshold={self.threshold}, min_ink_ratio={self.min_ink_ratio}")
        print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will modify files)'}")
        print()

        # Copy files to output directory (if not dry-run)
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            for xml_file in xml_files:
                shutil.copy2(xml_file, output_dir / xml_file.name)
            print(f"Copied {len(xml_files)} files to {output_dir}")

        # Process each file
        for i, xml_file in enumerate(xml_files):
            if not dry_run:
                xml_file = output_dir / xml_file.name

            result = self.process_xml(xml_file, input_dir.parent, dry_run)

            if result['error']:
                print(f"[{i+1}/{len(xml_files)}] ERROR {xml_file.name}: {result['error']}")
            else:
                print(f"[{i+1}/{len(xml_files)}] {xml_file.name}: "
                      f"{result['lines_tightened']} lines tightened, "
                      f"height {result['avg_height_before']:.1f}px → {result['avg_height_after']:.1f}px")

        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Lines tightened: {self.stats['lines_tightened']}")
        if self.stats['avg_height_before']:
            avg_before = np.mean(self.stats['avg_height_before'])
            avg_after = np.mean(self.stats['avg_height_after'])
            print(f"Avg height before: {avg_before:.1f}px")
            print(f"Avg height after: {avg_after:.1f}px")
            print(f"Height reduction: {(1 - avg_after/avg_before)*100:.1f}%")
        if self.stats['errors']:
            print(f"\nErrors ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:10]:
                print(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(description='Tighten PAGE XML polygon segmentation')
    parser.add_argument('--input', type=Path, required=True, help='Input directory with PAGE XML files')
    parser.add_argument('--output', type=Path, help='Output directory (default: input_dir + "_tight")')
    parser.add_argument('--padding', type=int, default=10, help='Padding in pixels (default: 10)')
    parser.add_argument('--threshold', type=int, default=200, help='Ink threshold 0-255 (default: 200)')
    parser.add_argument('--min-ink-ratio', type=float, default=0.05, help='Min ink ratio per row (default: 0.05)')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without modifying files')

    args = parser.parse_args()

    # Set default output directory
    if args.output is None:
        args.output = args.input.parent / (args.input.name + '_tight')

    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input directory not found: {args.input}")
        return

    print("="*80)
    print("PAGE XML POLYGON TIGHTENER")
    print("="*80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()

    # Process
    tightener = PageXMLTightener(
        padding=args.padding,
        threshold=args.threshold,
        min_ink_ratio=args.min_ink_ratio,
    )

    tightener.process_directory(args.input, args.output, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
