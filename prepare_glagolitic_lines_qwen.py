"""
Prepare Glagolitic LINE-LEVEL data for Qwen3-VL training.

Extracts individual text lines from Transkribus PAGE XML exports,
similar to Ukrainian dataset preparation. This approach is better
for HTR training because:
1. HTR models are designed for line-level recognition
2. Much smaller images (0.1-0.3MP vs 16MP full pages)
3. Enables batch_size=4+ training
4. Consistent with proven Ukrainian training approach
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import csv
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count

def extract_line_transcriptions(xml_file: Path) -> List[Tuple[List[Tuple[int, int]], str]]:
    """
    Extract line-level polygons and transcriptions from PAGE XML.

    Returns:
        List of (polygon_points, transcription) tuples
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # PAGE XML namespace
        ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

        lines = []

        # Find all TextLine elements
        for text_line in root.findall('.//page:TextLine', ns):
            # Get baseline or polygon coordinates
            coords_elem = text_line.find('.//page:Coords', ns)
            if coords_elem is None:
                continue

            points_str = coords_elem.get('points')
            if not points_str:
                continue

            # Parse polygon points
            points = []
            for point_str in points_str.split():
                x, y = map(int, point_str.split(','))
                points.append((x, y))

            # Get transcription text
            unicode_elem = text_line.find('.//page:Unicode', ns)
            if unicode_elem is not None and unicode_elem.text:
                transcription = unicode_elem.text.strip()
                if transcription:  # Only include non-empty transcriptions
                    lines.append((points, transcription))

        return lines

    except Exception as e:
        print(f"  Error parsing {xml_file.name}: {e}")
        return []

def crop_line_from_polygon(image: Image.Image, polygon: List[Tuple[int, int]]) -> Image.Image:
    """
    Crop a text line from an image using polygon coordinates.
    Uses bounding box of polygon.
    """
    # Get bounding box
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add small padding
    padding = 5
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.width, x_max + padding)
    y_max = min(image.height, y_max + padding)

    # Crop line
    line_image = image.crop((x_min, y_min, x_max, y_max))

    return line_image

def process_single_page(args: Tuple[Path, Path, Path, str]) -> Tuple[int, int, List[str]]:
    """
    Process a single page (to be called in parallel).

    Returns:
        (success_count, skip_count, error_messages)
    """
    xml_file, input_dir, images_dir, split_name = args

    success_count = 0
    skip_count = 0
    errors = []

    try:
        # Extract line data from XML
        lines_data = extract_line_transcriptions(xml_file)

        if not lines_data:
            errors.append(f"{xml_file.name}: No lines found")
            return (0, 1, errors)

        # Find corresponding image file
        image_base = xml_file.stem
        image_path = None
        for ext in ['.tif', '.jpg', '.jpeg', '.png']:
            candidate = input_dir / f"{image_base}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if not image_path:
            errors.append(f"{xml_file.name}: Image file not found")
            return (0, 1, errors)

        # Open page image
        page_image = Image.open(image_path)
        if page_image.mode != 'RGB':
            page_image = page_image.convert('RGB')

        # Process each line
        valid_lines = []
        for line_idx, (polygon, transcription) in enumerate(lines_data):
            try:
                # Crop line from page
                line_image = crop_line_from_polygon(page_image, polygon)

                # Skip very small lines
                if line_image.width < 10 or line_image.height < 5:
                    skip_count += 1
                    continue

                # Save line image
                line_filename = f"{image_base}_line{line_idx:04d}.png"
                output_path = images_dir / line_filename
                line_image.save(output_path, "PNG")

                # Store relative path for CSV
                image_rel_path = f"{split_name}/images/{line_filename}"
                valid_lines.append((image_rel_path, transcription))
                success_count += 1

            except Exception as e:
                errors.append(f"{xml_file.name} line {line_idx}: {e}")
                skip_count += 1

        page_image.close()

        return (success_count, skip_count, errors)

    except Exception as e:
        errors.append(f"{xml_file.name}: {e}")
        return (0, 1, errors)

def process_dataset(input_dir: Path, output_dir: Path, split_name: str) -> List[Tuple[str, str]]:
    """
    Process a dataset (train or val) with parallel processing.

    Returns:
        List of (image_path, transcription) tuples
    """
    print(f"\nProcessing {split_name} set from {input_dir}...")

    # Create output directory
    images_dir = output_dir / split_name / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Find all XML files
    xml_files = list(input_dir.glob("**/*.xml"))
    if not xml_files:
        print(f"  Warning: No XML files found in {input_dir}")
        return []

    print(f"  Found {len(xml_files)} XML files")

    # Prepare arguments for parallel processing
    num_workers = cpu_count()
    print(f"  Using {num_workers} CPU cores for parallel processing")

    args_list = [(xml_file, input_dir, images_dir, split_name) for xml_file in xml_files]

    # Process in parallel
    with Pool(num_workers) as pool:
        results = pool.map(process_single_page, args_list)

    # Collect results
    total_success = sum(r[0] for r in results)
    total_skipped = sum(r[1] for r in results)
    all_errors = [err for r in results for err in r[2]]

    print(f"  Processed {total_success} lines successfully")
    print(f"  Skipped {total_skipped} lines (empty transcription or too small)")

    if all_errors:
        print(f"  Encountered {len(all_errors)} errors (first 10):")
        for error in all_errors[:10]:
            print(f"    {error}")

    # Re-scan images directory to build line list
    # (We need to do this because multiprocessing doesn't allow returning large lists)
    print(f"  Scanning output directory for line images...")
    valid_lines = []
    for image_file in sorted(images_dir.glob("*.png")):
        # Find corresponding transcription from XML
        # Parse filename: {page_name}_line{idx}.png
        filename = image_file.stem
        if '_line' not in filename:
            continue

        # Extract page name and line index
        parts = filename.rsplit('_line', 1)
        if len(parts) != 2:
            continue

        page_name = parts[0]
        line_idx = int(parts[1])

        # Find XML file
        xml_file = None
        for xml_candidate in xml_files:
            if xml_candidate.stem == page_name:
                xml_file = xml_candidate
                break

        if not xml_file:
            continue

        # Extract transcription for this line
        lines_data = extract_line_transcriptions(xml_file)
        if line_idx < len(lines_data):
            _, transcription = lines_data[line_idx]
            image_rel_path = f"{split_name}/images/{image_file.name}"
            valid_lines.append((image_rel_path, transcription))

    print(f"  Final dataset: {len(valid_lines)} lines")

    return valid_lines

def main():
    # Input directories (Transkribus exports)
    train_input_dir = Path("C:/Users/Achim/Documents/TrOCR/Glagolitic/Glagolitic_train")
    val_input_dir = Path("C:/Users/Achim/Documents/TrOCR/Glagolitic/Glagolitic_val")

    # Output directory
    output_dir = Path("./data/glagolitic_qwen_lines")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Glagolitic LINE-LEVEL Data Preparation for Qwen3-VL")
    print("=" * 80)
    print(f"Input directories:")
    print(f"  Training: {train_input_dir}")
    print(f"  Validation: {val_input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Check input directories
    if not train_input_dir.exists():
        print(f"ERROR: Training directory not found: {train_input_dir}")
        sys.exit(1)
    if not val_input_dir.exists():
        print(f"ERROR: Validation directory not found: {val_input_dir}")
        sys.exit(1)

    # Process training set
    train_lines = process_dataset(train_input_dir, output_dir, "train")

    # Process validation set
    val_lines = process_dataset(val_input_dir, output_dir, "val")

    # Write CSV files
    print("\nWriting CSV files...")

    train_csv = output_dir / "train.csv"
    with open(train_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'transcription'])
        writer.writerows(train_lines)
    print(f"  Training CSV: {train_csv} ({len(train_lines)} lines)")

    val_csv = output_dir / "val.csv"
    with open(val_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'transcription'])
        writer.writerows(val_lines)
    print(f"  Validation CSV: {val_csv} ({len(val_lines)} lines)")

    # Summary
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"Training lines: {len(train_lines)}")
    print(f"Validation lines: {len(val_lines)}")
    print(f"Total lines: {len(train_lines) + len(val_lines)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nNext step: Start training with:")
    print(f"  python start_glagolitic_training.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
