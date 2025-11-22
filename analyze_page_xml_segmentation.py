#!/usr/bin/env python3
"""
Analyze PAGE XML segmentation differences between Church Slavonic and Prosta Mova datasets.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import random
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image

def parse_coords(coords_str: str) -> List[Tuple[int, int]]:
    """Parse PAGE XML Coords points string into list of (x, y) tuples."""
    if not coords_str:
        return []
    points = []
    for point in coords_str.split():
        x, y = map(int, point.split(','))
        points.append((x, y))
    return points

def calculate_line_height_from_coords(coords: List[Tuple[int, int]]) -> int:
    """Calculate line height from polygon coordinates."""
    if not coords:
        return 0
    y_coords = [y for x, y in coords]
    return max(y_coords) - min(y_coords)

def get_baseline_from_coords(coords: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Get baseline y-range from polygon coordinates."""
    if not coords:
        return 0, 0
    y_coords = [y for x, y in coords]
    return min(y_coords), max(y_coords)

def analyze_page_xml(xml_path: Path) -> Dict:
    """Analyze a single PAGE XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Handle namespace
        ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

        # Find all TextLine elements
        text_lines = root.findall('.//ns:TextLine', ns)

        line_heights = []
        baseline_heights = []
        has_baseline = False

        for line in text_lines:
            # Get Coords polygon
            coords_elem = line.find('ns:Coords', ns)
            if coords_elem is not None:
                coords_str = coords_elem.get('points', '')
                coords = parse_coords(coords_str)
                if coords:
                    height = calculate_line_height_from_coords(coords)
                    line_heights.append(height)

            # Check if baseline exists
            baseline_elem = line.find('ns:Baseline', ns)
            if baseline_elem is not None:
                has_baseline = True
                baseline_str = baseline_elem.get('points', '')
                baseline_coords = parse_coords(baseline_str)
                if baseline_coords:
                    # Baseline is typically a horizontal line, measure y-variation
                    y_coords = [y for x, y in baseline_coords]
                    baseline_height = max(y_coords) - min(y_coords)
                    baseline_heights.append(baseline_height)

        return {
            'xml_path': str(xml_path),
            'num_lines': len(text_lines),
            'line_heights': line_heights,
            'avg_line_height': np.mean(line_heights) if line_heights else 0,
            'median_line_height': np.median(line_heights) if line_heights else 0,
            'min_line_height': min(line_heights) if line_heights else 0,
            'max_line_height': max(line_heights) if line_heights else 0,
            'has_baseline': has_baseline,
            'baseline_heights': baseline_heights,
            'avg_baseline_height': np.mean(baseline_heights) if baseline_heights else 0,
        }
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return None

def analyze_extracted_line_image(image_path: Path) -> Dict:
    """Analyze whitespace in extracted line image."""
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)

        # Calculate histogram of pixel values
        hist, bins = np.histogram(img_array, bins=256, range=(0, 256))

        # Detect whitespace (threshold at 200)
        whitespace_threshold = 200
        whitespace_pixels = np.sum(img_array > whitespace_threshold)
        total_pixels = img_array.size
        whitespace_ratio = whitespace_pixels / total_pixels

        # Calculate vertical projection (sum along width)
        vertical_proj = np.sum(img_array < whitespace_threshold, axis=1)

        # Find first and last rows with significant ink
        ink_rows = np.where(vertical_proj > img_array.shape[1] * 0.05)[0]
        if len(ink_rows) > 0:
            first_ink_row = ink_rows[0]
            last_ink_row = ink_rows[-1]
            ink_height = last_ink_row - first_ink_row + 1
            top_margin = first_ink_row
            bottom_margin = img_array.shape[0] - last_ink_row - 1
        else:
            ink_height = 0
            top_margin = 0
            bottom_margin = 0

        return {
            'image_path': str(image_path),
            'width': img_array.shape[1],
            'height': img_array.shape[0],
            'whitespace_ratio': whitespace_ratio,
            'ink_height': ink_height,
            'top_margin': top_margin,
            'bottom_margin': bottom_margin,
            'margin_ratio': (top_margin + bottom_margin) / img_array.shape[0] if img_array.shape[0] > 0 else 0,
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    # Paths
    church_slavonic_xml_dir = Path('/home/achimrabus/htr_gui/Church_Slavonic/Church_Slavonic_Train/page')
    prosta_mova_xml_dir = Path('/home/achimrabus/htr_gui/Prosta_Mova/Prosta_Mova_train/page')

    church_slavonic_images_dir = Path('/home/achimrabus/htr_gui/dhlab-slavistik/data/pylaia_church_slavonic_train/line_images')
    prosta_mova_images_dir = Path('/home/achimrabus/htr_gui/dhlab-slavistik/data/pylaia_prosta_mova_train/line_images')

    # Sample XML files
    num_samples = 10

    print("="*80)
    print("COMPARING PAGE XML SEGMENTATION: Church Slavonic vs Prosta Mova")
    print("="*80)

    # Get XML files
    church_slavonic_xmls = [f for f in church_slavonic_xml_dir.glob('*.xml') if f.is_file()]
    prosta_mova_xmls = [f for f in prosta_mova_xml_dir.glob('*.xml') if f.is_file()]

    print(f"\nFound {len(church_slavonic_xmls)} Church Slavonic XML files")
    print(f"Found {len(prosta_mova_xmls)} Prosta Mova XML files")

    # Sample random files
    random.seed(42)
    sampled_church_slavonic = random.sample(church_slavonic_xmls, min(num_samples, len(church_slavonic_xmls)))
    sampled_prosta_mova = random.sample(prosta_mova_xmls, min(num_samples, len(prosta_mova_xmls)))

    print("\n" + "="*80)
    print("ANALYZING PAGE XML FILES")
    print("="*80)

    # Analyze Church Slavonic XMLs
    print("\n--- Church Slavonic XMLs ---")
    church_slavonic_results = []
    for xml_path in sampled_church_slavonic:
        result = analyze_page_xml(xml_path)
        if result:
            church_slavonic_results.append(result)
            print(f"\n{xml_path.name}:")
            print(f"  Lines: {result['num_lines']}")
            print(f"  Avg line height: {result['avg_line_height']:.1f}px")
            print(f"  Median: {result['median_line_height']:.1f}px")
            print(f"  Range: {result['min_line_height']}-{result['max_line_height']}px")
            print(f"  Has baseline: {result['has_baseline']}")
            if result['baseline_heights']:
                print(f"  Avg baseline y-variation: {result['avg_baseline_height']:.1f}px")

    # Analyze Prosta Mova XMLs
    print("\n--- Prosta Mova XMLs ---")
    prosta_mova_results = []
    for xml_path in sampled_prosta_mova:
        result = analyze_page_xml(xml_path)
        if result:
            prosta_mova_results.append(result)
            print(f"\n{xml_path.name}:")
            print(f"  Lines: {result['num_lines']}")
            print(f"  Avg line height: {result['avg_line_height']:.1f}px")
            print(f"  Median: {result['median_line_height']:.1f}px")
            print(f"  Range: {result['min_line_height']}-{result['max_line_height']}px")
            print(f"  Has baseline: {result['has_baseline']}")
            if result['baseline_heights']:
                print(f"  Avg baseline y-variation: {result['avg_baseline_height']:.1f}px")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (from PAGE XML Coords)")
    print("="*80)

    all_church_slavonic_heights = [h for r in church_slavonic_results for h in r['line_heights']]
    all_prosta_mova_heights = [h for r in prosta_mova_results for h in r['line_heights']]

    print(f"\nChurch Slavonic (n={len(all_church_slavonic_heights)} lines):")
    print(f"  Mean: {np.mean(all_church_slavonic_heights):.1f}px")
    print(f"  Median: {np.median(all_church_slavonic_heights):.1f}px")
    print(f"  Std: {np.std(all_church_slavonic_heights):.1f}px")
    print(f"  Range: {min(all_church_slavonic_heights)}-{max(all_church_slavonic_heights)}px")

    print(f"\nProsta Mova (n={len(all_prosta_mova_heights)} lines):")
    print(f"  Mean: {np.mean(all_prosta_mova_heights):.1f}px")
    print(f"  Median: {np.median(all_prosta_mova_heights):.1f}px")
    print(f"  Std: {np.std(all_prosta_mova_heights):.1f}px")
    print(f"  Range: {min(all_prosta_mova_heights)}-{max(all_prosta_mova_heights)}px")

    print(f"\nHeight difference: {np.mean(all_prosta_mova_heights) - np.mean(all_church_slavonic_heights):.1f}px")
    print(f"Ratio: {np.mean(all_prosta_mova_heights) / np.mean(all_church_slavonic_heights):.2f}x")

    # Analyze extracted line images
    print("\n" + "="*80)
    print("ANALYZING EXTRACTED LINE IMAGES (Whitespace)")
    print("="*80)

    # Sample line images
    church_slavonic_images = list(church_slavonic_images_dir.glob('*.png'))[:20]
    prosta_mova_images = list(prosta_mova_images_dir.glob('*.png'))[:20]

    print(f"\nSampling {len(church_slavonic_images)} Church Slavonic line images")
    print(f"Sampling {len(prosta_mova_images)} Prosta Mova line images")

    church_slavonic_image_results = []
    for img_path in church_slavonic_images:
        result = analyze_extracted_line_image(img_path)
        if result:
            church_slavonic_image_results.append(result)

    prosta_mova_image_results = []
    for img_path in prosta_mova_images:
        result = analyze_extracted_line_image(img_path)
        if result:
            prosta_mova_image_results.append(result)

    print("\n--- Church Slavonic Line Images ---")
    print(f"  Avg height: {np.mean([r['height'] for r in church_slavonic_image_results]):.1f}px")
    print(f"  Avg ink height: {np.mean([r['ink_height'] for r in church_slavonic_image_results]):.1f}px")
    print(f"  Avg top margin: {np.mean([r['top_margin'] for r in church_slavonic_image_results]):.1f}px")
    print(f"  Avg bottom margin: {np.mean([r['bottom_margin'] for r in church_slavonic_image_results]):.1f}px")
    print(f"  Avg margin ratio: {np.mean([r['margin_ratio'] for r in church_slavonic_image_results]):.2%}")
    print(f"  Avg whitespace ratio: {np.mean([r['whitespace_ratio'] for r in church_slavonic_image_results]):.2%}")

    print("\n--- Prosta Mova Line Images ---")
    print(f"  Avg height: {np.mean([r['height'] for r in prosta_mova_image_results]):.1f}px")
    print(f"  Avg ink height: {np.mean([r['ink_height'] for r in prosta_mova_image_results]):.1f}px")
    print(f"  Avg top margin: {np.mean([r['top_margin'] for r in prosta_mova_image_results]):.1f}px")
    print(f"  Avg bottom margin: {np.mean([r['bottom_margin'] for r in prosta_mova_image_results]):.1f}px")
    print(f"  Avg margin ratio: {np.mean([r['margin_ratio'] for r in prosta_mova_image_results]):.2%}")
    print(f"  Avg whitespace ratio: {np.mean([r['whitespace_ratio'] for r in prosta_mova_image_results]):.2%}")

    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)

    xml_height_ratio = np.mean(all_prosta_mova_heights) / np.mean(all_church_slavonic_heights)
    extracted_height_ratio = np.mean([r['height'] for r in prosta_mova_image_results]) / np.mean([r['height'] for r in church_slavonic_image_results])

    print(f"\n1. PAGE XML Coords line height ratio: {xml_height_ratio:.2f}x")
    print(f"2. Extracted line image height ratio: {extracted_height_ratio:.2f}x")

    church_margin = np.mean([r['margin_ratio'] for r in church_slavonic_image_results])
    prosta_margin = np.mean([r['margin_ratio'] for r in prosta_mova_image_results])

    print(f"\n3. Church Slavonic margin ratio: {church_margin:.1%}")
    print(f"4. Prosta Mova margin ratio: {prosta_margin:.1%}")
    print(f"5. Difference: {(prosta_margin - church_margin):.1%}")

    if xml_height_ratio > 1.2:
        print("\n=> Prosta Mova PAGE XMLs have significantly taller line polygons")
        print("=> This suggests different Transkribus segmentation settings or baseline detection")

    if prosta_margin > church_margin * 1.2:
        print("\n=> Prosta Mova extracted images have more whitespace/margins")
        print("=> This contributes to the height difference after extraction")

if __name__ == '__main__':
    main()
