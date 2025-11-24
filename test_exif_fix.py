#!/usr/bin/env python3
"""
Quick test to verify EXIF rotation fix works correctly.

Tests that images with EXIF rotation tags are loaded with correct dimensions
that match their PAGE XML files.
"""

from pathlib import Path
from PIL import Image, ImageOps
import xml.etree.ElementTree as ET

def check_image_xml_match(image_path: Path, xml_path: Path):
    """Check if image dimensions match PAGE XML after EXIF correction."""
    
    # Load image WITHOUT EXIF correction (old buggy way)
    with Image.open(image_path) as img_no_exif:
        w_no_exif, h_no_exif = img_no_exif.size
    
    # Load image WITH EXIF correction (new fixed way)
    with Image.open(image_path) as img_with_exif:
        img_with_exif = ImageOps.exif_transpose(img_with_exif)
        w_with_exif, h_with_exif = img_with_exif.size
    
    # Parse PAGE XML dimensions
    tree = ET.parse(xml_path)
    root = tree.getroot()
    NS = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    page = root.find('.//page:Page', NS)
    
    xml_w = int(page.get('imageWidth'))
    xml_h = int(page.get('imageHeight'))
    
    print(f"\nFile: {image_path.name}")
    print(f"  PAGE XML:     {xml_w}x{xml_h}")
    print(f"  Image (old):  {w_no_exif}x{h_no_exif} {'❌ MISMATCH' if (w_no_exif != xml_w or h_no_exif != xml_h) else '✓'}")
    print(f"  Image (new):  {w_with_exif}x{h_with_exif} {'✓ MATCHES' if (w_with_exif == xml_w and h_with_exif == xml_h) else '❌ STILL WRONG'}")
    
    # Check if EXIF rotation was applied
    if w_no_exif != w_with_exif or h_no_exif != h_with_exif:
        print(f"  → EXIF rotation WAS applied (dimensions changed)")
    else:
        print(f"  → No EXIF rotation needed (dimensions unchanged)")
    
    return w_with_exif == xml_w and h_with_exif == xml_h


def main():
    """Test EXIF fix on Ukrainian validation data."""
    
    print("="*70)
    print("EXIF ROTATION FIX VERIFICATION")
    print("="*70)
    
    # Find Ukrainian validation data
    val_folder = Path("/home/achimrabus/htr_gui/Ukrainian_Data/validation_set")
    
    if not val_folder.exists():
        print(f"❌ Validation folder not found: {val_folder}")
        return
    
    # Get image-xml pairs (XML files are in page/ subdirectory)
    images = sorted(val_folder.glob("*.jpg")) + sorted(val_folder.glob("*.png")) + sorted(val_folder.glob("*.tif"))
    xml_folder = val_folder / "page"
    xml_files = sorted(xml_folder.glob("*.xml")) if xml_folder.exists() else []
    
    print(f"\nFound {len(images)} images and {len(xml_files)} XML files")
    
    # Match image-xml pairs
    pairs = []
    for img_path in images:
        xml_path = xml_folder / f"{img_path.stem}.xml"
        if xml_path.exists():
            pairs.append((img_path, xml_path))
    
    print(f"Matched {len(pairs)} image-XML pairs\n")
    
    if not pairs:
        print("❌ No image-XML pairs found")
        return
    
    # Test first 5 pairs (those that had mismatch errors)
    test_count = min(5, len(pairs))
    matches = 0
    
    for img_path, xml_path in pairs[:test_count]:
        if check_image_xml_match(img_path, xml_path):
            matches += 1
    
    print("\n" + "="*70)
    print(f"SUMMARY: {matches}/{test_count} images match PAGE XML after EXIF fix")
    
    if matches == test_count:
        print("✓ ALL TESTS PASSED - EXIF fix working correctly!")
    else:
        print(f"⚠️  {test_count - matches} images still have dimension mismatches")
    
    print("="*70)


if __name__ == "__main__":
    main()
