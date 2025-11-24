#!/usr/bin/env python3
"""
Convert PAGE XML files to plain text files.

Extracts transcriptions from PAGE XML files following reading order,
preserving line structure with empty lines for missing text.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# PAGE XML namespace
NAMESPACES = {
    'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
}


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def parse_reading_order(root: ET.Element) -> Optional[List[str]]:
    """
    Extract reading order from PAGE XML.
    
    Returns:
        List of region IDs in reading order, or None if no reading order defined
    """
    reading_order_elem = root.find('.//page:ReadingOrder', NAMESPACES)
    if reading_order_elem is None:
        return None
    
    # Look for OrderedGroup or UnorderedGroup
    ordered_group = reading_order_elem.find('.//page:OrderedGroup', NAMESPACES)
    if ordered_group is None:
        return None
    
    # Extract region refs in order
    region_refs = ordered_group.findall('.//page:RegionRefIndexed', NAMESPACES)
    if not region_refs:
        return None
    
    # Sort by index attribute and extract regionRef
    try:
        sorted_refs = sorted(region_refs, key=lambda x: int(x.get('index', 0)))
        return [ref.get('regionRef') for ref in sorted_refs if ref.get('regionRef')]
    except (ValueError, TypeError) as e:
        logging.warning(f"Error parsing reading order indices: {e}")
        return None


def get_text_regions_by_reading_order(root: ET.Element, reading_order: List[str]) -> List[ET.Element]:
    """
    Get TextRegion elements sorted by reading order.
    
    Args:
        root: PAGE XML root element
        reading_order: List of region IDs in order
        
    Returns:
        List of TextRegion elements in reading order
    """
    # Build dict of region ID -> element
    regions = {}
    for region in root.findall('.//page:TextRegion', NAMESPACES):
        region_id = region.get('id')
        if region_id:
            regions[region_id] = region
    
    # Return regions in reading order
    ordered_regions = []
    for region_id in reading_order:
        if region_id in regions:
            ordered_regions.append(regions[region_id])
        else:
            logging.warning(f"Region ID {region_id} in reading order not found in document")
    
    # Add any regions not in reading order at the end
    for region_id, region in regions.items():
        if region_id not in reading_order:
            logging.debug(f"Region {region_id} not in reading order, appending at end")
            ordered_regions.append(region)
    
    return ordered_regions


def get_text_regions_spatial(root: ET.Element) -> List[ET.Element]:
    """
    Get TextRegion elements sorted spatially (top to bottom).
    
    Args:
        root: PAGE XML root element
        
    Returns:
        List of TextRegion elements sorted by vertical position
    """
    regions = root.findall('.//page:TextRegion', NAMESPACES)
    
    # Extract Y coordinate from first point in Coords
    def get_y_coord(region: ET.Element) -> float:
        coords = region.find('.//page:Coords', NAMESPACES)
        if coords is not None:
            points = coords.get('points', '')
            if points:
                first_point = points.split()[0]
                try:
                    x, y = first_point.split(',')
                    return float(y)
                except (ValueError, IndexError):
                    pass
        return float('inf')
    
    return sorted(regions, key=get_y_coord)


def extract_text_from_region(region: ET.Element) -> List[str]:
    """
    Extract text lines from a TextRegion.
    
    Args:
        region: TextRegion element
        
    Returns:
        List of text strings (empty string for lines without text)
    """
    lines = []
    text_lines = region.findall('.//page:TextLine', NAMESPACES)
    
    for text_line in text_lines:
        # Look for TextEquiv/Unicode element
        unicode_elem = text_line.find('.//page:TextEquiv/page:Unicode', NAMESPACES)
        if unicode_elem is not None and unicode_elem.text:
            lines.append(unicode_elem.text)
        else:
            # Empty line for TextLine without text
            lines.append('')
    
    return lines


def convert_pagexml_to_text(xml_path: Path, output_path: Path) -> bool:
    """
    Convert a single PAGE XML file to plain text.
    
    Args:
        xml_path: Path to input PAGE XML file
        output_path: Path to output text file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get reading order
        reading_order = parse_reading_order(root)
        
        # Get regions in order
        if reading_order:
            logging.debug(f"{xml_path.name}: Using reading order with {len(reading_order)} regions")
            regions = get_text_regions_by_reading_order(root, reading_order)
        else:
            logging.debug(f"{xml_path.name}: No reading order found, using spatial ordering")
            regions = get_text_regions_spatial(root)
        
        if not regions:
            logging.warning(f"{xml_path.name}: No TextRegion elements found")
            # Create empty file
            output_path.write_text('', encoding='utf-8')
            return True
        
        # Extract text from all regions
        all_lines = []
        for region in regions:
            lines = extract_text_from_region(region)
            all_lines.extend(lines)
        
        # Write to output file (Unix line endings)
        output_text = '\n'.join(all_lines) + '\n' if all_lines else ''
        output_path.write_text(output_text, encoding='utf-8', newline='\n')
        
        logging.info(f"✓ {xml_path.name} → {output_path.name} ({len(all_lines)} lines)")
        return True
        
    except ET.ParseError as e:
        logging.error(f"{xml_path.name}: XML parsing error: {e}")
        return False
    except Exception as e:
        logging.error(f"{xml_path.name}: Unexpected error: {e}")
        return False


def batch_convert(input_dir: Path, output_dir: Path, verbose: bool = False) -> Tuple[int, int]:
    """
    Batch convert PAGE XML files to text.
    
    Args:
        input_dir: Directory containing PAGE XML files (or page/ subdirectory)
        output_dir: Directory for output text files
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    # Check for page/ subdirectory (Transkribus export format)
    page_subdir = input_dir / 'page'
    if page_subdir.is_dir():
        logging.info(f"Found page/ subdirectory, using: {page_subdir}")
        xml_dir = page_subdir
    else:
        xml_dir = input_dir
    
    # Find all XML files
    xml_files = sorted(xml_dir.glob('*.xml'))
    
    if not xml_files:
        logging.error(f"No XML files found in {xml_dir}")
        return 0, 0
    
    logging.info(f"Found {len(xml_files)} XML file(s)")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    success_count = 0
    fail_count = 0
    
    for xml_path in xml_files:
        # Output file has same basename with .txt extension
        output_path = output_dir / (xml_path.stem + '.txt')
        
        if convert_pagexml_to_text(xml_path, output_path):
            success_count += 1
        else:
            fail_count += 1
    
    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(
        description='Convert PAGE XML files to plain text files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Convert single file
  %(prog)s input.xml output.txt
  
  # Batch convert folder (auto-detects page/ subdirectory)
  %(prog)s input_folder/ output_folder/
  
  # Batch convert with verbose logging
  %(prog)s input_folder/ output_folder/ -v
        '''
    )
    
    parser.add_argument(
        'input',
        type=Path,
        help='Input PAGE XML file or directory'
    )
    
    parser.add_argument(
        'output',
        type=Path,
        help='Output text file or directory'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Single file mode
    if args.input.is_file():
        if not args.input.suffix.lower() == '.xml':
            logging.error(f"Input file must be XML: {args.input}")
            return 1
        
        # Ensure output is a file path
        if args.output.is_dir():
            args.output = args.output / (args.input.stem + '.txt')
        
        success = convert_pagexml_to_text(args.input, args.output)
        return 0 if success else 1
    
    # Batch mode
    elif args.input.is_dir():
        if not args.output.suffix:  # Output should be a directory
            success_count, fail_count = batch_convert(args.input, args.output, args.verbose)
            
            logging.info(f"\nResults: {success_count} successful, {fail_count} failed")
            return 0 if fail_count == 0 else 1
        else:
            logging.error("For batch conversion, output must be a directory")
            return 1
    
    else:
        logging.error(f"Input not found: {args.input}")
        return 1


if __name__ == '__main__':
    exit(main())
