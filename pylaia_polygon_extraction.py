"""
Extract text lines from PAGE XML using polygons and baselines.

This script properly extracts text lines from PAGE XML format used by Transkribus,
applying baseline-based deskewing and polygon-based cropping for optimal quality.

Usage:
    python extract_lines_from_pagexml.py --input data/ukrainian_pagexml --output data/pylaia_ukrainian_from_pagexml
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
import logging
from collections import Counter
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PAGEXMLLineExtractor:
    """
    Extract text lines from PAGE XML format.
    
    Implements Transkribus-style extraction:
    - Polygon-based line regions
    - Baseline-based deskewing
    - Height normalization
    """
    
    # PAGE XML namespace
    NAMESPACES = {
        'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
        'pc2019': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'
    }
    
    def __init__(
        self,
        target_height: int = 64,
        min_line_width: int = 20,
        min_line_height: int = 10,
        padding: int = 5,
        deskew: bool = True
    ):
        """
        Args:
            target_height: Target height for normalized lines (pixels)
            min_line_width: Minimum line width to keep (pixels)
            min_line_height: Minimum line height to keep (pixels)
            padding: Padding around text line (pixels)
            deskew: Apply baseline-based deskewing
        """
        self.target_height = target_height
        self.min_line_width = min_line_width
        self.min_line_height = min_line_height
        self.padding = padding
        self.deskew = deskew
    
    def parse_points(self, points_str: str) -> List[Tuple[int, int]]:
        """
        Parse PAGE XML points string to list of (x, y) tuples.
        
        Args:
            points_str: Space-separated "x1,y1 x2,y2 ..." string
        
        Returns:
            List of (x, y) coordinate tuples
        """
        if not points_str:
            return []
        
        points = []
        for point in points_str.strip().split():
            try:
                x, y = point.split(',')
                points.append((int(float(x)), int(float(y))))
            except (ValueError, IndexError):
                logger.warning(f"Invalid point format: {point}")
                continue
        
        return points
    
    def calculate_baseline_angle(self, baseline_points: List[Tuple[int, int]]) -> float:
        """
        Calculate angle of baseline using linear regression.
        
        Args:
            baseline_points: List of (x, y) baseline coordinates
        
        Returns:
            Angle in degrees (positive = counterclockwise)
        """
        if len(baseline_points) < 2:
            return 0.0
        
        # Convert to numpy array
        points = np.array(baseline_points, dtype=np.float32)
        
        # Linear regression to find baseline angle
        # y = mx + b -> angle = arctan(m)
        x = points[:, 0]
        y = points[:, 1]
        
        if len(x) < 2:
            return 0.0
        
        # Fit line using least squares
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Calculate angle in degrees
        angle_rad = np.arctan(m)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def get_polygon_bbox(self, polygon: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """
        Get bounding box of polygon.
        
        Returns:
            (x_min, y_min, x_max, y_max)
        """
        if not polygon:
            return (0, 0, 0, 0)
        
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        
        return (min(xs), min(ys), max(xs), max(ys))
    
    def rotate_image(
        self,
        image: np.ndarray,
        angle: float,
        center: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Rotate image around center point.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counterclockwise)
            center: Rotation center (default: image center)
        
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        
        if center is None:
            center = (w / 2, h / 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix for new dimensions
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Rotate image
        rotated = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def rotate_points(
        self,
        points: List[Tuple[int, int]],
        angle: float,
        center: Tuple[float, float]
    ) -> List[Tuple[int, int]]:
        """
        Rotate points around center.
        
        Args:
            points: List of (x, y) points
            angle: Rotation angle in degrees
            center: Rotation center (x, y)
        
        Returns:
            Rotated points
        """
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        cx, cy = center
        rotated_points = []
        
        for x, y in points:
            # Translate to origin
            x_temp = x - cx
            y_temp = y - cy
            
            # Rotate
            x_rot = x_temp * cos_a - y_temp * sin_a
            y_rot = x_temp * sin_a + y_temp * cos_a
            
            # Translate back
            x_new = x_rot + cx
            y_new = y_rot + cy
            
            rotated_points.append((int(x_new), int(y_new)))
        
        return rotated_points
    
    def extract_line_polygon(
        self,
        image: np.ndarray,
        polygon: List[Tuple[int, int]],
        baseline: List[Tuple[int, int]]
    ) -> Optional[np.ndarray]:
        """
        Extract and deskew text line using polygon and baseline.
        
        Args:
            image: Full page image
            polygon: Polygon coordinates surrounding text line
            baseline: Baseline coordinates
        
        Returns:
            Extracted and deskewed line image, or None if invalid
        """
        if not polygon or len(polygon) < 3:
            return None
        
        # Get bounding box
        x_min, y_min, x_max, y_max = self.get_polygon_bbox(polygon)
        width = x_max - x_min
        height = y_max - y_min
        
        # Filter too small regions
        if width < self.min_line_width or height < self.min_line_height:
            return None
        
        # Add padding
        x_min = max(0, x_min - self.padding)
        y_min = max(0, y_min - self.padding)
        x_max = min(image.shape[1], x_max + self.padding)
        y_max = min(image.shape[0], y_max + self.padding)
        
        # Crop region
        cropped = image[y_min:y_max, x_min:x_max].copy()
        
        if cropped.size == 0:
            return None
        
        # Apply deskewing if requested
        if self.deskew and baseline and len(baseline) >= 2:
            # Calculate baseline angle
            angle = self.calculate_baseline_angle(baseline)
            
            # Only deskew if angle is significant (> 0.5 degrees)
            if abs(angle) > 0.5:
                # Adjust baseline coordinates to cropped image
                adjusted_baseline = [(x - x_min, y - y_min) for x, y in baseline]
                
                # Calculate center of baseline
                baseline_center_x = np.mean([p[0] for p in adjusted_baseline])
                baseline_center_y = np.mean([p[1] for p in adjusted_baseline])
                
                # Rotate image
                cropped = self.rotate_image(
                    cropped,
                    -angle,  # Negative to straighten
                    center=(baseline_center_x, baseline_center_y)
                )
        
        # Create mask from polygon
        # Adjust polygon coordinates to cropped region
        adjusted_polygon = [(x - x_min, y - y_min) for x, y in polygon]
        
        # Apply polygon mask
        mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(adjusted_polygon, dtype=np.int32)], 255)
        
        # Apply mask to image
        if len(cropped.shape) == 3:
            masked = cv2.bitwise_and(cropped, cropped, mask=mask)
        else:
            masked = cv2.bitwise_and(cropped, cropped, mask=mask)
        
        # Crop to content (remove excess padding)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None
        
        x, y, w, h = cv2.boundingRect(coords)
        final_crop = masked[y:y+h, x:x+w]
        
        if final_crop.size == 0 or final_crop.shape[0] < 5 or final_crop.shape[1] < 5:
            return None
        
        return final_crop
    
    def extract_lines_from_page(
        self,
        image_path: Path,
        xml_path: Path
    ) -> List[Tuple[np.ndarray, str, str]]:
        """
        Extract all text lines from a page.
        
        Args:
            image_path: Path to page image
            xml_path: Path to PAGE XML file
        
        Returns:
            List of (line_image, text, line_id) tuples
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        # Parse XML
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Could not parse XML {xml_path}: {e}")
            return []
        
        # Try different namespaces
        namespace = None
        for ns_key, ns_uri in self.NAMESPACES.items():
            if root.tag.startswith('{' + ns_uri):
                namespace = {'ns': ns_uri}
                break
        
        if namespace is None:
            # Try without namespace
            namespace = {}
            ns_prefix = ''
        else:
            ns_prefix = 'ns:'
        
        # Find all TextLine elements
        lines_data = []
        
        # Search in TextRegion -> TextLine
        if namespace:
            text_lines = root.findall(f'.//{{{namespace["ns"]}}}TextLine')
        else:
            text_lines = root.findall('.//TextLine')
        
        for text_line in text_lines:
            # Get line ID
            line_id = text_line.get('id', 'unknown')
            
            # Get text content
            if namespace:
                text_equiv = text_line.find(f'{{{namespace["ns"]}}}TextEquiv')
                if text_equiv is not None:
                    unicode_elem = text_equiv.find(f'{{{namespace["ns"]}}}Unicode')
                    text = unicode_elem.text if unicode_elem is not None and unicode_elem.text else ''
                else:
                    text = ''
            else:
                text_equiv = text_line.find('TextEquiv')
                if text_equiv is not None:
                    unicode_elem = text_equiv.find('Unicode')
                    text = unicode_elem.text if unicode_elem is not None and unicode_elem.text else ''
                else:
                    text = ''
            
            text = text.strip() if text else ''
            
            # Skip empty lines
            if not text:
                continue
            
            # Get Coords (polygon)
            if namespace:
                coords_elem = text_line.find(f'{{{namespace["ns"]}}}Coords')
            else:
                coords_elem = text_line.find('Coords')
            
            if coords_elem is None:
                logger.warning(f"No Coords found for line {line_id}")
                continue
            
            points_str = coords_elem.get('points', '')
            polygon = self.parse_points(points_str)
            
            if not polygon:
                logger.warning(f"Empty polygon for line {line_id}")
                continue
            
            # Get Baseline
            if namespace:
                baseline_elem = text_line.find(f'{{{namespace["ns"]}}}Baseline')
            else:
                baseline_elem = text_line.find('Baseline')
            
            if baseline_elem is not None:
                baseline_str = baseline_elem.get('points', '')
                baseline = self.parse_points(baseline_str)
            else:
                baseline = []
            
            # Extract line image
            line_image = self.extract_line_polygon(gray_image, polygon, baseline)
            
            if line_image is not None:
                lines_data.append((line_image, text, line_id))
        
        logger.info(f"Extracted {len(lines_data)} lines from {image_path.name}")
        return lines_data
    
    def process_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        image_subdir: str = 'images',
        xml_subdir: str = 'page'
    ):
        """
        Process entire dataset of PAGE XML files.
        
        Args:
            input_dir: Input directory with images/ and page/ subdirectories
            output_dir: Output directory for PyLaia format
            image_subdir: Name of images subdirectory
            xml_subdir: Name of PAGE XML subdirectory
        """
        images_dir = input_dir / image_subdir
        xml_dir = input_dir / xml_subdir
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        if not xml_dir.exists():
            raise FileNotFoundError(f"XML directory not found: {xml_dir}")
        
        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        output_images_dir = output_dir / 'images'
        output_gt_dir = output_dir / 'gt'
        output_images_dir.mkdir(exist_ok=True)
        output_gt_dir.mkdir(exist_ok=True)
        
        # Find all XML files
        xml_files = list(xml_dir.glob('*.xml'))
        logger.info(f"Found {len(xml_files)} XML files")
        
        all_sample_ids = []
        all_texts = []
        total_lines = 0
        skipped_pages = 0
        
        # Process each page
        for xml_path in tqdm(xml_files, desc="Processing pages"):
            # Find corresponding image
            image_name = xml_path.stem
            image_path = None
            
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                candidate = images_dir / f"{image_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path is None:
                logger.warning(f"Image not found for {xml_path.name}")
                skipped_pages += 1
                continue
            
            # Extract lines from page
            try:
                lines_data = self.extract_lines_from_page(image_path, xml_path)
            except Exception as e:
                logger.error(f"Error processing {xml_path.name}: {e}")
                skipped_pages += 1
                continue
            
            # Save each line
            for line_image, text, line_id in lines_data:
                # Generate unique sample ID
                sample_id = f"{image_name}_{line_id}"
                
                # Save image
                output_image_path = output_images_dir / f"{sample_id}.png"
                cv2.imwrite(str(output_image_path), line_image)
                
                # Save ground truth
                output_gt_path = output_gt_dir / f"{sample_id}.txt"
                with open(output_gt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                all_sample_ids.append(sample_id)
                all_texts.append(text)
                total_lines += 1
        
        logger.info(f"\nProcessed {len(xml_files) - skipped_pages} pages")
        logger.info(f"Skipped {skipped_pages} pages")
        logger.info(f"Extracted {total_lines} text lines")
        
        if total_lines == 0:
            logger.error("No lines extracted! Check your input paths and XML format.")
            return
        
        # Write lines.txt
        lines_file = output_dir / 'lines.txt'
        with open(lines_file, 'w', encoding='utf-8') as f:
            for sample_id in all_sample_ids:
                f.write(f"{sample_id}\n")
        
        logger.info(f"Wrote sample IDs to {lines_file}")
        
        # Collect symbols
        char_counter = Counter()
        for text in all_texts:
            for char in text:
                if char == ' ':
                    char_counter['<SPACE>'] += 1
                else:
                    char_counter[char] += 1
        
        symbols = [char for char, _ in char_counter.most_common()]
        
        # Write symbols.txt
        symbols_file = output_dir / 'symbols.txt'
        with open(symbols_file, 'w', encoding='utf-8') as f:
            for symbol in symbols:
                f.write(f"{symbol}\n")
        
        logger.info(f"Wrote {len(symbols)} symbols to {symbols_file}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Input directory:   {input_dir}")
        logger.info(f"Output directory:  {output_dir}")
        logger.info(f"Pages processed:   {len(xml_files) - skipped_pages}")
        logger.info(f"Total lines:       {total_lines}")
        logger.info(f"Vocabulary size:   {len(symbols)}")
        logger.info(f"Images saved to:   {output_images_dir}")
        logger.info(f"GT saved to:       {output_gt_dir}")
        
        # Sample texts
        logger.info("\nSample texts:")
        for i, text in enumerate(all_texts[:5], 1):
            logger.info(f"  {i}: {text[:80]}{'...' if len(text) > 80 else ''}")
        
        # Character distribution
        logger.info("\nTop 20 characters:")
        for char, count in char_counter.most_common(20):
            display_char = '<SPACE>' if char == ' ' else char
            logger.info(f"  '{display_char}': {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract text lines from PAGE XML using polygons and baselines"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing images/ and page/ subdirectories'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for PyLaia format'
    )
    parser.add_argument(
        '--image-subdir',
        type=str,
        default='images',
        help='Name of images subdirectory (default: images)'
    )
    parser.add_argument(
        '--xml-subdir',
        type=str,
        default='page',
        help='Name of PAGE XML subdirectory (default: page)'
    )
    parser.add_argument(
        '--target-height',
        type=int,
        default=64,
        help='Target height for extracted lines (default: 64)'
    )
    parser.add_argument(
        '--no-deskew',
        action='store_true',
        help='Disable baseline-based deskewing'
    )
    parser.add_argument(
        '--min-width',
        type=int,
        default=20,
        help='Minimum line width in pixels (default: 20)'
    )
    parser.add_argument(
        '--min-height',
        type=int,
        default=10,
        help='Minimum line height in pixels (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = PAGEXMLLineExtractor(
        target_height=args.target_height,
        min_line_width=args.min_width,
        min_line_height=args.min_height,
        deskew=not args.no_deskew
    )
    
    # Process dataset
    extractor.process_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        image_subdir=args.image_subdir,
        xml_subdir=args.xml_subdir
    )
    
    logger.info("\n✓ Extraction complete!")
    logger.info("Ready for PyLaia training with proper line extraction.")


if __name__ == '__main__':
    main()