"""
PAGE XML Exporter

Exports line segmentation and transcription data to PAGE XML format.
Compatible with party and other PAGE XML processors.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from inference_page import LineSegment


class PageXMLExporter:
    """Export line segmentation data to PAGE XML format."""

    # PAGE XML namespace
    NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

    def __init__(self, image_path: str, image_width: int, image_height: int):
        """
        Initialize PAGE XML exporter.

        Args:
            image_path: Path to the page image file
            image_width: Width of the page image in pixels
            image_height: Height of the page image in pixels
        """
        self.image_path = Path(image_path)
        self.image_width = image_width
        self.image_height = image_height

    def export(self, segments: List[LineSegment], output_path: str,
               creator: str = "TrOCR-GUI", comments: Optional[str] = None) -> None:
        """
        Export line segments to PAGE XML file.

        Args:
            segments: List of LineSegment objects
            output_path: Path where to save the PAGE XML file
            creator: Software/tool that created this PAGE XML
            comments: Optional comments about the document
        """
        # Register namespace
        ET.register_namespace('', self.NAMESPACE)

        # Create root element
        root = ET.Element('PcGts', {
            'xmlns': self.NAMESPACE,
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': f'{self.NAMESPACE} http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd',
            'pcGtsId': f'pc-{self.image_path.stem}'
        })

        # Metadata
        metadata = ET.SubElement(root, 'Metadata')
        ET.SubElement(metadata, 'Creator').text = creator
        ET.SubElement(metadata, 'Created').text = datetime.now().isoformat()
        ET.SubElement(metadata, 'LastChange').text = datetime.now().isoformat()
        if comments:
            ET.SubElement(metadata, 'Comments').text = comments

        # Page
        page = ET.SubElement(root, 'Page', {
            'imageFilename': str(self.image_path.name),
            'imageWidth': str(self.image_width),
            'imageHeight': str(self.image_height)
        })

        # Reading order (simple top-to-bottom)
        reading_order = ET.SubElement(page, 'ReadingOrder')
        ordered_group = ET.SubElement(reading_order, 'OrderedGroup', {
            'id': 'ro_1',
            'caption': 'Regions reading order'
        })

        # Text region containing all lines
        text_region = ET.SubElement(page, 'TextRegion', {
            'id': 'region_1',
            'type': 'paragraph'
        })

        # Add region to reading order
        ET.SubElement(ordered_group, 'RegionRefIndexed', {
            'index': '0',
            'regionRef': 'region_1'
        })

        # Region coordinates (bounding box of all lines)
        if segments:
            x1 = min(seg.bbox[0] for seg in segments)
            y1 = min(seg.bbox[1] for seg in segments)
            x2 = max(seg.bbox[2] for seg in segments)
            y2 = max(seg.bbox[3] for seg in segments)

            region_coords = ET.SubElement(text_region, 'Coords')
            region_coords.set('points', f'{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}')

        # Add each line
        for idx, segment in enumerate(segments):
            # Build custom attributes only if confidence exists
            line_attrs = {'id': f'line_{idx + 1}'}
            if hasattr(segment, 'confidence') and segment.confidence is not None:
                line_attrs['custom'] = f'readingOrder {{index:{idx};}}'

            line = ET.SubElement(text_region, 'TextLine', line_attrs)

            # Line coordinates
            coords = ET.SubElement(line, 'Coords')
            if hasattr(segment, 'coords') and segment.coords:
                # Use polygon coordinates if available
                points_str = ' '.join(f'{x},{y}' for x, y in segment.coords)
            else:
                # Use bounding box
                x1, y1, x2, y2 = segment.bbox
                points_str = f'{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}'
            coords.set('points', points_str)

            # Baseline (approximate from bbox)
            baseline = ET.SubElement(line, 'Baseline')
            x1, y1, x2, y2 = segment.bbox
            baseline_y = y2 - 5  # Approximate baseline near bottom
            baseline.set('points', f'{x1},{baseline_y} {x2},{baseline_y}')

            # Text content (only if text attribute exists and is not empty)
            if hasattr(segment, 'text') and segment.text:
                # Add confidence if available
                conf_value = '1.0'
                if hasattr(segment, 'confidence') and segment.confidence is not None:
                    conf_value = str(segment.confidence)

                text_equiv = ET.SubElement(line, 'TextEquiv', {'conf': conf_value})
                unicode_elem = ET.SubElement(text_equiv, 'Unicode')
                unicode_elem.text = segment.text

        # Write to file with pretty formatting
        xml_str = ET.tostring(root, encoding='utf-8', method='xml')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent='  ', encoding='utf-8')

        with open(output_path, 'wb') as f:
            f.write(pretty_xml)

    @staticmethod
    def quick_export(image_path: str, segments: List[LineSegment],
                     output_path: Optional[str] = None) -> str:
        """
        Quick export helper that automatically determines output path and image dimensions.

        Args:
            image_path: Path to the page image
            segments: List of LineSegment objects
            output_path: Optional output path (default: same as image with .xml extension)

        Returns:
            Path to the exported PAGE XML file
        """
        from PIL import Image

        # Load image to get dimensions
        img = Image.open(image_path)
        width, height = img.size

        # Determine output path
        if output_path is None:
            output_path = Path(image_path).with_suffix('.xml')

        # Export
        exporter = PageXMLExporter(image_path, width, height)
        exporter.export(segments, str(output_path))

        return str(output_path)


if __name__ == "__main__":
    # Example usage
    from PIL import Image

    # Create a dummy segment for testing
    dummy_img = Image.new('L', (100, 30))
    dummy_segment = LineSegment(
        image=dummy_img,
        bbox=(10, 10, 200, 40),
        text="Example text",
        confidence=0.95
    )

    exporter = PageXMLExporter("test_page.jpg", 800, 1200)
    exporter.export([dummy_segment], "test_output.xml",
                   creator="PAGE XML Exporter Test",
                   comments="This is a test export")

    print("Test PAGE XML created: test_output.xml")
