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

    def _make_root(self, creator: str, comments: Optional[str]) -> tuple:
        """Build root PcGts element with Metadata and Page. Returns (root, page)."""
        ET.register_namespace('', self.NAMESPACE)
        root = ET.Element('PcGts', {
            'xmlns': self.NAMESPACE,
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': (
                f'{self.NAMESPACE} '
                'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd'
            ),
            'pcGtsId': f'pc-{self.image_path.stem}'
        })
        metadata = ET.SubElement(root, 'Metadata')
        ET.SubElement(metadata, 'Creator').text = creator
        ET.SubElement(metadata, 'Created').text = datetime.now().isoformat()
        ET.SubElement(metadata, 'LastChange').text = datetime.now().isoformat()
        if comments:
            ET.SubElement(metadata, 'Comments').text = comments
        page = ET.SubElement(root, 'Page', {
            'imageFilename': str(self.image_path.name),
            'imageWidth': str(self.image_width),
            'imageHeight': str(self.image_height)
        })
        return root, page

    @staticmethod
    def _write_xml(root: ET.Element, output_path: str) -> None:
        xml_str = ET.tostring(root, encoding='utf-8', method='xml')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent='  ', encoding='utf-8')
        with open(output_path, 'wb') as f:
            f.write(pretty_xml)

    @staticmethod
    def _baseline_points(segment) -> str:
        """Return PAGE XML baseline points string for a segment."""
        if hasattr(segment, 'baseline') and segment.baseline:
            return ' '.join(f'{x},{y}' for x, y in segment.baseline)
        x1, y1, x2, y2 = segment.bbox
        bl_y = y2 - 5
        return f'{x1},{bl_y} {x2},{bl_y}'

    @staticmethod
    def _coords_points(segment) -> str:
        """Return PAGE XML coords points string for a segment."""
        if hasattr(segment, 'coords') and segment.coords:
            return ' '.join(f'{x},{y}' for x, y in segment.coords)
        x1, y1, x2, y2 = segment.bbox
        return f'{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}'

    def _add_text_line(self, parent: ET.Element, line_id: str, segment,
                       text: Optional[str], line_idx: int) -> None:
        """Add a TextLine element to parent with coords, baseline and optional text."""
        line_elem = ET.SubElement(parent, 'TextLine', {
            'id': line_id,
            'custom': f'readingOrder {{index:{line_idx};}}'
        })
        ET.SubElement(line_elem, 'Coords').set('points', self._coords_points(segment))
        ET.SubElement(line_elem, 'Baseline').set('points', self._baseline_points(segment))
        if text:
            conf = '1.0'
            if hasattr(segment, 'confidence') and segment.confidence is not None:
                conf = str(segment.confidence)
            text_equiv = ET.SubElement(line_elem, 'TextEquiv', {'conf': conf})
            ET.SubElement(text_equiv, 'Unicode').text = text

    def export(self, segments: List[LineSegment], output_path: str,
               creator: str = "TrOCR-GUI", comments: Optional[str] = None) -> None:
        """
        Export line segments to PAGE XML (single TextRegion, no region info).

        Args:
            segments: List of LineSegment objects (may carry .text attribute)
            output_path: Path where to save the PAGE XML file
            creator: Software/tool that created this PAGE XML
            comments: Optional comments about the document
        """
        root, page = self._make_root(creator, comments)

        # Reading order
        reading_order = ET.SubElement(page, 'ReadingOrder')
        ordered_group = ET.SubElement(reading_order, 'OrderedGroup', {
            'id': 'ro_1',
            'caption': 'Regions reading order'
        })
        ET.SubElement(ordered_group, 'RegionRefIndexed', {
            'index': '0',
            'regionRef': 'region_1'
        })

        # Single text region spanning all lines
        text_region = ET.SubElement(page, 'TextRegion', {
            'id': 'region_1',
            'type': 'paragraph',
            'custom': 'readingOrder {index:0;}'
        })
        if segments:
            x1 = min(seg.bbox[0] for seg in segments)
            y1 = min(seg.bbox[1] for seg in segments)
            x2 = max(seg.bbox[2] for seg in segments)
            y2 = max(seg.bbox[3] for seg in segments)
            ET.SubElement(text_region, 'Coords').set(
                'points', f'{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}'
            )

        for idx, segment in enumerate(segments):
            text = getattr(segment, 'text', None) or None
            self._add_text_line(text_region, f'line_{idx + 1}', segment, text, idx)

        self._write_xml(root, output_path)

    def export_with_regions(
        self,
        regions,
        lines,
        output_path: str,
        transcriptions: Optional[List[str]] = None,
        creator: str = "TrOCR-GUI",
        comments: Optional[str] = None,
    ) -> None:
        """
        Export with proper multi-region PAGE XML structure.

        Creates one TextRegion per detected region (e.g. columns, marginalia),
        with TextLines nested inside their region and actual baseline polylines.
        ReadingOrder lists regions left-to-right and lines top-to-bottom within
        each region, matching how blla / column clustering ordered them.

        Args:
            regions:         List of SegRegion objects (duck-typed: .id, .line_ids,
                             .bbox, optional .polygon).
            lines:           Flat list of LineSegment objects, already ordered by
                             region (region[0]'s lines first, then region[1]'s, …).
                             The count of lines per region is len(region.line_ids).
            output_path:     Where to write the PAGE XML file.
            transcriptions:  Optional list of text strings, parallel to *lines*.
                             Pass self.transcriptions from the GUI when available.
            creator:         Creator string for Metadata.
            comments:        Optional comments string for Metadata.
        """
        root, page = self._make_root(creator, comments)

        # ReadingOrder — one RegionRefIndexed per region
        reading_order = ET.SubElement(page, 'ReadingOrder')
        ordered_group = ET.SubElement(reading_order, 'OrderedGroup', {
            'id': 'ro_1',
            'caption': 'Regions reading order'
        })
        for ri, region in enumerate(regions):
            ET.SubElement(ordered_group, 'RegionRefIndexed', {
                'index': str(ri),
                'regionRef': region.id
            })

        # TextRegions — one per region, lines nested inside
        line_offset = 0
        for ri, region in enumerate(regions):
            n = len(region.line_ids) if hasattr(region, 'line_ids') else 0
            region_lines = lines[line_offset:line_offset + n]
            line_offset += n

            text_region = ET.SubElement(page, 'TextRegion', {
                'id': region.id,
                'type': 'paragraph',
                'custom': f'readingOrder {{index:{ri};}}'
            })

            # Region polygon (prefer neural boundary over convex hull over bbox)
            if hasattr(region, 'polygon') and region.polygon and len(region.polygon) >= 3:
                pts = ' '.join(f'{x},{y}' for x, y in region.polygon)
            else:
                x1, y1, x2, y2 = region.bbox
                pts = f'{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}'
            ET.SubElement(text_region, 'Coords').set('points', pts)

            for li, segment in enumerate(region_lines):
                global_line_idx = line_offset - n + li  # index in the flat lines list
                text = None
                if transcriptions and global_line_idx < len(transcriptions):
                    text = transcriptions[global_line_idx] or None
                elif hasattr(segment, 'text'):
                    text = getattr(segment, 'text', None) or None
                self._add_text_line(
                    text_region,
                    f'line_{ri + 1}_{li + 1}',
                    segment,
                    text,
                    li,
                )

        self._write_xml(root, output_path)

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
