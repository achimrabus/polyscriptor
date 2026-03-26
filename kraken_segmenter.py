"""
Kraken-based line segmentation for historical document OCR.

This module provides an alternative to the classical HPP (Horizontal Projection Profile)
segmentation using Kraken's pre-trained neural models.

Supports two modes:
- Classical: pageseg.segment() — fast, lines only, no regions
- Neural (blla): blla.segment() — GPU-accelerated, returns regions AND baselines,
  handles multi-column layouts
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, NamedTuple, Tuple, Dict
from PIL import Image
import numpy as np

# Module-level cache: maps model path -> loaded TorchVGSLModel.
# Shared across all KrakenLineSegmenter instances so that the model is loaded
# from disk only once per process, even in batch processing loops.
_MODEL_CACHE: Dict[str, Any] = {}


class LineSegment(NamedTuple):
    """Represents a segmented text line."""
    image: Image.Image
    bbox: tuple  # (x1, y1, x2, y2)
    baseline: Optional[List[tuple]] = None  # List of (x, y) points


@dataclass
class SegRegion:
    """Represents a detected text region (column, marginalia, etc.)."""
    id: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    line_ids: List[str] = field(default_factory=list)
    polygon: Optional[List[Tuple[int, int]]] = None  # Convex hull or neural polygon
    mode: str = "neural"  # "neural" or "classical"


class KrakenLineSegmenter:
    """
    Line segmentation using Kraken with pre-trained models.

    Kraken is specifically designed for historical document OCR and provides:
    - Pre-trained models that work out-of-the-box
    - Baseline detection (not just bounding boxes)
    - Robust handling of degraded/faded text
    - Support for rotated and multi-column layouts

    Performance: ~3-8s per page (CPU), ~1-3s (GPU)
    Accuracy: 90-95% on historical documents
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize Kraken segmenter.

        Args:
            model_path: Path to custom segmentation model (.mlmodel file).
                       Note: Kraken 5.x uses classical segmentation by default.
                       Neural baseline segmentation requires additional setup.
            device: 'cpu' or 'cuda' for GPU acceleration (not used by classical segmenter)
        """
        self.model_path = model_path
        self.device = device

        # Import kraken components
        try:
            from kraken import binarization, pageseg
            self.binarization = binarization
            self.pageseg = pageseg
        except ImportError as e:
            raise ImportError(
                "Kraken is not installed. Install it with: pip install kraken\n"
                f"Original error: {e}"
            )

        # Note: model_path is currently not used as pageseg.segment() doesn't accept models
        # The classical segmentation algorithm is robust and works well for most documents
        if model_path:
            print(f"[KrakenSegmenter] Warning: Custom model path provided but not used.")
            print(f"[KrakenSegmenter] Kraken 5.x pageseg.segment() uses classical algorithm.")
            print(f"[KrakenSegmenter] Neural baseline segmentation requires kraken.lib.models workflow.")

    def segment_lines(
        self,
        image: Image.Image,
        text_direction: str = 'horizontal-lr',
        use_binarization: bool = True
    ) -> List[LineSegment]:
        """
        Segment image into text lines using Kraken.

        Args:
            image: PIL Image to segment
            text_direction: Text direction - 'horizontal-lr' (left-to-right),
                          'horizontal-rl', 'vertical-lr', 'vertical-rl'
            use_binarization: Whether to apply neural binarization preprocessing
                            (recommended for degraded documents)

        Returns:
            List of LineSegment objects sorted top to bottom
        """
        print(f"[KrakenSegmenter] Segmenting image (size={image.size}, mode={image.mode}, "
              f"direction={text_direction}, binarize={use_binarization})")

        try:
            # Step 0: Convert to grayscale if needed (Kraken works better with grayscale)
            if image.mode not in ('L', '1'):
                print(f"[KrakenSegmenter] Converting from {image.mode} to grayscale...")
                image = image.convert('L')

            # Step 1: Binarize (required by pageseg.segment)
            # pageseg.segment REQUIRES binary images
            if use_binarization:
                print(f"[KrakenSegmenter] Applying neural binarization...")
                processed_img = self.binarization.nlbin(image)
            else:
                # Simple Otsu binarization as fallback
                print(f"[KrakenSegmenter] Applying Otsu binarization...")
                import numpy as np
                from PIL import ImageOps
                # Otsu's method
                img_array = np.array(image)
                threshold = np.median(img_array)  # Simple threshold
                binary = img_array > threshold
                processed_img = Image.fromarray((binary * 255).astype(np.uint8), mode='L')

            # Step 2: Line segmentation using Kraken's classical algorithm
            # This is more robust than basic HPP and works well on historical documents
            print(f"[KrakenSegmenter] Running line segmentation...")
            seg_result = self.pageseg.segment(
                processed_img,
                text_direction=text_direction
            )

            # Handle both dict (old Kraken) and Segmentation object (new Kraken)
            if isinstance(seg_result, dict):
                print(f"[KrakenSegmenter] pageseg.segment returned dict (old Kraken API)")
                # Old API: seg_result is a dict with 'boxes' key
                seg_lines = seg_result.get('boxes', seg_result.get('lines', []))
            else:
                print(f"[KrakenSegmenter] pageseg.segment returned Segmentation object")
                seg_lines = seg_result.lines

            print(f"[KrakenSegmenter] Processing {len(seg_lines)} lines...")

            # Step 3: Extract line information
            lines = []
            for idx, line in enumerate(seg_lines):
                # Extract bounding box
                bbox = line.bbox  # (x_min, y_min, x_max, y_max)

                # Extract baseline (list of (x, y) points)
                baseline = line.baseline if hasattr(line, 'baseline') else None

                # Crop line image from original (not binarized)
                line_img = image.crop(bbox)

                lines.append(LineSegment(
                    image=line_img,
                    bbox=bbox,
                    baseline=baseline
                ))

            # Sort lines top to bottom by Y coordinate
            lines = sorted(lines, key=lambda x: x.bbox[1])

            print(f"[KrakenSegmenter] Detected {len(lines)} lines")

            return lines

        except Exception as e:
            print(f"[KrakenSegmenter] ERROR: Segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def segment_with_regions(
        self,
        image: Image.Image,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        min_line_height: int = 8,
        max_columns: int = 4,
        split_width_fraction: float = 0.40,
        min_lines_to_split: int = 10,
        text_direction: str = 'horizontal-lr',
    ) -> Tuple[List[SegRegion], List[LineSegment]]:
        """
        Neural baseline segmentation using blla.segment().

        Returns regions AND lines with baselines.  Handles multi-column layouts
        by using blla's region detection, with a column-clustering fallback when
        blla returns a single region with many lines (≥30).

        Falls back to classical pageseg.segment() + column clustering if blla
        fails or the model file is missing.

        Args:
            image: PIL Image to segment (RGB or grayscale)
            model_path: Path to blla .mlmodel file.  Defaults to
                        ``pagexml/blla.mlmodel`` relative to this script.
            device: 'cpu' or 'cuda' / 'cuda:0'.  Defaults to self.device.
            min_line_height: Discard lines shorter than this (pixels).
            max_columns: Maximum number of columns to detect per region (1-8).
            split_width_fraction: Minimum region width as fraction of page width
                        to trigger sub-column splitting (0.0-1.0).  Lower values
                        split narrower regions.  Default 0.40 (40%).
                        For landscape double-page spreads, try 0.20 (20%).
            min_lines_to_split: Minimum number of lines in a region before
                        attempting to split it into sub-columns.

        Returns:
            (regions, lines) where *lines* carry a ``region_id`` attribute via
            the companion ``SegRegion`` that owns them.
        """
        device = device or self.device
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'pagexml', 'blla.mlmodel')

        print(f"[KrakenSegmenter] Neural segmentation (blla) on {image.size}, device={device}")

        # ── Try neural (blla) first ──────────────────────────────────
        if os.path.isfile(model_path):
            try:
                regions, lines = self._segment_neural(
                    image, model_path, device, min_line_height,
                    max_columns=max_columns,
                    split_width_fraction=split_width_fraction,
                    min_lines_to_split=min_lines_to_split,
                    text_direction=text_direction,
                )
                if regions:
                    print(f"[KrakenSegmenter] blla: {len(regions)} regions, {len(lines)} lines")
                    return regions, lines
                print("[KrakenSegmenter] blla returned no regions; falling back to classical + clustering")
            except Exception as e:
                print(f"[KrakenSegmenter] blla failed ({e}); falling back to classical + clustering")
                import traceback
                traceback.print_exc()
        else:
            print(f"[KrakenSegmenter] blla model not found at {model_path}; using classical fallback")

        # ── Fallback: classical pageseg + column clustering ──────────
        return self._segment_classical_with_regions(image, min_line_height)

    # ── internal: neural blla ────────────────────────────────────────

    def _segment_neural(
        self,
        image: Image.Image,
        model_path: str,
        device: str,
        min_line_height: int,
        max_columns: int = 4,
        split_width_fraction: float = 0.40,
        min_lines_to_split: int = 10,
        text_direction: str = 'horizontal-lr',
    ) -> Tuple[List[SegRegion], List[LineSegment]]:
        """Run blla.segment() and build SegRegion / LineSegment lists."""
        from kraken import blla
        from kraken.lib import vgsl
        import torch

        start = time.time()

        # Validate device
        if device.startswith('cuda') and not torch.cuda.is_available():
            print(f"[KrakenSegmenter] WARNING: device={device} but CUDA not available, falling back to cpu")
            device = 'cpu'

        # Load model once and cache keyed by (path, device) — repeated calls
        # reuse the already-loaded, already-placed model. Keying by device means
        # a CPU and a CUDA instance don't share the same cached object.
        cache_key = (model_path, device)
        if cache_key not in _MODEL_CACHE:
            print(f"[KrakenSegmenter] Loading blla model: {model_path}")
            m = vgsl.TorchVGSLModel.load_model(model_path)
            # blla.segment()'s device= parameter does NOT move the model —
            # it must be placed on the target device explicitly before the call.
            m.nn.to(device)
            _MODEL_CACHE[cache_key] = m
        model = _MODEL_CACHE[cache_key]

        # Diagnostic: confirm model parameters are on the expected device.
        try:
            actual_device = next(model.nn.parameters()).device
            print(f"[KrakenSegmenter] blla model on: {actual_device} (requested: {device})")
            if device.startswith('cuda') and actual_device.type != 'cuda':
                print(f"[KrakenSegmenter] WARNING: model is on {actual_device}, not GPU")
        except Exception:
            print(f"[KrakenSegmenter] blla running on device={device}")

        # blla wants RGB
        img = image.convert('RGB') if image.mode != 'RGB' else image

        # blla has built-in autocast support (disabled by default). Enable it
        # on CUDA for faster fp16 forward pass.
        baseline_seg = blla.segment(img, model=model, device=device,
                                    autocast=device.startswith('cuda'),
                                    text_direction=text_direction)

        w, h = image.size
        seg_lines: List[LineSegment] = []
        # region_id -> {'lines': [...], 'blla_region': ...}
        regions_dict: Dict[str, dict] = {}

        for idx, line in enumerate(baseline_seg.lines):
            bbox = self._extract_bbox(line)
            if bbox is None:
                continue

            # Filter tiny lines
            if (bbox[3] - bbox[1]) < min_line_height:
                continue

            baseline = (
                [(int(p[0]), int(p[1])) for p in line.baseline]
                if hasattr(line, 'baseline') and line.baseline
                else None
            )

            line_img = image.crop(bbox)
            seg_line = LineSegment(image=line_img, bbox=bbox, baseline=baseline)
            seg_lines.append(seg_line)

            # Assign to region
            region_id, blla_region = self._find_region_for_line(
                bbox, line, baseline_seg
            )
            if region_id not in regions_dict:
                regions_dict[region_id] = {'lines': [], 'blla_region': blla_region}
            regions_dict[region_id]['lines'].append((len(seg_lines) - 1, seg_line))

        # Sub-split wide regions that likely contain multiple columns.
        # blla often detects "left page" and "right page" as two regions on a
        # double-page spread, but each page may have 2 columns internally.
        # Loop until convergence: a single pass may leave wide sub-regions that
        # need further splitting (e.g. a 3-column area assigned as one bucket).
        for _round in range(max_columns):
            prev_size = len(regions_dict)
            regions_dict = self._split_wide_regions(
                regions_dict, w,
                min_lines_to_split=min_lines_to_split,
                split_width_fraction=split_width_fraction,
                max_columns=max_columns,
            )
            if len(regions_dict) == prev_size:
                break  # no new splits — converged

        # Build SegRegion objects
        regions, ordered_lines = self._build_regions(regions_dict, seg_lines, w,
                                                      text_direction=text_direction)

        elapsed = time.time() - start
        print(f"[KrakenSegmenter] blla completed in {elapsed:.2f}s")
        return regions, ordered_lines

    # ── internal: classical fallback with column clustering ──────────

    def segment_classical_with_regions(
        self,
        image: Image.Image,
        min_line_height: int = 15,
        max_columns: int = 4,
    ) -> Tuple[List[SegRegion], List[LineSegment]]:
        """Public wrapper: classical pageseg + heuristic column clustering."""
        return self._segment_classical_with_regions(image, min_line_height, max_columns)

    def _segment_classical_with_regions(
        self,
        image: Image.Image,
        min_line_height: int = 15,
        max_columns: int = 4,
    ) -> Tuple[List[SegRegion], List[LineSegment]]:
        """Classical pageseg + heuristic column clustering."""
        raw_lines = self.segment_lines(image)
        if not raw_lines:
            return [], []

        # Filter small lines
        raw_lines = [l for l in raw_lines if (l.bbox[3] - l.bbox[1]) >= min_line_height]

        w = image.size[0]
        # Cluster into columns (pass max_columns so 4-column spreads are handled)
        regions_dict = self._cluster_into_columns(raw_lines, w, max_columns=max_columns)
        regions, ordered_lines = self._build_regions(regions_dict, raw_lines, w)
        for r in regions:
            r.mode = "classical"
        return regions, ordered_lines

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_bbox(line) -> Optional[Tuple[int, int, int, int]]:
        """Extract (x1,y1,x2,y2) bbox from a blla line object."""
        if hasattr(line, 'bbox'):
            return tuple(int(v) for v in line.bbox)
        if hasattr(line, 'baseline') and line.baseline:
            xs = [p[0] for p in line.baseline]
            ys = [p[1] for p in line.baseline]
            avg_h = 30
            return (int(min(xs)), int(min(ys) - avg_h // 2),
                    int(max(xs)), int(max(ys) + avg_h // 2))
        return None

    @staticmethod
    def _find_region_for_line(bbox, line, baseline_seg) -> Tuple[str, object]:
        """Determine which blla region a line belongs to."""
        # Check tags first
        if hasattr(line, 'tags') and isinstance(line.tags, dict):
            rtype = line.tags.get('type')
            if rtype and isinstance(rtype, str):
                return rtype, None

        # Check region boundaries
        if hasattr(baseline_seg, 'regions') and baseline_seg.regions:
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            for rtype, region_list in baseline_seg.regions.items():
                for ri, region in enumerate(region_list):
                    if hasattr(region, 'boundary') and region.boundary:
                        bxs = [p[0] for p in region.boundary]
                        bys = [p[1] for p in region.boundary]
                        if (min(bxs) <= cx <= max(bxs) and
                                min(bys) <= cy <= max(bys)):
                            return f"{rtype}_{ri}", region

        return 'r_1', None

    @staticmethod
    def _estimate_columns(
        lines: list,
        page_w: int,
        max_columns: int = 4,
        min_gap_fraction: float = 0.03,
    ) -> List[int]:
        """
        Gap-based column clustering.

        Finds natural breaks in the x-center distribution by looking for the
        largest gaps in the sorted sequence of line x-centers.  This is more
        robust than histogram peak-finding for closely spaced columns, because
        a column gap is a region with *no* line centers — it shows up as a large
        jump in the sorted sequence regardless of how close the columns are.

        Args:
            lines:             List of LineSegment objects.
            page_w:            Width of the region being analysed (pixels).
            max_columns:       Maximum number of columns to return (≥1).
            min_gap_fraction:  Minimum gap size as a fraction of *page_w* to be
                               considered a column boundary.  Default 0.03 (3%).
                               Increase if spurious splits occur within a column.
        """
        if not lines:
            return []

        # Lines wider than 60% of the region are likely headers/footers that
        # span columns — exclude them from clustering to avoid false splits.
        orig_centers = [((l.bbox[0] + l.bbox[2]) // 2) for l in lines]
        line_widths = [(l.bbox[2] - l.bbox[0]) for l in lines]
        clustering_centers = [
            cx for cx, w in zip(orig_centers, line_widths)
            if w < 0.60 * page_w
        ]

        if not clustering_centers:
            # All lines are wide (e.g. single full-width text block)
            return [0] * len(lines)

        min_gap_px = max(10, int(min_gap_fraction * page_w))
        sorted_cx = sorted(clustering_centers)

        # Compute gaps between consecutive sorted x-centers
        gaps = [
            (sorted_cx[i + 1] - sorted_cx[i], (sorted_cx[i] + sorted_cx[i + 1]) // 2)
            for i in range(len(sorted_cx) - 1)
            if sorted_cx[i + 1] - sorted_cx[i] >= min_gap_px
        ]

        if not gaps:
            return [0] * len(lines)

        # Take the largest max_columns-1 gaps as column boundaries
        split_midpoints = sorted(
            mid for _, mid in sorted(gaps, reverse=True)[: max_columns - 1]
        )

        # Assign each line (using original center) to a column
        assignments = []
        for cx in orig_centers:
            col = sum(1 for sp in split_midpoints if cx > sp)
            assignments.append(col)

        return assignments

    def _split_wide_regions(
        self,
        regions_dict: Dict[str, dict],
        page_w: int,
        min_lines_to_split: int = 10,
        split_width_fraction: float = 0.40,
        max_columns: int = 4,
    ) -> Dict[str, dict]:
        """
        Split blla regions that are wide enough to contain multiple columns.

        A region whose width exceeds *split_width_fraction* of the page width
        and has enough lines is run through column clustering internally.

        For landscape double-page spreads, lower split_width_fraction (e.g. 0.20)
        to trigger splitting on narrower regions.
        """
        new_dict: Dict[str, dict] = {}
        split_counter = 0

        for key, rdata in regions_dict.items():
            region_lines = rdata['lines']  # list of (idx, LineSegment)
            if len(region_lines) < min_lines_to_split:
                new_dict[key] = rdata
                continue

            # Compute region width from line bboxes
            bboxes = [l.bbox for _, l in region_lines]
            rx1 = min(b[0] for b in bboxes)
            rx2 = max(b[2] for b in bboxes)
            region_w = rx2 - rx1

            if region_w < split_width_fraction * page_w:
                # Narrow enough to be a single column
                new_dict[key] = rdata
                continue

            # Wide region — try column clustering within it.
            # _estimate_columns bins x-centers into [0, page_w), so we need to
            # shift line coordinates so that rx1 maps to 0.
            just_lines = [l for _, l in region_lines]
            shifted_lines = []
            for l in just_lines:
                shifted_bbox = (l.bbox[0] - rx1, l.bbox[1],
                                l.bbox[2] - rx1, l.bbox[3])
                shifted_lines.append(LineSegment(l.image, shifted_bbox, l.baseline))
            assignments = self._estimate_columns(shifted_lines, page_w=region_w,
                                                  max_columns=max_columns)

            n_cols = len(set(assignments))
            if n_cols <= 1:
                # Clustering didn't find multiple columns
                new_dict[key] = rdata
                continue

            print(f"[KrakenSegmenter] Splitting region '{key}' ({len(region_lines)} lines, "
                  f"width={region_w}px) into {n_cols} sub-columns")

            # Re-compute x-centers relative to region left edge for clustering
            # (already done inside _estimate_columns via absolute coords, which
            # works fine since columns are spatially separated)
            for col_id in sorted(set(assignments)):
                sub_key = f"{key}_col{split_counter}"
                split_counter += 1
                sub_lines = [
                    region_lines[i]
                    for i, a in enumerate(assignments)
                    if a == col_id
                ]
                new_dict[sub_key] = {'lines': sub_lines, 'blla_region': None}

        return new_dict

    def _cluster_into_columns(
        self,
        lines: list,
        page_w: int,
        max_columns: int = 4,
    ) -> Dict[str, dict]:
        """Cluster lines into columns and return regions_dict."""
        assignments = self._estimate_columns(lines, page_w, max_columns=max_columns)
        regions_dict: Dict[str, dict] = {}
        for idx, (col, line) in enumerate(zip(assignments, lines)):
            key = f"col_{col}"
            if key not in regions_dict:
                regions_dict[key] = {'lines': [], 'blla_region': None}
            regions_dict[key]['lines'].append((idx, line))
        return regions_dict

    @staticmethod
    def _convex_hull(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Monotonic chain convex hull."""
        pts = sorted(set(points))
        if len(pts) <= 2:
            return pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    def _build_regions(
        self,
        regions_dict: Dict[str, dict],
        all_lines: list,
        page_w: int,
        text_direction: str = 'horizontal-lr',
    ) -> Tuple[List[SegRegion], List[LineSegment]]:
        """
        Build SegRegion objects from regions_dict.

        Returns (regions, ordered_lines) where ordered_lines is sorted by
        region (left-to-right for LTR, right-to-left for RTL) then
        top-to-bottom within each region.
        """
        rtl = text_direction.endswith('-rl')

        # Sort regions by mean x-center: LTR = ascending, RTL = descending
        def _region_mean_x(item):
            lines = item[1]['lines']
            if not lines:
                return 0
            return sum((l.bbox[0] + l.bbox[2]) / 2 for _, l in lines) / len(lines)

        sorted_regions = sorted(regions_dict.items(), key=_region_mean_x, reverse=rtl)

        regions: List[SegRegion] = []
        ordered_lines: List[LineSegment] = []

        for ri, (region_key, rdata) in enumerate(sorted_regions, start=1):
            region_lines = rdata['lines']
            blla_region = rdata['blla_region']

            # Sort lines top-to-bottom within region
            region_lines.sort(key=lambda item: item[1].bbox[1])

            region_id = f"r_{ri}"
            line_ids = [f"l_{i + 1}" for i, _ in region_lines]

            bboxes = [l.bbox for _, l in region_lines]
            rbbox = (
                min(b[0] for b in bboxes),
                min(b[1] for b in bboxes),
                max(b[2] for b in bboxes),
                max(b[3] for b in bboxes),
            )

            # Polygon: prefer blla boundary, else convex hull
            polygon = None
            if blla_region and hasattr(blla_region, 'boundary') and blla_region.boundary:
                polygon = [(int(p[0]), int(p[1])) for p in blla_region.boundary]
            else:
                pts = []
                for _, l in region_lines:
                    x1, y1, x2, y2 = l.bbox
                    pts.extend([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                hull = self._convex_hull(pts)
                polygon = hull if len(hull) >= 3 else None

            regions.append(SegRegion(
                id=region_id,
                bbox=rbbox,
                line_ids=line_ids,
                polygon=polygon,
            ))

            for _, line in region_lines:
                ordered_lines.append(line)

        return regions, ordered_lines

    def segment_lines_to_dict(
        self,
        image: Image.Image,
        text_direction: str = 'horizontal-lr',
        use_binarization: bool = True
    ) -> List[dict]:
        """
        Segment image and return results as dictionaries (for compatibility).

        Returns:
            List of dicts with 'image', 'bbox', and 'baseline' keys
        """
        segments = self.segment_lines(image, text_direction, use_binarization)
        return [
            {
                'image': seg.image,
                'bbox': seg.bbox,
                'baseline': seg.baseline
            }
            for seg in segments
        ]


def test_kraken_segmenter():
    """Test Kraken segmenter on a sample image."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kraken_segmenter.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Testing Kraken segmenter on: {image_path}")

    # Load image
    image = Image.open(image_path)
    print(f"Image size: {image.size}")

    # Create segmenter
    segmenter = KrakenLineSegmenter()

    # Segment lines
    lines = segmenter.segment_lines(image, use_binarization=True)

    # Print results
    print(f"\nDetected {len(lines)} lines:")
    for i, line in enumerate(lines):
        print(f"  Line {i+1}: bbox={line.bbox}, "
              f"baseline_points={len(line.baseline) if line.baseline else 0}")

    # Save line images
    import os
    output_dir = "kraken_test_output"
    os.makedirs(output_dir, exist_ok=True)

    for i, line in enumerate(lines):
        output_path = os.path.join(output_dir, f"line_{i+1:03d}.png")
        line.image.save(output_path)

    print(f"\nLine images saved to: {output_dir}/")


if __name__ == "__main__":
    test_kraken_segmenter()
