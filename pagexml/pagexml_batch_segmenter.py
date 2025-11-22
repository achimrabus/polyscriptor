"""
Standalone batch tool to generate Transkribus/PAGE XML from a folder of page images.

Important: This module does NOT touch the GUI plugin. It is a separate batch processor.

- Uses Kraken classical pageseg for line detection
- Adds heuristic region clustering for 1–4 columns
- Writes PAGE XML with TextRegions + TextLines (Coords and Baseline when available)
- Optional overlays for QC
"""
from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
from enum import Enum

from PIL import Image, ImageDraw
import threading
import time

try:
    # Local module
    from kraken_segmenter import KrakenLineSegmenter, LineSegment
except Exception as e:
    raise ImportError("kraken_segmenter.py not found or Kraken is not installed. Install with: pip install kraken")

import xml.etree.ElementTree as ET

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
ET.register_namespace('', PAGE_NS)


class SegmentationMode(Enum):
    CLASSICAL = "classical"
    NEURAL = "neural"


@dataclass
class SegLine:
    id: str
    bbox: Tuple[int, int, int, int]
    baseline: Optional[List[Tuple[int, int]]]
    region_id: Optional[str] = None


@dataclass
class Region:
    id: str
    bbox: Tuple[int, int, int, int]
    line_ids: List[str]
    polygon: Optional[List[Tuple[int,int]]] = None  # Convex hull or neural polygon
    mode: str = "classical"  # Segmentation mode used


def _list_images(input_dir: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = []
    for name in os.listdir(input_dir):
        p = os.path.join(input_dir, name)
        if os.path.isfile(p) and os.path.splitext(name.lower())[1] in exts:
            files.append(p)
    return sorted(files)


def _deskew_if_needed(img: Image.Image) -> Image.Image:
    # Placeholder: keep simple for MVP; real deskew can be added later
    return img


def _line_center_x(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2


def _line_top_y(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return y1


def _bbox_union(bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    xs1 = [b[0] for b in bboxes]
    ys1 = [b[1] for b in bboxes]
    xs2 = [b[2] for b in bboxes]
    ys2 = [b[3] for b in bboxes]
    return (min(xs1), min(ys1), max(xs2), max(ys2))


def _convex_hull(points: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """Monotonic chain convex hull. Returns CCW hull. If <3 points returns unique sorted points."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
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


def _region_convex_hull(lines: List['SegLine']) -> List[Tuple[int,int]]:
    pts: List[Tuple[int,int]] = []
    for ln in lines:
        x1,y1,x2,y2 = ln.bbox
        pts.extend([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    return _convex_hull(pts)


def _estimate_columns(lines: List[SegLine], page_w: int, max_columns: int = 4) -> List[int]:
    """
    Heuristic: build histogram of line x-centers, pick up to max_columns peaks,
    and assign each line to nearest peak center.
    Returns a list of cluster indices per line (0..k-1).
    """
    if not lines:
        return []

    centers = [_line_center_x(l.bbox) for l in lines]
    bins = 40
    hist = [0] * bins
    for cx in centers:
        idx = min(bins - 1, max(0, int(cx / (page_w / bins))))
        hist[idx] += 1

    # Find peaks above threshold (relative to max)
    max_count = max(hist) if hist else 0
    if max_count == 0:
        return [0] * len(lines)

    threshold = max(2, int(0.25 * max_count))
    peak_idxs = [i for i, v in enumerate(hist) if v >= threshold]

    # Merge nearby peaks
    merged = []
    prev = None
    for idx in peak_idxs:
        if prev is None or idx - prev > 2:  # keep some spacing
            merged.append(idx)
        prev = idx

    # Limit to max_columns
    if not merged:
        merged = [hist.index(max_count)]
    centers_est = [int((i + 0.5) * (page_w / bins)) for i in merged[:max_columns]]

    # Assign lines to nearest estimated center
    assignments = []
    for cx in centers:
        nearest = min(range(len(centers_est)), key=lambda j: abs(cx - centers_est[j]))
        assignments.append(nearest)
    return assignments


def _filter_small_lines(lines: List[SegLine], min_h: int) -> List[SegLine]:
    out = []
    for l in lines:
        x1, y1, x2, y2 = l.bbox
        if (y2 - y1) >= min_h:
            out.append(l)
    return out


def _draw_overlay(base_img: Image.Image, regions: List[Region], line_map: Dict[str, SegLine]) -> Image.Image:
    img = base_img.convert('RGB')
    draw = ImageDraw.Draw(img)

    # Colors per region id order
    palette = [(255,0,0),(0,128,0),(0,0,255),(255,128,0),(128,0,128)]

    for ri, region in enumerate(regions):
        color = palette[ri % len(palette)]
        # Region shape: polygon if available else bbox rectangle
        if region.polygon and len(region.polygon) >= 3:
            draw.line(region.polygon + [region.polygon[0]], fill=color, width=3)
        else:
            draw.rectangle(region.bbox, outline=color, width=3)
        # Lines
        for lid in region.line_ids:
            ln = line_map[lid]
            draw.rectangle(ln.bbox, outline=color, width=1)
            # Baseline
            if ln.baseline and len(ln.baseline) >= 2:
                draw.line(ln.baseline, fill=color, width=2)
    return img


def _to_points_str(points: List[Tuple[int,int]]) -> str:
    return ' '.join([f"{x},{y}" for x,y in points])


def _bbox_to_polygon(b: Tuple[int,int,int,int]) -> List[Tuple[int,int]]:
    x1,y1,x2,y2 = b
    return [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]


def _write_page_xml(
    image_path: str,
    page_size: Tuple[int,int],
    regions: List[Region],
    line_map: Dict[str, SegLine],
    out_xml_path: str
):
    w,h = page_size

    PcGts = ET.Element(ET.QName(PAGE_NS, 'PcGts'))
    Metadata = ET.SubElement(PcGts, ET.QName(PAGE_NS, 'Metadata'))
    ET.SubElement(Metadata, ET.QName(PAGE_NS, 'Creator')).text = 'pagexml_batch_segmenter'
    Page = ET.SubElement(PcGts, ET.QName(PAGE_NS, 'Page'), attrib={'imageFilename': os.path.basename(image_path), 'imageWidth': str(w), 'imageHeight': str(h)})

    # Regions and lines
    for region in regions:
        tr = ET.SubElement(Page, ET.QName(PAGE_NS, 'TextRegion'), attrib={'id': region.id})
        if region.polygon and len(region.polygon) >= 3:
            ET.SubElement(tr, ET.QName(PAGE_NS, 'Coords'), attrib={'points': _to_points_str(region.polygon)})
        else:
            ET.SubElement(tr, ET.QName(PAGE_NS, 'Coords'), attrib={'points': _to_points_str(_bbox_to_polygon(region.bbox))})
        # Lines within region (reading order: top->bottom)
        for lid in region.line_ids:
            ln = line_map[lid]
            tl = ET.SubElement(tr, ET.QName(PAGE_NS, 'TextLine'), attrib={'id': ln.id})
            ET.SubElement(tl, ET.QName(PAGE_NS, 'Coords'), attrib={'points': _to_points_str(_bbox_to_polygon(ln.bbox))})
            if ln.baseline and len(ln.baseline) >= 2:
                ET.SubElement(tl, ET.QName(PAGE_NS, 'Baseline'), attrib={'points': _to_points_str(ln.baseline)})

    # Simple reading order: regions left->right; lines already in order
    ro = ET.SubElement(Page, ET.QName(PAGE_NS, 'ReadingOrder'))
    og = ET.SubElement(ro, ET.QName(PAGE_NS, 'OrderedGroup'), attrib={'id': 'ro_1'})
    for region in regions:
        ET.SubElement(og, ET.QName(PAGE_NS, 'RegionRefIndexed'), attrib={'index': str(regions.index(region)), 'regionRef': region.id})

    tree = ET.ElementTree(PcGts)
    os.makedirs(os.path.dirname(out_xml_path), exist_ok=True)
    tree.write(out_xml_path, encoding='utf-8', xml_declaration=True)


def process_image_neural(
    image_path: str,
    model_path: str,
    device: str = 'cpu',
    min_line_height: int = 8
) -> Tuple[List[Region], Dict[str, SegLine], Tuple[int,int], float]:
    """
    Neural segmentation using blla.mlmodel. Returns (regions, line_map, page_size, elapsed_time).
    """
    from kraken import blla
    from kraken.lib import vgsl
    
    start = time.time()
    img = Image.open(image_path)
    w, h = img.size
    
    try:
        # Load segmentation model
        model = vgsl.TorchVGSLModel.load_model(model_path)
        
        # Run baseline and region segmentation
        baseline_seg = blla.segment(img, model=model, device=device)
        
        regions_dict = {}
        seg_lines = []
        
        # Extract lines from segmentation result
        for idx, line in enumerate(baseline_seg.lines):
            # Extract bbox
            if hasattr(line, 'bbox'):
                bbox = tuple(int(v) for v in line.bbox)
            else:
                # Compute from baseline if bbox not available
                if hasattr(line, 'baseline') and line.baseline:
                    xs = [p[0] for p in line.baseline]
                    ys = [p[1] for p in line.baseline]
                    # Estimate bbox from baseline with padding
                    avg_height = 30  # rough estimate
                    bbox = (int(min(xs)), int(min(ys) - avg_height//2), 
                           int(max(xs)), int(max(ys) + avg_height//2))
                else:
                    continue
            
            # Baseline
            baseline = [(int(p[0]), int(p[1])) for p in line.baseline] if hasattr(line, 'baseline') and line.baseline else None
            
            # Filter small lines
            if (bbox[3] - bbox[1]) < min_line_height:
                continue
            
            seg_line = SegLine(id=f"l_{idx+1}", bbox=bbox, baseline=baseline)
            seg_lines.append(seg_line)
            
            # Group by region (use tags/region info if available)
            region_id = 'r_1'  # default
            
            if hasattr(line, 'tags') and isinstance(line.tags, dict):
                region_type = line.tags.get('type')
                if region_type and isinstance(region_type, str):
                    region_id = region_type
            elif hasattr(baseline_seg, 'regions') and baseline_seg.regions:
                # Try to find which region contains this line
                for reg_idx, region in enumerate(baseline_seg.regions):
                    if hasattr(region, 'bbox'):
                        rbbox = region.bbox
                        # Ensure bbox is iterable and has 4 values
                        if hasattr(rbbox, '__iter__') and len(list(rbbox)) == 4:
                            rx1, ry1, rx2, ry2 = rbbox
                            lx1, ly1, lx2, ly2 = bbox
                            # Check if line center is inside region
                            cx, cy = (lx1 + lx2) // 2, (ly1 + ly2) // 2
                            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                                region_id = f"r_{reg_idx+1}"
                                break
            
            # Ensure region_id is always a string
            if not isinstance(region_id, str):
                region_id = 'r_1'
            
            regions_dict.setdefault(region_id, []).append(seg_line)
        
        # Build regions
        regions = []
        line_map = {l.id: l for l in seg_lines}
        
        # If blla returned only one region but we have many lines, apply column clustering
        # Higher threshold: only apply if we have substantial content (30+ lines typical for double-page)
        if len(regions_dict) == 1 and len(seg_lines) >= 30:
            # Use classical column clustering for better multi-column handling
            max_columns = 4  # typical for manuscripts
            col_assignments = _estimate_columns(seg_lines, w, max_columns)
            
            # Regroup by column (col_assignments is a list parallel to seg_lines)
            regions_dict = {}
            for idx, ln in enumerate(seg_lines):
                col_id = col_assignments[idx] if idx < len(col_assignments) else 0
                regions_dict.setdefault(f"col_{col_id}", []).append(ln)
        
        for ri, (region_id, lines_in_region) in enumerate(sorted(regions_dict.items()), start=1):
            lines_in_region.sort(key=lambda l: _line_top_y(l.bbox))
            final_id = f"r_{ri}"
            for ln in lines_in_region:
                ln.region_id = final_id
            rbbox = _bbox_union([l.bbox for l in lines_in_region])
            hull = _region_convex_hull(lines_in_region)
            regions.append(Region(
                id=final_id,
                bbox=rbbox,
                line_ids=[l.id for l in lines_in_region],
                polygon=hull if len(hull) >= 3 else None,
                mode="neural"
            ))
        
        elapsed = time.time() - start
        return regions, line_map, (w, h), elapsed
    
    except Exception as e:
        # Fallback: return empty and let caller handle classical fallback
        elapsed = time.time() - start
        print(f"[ERROR] Neural segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return [], {}, (w, h), elapsed


def process_image(
    image_path: str,
    segmenter: KrakenLineSegmenter,
    max_columns: int = 4,
    min_line_height: int = 8
) -> Tuple[List[Region], Dict[str, SegLine], Tuple[int,int]]:
    img = Image.open(image_path)
    img = _deskew_if_needed(img)
    w,h = img.size

    # Kraken segmentation (grayscale+binarization inside segmenter)
    lines_raw = segmenter.segment_lines(img, use_binarization=True)
    seg_lines: List[SegLine] = []
    for i, ln in enumerate(lines_raw):
        seg_lines.append(SegLine(id=f"l_{i+1}", bbox=ln.bbox, baseline=ln.baseline))

    # Filter tiny noise lines
    seg_lines = _filter_small_lines(seg_lines, min_line_height)

    if not seg_lines:
        # Return empty result; caller can log
        return [], {}, (w,h)

    # Estimate column assignments
    assignments = _estimate_columns(seg_lines, page_w=w, max_columns=max_columns)

    # Build regions dict
    regions_map: Dict[int, List[SegLine]] = {}
    for line, col in zip(seg_lines, assignments):
        regions_map.setdefault(col, []).append(line)

    # Sort regions left->right by their mean x center
    col_indices = list(regions_map.keys())
    col_indices.sort(key=lambda c: sum(_line_center_x(l.bbox) for l in regions_map[c]) / max(1,len(regions_map[c])))

    regions: List[Region] = []
    line_map: Dict[str, SegLine] = {l.id: l for l in seg_lines}

    for ri, col in enumerate(col_indices):
        lines_in_col = regions_map[col]
        # Order lines top->bottom
        lines_in_col.sort(key=lambda l: _line_top_y(l.bbox))
        region_id = f"r_{ri+1}"
        for l in lines_in_col:
            l.region_id = region_id
        rbbox = _bbox_union([l.bbox for l in lines_in_col])
        hull = _region_convex_hull(lines_in_col)
        regions.append(Region(
            id=region_id,
            bbox=rbbox,
            line_ids=[l.id for l in lines_in_col],
            polygon=hull if len(hull) >= 3 else None,
            mode="classical"
        ))

    return regions, line_map, (w,h)


@dataclass
class PageQCMetrics:
    filename: str
    mode: str
    regions_count: int
    lines_count: int
    mean_line_height: float
    height_variance: float
    baseline_ratio: float  # avg baseline_length / bbox_width
    processing_time: float
    fallback_used: bool = False


def _compute_qc_metrics(
    filename: str,
    mode: str,
    regions: List[Region],
    line_map: Dict[str, SegLine],
    elapsed: float,
    fallback: bool = False
) -> PageQCMetrics:
    """Compute QC metrics for a segmented page."""
    lines = list(line_map.values())
    if not lines:
        return PageQCMetrics(
            filename=filename, mode=mode, regions_count=len(regions),
            lines_count=0, mean_line_height=0, height_variance=0,
            baseline_ratio=0, processing_time=elapsed, fallback_used=fallback
        )
    
    heights = [(l.bbox[3] - l.bbox[1]) for l in lines]
    mean_h = sum(heights) / len(heights)
    variance = sum((h - mean_h)**2 for h in heights) / len(heights)
    
    baseline_ratios = []
    for ln in lines:
        if ln.baseline and len(ln.baseline) >= 2:
            bl_len = sum(
                math.sqrt((ln.baseline[i+1][0]-ln.baseline[i][0])**2 + (ln.baseline[i+1][1]-ln.baseline[i][1])**2)
                for i in range(len(ln.baseline)-1)
            )
            bbox_w = ln.bbox[2] - ln.bbox[0]
            if bbox_w > 0:
                baseline_ratios.append(bl_len / bbox_w)
    
    avg_baseline_ratio = sum(baseline_ratios) / len(baseline_ratios) if baseline_ratios else 0.0
    
    return PageQCMetrics(
        filename=filename,
        mode=mode,
        regions_count=len(regions),
        lines_count=len(lines),
        mean_line_height=mean_h,
        height_variance=variance,
        baseline_ratio=avg_baseline_ratio,
        processing_time=elapsed,
        fallback_used=fallback
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="Batch-create PAGE XML from page images (standalone; GUI untouched)")
    p.add_argument('--input', required=True, help='Folder with page images')
    p.add_argument('--output', default='./pagexml/xml', help='Output folder for PAGE XML files')
    p.add_argument('--overlays', default='./pagexml/overlays', help='Overlay images folder (omit or set empty to disable)')
    p.add_argument('--device', default='cpu', choices=['cpu','cuda'], help='Device for Kraken (classical uses CPU)')
    p.add_argument('--max-columns', type=int, default=4, help='Upper bound for column clustering (1–4 typical)')
    p.add_argument('--min-line-height', type=int, default=8, help='Filter lines below this pixel height')
    p.add_argument('--deskew', action='store_true', help='Attempt light deskew before segmentation')
    p.add_argument('--mode', default='classical', choices=['classical', 'neural', 'auto'], help='Segmentation mode: classical, neural (blla), or auto (neural with classical fallback)')
    # Default to pagexml/blla.mlmodel relative to script location
    default_model = os.path.join(os.path.dirname(__file__), 'blla.mlmodel')
    p.add_argument('--neural-model', default=default_model, help='Path to neural segmentation model (for neural/auto mode)')
    p.add_argument('--qc-csv', default=None, help='Path to export QC metrics CSV (optional)')
    args = p.parse_args(argv)

    summary = run_batch(
        input_dir=args.input,
        output_dir=args.output,
        overlays_dir=(args.overlays if args.overlays else None),
        device=args.device,
        max_columns=args.max_columns,
        min_line_height=args.min_line_height,
        deskew=args.deskew,
        mode=args.mode,
        neural_model_path=args.neural_model,
        qc_csv_path=args.qc_csv,
        log=print,
        on_progress=lambda c,t: None,
        on_file=lambda name, regions, lines: None,
        stop_event=None,
    )
    # CLI exit codes: 0 ok, 1 no images, 2 bad input dir
    return summary.get('exit_code', 0)


def run_batch(
    input_dir: str,
    output_dir: str = './pagexml/xml',
    overlays_dir: Optional[str] = './pagexml/overlays',
    device: str = 'cpu',
    max_columns: int = 4,
    min_line_height: int = 8,
    deskew: bool = False,
    mode: str = 'classical',
    neural_model_path: str = 'blla.mlmodel',
    qc_csv_path: Optional[str] = None,
    log: Optional[callable] = None,
    on_progress: Optional[callable] = None,
    on_file: Optional[callable] = None,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, int]:
    """
    Executes the batch segmentation. Designed for reuse from a GUI.

    Callbacks:
      - log(msg: str)
      - on_progress(current: int, total: int)
      - on_file(name: str, regions: int, lines: int)
    Returns summary dict with keys: total, processed, errors, exit_code
    """
    def _log(msg: str):
        if log:
            log(msg)
        else:
            print(msg)

    if not os.path.isdir(input_dir):
        _log(f"Input dir not found: {input_dir}")
        return {'total': 0, 'processed': 0, 'errors': 0, 'exit_code': 2}

    os.makedirs(output_dir, exist_ok=True)
    if overlays_dir:
        os.makedirs(overlays_dir, exist_ok=True)

    # Initialize segmenter
    segmenter = KrakenLineSegmenter(device=device)

    images = _list_images(input_dir)
    total = len(images)
    if total == 0:
        _log(f"No images found in {input_dir}")
        return {'total': 0, 'processed': 0, 'errors': 0, 'exit_code': 1}

    _log(f"Processing {total} pages...")
    _log(f"Segmentation mode: {mode}")
    processed = 0
    errors = 0
    qc_metrics_list: List[PageQCMetrics] = []

    # Prepare QC CSV if requested
    if qc_csv_path:
        import csv
        qc_file = open(qc_csv_path, 'w', newline='', encoding='utf-8')
        qc_writer = csv.writer(qc_file)
        qc_writer.writerow(['filename', 'mode', 'regions', 'lines', 'mean_line_height', 'height_variance', 'baseline_ratio', 'time_sec', 'fallback'])
    else:
        qc_file = None
        qc_writer = None

    if on_progress:
        on_progress(0, total)
    
    try:
        for idx, img_path in enumerate(images, start=1):
            if stop_event and stop_event.is_set():
                _log("[INFO] Stop requested; terminating batch after current file.")
                break

            base = os.path.splitext(os.path.basename(img_path))[0]
            xml_path = os.path.join(output_dir, f"{base}.xml")
            try:
                fallback_used = False
                start_time = time.time()
                
                # Select segmentation mode
                if mode == 'neural':
                    regions, line_map, size, elapsed = process_image_neural(
                        img_path, neural_model_path, device, min_line_height
                    )
                    if not regions:
                        _log(f"[WARN] Neural segmentation returned no regions for {base}; writing minimal PAGE XML")
                    used_mode = 'neural'
                elif mode == 'auto':
                    # Try neural first, fallback to classical if fails
                    regions, line_map, size, elapsed = process_image_neural(
                        img_path, neural_model_path, device, min_line_height
                    )
                    if not regions or len(line_map) < 3:  # Fallback threshold
                        _log(f"[INFO] Neural segmentation insufficient for {base}; falling back to classical")
                        fallback_used = True
                        regions, line_map, size = process_image(
                            img_path, segmenter, max(1, min(max_columns, 8)), min_line_height
                        )
                        elapsed = time.time() - start_time
                    used_mode = 'neural' if not fallback_used else 'classical'
                else:  # classical
                    regions, line_map, size = process_image(
                        img_path, segmenter, max(1, min(max_columns, 8)), min_line_height
                    )
                    elapsed = time.time() - start_time
                    used_mode = 'classical'
                
                # Compute QC metrics
                metrics = _compute_qc_metrics(base, used_mode, regions, line_map, elapsed, fallback_used)
                qc_metrics_list.append(metrics)
                if qc_writer:
                    qc_writer.writerow([
                        metrics.filename, metrics.mode, metrics.regions_count, metrics.lines_count,
                        f"{metrics.mean_line_height:.1f}", f"{metrics.height_variance:.1f}",
                        f"{metrics.baseline_ratio:.2f}", f"{metrics.processing_time:.2f}",
                        'yes' if metrics.fallback_used else 'no'
                    ])

                # Check stop before writing outputs
                if stop_event and stop_event.is_set():
                    break
                    
                if not regions:
                    _log(f"[WARN] No lines detected for {base}; writing minimal PAGE XML")
                    _write_page_xml(img_path, size, [], {}, xml_path)
                else:
                    _write_page_xml(img_path, size, regions, line_map, xml_path)

                    # Check stop before overlay generation (can be slow)
                    if stop_event and stop_event.is_set():
                        break
                        
                    if overlays_dir:
                        try:
                            img = Image.open(img_path)
                            overlay = _draw_overlay(img, regions, line_map)
                            overlay_path = os.path.join(overlays_dir, f"{base}_overlay.png")
                            overlay.save(overlay_path)
                        except Exception as overlay_err:
                            import traceback
                            _log(f"[WARN] Failed to create overlay for {base}: {overlay_err}")
                            traceback.print_exc()

                if on_file:
                    on_file(base, len(regions), len(line_map))
                _log(f"[OK] {base} ({used_mode}): regions={len(regions)} lines={len(line_map)} time={elapsed:.2f}s")
                processed += 1
            except Exception as e:
                errors += 1
                _log(f"[ERROR] Failed on {base}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if on_progress:
                    on_progress(idx, total)
    finally:
        # Ensure QC file is closed even if loop exits early or has errors
        if qc_file:
            try:
                qc_file.close()
                _log(f"QC metrics exported to {qc_csv_path}")
            except Exception as close_err:
                _log(f"[WARN] Error closing QC file: {close_err}")
    
    # Summary stats
    if qc_metrics_list:
        avg_time = sum(m.processing_time for m in qc_metrics_list) / len(qc_metrics_list)
        avg_lines = sum(m.lines_count for m in qc_metrics_list) / len(qc_metrics_list)
        fallback_count = sum(1 for m in qc_metrics_list if m.fallback_used)
        _log(f"Summary: avg_time={avg_time:.2f}s avg_lines={avg_lines:.1f} fallbacks={fallback_count}")
    
    _log("Done.")
    return {'total': total, 'processed': processed, 'errors': errors, 'exit_code': 0 if processed>0 else 1}


if __name__ == '__main__':
    sys.exit(main())
