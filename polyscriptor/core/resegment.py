"""Targeted re-segmentation inside user-drawn rectangles.

Ported from the desktop GUI: re-run Kraken blla segmentation on each drawn
crop, drop the regions overlapping the crops, keep everything else, and
re-index the merged layout.
"""

from typing import List, Optional, Tuple


def _overlaps(b1, b2) -> bool:
    return not (b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1])


def resegment_rects(
    pil_image,
    lines: list,
    seg_regions: List[dict],
    line_regions: List[int],
    rects: List[List[int]],
    *,
    device: str = "cpu",
    model_path: Optional[str] = None,
    max_columns: int = 4,
    split_width_fraction: float = 0.40,
    min_lines_to_split: int = 10,
    text_direction: str = "horizontal-lr",
) -> Tuple[list, List[dict], List[int]]:
    """Return (lines, seg_regions, line_regions) after replacing the regions
    overlapping *rects* with freshly segmented content.

    *lines* are LineSegment objects; *seg_regions* dicts as produced by
    ``_run_segmentation`` ({id, bbox, num_lines}). Raises RuntimeError when
    the segmenter cannot crop a rectangle.
    """
    from polyscriptor.core.kraken_segmenter import KrakenLineSegmenter
    from polyscriptor.inference.page import LineSegment

    W, H = pil_image.size

    # Region pairs for KEPT regions (no overlap with any drawn rect)
    kept_lines: list = []
    kept_regions: List[dict] = []
    for ri, region in enumerate(seg_regions):
        region_lines = [l for l, lr in zip(lines, line_regions) if lr == ri]
        if any(_overlaps(region["bbox"], r) for r in rects):
            continue
        kept_regions.append(dict(region))
        kept_lines.extend(region_lines)

    segmenter = KrakenLineSegmenter(device=device)
    new_pairs: list = []  # (region dict, lines)

    for di, (dx1, dy1, dx2, dy2) in enumerate(rects):
        dx1, dy1 = max(0, dx1), max(0, dy1)
        dx2, dy2 = min(W, dx2), min(H, dy2)
        if dx2 <= dx1 or dy2 <= dy1:
            continue

        crop = pil_image.crop((dx1, dy1, dx2, dy2))
        seg_regions_out, seg_lines_out = segmenter.segment_with_regions(
            crop,
            device=device,
            model_path=model_path or None,
            max_columns=max_columns,
            split_width_fraction=split_width_fraction,
            min_lines_to_split=min_lines_to_split,
            text_direction=text_direction,
        )

        # Shift crop coordinates into page space
        new_lines = []
        for seg in seg_lines_out:
            sbx1, sby1, sbx2, sby2 = seg.bbox
            baseline = (
                [(px + dx1, py + dy1) for px, py in seg.baseline]
                if getattr(seg, "baseline", None) else None
            )
            new_lines.append(LineSegment(
                image=seg.image,
                bbox=(sbx1 + dx1, sby1 + dy1, sbx2 + dx1, sby2 + dy1),
                coords=baseline,
            ))

        line_offset = 0
        for sr in seg_regions_out:
            n = len(sr.line_ids)
            bx1, by1, bx2, by2 = sr.bbox
            region_dict = {
                "id": f"r_draw_{di}_{sr.id}",
                "bbox": [bx1 + dx1, by1 + dy1, bx2 + dx1, by2 + dy1],
                "num_lines": n,
            }
            new_pairs.append((region_dict, new_lines[line_offset:line_offset + n]))
            line_offset += n

    # Merge: kept first, then fresh; sort regions left-to-right by x-center
    merged = [(r, ls) for r, ls in zip(kept_regions, _group(kept_lines, kept_regions))] + new_pairs
    merged.sort(key=lambda pair: (pair[0]["bbox"][0] + pair[0]["bbox"][2]) / 2)

    out_lines: list = []
    out_regions: List[dict] = []
    out_line_regions: List[int] = []
    for ri, (region, region_lines) in enumerate(merged):
        region = dict(region)
        region["num_lines"] = len(region_lines)
        out_regions.append(region)
        for _ in region_lines:
            out_line_regions.append(ri)
        out_lines.extend(region_lines)

    return out_lines, out_regions, out_line_regions


def _group(lines: list, regions: List[dict]) -> list:
    """Split *lines* back into per-region chunks given each region's num_lines."""
    out = []
    offset = 0
    for r in regions:
        n = r.get("num_lines", 0)
        out.append(lines[offset:offset + n])
        offset += n
    return out
