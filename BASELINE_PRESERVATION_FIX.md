# Baseline Preservation Fix

**Date**: 2025-12-03
**Branch**: codex/review-repo-for-public-release-cleanliness
**Status**: ✅ Fixed and tested

## Problem

Baselines in PAGE XML were being lost or converted to straight horizontal lines during batch processing, regardless of whether the input had curved baselines (from Kraken segmentation or ground truth PAGE XML).

### What Was Happening

1. **Ground Truth PAGE XML**: Curved baselines were ignored during import
2. **Kraken Segmentation**: Curved baselines were extracted but then discarded
3. **Export**: All baselines became straight horizontal lines (`y = bbox_bottom - 5`)

### Impact

- **Minimal for TrOCR/PyLaia/Qwen3**: These engines don't use baseline information
- **Potential impact for Party OCR**: Party can use baselines for better text alignment on curved manuscripts
- **Metadata loss**: Ground truth annotations were not preserved

## Solution

### Files Modified

1. **[inference_page.py](inference_page.py)**:
   - Added `baseline` field to `LineSegment` dataclass
   - Updated `PageXMLSegmenter.segment_lines()` to extract baselines from `<Baseline>` elements

2. **[page_xml_exporter.py](page_xml_exporter.py)**:
   - Modified `export()` to use real baseline coordinates when available
   - Maintains fallback to synthetic straight baselines when baseline data is missing

3. **[batch_processing.py](batch_processing.py)**:
   - Fixed Kraken normalization to correctly preserve `baseline` attribute
   - Changed from incorrectly copying to `coords` → now correctly preserves in `baseline`

### Code Changes

#### LineSegment Dataclass
```python
@dataclass
class LineSegment:
    image: Image.Image
    bbox: Tuple[int, int, int, int]
    coords: Optional[List[Tuple[int, int]]] = None  # polygon coordinates
    baseline: Optional[List[Tuple[int, int]]] = None  # ← NEW: baseline coordinates
    text: Optional[str] = None
    confidence: Optional[float] = None
    char_confidences: Optional[List[float]] = None
```

#### PageXMLSegmenter Enhancement
```python
# Extract baseline (if available)
baseline = None
baseline_elem = text_line.find('page:Baseline', self.NS)
if baseline_elem is not None:
    baseline_str = baseline_elem.get('points')
    if baseline_str:
        baseline = self._parse_coords(baseline_str)

segment = LineSegment(
    image=line_img,
    bbox=bbox,
    coords=coords,
    baseline=baseline  # ← Baseline now extracted
)
```

#### PAGE XML Exporter Fix
```python
# Baseline (use real baseline if available, otherwise approximate from bbox)
baseline_elem = ET.SubElement(line, 'Baseline')
if hasattr(segment, 'baseline') and segment.baseline:
    # Use real baseline coordinates (curved)
    baseline_points_str = ' '.join(f'{x},{y}' for x, y in segment.baseline)
    baseline_elem.set('points', baseline_points_str)
else:
    # Fallback: approximate baseline from bbox (straight line)
    x1, y1, x2, y2 = segment.bbox
    baseline_y = y2 - 5
    baseline_elem.set('points', f'{x1},{baseline_y} {x2},{baseline_y}')
```

#### Batch Processing Kraken Normalization Fix
```python
# Before (WRONG - put baseline in coords):
coords=line.baseline if hasattr(line, 'baseline') else None,

# After (CORRECT - preserve baseline separately):
coords=None,  # Kraken doesn't provide polygon coords
baseline=line.baseline if hasattr(line, 'baseline') else None,
```

## Testing

All tests passed:

### ✅ Test 1: LineSegment Dataclass
- Verified `baseline` field exists
- Tested with and without baseline data
- Confirmed backward compatibility (baseline is optional)

### ✅ Test 2: PAGE XML Export
- Curved baselines correctly exported to PAGE XML
- Straight baselines used as fallback when baseline=None
- Proper XML formatting with curved coordinates

### ✅ Test 3: PAGE XML Import
- PageXMLSegmenter correctly extracts baselines from `<Baseline>` elements
- Baseline coordinates parsed as list of (x, y) tuples
- Works with complex curved baselines

### ✅ Test 4: Round-Trip Preservation
- PAGE XML → Parse → Export → Verify
- Curved baselines fully preserved through the pipeline
- Input baseline: `110,145 310,148 510,147 690,144`
- Output baseline: `110,145 310,148 510,147 690,144` ✅ **EXACT MATCH**

## Data Flow (After Fix)

```
Ground Truth PAGE XML
  ├─ <Baseline points="x1,y1 x2,y2 x3,y3 ..."/>  (curved)
  └─ <Coords points="..."/>                       (polygon)
          ↓
  PageXMLSegmenter.segment_lines()
  ├─ ✅ Extracts coords (polygon)
  └─ ✅ Extracts baseline (curved)
          ↓
  LineSegment(coords=polygon, baseline=curved_baseline)
  ├─ .coords = polygon coordinates
  └─ .baseline = curved baseline coordinates
          ↓
  page_xml_exporter.export()
  └─ Uses real baseline if available, else fallback
          ↓
  Output PAGE XML
  └─ <Baseline points="x1,y1 x2,y2 x3,y3 ..."/>  (✅ PRESERVED!)
```

## Backward Compatibility

✅ **Fully backward compatible**:
- `baseline` field is optional (defaults to `None`)
- Existing code that doesn't set baseline continues to work
- Fallback to straight baseline when baseline data is missing
- All existing functionality preserved

## Benefits

1. **Ground Truth Preservation**: PAGE XML baselines from Transkribus/manual annotation are now preserved
2. **Kraken Baseline Support**: Kraken's curved baselines are now exported correctly
3. **Party OCR Ready**: Party can now use accurate baselines for better text alignment
4. **Metadata Integrity**: Complete round-trip preservation of PAGE XML annotations
5. **Future-Proof**: Other engines that may use baselines can now benefit

## Verification Commands

```bash
# Verify LineSegment has baseline field
python3 -c "from inference_page import LineSegment; from dataclasses import fields; print([f.name for f in fields(LineSegment)])"

# Test round-trip preservation
python3 << 'EOF'
from inference_page import PageXMLSegmenter
from page_xml_exporter import PageXMLExporter
from PIL import Image

# Create test PAGE XML with curved baseline
# Parse → Export → Verify (see test scripts above)
EOF
```

## Notes

- This fix does NOT affect transcription quality for TrOCR/PyLaia/Qwen3 (they don't use baselines)
- The fix ensures metadata preservation for future use cases and Party OCR
- All automated tests pass
- No breaking changes to existing APIs
