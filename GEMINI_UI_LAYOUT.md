# Gemini Advanced UI Layout - Improved Design

## Visual Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Gemini Advanced                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ☑ Early exit on first chunk      ☐ Auto continuation          │
│                                                                 │
│  Max passes:    [  2  ]            Min new chars:    [ 50  ]   │
│                                                                 │
│  Low-mode tokens: [ 6144 ]         Fallback %:      [ 0.6 ]    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Default Values

| Setting | Default | Range | Purpose |
|---------|---------|-------|---------|
| **Early exit on first chunk** | ✓ Checked | Boolean | Stream returns after first non-empty text (faster) |
| **Auto continuation** | ✗ Unchecked | Boolean | Perform additional passes to capture missed text |
| **Max passes** | 2 | 1-5 | Number of continuation attempts |
| **Min new chars** | 50 | 10-200 | Minimum chars to accept continuation chunk |
| **Low-mode tokens** | 6144 | 4096-8192 | Initial token budget for LOW thinking |
| **Fallback %** | 0.6 | 0.3-0.9 | Internal token fraction triggering fallback |

## Design Improvements

### ✅ Symmetry
- **Row 1**: Two checkboxes side-by-side
- **Row 2**: Two labeled inputs (Max passes | Min new chars)
- **Row 3**: Two labeled inputs (Low-mode tokens | Fallback %)

### ✅ Consistency
- All numeric inputs: **60px fixed width**
- Clear label spacing: **20px between groups**
- Uniform alignment with stretch on right

### ✅ User Experience
- **Default values pre-filled** → no need to remember
- **Tooltips on all controls** → hover for help
- **Logical grouping** → continuation settings in row 2
- **Auto continuation OFF by default** → optimized for speed

## Recommended Profiles

### Quick (Default - for most users)
```
☑ Early exit on first chunk
☐ Auto continuation
Max passes: 2
Min new chars: 50
Low-mode tokens: 6144
Fallback %: 0.6
```
**Result**: Fast streaming with early exit, automatic fallback at 60% internal token burn.

### Thorough (for complex manuscripts)
```
☐ Early exit on first chunk
☑ Auto continuation
Max passes: 3
Min new chars: 50
Low-mode tokens: 7168
Fallback %: 0.7
```
**Result**: Full stream collection, 3 continuation passes, higher token budget & fallback threshold.

### Conservative (minimize API costs)
```
☑ Early exit on first chunk
☐ Auto continuation
Max passes: 1
Min new chars: 75
Low-mode tokens: 4096
Fallback %: 0.5
```
**Result**: Aggressive early exit & fallback, single-pass only, lower token budget.

## Tooltips Reference

### Early exit on first chunk
> If checked, streaming returns after first non-empty text chunk. Uncheck to collect full stream.

### Auto continuation
> If checked, performs additional continuation calls to capture missed trailing text.

### Max passes
> Maximum number of continuation attempts (2-3 recommended)

### Min new chars
> Minimum number of new characters required to accept a continuation chunk.

### Low-mode tokens
> Initial max_output_tokens for LOW thinking before fallback escalation (4096-8192).

### Fallback %
> Fraction of token budget consumed internally (no output) that triggers early fallback (0.5-0.8).

## Code Changes Summary

1. **Reorganized layout**: 3 rows instead of 4, better visual balance
2. **Fixed widths**: All inputs 60px for uniformity
3. **Pre-filled defaults**: No empty placeholders; actual values shown
4. **Checkbox grouping**: Both checkboxes in row 1 for scanning ease
5. **Label alignment**: Consistent "Label: [input]" pattern across rows 2-3

## Testing Checklist

- [x] All fields have default values
- [x] All fields compile without errors
- [x] Tooltips present on all controls
- [x] Layout symmetric and visually balanced
- [x] Input widths uniform (60px)
- [x] Labels clearly identify purpose
- [x] Defaults match recommended "Quick" profile
