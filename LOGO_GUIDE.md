# Logo Integration Guide

This document explains the logo system implemented in Polyscriptor and how to customize it.

## Current Status

✅ **Logo system implemented** (November 3, 2025)
- Automatic logo detection and loading
- Fallback text-based logo generation
- Window icon integration
- Demo logo created as placeholder

## Files Added

1. **[logo_handler.py](logo_handler.py)**: Logo management module
   - `LogoHandler` class: Searches for logo files, creates fallbacks
   - `get_logo_handler()`: Singleton accessor function
   - Supports PNG, JPG, SVG, ICO, BMP formats

2. **[create_demo_logo.py](create_demo_logo.py)**: Demo logo generator
   - Creates simple text-based placeholder logo
   - Generates application icon (256×256px circle with 'П')
   - Run: `python3 create_demo_logo.py`

3. **[assets/](assets/)**: Logo storage directory
   - `logo.png`: Main application logo (400×120px)
   - `icon.png`: Application icon (256×256px)
   - `README.md`: Detailed logo instructions

4. **[transcription_gui_plugin.py](transcription_gui_plugin.py)**: Modified GUI
   - Logo display at top of image panel (300px width)
   - Window icon set from logo/icon
   - Window title: "Polyscriptor - Multi-Engine HTR Tool"

## How the Logo System Works

### Detection Order

The `LogoHandler` searches for logo files in this order:
1. `assets/logo.png`
2. `assets/polyscript_logo.png`
3. `logo.png` (root directory)
4. `polyscript_logo.png` (root directory)
5. `assets/logo.svg`
6. `assets/polyscript_logo.svg`
7. Any file in `assets/` with 'logo' in the name

### Fallback Behavior

If no logo file is found, the system automatically generates a text-based logo:
- Main text: "Polyscriptor" (blue #2980b9, Arial Bold 36pt)
- Subtitle: "Multi-Engine HTR" (gray #7f8c8d, Arial 14pt)
- Transparent background, 400×120px

### Icon Generation

The application icon (taskbar/window icon) is:
- Loaded from logo file if available, or
- Generated as a circular blue badge with Cyrillic "П" (white, centered)

## Customizing the Logo

### Option 1: Replace Demo Logo

Simply replace `assets/logo.png` with your custom logo:

```bash
# Your logo should be ~400×120px (or similar 3.3:1 aspect ratio)
cp /path/to/your/logo.png assets/logo.png

# Restart the GUI
python3 transcription_gui_plugin.py
```

### Option 2: Add Professional Logo

Create a professional logo and save it as:
- **Primary**: `assets/logo.png` (400×120px, PNG with transparency)
- **Icon**: `assets/icon.png` (256×256px, square variant for taskbar)
- **Scalable**: `assets/logo.svg` (optional, vector format)

### Option 3: Use AI Generation

Generate a logo using AI tools:

**DALL-E Prompt Example**:
```
Logo for software application called "Polyscriptor", featuring Cyrillic and
Glagolitic manuscript letters combined with modern circuit board elements,
deep blue (#2980b9) and gold accents, clean minimalist design, 400x120
horizontal layout, professional software logo, transparent background
```

**Stable Diffusion Prompt Example**:
```
polyscript logo, medieval Cyrillic letters, Glagolitic script, modern AI
circuitry, deep blue and purple gradient, gold manuscript illumination,
software logo, clean professional design, white background, 4k, highly
detailed
```

**Midjourney Prompt Example**:
```
polyscript software logo, ancient Cyrillic and Glagolitic manuscript text
merged with futuristic AI neural network, deep blue #2980b9 and gold,
minimalist modern design, horizontal banner 400x120px --ar 10:3 --stylize 500
```

## Design Recommendations

### Visual Theme

**Concept**: "Polyscriptor" = Many Scripts + AI Technology
- **Historical**: Cyrillic (Ґ, Ѳ, Ѣ), Glagolitic glyphs (Ⰰ, Ⱂ, Ⱃ)
- **Modern**: Clean typography, subtle tech elements (circuit patterns, AI motifs)
- **Color palette**:
  - Primary: Deep blue (#2980b9) - trust, technology
  - Secondary: Purple (#8e44ad) - creativity, history
  - Accent: Gold (#f39c12) - illuminated manuscripts

### Technical Specs

**Main Logo** (`logo.png`):
- **Dimensions**: 400×120 pixels (3.3:1 aspect ratio)
- **Format**: PNG with transparency (alpha channel)
- **Resolution**: 72-96 DPI (screen optimized)
- **File size**: <100KB for fast loading

**Icon** (`icon.png`):
- **Dimensions**: 256×256 pixels (square)
- **Format**: PNG with transparency
- **Design**: Simplified version of main logo or standalone symbol
- **File size**: <50KB

**Vector Version** (`logo.svg`, optional):
- **Format**: SVG (Scalable Vector Graphics)
- **Use case**: Printing, large displays, future-proofing

### Typography

Recommended fonts for logo text:
- **Sans-serif**: Arial, Helvetica, Roboto, Inter (modern, clean)
- **Serif**: Playfair Display, Crimson Text (manuscript feel)
- **Display**: Custom Cyrillic fonts (e.g., PT Serif, Philosopher)

## Testing

After adding/updating a logo:

1. **Run the GUI**:
   ```bash
   python3 transcription_gui_plugin.py
   ```

2. **Check logo appearance**:
   - Logo appears at top of left panel (scaled to 300px width)
   - Window icon shows in title bar
   - Taskbar icon displays correctly

3. **Test fallback** (remove logo temporarily):
   ```bash
   mv assets/logo.png assets/logo.png.bak
   python3 transcription_gui_plugin.py
   # Should show text-based fallback logo
   mv assets/logo.png.bak assets/logo.png
   ```

## Professional Design Services

If you want a custom professional logo:

### Budget Options ($10-100)
- **Fiverr**: Search "software logo design" ($10-50)
- **99designs**: Logo contest, multiple designers compete ($199-499)
- **DesignCrowd**: Similar to 99designs ($100-300)

### AI Generation (Free-Low Cost)
- **DALL-E**: OpenAI, $15 for 115 credits (plenty for logo iterations)
- **Midjourney**: Discord-based, $10/month subscription
- **Stable Diffusion**: Free via DreamStudio or local installation

### Design Tools (DIY)
- **Canva**: canva.com, free tier with logo templates
- **Figma**: figma.com, free for individuals
- **Inkscape**: inkscape.org, free open-source SVG editor
- **GIMP**: gimp.org, free Photoshop alternative

## Examples

### Text-Based Logo (Current Demo)

```
┌────────────────────────────────────┐
│                                    │
│          Polyscriptor                │
│       Multi-Engine HTR             │
│                                    │
└────────────────────────────────────┘
```

### Unicode Art Logo Concept

```
ПоⰾⰐ𝕊𝕔𝕣𝕚𝕡𝕥
(Cyrillic П + Glagolitic Ⱂ + Latin)
```

### Professional Logo Ideas

1. **Manuscript + Circuit**: Illuminated Cyrillic letter 'П' with subtle circuit board traces
2. **Script Layers**: Overlapping Cyrillic, Glagolitic, and Latin alphabets with transparency
3. **AI Quill**: Modern AI chip combined with medieval quill pen, writing Cyrillic letters
4. **Book + Network**: Ancient manuscript book with neural network connections emerging

## Implementation Details

### Code Structure

**Logo Loading** ([logo_handler.py:82-100](logo_handler.py)):
```python
def get_logo_pixmap(self, width=None, height=None):
    if self._logo_pixmap is None:
        logo_file = self.find_logo_file()
        if logo_file:
            self._logo_pixmap = QPixmap(str(logo_file))
        else:
            self._logo_pixmap = self.create_text_logo()
    # Scale if needed...
```

**GUI Integration** ([transcription_gui_plugin.py:426-433](transcription_gui_plugin.py)):
```python
# Logo display at top
logo_handler = get_logo_handler()
logo_label = QLabel()
logo_pixmap = logo_handler.get_logo_pixmap(width=300)
logo_label.setPixmap(logo_pixmap)
logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
left_layout.addWidget(logo_label)
```

**Window Icon** ([transcription_gui_plugin.py:394-396](transcription_gui_plugin.py)):
```python
logo_handler = get_logo_handler()
self.setWindowIcon(logo_handler.get_icon())
```

## Future Enhancements

Potential improvements:
- [ ] Animated logo (fade-in on startup)
- [ ] Theme-aware logos (light/dark mode variants)
- [ ] Multiple logo sizes (small/medium/large presets)
- [ ] About dialog with logo and credits
- [ ] Splash screen with logo during startup
- [ ] Export results with logo watermark (optional)

## License

Logo assets are subject to the same MIT License as the project, unless you create a custom logo with different licensing terms.

---

**Status**: ✅ Logo system ready for use
**Created**: November 3, 2025
**Demo logo**: Available in `assets/logo.png` (replace with professional logo for production)
