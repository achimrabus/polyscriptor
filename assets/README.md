# Assets Directory

This directory contains logo files and other graphical assets for the Polyscriptor HTR application.

## Logo Files

Place your application logo in this directory. Supported formats:
- PNG (recommended): `logo.png` or `polyscriptor_logo.png`
- SVG (scalable): `logo.svg` or `polyscriptor_logo.svg`
- JPG/JPEG: `logo.jpg`
- ICO (Windows icon): `logo.ico`

### Recommended Logo Specifications

**For best results**:
- **Size**: 400×120 pixels (or similar aspect ratio)
- **Format**: PNG with transparency
- **Color scheme**: Blue (#2980b9) and purple (#8e44ad) to match Cyrillic manuscript themes
- **Style**: Modern + historical fusion

### Logo Design Ideas

**Theme**: "Polyscriptor" = Many Scripts
- Visual elements: Cyrillic letters (Ґ, Ѳ, Ѣ), Glagolitic glyphs
- Color palette: Deep blues/purples (manuscript ink), gold accents
- Style inspiration: Medieval manuscripts meet modern AI

**AI Generation Prompts**:
```
Logo for 'Polyscriptor' - medieval Cyrillic and Glagolitic manuscript letters
combined with modern AI circuitry, deep blue and gold colors, minimalist
tech aesthetic, professional software logo
```

**Text-based Logo** (Unicode):
```
ПоⰾⰐ𝕊𝕔𝕣𝕚𝕡𝕥
(Combines Cyrillic П, Glagolitic Ⱂ, and Latin Script characters)
```

## Fallback Behavior

If no logo file is found, the application will automatically generate a text-based logo with:
- Main text: "Polyscriptor" (blue, Arial Bold 36pt)
- Subtitle: "Multi-Engine HTR" (gray, Arial 14pt)

## How to Add Your Logo

1. Create or obtain a logo file (PNG recommended)
2. Save it as `logo.png` or `polyscriptor_logo.png` in this directory
3. Restart the application - the logo will be automatically detected and loaded

## Icon

The application icon (taskbar/window icon) is generated from:
- Logo file (if available), or
- A circular blue badge with Cyrillic "П" (fallback)

## Creating a Logo

### Option 1: AI Generation (Free)
- **Stable Diffusion**: Local or via DreamStudio
- **DALL-E**: OpenAI (free tier available)
- **Midjourney**: Discord-based, free trial

### Option 2: Online Tools (Free)
- **Canva**: canva.com (free templates)
- **Figma**: figma.com (design tool)
- **Inkscape**: inkscape.org (free SVG editor)

### Option 3: Professional Design
- **Fiverr**: Starting at $10-50
- **99designs**: $200-500 (logo contest)
- **Upwork**: Hourly freelancers

## Example Logo Structure

```
assets/
├── logo.png              # Main logo (400×120px)
├── logo_square.png       # Square variant for icon (256×256px)
├── logo.svg              # Scalable vector version
└── README.md             # This file
```

## Testing

After adding a logo, you can test it by running:
```bash
python3 transcription_gui_plugin.py
```

The logo will appear:
1. At the top of the image panel (scaled to 300px width)
2. In the window title bar (as icon)
3. In the taskbar (as icon)

## License

Logo assets are subject to the same MIT License as the project, unless otherwise specified.
