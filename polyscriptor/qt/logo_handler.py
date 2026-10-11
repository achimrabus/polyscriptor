"""
Logo Handler for Polyscriptor HTR Application

Manages application logo loading, fallback text logo, and icon generation.
"""

from pathlib import Path
from typing import Optional, Tuple
from PyQt6.QtGui import QPixmap, QIcon, QImage, QPainter, QFont, QColor
from PyQt6.QtCore import Qt, QSize


class LogoHandler:
    """Handles logo loading with fallback to text-based logo."""

    # Logo file search paths (in order of preference)
    LOGO_PATHS = [
        "assets/logo.png",
        "assets/polyscriptor_logo.png",
        "assets/polyscript_logo.png",  # Legacy support
        "logo.png",
        "polyscriptor_logo.png",
        "polyscript_logo.png",  # Legacy support
        "assets/logo.svg",
        "assets/polyscriptor_logo.svg",
        "assets/polyscript_logo.svg",  # Legacy support
    ]

    # Supported image formats
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.svg', '.ico', '.bmp']

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize logo handler.

        Args:
            base_dir: Base directory for asset search (defaults to script directory)
        """
        self.base_dir = base_dir or Path(__file__).parent
        self._logo_pixmap: Optional[QPixmap] = None
        self._logo_icon: Optional[QIcon] = None

    def find_logo_file(self) -> Optional[Path]:
        """
        Search for logo file in predefined paths.

        Returns:
            Path to logo file, or None if not found
        """
        for logo_path in self.LOGO_PATHS:
            full_path = self.base_dir / logo_path
            if full_path.exists():
                return full_path

        # Search assets directory for any supported image
        assets_dir = self.base_dir / "assets"
        if assets_dir.exists():
            for ext in self.SUPPORTED_FORMATS:
                for file in assets_dir.glob(f"*{ext}"):
                    if 'logo' in file.stem.lower():
                        return file

        return None

    def create_text_logo(self, width: int = 400, height: int = 120) -> QPixmap:
        """
        Create a text-based logo as fallback.

        Args:
            width: Logo width in pixels
            height: Logo height in pixels

        Returns:
            QPixmap with text-based logo
        """
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Gradient background (optional)
        # gradient = QLinearGradient(0, 0, width, height)
        # gradient.setColorAt(0, QColor(41, 128, 185))  # Blue
        # gradient.setColorAt(1, QColor(142, 68, 173))  # Purple
        # painter.fillRect(0, 0, width, height, gradient)

        # Main text: "Polyscriptor"
        font = QFont("Arial", 36, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(41, 128, 185))  # Blue
        painter.drawText(0, 0, width, height * 0.6,
                        Qt.AlignmentFlag.AlignCenter, "Polyscriptor")

        # Subtitle: "Multi-Engine HTR"
        font_small = QFont("Arial", 14)
        painter.setFont(font_small)
        painter.setPen(QColor(127, 140, 141))  # Gray
        painter.drawText(0, height * 0.5, width, height * 0.5,
                        Qt.AlignmentFlag.AlignCenter, "Multi-Engine HTR")

        painter.end()
        return pixmap

    def create_icon_from_text(self, size: int = 64) -> QIcon:
        """
        Create application icon from text.

        Args:
            size: Icon size in pixels

        Returns:
            QIcon with text-based icon
        """
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background circle
        painter.setBrush(QColor(41, 128, 185))  # Blue
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, size - 4, size - 4)

        # Letter "P"
        font = QFont("Arial", int(size * 0.6), QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))  # White
        painter.drawText(0, 0, size, size,
                        Qt.AlignmentFlag.AlignCenter, "П")  # Cyrillic P

        painter.end()
        return QIcon(pixmap)

    def get_logo_pixmap(self, width: Optional[int] = None,
                       height: Optional[int] = None) -> QPixmap:
        """
        Get logo pixmap, loading from file or creating fallback.

        Args:
            width: Desired width (maintains aspect ratio if only one dimension given)
            height: Desired height (maintains aspect ratio if only one dimension given)

        Returns:
            QPixmap with logo
        """
        if self._logo_pixmap is None:
            logo_file = self.find_logo_file()

            if logo_file:
                print(f"[Logo] Loading logo from: {logo_file}")
                self._logo_pixmap = QPixmap(str(logo_file))
            else:
                print("[Logo] No logo file found, creating text-based logo")
                self._logo_pixmap = self.create_text_logo()

        # Scale if dimensions provided
        if width or height:
            if width and height:
                return self._logo_pixmap.scaled(width, height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
            elif width:
                return self._logo_pixmap.scaledToWidth(width,
                    Qt.TransformationMode.SmoothTransformation)
            else:
                return self._logo_pixmap.scaledToHeight(height,
                    Qt.TransformationMode.SmoothTransformation)

        return self._logo_pixmap

    def get_icon(self) -> QIcon:
        """
        Get application icon.

        Returns:
            QIcon for window/taskbar
        """
        if self._logo_icon is None:
            logo_file = self.find_logo_file()

            if logo_file:
                self._logo_icon = QIcon(str(logo_file))
            else:
                self._logo_icon = self.create_icon_from_text()

        return self._logo_icon


# Global singleton instance
_logo_handler: Optional[LogoHandler] = None


def get_logo_handler(base_dir: Optional[Path] = None) -> LogoHandler:
    """
    Get global logo handler instance.

    Args:
        base_dir: Base directory for asset search

    Returns:
        Global LogoHandler instance
    """
    global _logo_handler
    if _logo_handler is None:
        _logo_handler = LogoHandler(base_dir)
    return _logo_handler
