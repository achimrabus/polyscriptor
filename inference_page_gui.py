"""
GUI for whole-page OCR inference using TrOCR.

This provides a simple graphical interface for transcribing Ukrainian handwritten documents.

Usage:
    python inference_page_gui.py

Requirements:
    pip install tkinter pillow torch transformers scipy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from PIL import Image, ImageTk
import threading
from typing import Optional
import sys

# Import inference components
from inference_page import LineSegmenter, PageXMLSegmenter, TrOCRInference, LineSegment


class TrOCRGUI:
    """Simple GUI for TrOCR whole-page inference."""

    def __init__(self, root):
        self.root = root
        self.root.title("TrOCR Ukrainian Handwriting Recognition")
        self.root.geometry("1200x800")

        # State
        self.image_path: Optional[Path] = None
        self.xml_path: Optional[Path] = None
        self.checkpoint_path: Optional[Path] = None
        self.current_image: Optional[Image.Image] = None
        self.ocr: Optional[TrOCRInference] = None
        self.segments = []

        # Setup UI
        self._create_widgets()

    def _create_widgets(self):
        """Create all GUI widgets."""

        # Top frame - file selection
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Image selection
        ttk.Label(top_frame, text="Image:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.image_label = ttk.Label(top_frame, text="No image selected", foreground="gray")
        self.image_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(top_frame, text="Browse...", command=self._select_image).grid(row=0, column=2, padx=5)

        # Optional XML
        ttk.Label(top_frame, text="PAGE XML (optional):").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.xml_label = ttk.Label(top_frame, text="Not selected (will use automatic segmentation)", foreground="gray")
        self.xml_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Button(top_frame, text="Browse...", command=self._select_xml).grid(row=1, column=2, padx=5)

        # Checkpoint selection
        ttk.Label(top_frame, text="Model Checkpoint:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.checkpoint_label = ttk.Label(top_frame, text="No checkpoint selected", foreground="gray")
        self.checkpoint_label.grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Button(top_frame, text="Browse...", command=self._select_checkpoint).grid(row=2, column=2, padx=5)

        # Settings frame
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding="10")
        settings_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=5)

        ttk.Label(settings_frame, text="Beam Search:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.num_beams_var = tk.IntVar(value=4)
        ttk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.num_beams_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(settings_frame, text="Max Length:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.max_length_var = tk.IntVar(value=128)
        ttk.Spinbox(settings_frame, from_=64, to=256, textvariable=self.max_length_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5)

        # Background normalization checkbox
        self.normalize_bg_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            settings_frame,
            text="Normalize Background (enable if model was trained with normalization)",
            variable=self.normalize_bg_var
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=5, pady=5)

        # Action buttons
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.process_btn = ttk.Button(button_frame, text="Process Page", command=self._process_page, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(button_frame, text="Save Transcription", command=self._save_transcription, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(button_frame, text="Clear", command=self._clear)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=5)

        # Main content area
        content_frame = ttk.Frame(self.root)
        content_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)

        # Image preview (left)
        image_frame = ttk.LabelFrame(content_frame, text="Image Preview", padding="5")
        image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        self.image_canvas = tk.Canvas(image_frame, bg="white", width=400, height=500)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        # Transcription output (right)
        output_frame = ttk.LabelFrame(content_frame, text="Transcription", padding="5")
        output_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=50, height=30)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(4, weight=1)

    def _select_image(self):
        """Select input image."""
        path = filedialog.askopenfilename(
            title="Select Page Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.image_path = Path(path)
            self.image_label.config(text=self.image_path.name, foreground="black")
            self._load_image()
            self._update_buttons()

    def _select_xml(self):
        """Select optional PAGE XML."""
        path = filedialog.askopenfilename(
            title="Select PAGE XML (Optional)",
            filetypes=[
                ("XML files", "*.xml"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.xml_path = Path(path)
            self.xml_label.config(text=self.xml_path.name, foreground="black")
        self._update_buttons()

    def _select_checkpoint(self):
        """Select model checkpoint directory."""
        path = filedialog.askdirectory(
            title="Select TrOCR Checkpoint Directory"
        )
        if path:
            self.checkpoint_path = Path(path)
            self.checkpoint_label.config(text=self.checkpoint_path.name, foreground="black")
            self._update_buttons()

    def _load_image(self):
        """Load and display image preview."""
        if not self.image_path:
            return

        try:
            from PIL import ImageOps
            Image.MAX_IMAGE_PIXELS = None
            self.current_image = Image.open(self.image_path)
            self.current_image = ImageOps.exif_transpose(self.current_image)  # Fix EXIF orientation
            self.current_image = self.current_image.convert('RGB')

            # Create thumbnail for preview
            thumb = self.current_image.copy()
            thumb.thumbnail((400, 500), Image.Resampling.LANCZOS)

            # Display on canvas
            self.photo = ImageTk.PhotoImage(thumb)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(200, 250, image=self.photo)

            self.status_var.set(f"Loaded image: {self.image_path.name} ({self.current_image.width}x{self.current_image.height})")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def _update_buttons(self):
        """Update button states based on selections."""
        can_process = (self.image_path is not None and
                      self.checkpoint_path is not None)

        self.process_btn.config(state=tk.NORMAL if can_process else tk.DISABLED)

    def _process_page(self):
        """Process the page in a background thread."""
        # Disable buttons during processing
        self.process_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)

        # Start progress bar
        self.progress.start()
        self.status_var.set("Processing...")

        # Run in thread to avoid freezing GUI
        thread = threading.Thread(target=self._run_inference, daemon=True)
        thread.start()

    def _run_inference(self):
        """Run OCR inference (called in background thread)."""
        try:
            # Load model if not already loaded or if normalization setting changed
            normalize_bg = self.normalize_bg_var.get()
            if (self.ocr is None or
                self.ocr.checkpoint_path != self.checkpoint_path or
                self.ocr.normalize_bg != normalize_bg):
                self._update_status("Loading model...")
                # Use base model for processor (checkpoints don't include preprocessor_config.json)
                self.ocr = TrOCRInference(
                    str(self.checkpoint_path),
                    base_model="kazars24/trocr-base-handwritten-ru",
                    normalize_bg=normalize_bg  # NEW: pass normalization flag
                )

            # Segment lines
            self._update_status("Segmenting lines...")
            if self.xml_path:
                segmenter = PageXMLSegmenter(str(self.xml_path))
                self.segments = segmenter.segment_lines(self.current_image)
            else:
                segmenter = LineSegmenter()
                self.segments = segmenter.segment_lines(self.current_image)

            if not self.segments:
                self._show_error("No lines detected in the image!")
                return

            self._update_status(f"Transcribing {len(self.segments)} lines...")

            # Transcribe
            num_beams = self.num_beams_var.get()
            max_length = self.max_length_var.get()

            self.segments = self.ocr.transcribe_segments(
                self.segments,
                num_beams=num_beams,
                max_length=max_length,
                show_progress=False  # No tqdm in GUI
            )

            # Display results
            transcription = "\n".join(seg.text for seg in self.segments if seg.text)
            self._display_transcription(transcription)

            self._update_status(f"Complete! Transcribed {len(self.segments)} lines.")
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))

        except Exception as e:
            self._show_error(f"Processing failed:\n{e}")

        finally:
            # Stop progress and re-enable button
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))

    def _update_status(self, message: str):
        """Update status bar from background thread."""
        self.root.after(0, lambda: self.status_var.set(message))

    def _show_error(self, message: str):
        """Show error dialog from background thread."""
        self.root.after(0, lambda: messagebox.showerror("Error", message))
        self.root.after(0, lambda: self.status_var.set("Error occurred"))

    def _display_transcription(self, text: str):
        """Display transcription in text widget."""
        def update():
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(1.0, text)

        self.root.after(0, update)

    def _save_transcription(self):
        """Save transcription to file."""
        if not self.segments:
            messagebox.showwarning("Warning", "No transcription to save!")
            return

        # Default filename
        default_name = f"{self.image_path.stem}_transcription.txt" if self.image_path else "transcription.txt"

        path = filedialog.asksaveasfilename(
            title="Save Transcription",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if path:
            try:
                transcription = "\n".join(seg.text for seg in self.segments if seg.text)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(transcription)

                self.status_var.set(f"Saved to {Path(path).name}")
                messagebox.showinfo("Success", f"Transcription saved to:\n{path}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save:\n{e}")

    def _clear(self):
        """Clear all selections and outputs."""
        self.image_path = None
        self.xml_path = None
        self.current_image = None
        self.segments = []

        self.image_label.config(text="No image selected", foreground="gray")
        self.xml_label.config(text="Not selected (will use automatic segmentation)", foreground="gray")

        self.image_canvas.delete("all")
        self.output_text.delete(1.0, tk.END)

        self.save_btn.config(state=tk.DISABLED)
        self._update_buttons()

        self.status_var.set("Ready")


def main():
    """Launch the GUI."""
    root = tk.Tk()
    app = TrOCRGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
