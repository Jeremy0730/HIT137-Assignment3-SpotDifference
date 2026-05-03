"""Tkinter GUI application."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from core import DifferenceGenerator, ImageLoader


class ImagePanel:
    """Render one BGR image into a canvas dynamically scaling with the window."""

    def __init__(self, canvas: tk.Canvas) -> None:
        self._canvas = canvas
        self._photo: ImageTk.PhotoImage | None = None
        self._bgr_image: np.ndarray | None = None
        
        # Bind the canvas resize event to redraw the image responsively
        self._canvas.bind("<Configure>", self._on_resize)

    def draw(self, bgr_image: np.ndarray) -> None:
        """Store the original image matrix and trigger a draw."""
        self._bgr_image = bgr_image
        self._redraw()

    def _on_resize(self, event: tk.Event) -> None:
        """Handle window resizing dynamically."""
        if self._bgr_image is not None:
            self._redraw()

    def _redraw(self) -> None:
        """Calculate dynamic scaling and draw the image centered on the canvas."""
        if self._bgr_image is None:
            return

        # Get current canvas dimensions
        canvas_w = self._canvas.winfo_width()
        canvas_h = self._canvas.winfo_height()

        # Prevent drawing if the canvas is too small (e.g., during initialization)
        if canvas_w < 10 or canvas_h < 10:
            return

        rgb = cv2.cvtColor(self._bgr_image, cv2.COLOR_BGR2RGB)
        img_h, img_w = rgb.shape[:2]

        # Calculate scale to fit within the canvas while maintaining aspect ratio
        scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        self._photo = ImageTk.PhotoImage(
            Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
        )
        
        self._canvas.delete("all")
        
        # Calculate offsets to center the image horizontally and vertically
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        
        self._canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self._photo)


class SpotDiffApp(tk.Tk):
    """Main application window for the Spot the Difference game."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Spot Difference")
        self.geometry("1100x700")

        # Top Control Panel
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Load Image", command=self._on_load).pack(side=tk.LEFT)

        # Body Frame for Canvases
        body = ttk.Frame(self, padding=(8, 0, 8, 8))
        body.pack(fill=tk.BOTH, expand=True)
        
        left_canvas = tk.Canvas(body, bg="#222", highlightthickness=0)
        right_canvas = tk.Canvas(body, bg="#222", highlightthickness=0)
        
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        self._left = ImagePanel(left_canvas)
        self._right = ImagePanel(right_canvas)
        
        # Store the coordinates of the generated differences
        self._differences: list[tuple[int, int, int]] = []

    def _on_load(self) -> None:
        """Triggered when the user wants to load a new image."""
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            # Step 1: Load original image
            src = ImageLoader.load_bgr(path)
            
            # Step 2: Generate 5 random differences using Polymorphic modifiers
            generator = DifferenceGenerator(num_differences=5)
            modified_src, self._differences = generator.generate(src)
            
            # Step 3: Draw both original and modified images
            self._left.draw(src)
            self._right.draw(modified_src)
            
        except Exception as exc:
            messagebox.showerror("Error", str(exc))