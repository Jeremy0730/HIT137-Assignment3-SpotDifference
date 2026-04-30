"""Tkinter app"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from core import ImageLoader


class ImagePanel:
    """Render one BGR image into a canvas."""

    def __init__(self, canvas: tk.Canvas) -> None:
        self._canvas = canvas
        self._photo: ImageTk.PhotoImage | None = None

    def draw(self, bgr_image) -> None:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(470 / w, 520 / h, 1.0)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS))
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)


class SpotDiffApp(tk.Tk):
    """Load one image and show original/clone."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Spot Difference")
        self.geometry("980x620")

        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Load Image", command=self._on_load).pack(side=tk.LEFT)

        body = ttk.Frame(self, padding=(8, 0, 8, 8))
        body.pack(fill=tk.BOTH, expand=True)
        left_canvas = tk.Canvas(body, bg="#222", highlightthickness=0)
        right_canvas = tk.Canvas(body, bg="#222", highlightthickness=0)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        self._left = ImagePanel(left_canvas)
        self._right = ImagePanel(right_canvas)

    def _on_load(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            src = ImageLoader.load_bgr(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))
            return
        self._left.draw(src)
        self._right.draw(src.copy())

