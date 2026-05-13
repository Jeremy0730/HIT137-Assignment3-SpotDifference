"""Tkinter GUI application."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import cv2
from PIL import Image, ImageTk

from core import DifferenceGenerator, GameState, ImageLoader


class UIConstants:
    """Constants for UI configuration."""
    WINDOW_GEOMETRY = "1100x700"
    BG_COLOR = "#222"
    PADDING_OUTER = 8
    PADDING_INNER = 4
    MIN_CANVAS_DRAW_SIZE = 10
    CIRCLE_WIDTH = 3
    COLOR_FOUND = "red"
    COLOR_REVEAL = "blue"


class ImagePanel:
    """Render one BGR image into a canvas dynamically scaling with the window."""

    def __init__(self, canvas: tk.Canvas) -> None:
        self._canvas = canvas
        self._photo: ImageTk.PhotoImage | None = None
        self._bgr_image: np.ndarray | None = None
        
        self._scale: float = 1.0
        self._x_offset: int = 0
        self._y_offset: int = 0
        
        self._circles: list[tuple[int, int, int, str]] = []
        self.on_click_callback: Callable[[int, int], None] | None = None
        
        self._canvas.bind("<Configure>", self._on_resize)
        self._canvas.bind("<Button-1>", self._on_click)

    def draw(self, bgr_image: np.ndarray) -> None:
        self._bgr_image = bgr_image
        self._redraw()

    def add_circle(self, x: int, y: int, radius: int, color: str = UIConstants.COLOR_FOUND) -> None:
        self._circles.append((x, y, radius, color))
        self._redraw()

    def clear_circles(self) -> None:
        self._circles.clear()
        self._redraw()

    def _on_click(self, event: tk.Event) -> None:
        if self.on_click_callback is None or self._bgr_image is None or self._scale <= 0:
            return

        img_x = (event.x - self._x_offset) / self._scale
        img_y = (event.y - self._y_offset) / self._scale

        h, w = self._bgr_image.shape[:2]
        if 0 <= img_x < w and 0 <= img_y < h:
            self.on_click_callback(int(img_x), int(img_y))

    def _on_resize(self, event: tk.Event) -> None:
        if self._bgr_image is not None:
            self._redraw()

    def _redraw(self) -> None:
        if self._bgr_image is None:
            return

        canvas_w = self._canvas.winfo_width()
        canvas_h = self._canvas.winfo_height()

        if canvas_w < UIConstants.MIN_CANVAS_DRAW_SIZE or canvas_h < UIConstants.MIN_CANVAS_DRAW_SIZE:
            return

        rgb = cv2.cvtColor(self._bgr_image, cv2.COLOR_BGR2RGB)
        img_h, img_w = rgb.shape[:2]

        self._scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        new_w = max(1, int(img_w * self._scale))
        new_h = max(1, int(img_h * self._scale))

        self._photo = ImageTk.PhotoImage(
            Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
        )
        
        self._canvas.delete("all")
        
        self._x_offset = (canvas_w - new_w) // 2
        self._y_offset = (canvas_h - new_h) // 2
        
        self._canvas.create_image(self._x_offset, self._y_offset, anchor=tk.NW, image=self._photo)
        
        for cx, cy, r, color in self._circles:
            scaled_x = cx * self._scale + self._x_offset
            scaled_y = cy * self._scale + self._y_offset
            scaled_r = r * self._scale
            self._canvas.create_oval(
                scaled_x - scaled_r, scaled_y - scaled_r,
                scaled_x + scaled_r, scaled_y + scaled_r,
                outline=color, width=UIConstants.CIRCLE_WIDTH
            )


class SpotDiffApp(tk.Tk):
    """Main application window for the Spot the Difference game."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Spot Difference")
        self.geometry(UIConstants.WINDOW_GEOMETRY)

        self._game_state: GameState | None = None

        top = ttk.Frame(self, padding=UIConstants.PADDING_OUTER)
        top.pack(fill=tk.X)
        
        ttk.Button(top, text="Load Image", command=self._on_load).pack(side=tk.LEFT)
        
        self._reveal_btn = ttk.Button(
            top, text="Reveal", command=self._on_reveal, state=tk.DISABLED
        )
        self._reveal_btn.pack(side=tk.LEFT, padx=(UIConstants.PADDING_INNER, 0))
        
        self._status_var = tk.StringVar(value="Welcome! Please load an image to start.")
        ttk.Label(top, textvariable=self._status_var, font=("Arial", 11, "bold")).pack(side=tk.RIGHT)

        body = ttk.Frame(
            self, padding=(UIConstants.PADDING_OUTER, 0, UIConstants.PADDING_OUTER, UIConstants.PADDING_OUTER)
        )
        body.pack(fill=tk.BOTH, expand=True)
        
        left_canvas = tk.Canvas(body, bg=UIConstants.BG_COLOR, highlightthickness=0)
        right_canvas = tk.Canvas(body, bg=UIConstants.BG_COLOR, highlightthickness=0)
        
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, UIConstants.PADDING_INNER))
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(UIConstants.PADDING_INNER, 0))

        self._left = ImagePanel(left_canvas)
        self._right = ImagePanel(right_canvas)
        self._right.on_click_callback = self._handle_click

    def _update_status_bar(self) -> None:
        if not self._game_state:
            return
        left = len(self._game_state.unfound)
        mistakes = self._game_state.mistakes
        max_m = self._game_state.max_mistakes
        self._status_var.set(f"Differences Left: {left}   |   Mistakes: {mistakes} / {max_m}")

    def _on_reveal(self) -> None:
        """Reveal remaining differences. Can be used even if game is lost."""
        if not self._game_state or len(self._game_state.unfound) == 0:
            return
            
        remaining = self._game_state.reveal_all()
        for cx, cy, cr in remaining:
            self._left.add_circle(cx, cy, cr, color=UIConstants.COLOR_REVEAL)
            self._right.add_circle(cx, cy, cr, color=UIConstants.COLOR_REVEAL)
            
        self._update_status_bar()
        self._status_var.set("Game Over! All remaining differences revealed.")
        self._reveal_btn.config(state=tk.DISABLED) # Disable only after revealing

    def _handle_click(self, x: int, y: int) -> None:
        if not self._game_state or not self._game_state.is_active:
            return

        result, diff = self._game_state.check_click(x, y)

        if result == "hit" and diff is not None:
            self._left.add_circle(diff[0], diff[1], diff[2], color=UIConstants.COLOR_FOUND)
            self._right.add_circle(diff[0], diff[1], diff[2], color=UIConstants.COLOR_FOUND)
            self._update_status_bar()
            
            if self._game_state.is_won():
                self._status_var.set("Congratulations! You found all differences!")
                self._reveal_btn.config(state=tk.DISABLED) # Victory = no need to reveal
                messagebox.showinfo("Victory", "You successfully found all differences!")

        elif result == "miss":
            self._update_status_bar()
            if self._game_state.is_lost():
                self._status_var.set("Game Over! Too many mistakes. You can 'Reveal' to see remaining.")
                # Note: We DO NOT disable the reveal button here anymore.
                messagebox.showerror("Game Over", "You've reached the maximum number of mistakes.")

    def _on_load(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            src = ImageLoader.load_bgr(path)
            
            generator = DifferenceGenerator(num_differences=5)
            modified_src, differences = generator.generate(src)
            
            self._game_state = GameState(differences)
            self._left.clear_circles()
            self._right.clear_circles()
            
            self._left.draw(src)
            self._right.draw(modified_src)
            
            self._update_status_bar()
            # Enable reveal button whenever a valid image is loaded
            self._reveal_btn.config(state=tk.NORMAL)
            
        except Exception as exc:
            messagebox.showerror("Error", str(exc))