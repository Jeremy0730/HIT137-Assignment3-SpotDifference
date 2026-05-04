"""Tkinter GUI application."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import cv2
from PIL import Image, ImageTk

from core import DifferenceGenerator, GameState, ImageLoader


class ImagePanel:
    """Render one BGR image into a canvas dynamically scaling with the window."""

    def __init__(self, canvas: tk.Canvas) -> None:
        self._canvas = canvas
        self._photo: ImageTk.PhotoImage | None = None
        self._bgr_image: np.ndarray | None = None
        
        # Scaling and offset values for coordinate mapping
        self._scale: float = 1.0
        self._x_offset: int = 0
        self._y_offset: int = 0
        
        # List of circles to draw (x, y, radius, color)
        self._circles: list[tuple[int, int, int, str]] = []
        
        # Callback for when the user clicks the image
        self.on_click_callback: Callable[[int, int], None] | None = None
        
        self._canvas.bind("<Configure>", self._on_resize)
        self._canvas.bind("<Button-1>", self._on_click)

    def draw(self, bgr_image: np.ndarray) -> None:
        self._bgr_image = bgr_image
        self._redraw()

    def add_circle(self, x: int, y: int, radius: int, color: str = "red") -> None:
        """Add a marker circle to be drawn on top of the image."""
        self._circles.append((x, y, radius, color))
        self._redraw()

    def clear_circles(self) -> None:
        """Remove all marker circles."""
        self._circles.clear()
        self._redraw()

    def _on_click(self, event: tk.Event) -> None:
        """Handle canvas click and map canvas coordinates back to image coordinates."""
        if self.on_click_callback is None or self._bgr_image is None or self._scale <= 0:
            return

        # Reverse the scaling and offset applied during _redraw
        img_x = (event.x - self._x_offset) / self._scale
        img_y = (event.y - self._y_offset) / self._scale

        h, w = self._bgr_image.shape[:2]
        
        # Ensure the click was actually within the bounds of the image
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

        if canvas_w < 10 or canvas_h < 10:
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
        
        # Draw the image
        self._canvas.create_image(self._x_offset, self._y_offset, anchor=tk.NW, image=self._photo)
        
        # Draw the circles correctly scaled and positioned
        for cx, cy, r, color in self._circles:
            scaled_x = cx * self._scale + self._x_offset
            scaled_y = cy * self._scale + self._y_offset
            scaled_r = r * self._scale
            self._canvas.create_oval(
                scaled_x - scaled_r, scaled_y - scaled_r,
                scaled_x + scaled_r, scaled_y + scaled_r,
                outline=color, width=3
            )


class SpotDiffApp(tk.Tk):
    """Main application window for the Spot the Difference game."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Spot Difference")
        self.geometry("1100x700")

        # Game State Tracker
        self._game_state: GameState | None = None

        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        
        ttk.Button(top, text="Load Image", command=self._on_load).pack(side=tk.LEFT)
        
        # Status Label to display game progress
        self._status_var = tk.StringVar(value="Welcome! Please load an image to start.")
        ttk.Label(top, textvariable=self._status_var, font=("Arial", 11, "bold")).pack(side=tk.RIGHT)

        body = ttk.Frame(self, padding=(8, 0, 8, 8))
        body.pack(fill=tk.BOTH, expand=True)
        
        left_canvas = tk.Canvas(body, bg="#222", highlightthickness=0)
        right_canvas = tk.Canvas(body, bg="#222", highlightthickness=0)
        
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        self._left = ImagePanel(left_canvas)
        self._right = ImagePanel(right_canvas)
        
        # Bind the click callback to the right panel ONLY (left panel is for reference)
        self._right.on_click_callback = self._handle_click

    def _update_status_bar(self) -> None:
        """Update the UI text with current game statistics."""
        if not self._game_state:
            return
        left = len(self._game_state.unfound)
        mistakes = self._game_state.mistakes
        max_m = self._game_state.max_mistakes
        self._status_var.set(f"Differences Left: {left}   |   Mistakes: {mistakes} / {max_m}")

    def _handle_click(self, x: int, y: int) -> None:
        """Process a user click on the right image panel."""
        if not self._game_state or not self._game_state.is_active:
            return

        result, diff = self._game_state.check_click(x, y)

        if result == "hit" and diff is not None:
            # Draw red circles on both images
            self._left.add_circle(diff[0], diff[1], diff[2], color="red")
            self._right.add_circle(diff[0], diff[1], diff[2], color="red")
            self._update_status_bar()
            
            if self._game_state.is_won():
                self._status_var.set("Congratulations! You found all differences!")
                messagebox.showinfo("Victory", "You successfully found all differences!")
                self._game_state.is_active = False # End game

        elif result == "miss":
            self._update_status_bar()
            if self._game_state.is_lost():
                self._status_var.set("Game Over! Too many mistakes.")
                messagebox.showerror("Game Over", "You've reached the maximum number of mistakes.")

    def _on_load(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            src = ImageLoader.load_bgr(path)
            
            generator = DifferenceGenerator(num_differences=5)
            modified_src, differences = generator.generate(src)
            
            # Reset game state
            self._game_state = GameState(differences)
            self._left.clear_circles()
            self._right.clear_circles()
            
            self._left.draw(src)
            self._right.draw(modified_src)
            
            self._update_status_bar()
            
        except Exception as exc:
            messagebox.showerror("Error", str(exc))