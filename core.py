"""Core classes for image processing, loading, difference generation, and game state."""

from __future__ import annotations

import abc
import math
import os
import random

import cv2
import numpy as np


class ImageLoader:
    """Load BGR image from local file path with robust error handling."""

    @staticmethod
    def load_bgr(path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError("Image file is empty (0 bytes).")

        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            
        if data.size == 0:
            raise ValueError("Failed to read image data stream.")

        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("Corrupted or unsupported image file format.")
            
        return image


class BaseModifier(abc.ABC):
    """Abstract base class representing an image modification strategy."""

    @abc.abstractmethod
    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pass


class BlurModifier(BaseModifier):
    """Applies a Gaussian blur to the masked area."""
    KERNEL_SIZE = (25, 25)

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, self.KERNEL_SIZE, 0)
        return np.where(mask[:, :, np.newaxis] == 255, blurred, image)


class ColorShiftModifier(BaseModifier):
    """Shifts the hue of the masked area."""
    HUE_SHIFT = 45
    MAX_HUE = 180

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + self.HUE_SHIFT) % self.MAX_HUE
        shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return np.where(mask[:, :, np.newaxis] == 255, shifted, image)


class PixelateModifier(BaseModifier):
    """Applies a pixelation (mosaic) effect to the masked area."""
    DOWN_RATIO = 15

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        target_w = max(1, w // self.DOWN_RATIO)
        target_h = max(1, h // self.DOWN_RATIO)
        small = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.where(mask[:, :, np.newaxis] == 255, pixelated, image)


class DifferenceGenerator:
    """Handles the creation of non-overlapping differences on an image."""
    MARGIN = 10
    SPACING_PAD = 20
    MIN_RADIUS = 15
    MAX_ATTEMPTS = 500  # Prevent infinite loops on small/cluttered images

    def __init__(self, num_differences: int = 5) -> None:
        self.num_differences = num_differences
        self.modifiers: list[BaseModifier] = [
            BlurModifier(),
            ColorShiftModifier(),
            PixelateModifier()
        ]

    def generate(self, image: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        modified_image = image.copy()
        h, w = image.shape[:2]
        
        radius = max(min(h, w) // 20, self.MIN_RADIUS)
        safe_margin = radius + self.MARGIN
        
        # Guard against extremely small images
        if w <= safe_margin * 2 or h <= safe_margin * 2:
            raise ValueError(f"Image is too small to generate {self.num_differences} distinct differences.")

        differences: list[tuple[int, int, int]] = []
        attempts = 0

        while len(differences) < self.num_differences and attempts < self.MAX_ATTEMPTS:
            attempts += 1
            x = random.randint(safe_margin, w - safe_margin - 1)
            y = random.randint(safe_margin, h - safe_margin - 1)

            overlap = False
            for dx, dy, dr in differences:
                if math.hypot(x - dx, y - dy) < (radius + dr + self.SPACING_PAD):
                    overlap = True
                    break

            if not overlap:
                differences.append((x, y, radius))
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (x, y), radius, 255, -1)
                modifier = random.choice(self.modifiers)
                modified_image = modifier.apply(modified_image, mask)

        if len(differences) < self.num_differences:
            raise RuntimeError("Algorithm timed out: Could not find enough non-overlapping spots.")

        return modified_image, differences


class GameState:
    """Manages the logic and state of the current game session."""
    CLICK_TOLERANCE = 10

    def __init__(self, differences: list[tuple[int, int, int]], max_mistakes: int = 3) -> None:
        self.all_differences = differences
        self.unfound = list(differences)
        self.found: list[tuple[int, int, int]] = []
        self.mistakes = 0
        self.max_mistakes = max_mistakes
        self.is_active = True

    def check_click(self, x: int, y: int) -> tuple[str, tuple[int, int, int] | None]:
        if not self.is_active:
            return "inactive", None

        for diff in self.found:
            cx, cy, cr = diff
            if math.hypot(x - cx, y - cy) <= cr + self.CLICK_TOLERANCE:
                return "ignored", diff

        for diff in self.unfound:
            cx, cy, cr = diff
            if math.hypot(x - cx, y - cy) <= cr + self.CLICK_TOLERANCE:
                self.unfound.remove(diff)
                self.found.append(diff)
                
                # Check win condition internally
                if len(self.unfound) == 0:
                    self.is_active = False
                return "hit", diff

        self.mistakes += 1
        if self.mistakes >= self.max_mistakes:
            self.is_active = False
            
        return "miss", None

    def reveal_all(self) -> list[tuple[int, int, int]]:
        remaining = list(self.unfound)
        self.unfound.clear()
        self.is_active = False
        return remaining

    def is_won(self) -> bool:
        return len(self.unfound) == 0 and self.mistakes < self.max_mistakes

    def is_lost(self) -> bool:
        return self.mistakes >= self.max_mistakes