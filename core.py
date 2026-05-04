"""Core classes for image processing, loading, difference generation, and game state."""

from __future__ import annotations

import abc
import math
import random

import cv2
import numpy as np


class ImageLoader:
    """Load BGR image from local file path."""

    @staticmethod
    def load_bgr(path: str) -> np.ndarray:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode image file.")
        return image


class BaseModifier(abc.ABC):
    """Abstract base class representing an image modification strategy (Polymorphism)."""

    @abc.abstractmethod
    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply a specific visual modification to the masked area."""
        pass


class BlurModifier(BaseModifier):
    """Applies a Gaussian blur to the masked area."""

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (25, 25), 0)
        return np.where(mask[:, :, np.newaxis] == 255, blurred, image)


class ColorShiftModifier(BaseModifier):
    """Shifts the hue of the masked area."""

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + 45) % 180
        shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return np.where(mask[:, :, np.newaxis] == 255, shifted, image)


class PixelateModifier(BaseModifier):
    """Applies a pixelation (mosaic) effect to the masked area."""

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        target_w = max(1, w // 15)
        target_h = max(1, h // 15)
        small = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.where(mask[:, :, np.newaxis] == 255, pixelated, image)


class DifferenceGenerator:
    """Handles the creation of non-overlapping differences on an image."""

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
        radius = max(min(h, w) // 20, 15)
        differences: list[tuple[int, int, int]] = []

        while len(differences) < self.num_differences:
            x = random.randint(radius + 10, w - radius - 10)
            y = random.randint(radius + 10, h - radius - 10)

            overlap = False
            for dx, dy, dr in differences:
                if math.hypot(x - dx, y - dy) < (radius + dr + 20):
                    overlap = True
                    break

            if not overlap:
                differences.append((x, y, radius))
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (x, y), radius, 255, -1)
                modifier = random.choice(self.modifiers)
                modified_image = modifier.apply(modified_image, mask)

        return modified_image, differences


class GameState:
    """Manages the logic and state of the current game session."""

    def __init__(self, differences: list[tuple[int, int, int]]) -> None:
        self.all_differences = differences
        self.unfound = list(differences)
        self.found: list[tuple[int, int, int]] = []
        self.mistakes = 0
        self.max_mistakes = 3
        self.is_active = True

    def check_click(self, x: int, y: int) -> tuple[str, tuple[int, int, int] | None]:
        """Check if a clicked coordinate hits a difference. Includes a small tolerance."""
        if not self.is_active:
            return "inactive", None

        # Tolerance makes clicking slightly easier for the user
        tolerance = 10 

        # Check if they clicked an already found difference (ignore it, no penalty)
        for diff in self.found:
            cx, cy, cr = diff
            if math.hypot(x - cx, y - cy) <= cr + tolerance:
                return "ignored", diff

        # Check if they hit a new difference
        for diff in self.unfound:
            cx, cy, cr = diff
            if math.hypot(x - cx, y - cy) <= cr + tolerance:
                self.unfound.remove(diff)
                self.found.append(diff)
                return "hit", diff

        # If we reach here, it's a completely wrong click
        self.mistakes += 1
        if self.mistakes >= self.max_mistakes:
            self.is_active = False
            
        return "miss", None

    def is_won(self) -> bool:
        return len(self.unfound) == 0 and self.is_active

    def is_lost(self) -> bool:
        return self.mistakes >= self.max_mistakes