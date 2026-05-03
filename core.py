"""Core classes for image processing, loading, and difference generation."""

from __future__ import annotations

import abc
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
        # Create a heavily blurred version of the entire image
        blurred = cv2.GaussianBlur(image, (25, 25), 0)
        # Blend the blurred image with the original using the provided mask
        return np.where(mask[:, :, np.newaxis] == 255, blurred, image)


class ColorShiftModifier(BaseModifier):
    """Shifts the hue of the masked area."""

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Convert to HSV color space to easily manipulate hue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        # Shift hue by 45 degrees
        hsv[:, :, 0] = (hsv[:, :, 0] + 45) % 180 
        # Convert back to BGR (Fixed typo here)
        shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return np.where(mask[:, :, np.newaxis] == 255, shifted, image)


class PixelateModifier(BaseModifier):
    """Applies a pixelation (mosaic) effect to the masked area."""

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        # Shrink the image down, ensuring width and height are at least 1 pixel
        target_w = max(1, w // 15)
        target_h = max(1, h // 15)
        small = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        # Scale it back up without interpolation to create the blocky effect
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.where(mask[:, :, np.newaxis] == 255, pixelated, image)


class DifferenceGenerator:
    """Handles the creation of non-overlapping differences on an image."""

    def __init__(self, num_differences: int = 5) -> None:
        self.num_differences = num_differences
        # Instantiate the available modifier strategies
        self.modifiers: list[BaseModifier] = [
            BlurModifier(),
            ColorShiftModifier(),
            PixelateModifier()
        ]

    def generate(self, image: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        """Generate differences and return the modified image and their coordinates."""
        modified_image = image.copy()
        h, w = image.shape[:2]
        
        # Calculate a dynamic radius based on image size (approx 5% of the smaller dimension)
        radius = max(min(h, w) // 20, 15)
        differences: list[tuple[int, int, int]] = []

        while len(differences) < self.num_differences:
            # Generate random center points, keeping a safe margin from the edges
            x = random.randint(radius + 10, w - radius - 10)
            y = random.randint(radius + 10, h - radius - 10)

            # Check for overlap with existing generated differences
            overlap = False
            for dx, dy, dr in differences:
                distance = np.sqrt((x - dx)**2 + (y - dy)**2)
                # Ensure they are separated by at least twice the radius + safe margin
                if distance < (radius + dr + 20):
                    overlap = True
                    break

            if not overlap:
                differences.append((x, y, radius))
                
                # Create a binary mask for the circular area
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (x, y), radius, 255, -1)
                
                # Randomly select and apply a modification strategy
                modifier = random.choice(self.modifiers)
                modified_image = modifier.apply(modified_image, mask)

        return modified_image, differences