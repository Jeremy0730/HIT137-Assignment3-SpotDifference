"""Core classes for v1 (load + clone preview)."""

from __future__ import annotations

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

