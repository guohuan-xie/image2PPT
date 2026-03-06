from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .config import PreprocessSettings


@dataclass(slots=True)
class PreprocessResult:
    source_path: Path
    image: Image.Image
    rgba: np.ndarray
    segmented_image: Image.Image
    segmented_rgba: np.ndarray
    original_size: tuple[int, int]
    processed_size: tuple[int, int]


def preprocess_image(source_path: Path, settings: PreprocessSettings) -> PreprocessResult:
    image = Image.open(source_path).convert("RGBA")
    original_size = image.size

    max_side = max(image.size)
    if max_side > settings.max_side_px:
        scale = settings.max_side_px / max_side
        resized = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        image = image.resize(resized, Image.Resampling.LANCZOS)

    if settings.blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(settings.blur_radius))

    rgba = np.array(image, dtype=np.uint8)
    quantized = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=settings.quantize_colors)
    segmented_image = quantized.convert("RGBA")
    segmented_rgba = np.array(segmented_image, dtype=np.uint8)

    return PreprocessResult(
        source_path=source_path,
        image=image,
        rgba=rgba,
        segmented_image=segmented_image,
        segmented_rgba=segmented_rgba,
        original_size=original_size,
        processed_size=image.size,
    )
