from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import OcrSettings
from .preprocess import PreprocessResult
from .scene_graph import BoundingBox, PrimitiveNode

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover - optional runtime dependency
    RapidOCR = None

_OCR_ENGINE = None


@dataclass(slots=True)
class OcrBox:
    text: str
    score: float
    bbox: BoundingBox
    polygon: tuple[tuple[float, float], ...]


def detect_text_boxes(preprocessed: PreprocessResult, settings: OcrSettings) -> list[OcrBox]:
    if not settings.enabled or RapidOCR is None:
        return []

    engine = _get_engine()
    if engine is None:
        return []

    try:
        result, _ = engine(preprocessed.rgba)
    except Exception:
        return []

    if not result:
        return []

    boxes: list[OcrBox] = []
    for index, item in enumerate(result):
        if len(item) < 3:
            continue
        raw_box, raw_text, raw_score = item[0], str(item[1]).strip(), float(item[2])
        if raw_score < settings.min_score or len(raw_text) < settings.min_text_length:
            continue

        points = np.array(raw_box, dtype=float)
        if points.shape != (4, 2):
            continue
        min_x = max(0.0, float(np.min(points[:, 0])) - settings.text_padding_px)
        min_y = max(0.0, float(np.min(points[:, 1])) - settings.text_padding_px)
        max_x = min(float(preprocessed.processed_size[0]), float(np.max(points[:, 0])) + settings.text_padding_px)
        max_y = min(float(preprocessed.processed_size[1]), float(np.max(points[:, 1])) + settings.text_padding_px)
        if max_x <= min_x or max_y <= min_y:
            continue
        boxes.append(
            OcrBox(
                text=raw_text,
                score=raw_score,
                bbox=BoundingBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y),
                polygon=tuple((float(point[0]), float(point[1])) for point in points.tolist()),
            )
        )
        if len(boxes) >= settings.max_text_boxes:
            break

    boxes.sort(key=lambda item: (item.bbox.y, item.bbox.x))
    return boxes


def build_text_nodes(preprocessed: PreprocessResult, boxes: list[OcrBox]) -> list[PrimitiveNode]:
    nodes: list[PrimitiveNode] = []
    for index, box in enumerate(boxes):
        text_color = estimate_text_color(preprocessed.rgba, box.bbox)
        font_size = max(8.0, box.bbox.height * 0.7)
        nodes.append(
            PrimitiveNode(
                id=f"text-{index}",
                primitive_type="text",
                bbox=box.bbox,
                z_index=0,
                fill_color=None,
                stroke_color=None,
                text=box.text,
                text_color=text_color,
                font_size=font_size,
            )
        )
    return nodes


def build_text_mask(rgba: np.ndarray, boxes: list[OcrBox], settings: OcrSettings) -> np.ndarray:
    mask = np.zeros(rgba.shape[:2], dtype=np.uint8)
    for box in boxes:
        refined = refine_text_mask(rgba, box, settings)
        if refined is not None:
            mask |= refined
    return mask


def estimate_text_color(rgba: np.ndarray, bbox: BoundingBox) -> tuple[int, int, int, int]:
    x1 = max(0, int(np.floor(bbox.x)))
    y1 = max(0, int(np.floor(bbox.y)))
    x2 = min(rgba.shape[1], int(np.ceil(bbox.x + bbox.width)))
    y2 = min(rgba.shape[0], int(np.ceil(bbox.y + bbox.height)))
    crop = rgba[y1:y2, x1:x2]
    if crop.size == 0:
        return (0, 0, 0, 255)

    alpha_mask = crop[:, :, 3] > 0
    if not np.any(alpha_mask):
        return (0, 0, 0, 255)

    pixels = crop[alpha_mask][:, :3]
    luminance = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    darkest = pixels[luminance <= np.quantile(luminance, 0.25)]
    if darkest.size == 0:
        darkest = pixels
    color = np.median(darkest, axis=0).astype(int)
    return (int(color[0]), int(color[1]), int(color[2]), 255)


def refine_text_mask(rgba: np.ndarray, box: OcrBox, settings: OcrSettings) -> np.ndarray | None:
    x1 = max(0, int(np.floor(box.bbox.x)))
    y1 = max(0, int(np.floor(box.bbox.y)))
    x2 = min(rgba.shape[1], int(np.ceil(box.bbox.x + box.bbox.width)))
    y2 = min(rgba.shape[0], int(np.ceil(box.bbox.y + box.bbox.height)))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = rgba[y1:y2, x1:x2]
    alpha_mask = crop[:, :, 3] > 0
    if not np.any(alpha_mask):
        return None

    polygon_mask = polygon_to_local_mask(crop.shape[:2], box.polygon, x1, y1)
    if not np.any(polygon_mask):
        polygon_mask = np.ones(crop.shape[:2], dtype=np.uint8)

    background_rgb = estimate_local_background_color(rgba, box, polygon_mask, x1, y1, x2, y2, settings)
    crop_rgb = crop[:, :, :3].astype(np.float32)
    color_distance = np.linalg.norm(crop_rgb - background_rgb[None, None, :], axis=2)
    luminance = rgb_to_luminance(crop_rgb)
    background_luminance = float(rgb_to_luminance(background_rgb))

    candidate_mask = (
        (polygon_mask > 0)
        & alpha_mask
        & (
            (color_distance >= settings.text_mask_min_color_distance)
            | (np.abs(luminance - background_luminance) >= settings.text_mask_min_luminance_delta)
        )
    )
    if not np.any(candidate_mask):
        return None

    if settings.text_mask_dilate_px > 0:
        kernel_size = settings.text_mask_dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        candidate_mask = cv2.dilate(candidate_mask.astype(np.uint8), kernel, iterations=1) > 0

    mask = np.zeros(rgba.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = candidate_mask.astype(np.uint8)
    return mask


def polygon_to_local_mask(
    shape: tuple[int, int],
    polygon: tuple[tuple[float, float], ...],
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    if not polygon:
        return np.zeros(shape, dtype=np.uint8)

    local_points = np.array(
        [[int(round(px - offset_x)), int(round(py - offset_y))] for px, py in polygon],
        dtype=np.int32,
    )
    if local_points.ndim != 2 or local_points.shape[0] < 3:
        return np.zeros(shape, dtype=np.uint8)

    local_points[:, 0] = np.clip(local_points[:, 0], 0, max(0, shape[1] - 1))
    local_points[:, 1] = np.clip(local_points[:, 1], 0, max(0, shape[0] - 1))
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [local_points], 1)
    return mask


def estimate_local_background_color(
    rgba: np.ndarray,
    box: OcrBox,
    polygon_mask: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    settings: OcrSettings,
) -> np.ndarray:
    crop = rgba[y1:y2, x1:x2]
    crop_alpha = crop[:, :, 3] > 0
    local_background_pixels = crop[(polygon_mask == 0) & crop_alpha][:, :3]
    if len(local_background_pixels) >= 8:
        return np.median(local_background_pixels, axis=0).astype(np.float32)

    pad = max(1, settings.text_background_sample_px)
    ex1 = max(0, x1 - pad)
    ey1 = max(0, y1 - pad)
    ex2 = min(rgba.shape[1], x2 + pad)
    ey2 = min(rgba.shape[0], y2 + pad)
    expanded = rgba[ey1:ey2, ex1:ex2]
    expanded_alpha = expanded[:, :, 3] > 0
    expanded_mask = np.ones(expanded.shape[:2], dtype=bool)
    expanded_mask[y1 - ey1 : y2 - ey1, x1 - ex1 : x2 - ex1] = False
    ring_pixels = expanded[expanded_mask & expanded_alpha][:, :3]
    if len(ring_pixels) >= 8:
        return np.median(ring_pixels, axis=0).astype(np.float32)

    fallback = crop[crop_alpha][:, :3]
    if len(fallback) == 0:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)
    return np.median(fallback, axis=0).astype(np.float32)


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _get_engine():
    global _OCR_ENGINE
    if RapidOCR is None:
        return None
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE
