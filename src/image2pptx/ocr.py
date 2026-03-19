from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import OcrSettings
from .preprocess import PreprocessResult
from .scene_graph import BoundingBox, Node, PictureAssetNode, PrimitiveNode

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
        box_width = max_x - min_x
        box_height = max_y - min_y
        if (
            box_width < settings.min_text_box_width_px
            or box_height < settings.min_text_box_height_px
            or (box_width * box_height) < settings.min_text_box_area_px
        ):
            continue
        boxes.append(
            OcrBox(
                text=raw_text,
                score=raw_score,
                bbox=BoundingBox(x=min_x, y=min_y, width=box_width, height=box_height),
                polygon=tuple((float(point[0]), float(point[1])) for point in points.tolist()),
            )
        )
        if len(boxes) >= settings.max_text_boxes:
            break

    boxes.sort(key=lambda item: (item.bbox.y, item.bbox.x))
    return boxes


def build_text_nodes(
    preprocessed: PreprocessResult,
    boxes: list[OcrBox],
    settings: OcrSettings,
    artifacts_dir: Path | None = None,
) -> list[Node]:
    excluded_indexes: set[int] = set()
    nodes: list[Node] = []
    if settings.rasterize_dense_text and artifacts_dir is not None:
        dense_nodes, excluded_indexes = build_dense_text_picture_nodes(preprocessed, boxes, settings, artifacts_dir)
        nodes.extend(dense_nodes)

    text_node_index = 0
    for index, box in enumerate(boxes):
        if index in excluded_indexes:
            continue
        if box.bbox.height < settings.min_editable_text_height_px:
            continue
        text_color = estimate_text_color(preprocessed.rgba, box.bbox)
        font_size = estimate_font_size(box)
        nodes.append(
            PrimitiveNode(
                id=f"text-{text_node_index}",
                primitive_type="text",
                bbox=box.bbox,
                z_index=0,
                fill_color=None,
                stroke_color=None,
                text=box.text,
                text_color=text_color,
                font_size=font_size,
                text_align=infer_text_alignment(box, preprocessed.processed_size[0]),
                bold=box.bbox.height >= settings.heading_text_height_px,
                single_line=should_render_as_single_line(box),
            )
        )
        text_node_index += 1
    return nodes


def build_dense_text_picture_nodes(
    preprocessed: PreprocessResult,
    boxes: list[OcrBox],
    settings: OcrSettings,
    artifacts_dir: Path,
) -> tuple[list[PictureAssetNode], set[int]]:
    candidate_indexes = [
        index for index, box in enumerate(boxes) if box.bbox.height <= settings.rasterize_text_height_px
    ]
    if not candidate_indexes:
        return [], set()

    clusters = cluster_text_boxes(boxes, candidate_indexes, settings.rasterize_cluster_gap_px)
    text_asset_dir = artifacts_dir / "text_assets"
    nodes: list[PictureAssetNode] = []
    excluded_indexes: set[int] = set()

    for cluster_index, cluster in enumerate(clusters):
        if len(cluster) < settings.rasterize_cluster_min_boxes:
            continue

        cluster_boxes = [boxes[index] for index in cluster]
        cluster_bbox = union_bounding_boxes(
            [box.bbox for box in cluster_boxes],
            width=preprocessed.rgba.shape[1],
            height=preprocessed.rgba.shape[0],
            padding=settings.rasterize_cluster_padding_px,
        )
        mask = build_text_mask(preprocessed.rgba, cluster_boxes, settings)
        crop = crop_masked_region(preprocessed.rgba, mask, cluster_bbox)
        if crop is None:
            continue

        text_asset_dir.mkdir(parents=True, exist_ok=True)
        image_path = text_asset_dir / f"text_cluster_{cluster_index}.png"
        Image.fromarray(crop, mode="RGBA").save(image_path)
        nodes.append(
            PictureAssetNode(
                id=f"text-cluster-{cluster_index}",
                bbox=cluster_bbox,
                z_index=0,
                fill_color=None,
                image_path=str(image_path),
                source_region_id=None,
            )
        )
        excluded_indexes.update(cluster)

    return nodes, excluded_indexes


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


def estimate_font_size(box: OcrBox) -> float:
    return max(8.0, box.bbox.height * 0.72)


def infer_text_alignment(box: OcrBox, canvas_width: int) -> str:
    center_x = box.bbox.x + (box.bbox.width / 2.0)
    if box.bbox.width >= canvas_width * 0.32 and abs(center_x - (canvas_width / 2.0)) <= canvas_width * 0.12:
        return "center"
    if box.bbox.x >= canvas_width * 0.72 and box.bbox.width <= canvas_width * 0.2:
        return "right"
    return "left"


def should_render_as_single_line(box: OcrBox) -> bool:
    return "\n" not in box.text and box.bbox.width >= box.bbox.height * 3.0


def cluster_text_boxes(boxes: list[OcrBox], candidate_indexes: list[int], gap_px: int) -> list[list[int]]:
    clusters: list[list[int]] = []
    visited: set[int] = set()

    for start in candidate_indexes:
        if start in visited:
            continue
        queue = [start]
        cluster: list[int] = []
        visited.add(start)

        while queue:
            current = queue.pop()
            cluster.append(current)
            for other in candidate_indexes:
                if other in visited:
                    continue
                if expanded_boxes_intersect(boxes[current].bbox, boxes[other].bbox, gap_px):
                    visited.add(other)
                    queue.append(other)

        clusters.append(sorted(cluster))

    return clusters


def expanded_boxes_intersect(first: BoundingBox, second: BoundingBox, gap_px: int) -> bool:
    return not (
        first.x + first.width + gap_px < second.x
        or second.x + second.width + gap_px < first.x
        or first.y + first.height + gap_px < second.y
        or second.y + second.height + gap_px < first.y
    )


def union_bounding_boxes(
    boxes: list[BoundingBox],
    width: int,
    height: int,
    padding: int,
) -> BoundingBox:
    x1 = max(0.0, min(box.x for box in boxes) - padding)
    y1 = max(0.0, min(box.y for box in boxes) - padding)
    x2 = min(float(width), max(box.x + box.width for box in boxes) + padding)
    y2 = min(float(height), max(box.y + box.height for box in boxes) + padding)
    return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def crop_masked_region(rgba: np.ndarray, mask: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
    x1 = max(0, int(np.floor(bbox.x)))
    y1 = max(0, int(np.floor(bbox.y)))
    x2 = min(rgba.shape[1], int(np.ceil(bbox.x + bbox.width)))
    y2 = min(rgba.shape[0], int(np.ceil(bbox.y + bbox.height)))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = rgba[y1:y2, x1:x2].copy()
    local_mask = mask[y1:y2, x1:x2] > 0
    if not np.any(local_mask):
        return None
    crop[:, :, 3] = np.where(local_mask, crop[:, :, 3], 0)
    return crop


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
    elif settings.text_mask_dilate_px > 0:
        polygon_mask = dilate_mask(
            polygon_mask,
            max(settings.text_mask_dilate_px, adaptive_text_margin(box)),
        )

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

    candidate_mask = postprocess_text_candidate_mask(candidate_mask.astype(np.uint8), box, settings)

    mask = np.zeros(rgba.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = candidate_mask.astype(np.uint8)
    return mask


def adaptive_text_margin(box: OcrBox) -> int:
    return max(1, int(round(box.bbox.height * 0.12)))


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    kernel_size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def postprocess_text_candidate_mask(mask: np.ndarray, box: OcrBox, settings: OcrSettings) -> np.ndarray:
    processed = mask.astype(np.uint8)
    dilation = max(settings.text_mask_dilate_px, adaptive_text_margin(box))
    if dilation > 0:
        processed = dilate_mask(processed, dilation)

    close_px = max(settings.text_mask_close_px, dilation // 2)
    if close_px > 0:
        kernel_size = close_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    if settings.text_mask_min_component_area_px > 0:
        processed = remove_small_components(processed, settings.text_mask_min_component_area_px)

    return processed > 0


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area_px:
            filtered[labels == label] = 1
    return filtered


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
