from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import ShapeDetectionSettings


@dataclass(slots=True)
class DetectedRegion:
    id: str
    color: tuple[int, int, int, int]
    bbox: tuple[int, int, int, int]
    contour: np.ndarray
    approx_points: list[tuple[int, int]]
    area: float
    classification: str
    mask: np.ndarray


def detect_regions(rgba: np.ndarray, settings: ShapeDetectionSettings) -> list[DetectedRegion]:
    colors = np.unique(rgba.reshape(-1, 4), axis=0)
    regions: list[DetectedRegion] = []
    region_index = 0

    for color in colors:
        rgba_color = tuple(int(v) for v in color.tolist())
        if rgba_color[3] == 0:
            continue
        mask = np.all(rgba == color, axis=2).astype(np.uint8)
        if not np.any(mask):
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, num_labels):
            x, y, width, height, area = stats[label]
            if area < settings.min_area_px:
                continue

            component_mask = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            epsilon = max(1.0, settings.polygon_epsilon_ratio * perimeter)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_points = [(int(point[0][0]), int(point[0][1])) for point in approx]

            classification = classify_region(
                contour=contour,
                approx_points=approx_points,
                bbox=(x, y, width, height),
                area=float(area),
                settings=settings,
            )

            regions.append(
                DetectedRegion(
                    id=f"region-{region_index}",
                    color=rgba_color,
                    bbox=(int(x), int(y), int(width), int(height)),
                    contour=contour,
                    approx_points=approx_points,
                    area=float(area),
                    classification=classification,
                    mask=component_mask,
                )
            )
            region_index += 1

    regions.sort(key=lambda item: item.area, reverse=True)
    return regions


def classify_region(
    contour: np.ndarray,
    approx_points: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    area: float,
    settings: ShapeDetectionSettings,
) -> str:
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return "freeform"

    bbox_area = width * height
    fill_ratio = area / bbox_area if bbox_area else 0.0

    perimeter = cv2.arcLength(contour, True)
    circularity = 0.0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)

    aspect_ratio = max(width, height) / max(1, min(width, height))

    if len(approx_points) == 4 and fill_ratio >= settings.rect_fill_ratio_min:
        return "rect"

    if circularity >= settings.circle_circularity_min:
        return "circle"

    if aspect_ratio >= settings.line_aspect_ratio_min and fill_ratio >= 0.4:
        return "line"

    if len(approx_points) <= 8:
        return "freeform"

    return "svg_candidate"
