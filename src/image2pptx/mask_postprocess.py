from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import SamSettings


@dataclass(slots=True)
class ComponentMask:
    id: str
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    score: float


def postprocess_component_masks(
    raw_components: list[ComponentMask],
    settings: SamSettings,
    image_shape: tuple[int, int],
    text_mask: np.ndarray | None = None,
) -> list[ComponentMask]:
    image_height, image_width = image_shape
    image_area = image_height * image_width

    filtered: list[ComponentMask] = []
    for component in raw_components:
        if component.area < settings.min_mask_area_px:
            continue
        if component.area / image_area > settings.max_mask_area_ratio:
            continue
        bbox_area = max(1, component.bbox[2] * component.bbox[3])
        bbox_area_ratio = bbox_area / image_area
        fill_ratio = component.area / bbox_area
        is_thin_component = is_thin_component_candidate(component, settings)
        if (
            not is_thin_component
            and bbox_area_ratio >= settings.sparse_mask_bbox_area_ratio
            and fill_ratio < settings.sparse_mask_fill_ratio
        ):
            continue
        if text_mask is not None and overlap_ratio(component.mask, text_mask > 0) >= settings.text_overlap_ratio_thresh:
            continue
        filtered.append(component)

    # Prefer smaller, more specific masks first and drop near-duplicates.
    filtered.sort(key=lambda item: (item.area, -item.score))
    deduped: list[ComponentMask] = []
    for component in filtered:
        duplicate = False
        for kept in deduped:
            if mask_iou(component.mask, kept.mask) >= settings.duplicate_iou_thresh:
                duplicate = True
                break
            if containment_ratio(component.mask, kept.mask) >= settings.containment_ratio_thresh:
                duplicate = True
                break
        if not duplicate:
            deduped.append(component)

    merged = merge_adjacent_components(deduped, settings.merge_gap_px)
    merged.sort(key=lambda item: item.area, reverse=True)
    return merged[: settings.max_components]


def merge_adjacent_components(components: list[ComponentMask], gap_px: int) -> list[ComponentMask]:
    if gap_px <= 0 or len(components) < 2:
        return components

    pending = components[:]
    merged: list[ComponentMask] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_px, gap_px))

    while pending:
        current = pending.pop(0)
        current_mask = current.mask.copy()
        current_score = current.score
        current_ids = [current.id]
        changed = True

        while changed:
            changed = False
            still_pending: list[ComponentMask] = []
            dilated_current = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1) > 0
            for candidate in pending:
                dilated_candidate = cv2.dilate(candidate.mask.astype(np.uint8), kernel, iterations=1) > 0
                if np.any(dilated_current & dilated_candidate):
                    current_mask = current_mask | candidate.mask
                    current_score = max(current_score, candidate.score)
                    current_ids.append(candidate.id)
                    changed = True
                else:
                    still_pending.append(candidate)
            pending = still_pending

        merged.append(
            ComponentMask(
                id="+".join(current_ids),
                mask=current_mask,
                bbox=bbox_from_mask(current_mask),
                area=int(np.count_nonzero(current_mask)),
                score=current_score,
            )
        )

    return merged


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return (x1, y1, x2 - x1, y2 - y1)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.count_nonzero(left & right)
    union = np.count_nonzero(left | right)
    if union == 0:
        return 0.0
    return intersection / union


def containment_ratio(inner: np.ndarray, outer: np.ndarray) -> float:
    inner_pixels = np.count_nonzero(inner)
    if inner_pixels == 0:
        return 0.0
    return np.count_nonzero(inner & outer) / inner_pixels


def overlap_ratio(left: np.ndarray, right: np.ndarray) -> float:
    left_pixels = np.count_nonzero(left)
    if left_pixels == 0:
        return 0.0
    return np.count_nonzero(left & right) / left_pixels


def is_thin_component_candidate(component: ComponentMask, settings: SamSettings) -> bool:
    width = max(1, component.bbox[2])
    height = max(1, component.bbox[3])
    aspect_ratio = max(width, height) / max(1, min(width, height))
    return (
        component.area >= settings.thin_component_min_area_px
        and max(width, height) >= settings.thin_component_min_length_px
        and aspect_ratio >= settings.thin_component_min_aspect_ratio
    )


def save_sam_debug_artifacts(
    rgba: np.ndarray,
    raw_components: list[ComponentMask],
    final_components: list[ComponentMask],
    artifacts_dir: Path,
) -> None:
    save_masks_json(raw_components, final_components, artifacts_dir / "sam_masks.json")
    save_mask_overlay(rgba, final_components, artifacts_dir / "sam_masks_overlay.png")
    save_component_boxes(rgba, final_components, artifacts_dir / "component_boxes.png")
    save_component_crops(rgba, final_components, artifacts_dir / "component_crops")


def save_masks_json(
    raw_components: list[ComponentMask],
    final_components: list[ComponentMask],
    output_path: Path,
) -> None:
    payload = {
        "raw_masks": [component_to_json(item) for item in raw_components],
        "final_masks": [component_to_json(item) for item in final_components],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def component_to_json(component: ComponentMask) -> dict:
    x, y, width, height = component.bbox
    return {
        "id": component.id,
        "bbox": {"x": x, "y": y, "width": width, "height": height},
        "area": component.area,
        "score": component.score,
    }


def save_mask_overlay(rgba: np.ndarray, components: list[ComponentMask], output_path: Path) -> None:
    base = Image.fromarray(rgba, mode="RGBA").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    colors = [
        (255, 99, 71, 90),
        (65, 105, 225, 90),
        (60, 179, 113, 90),
        (238, 130, 238, 90),
        (255, 165, 0, 90),
        (70, 130, 180, 90),
    ]
    for index, component in enumerate(components):
        color = colors[index % len(colors)]
        mask_image = Image.fromarray((component.mask.astype(np.uint8) * 255), mode="L")
        color_image = Image.new("RGBA", base.size, color)
        overlay.paste(color_image, (0, 0), mask_image)
    Image.alpha_composite(base, overlay).save(output_path)


def save_component_boxes(rgba: np.ndarray, components: list[ComponentMask], output_path: Path) -> None:
    image = Image.fromarray(rgba, mode="RGBA").convert("RGB")
    draw = ImageDraw.Draw(image)
    for component in components:
        x, y, width, height = component.bbox
        draw.rectangle((x, y, x + width, y + height), outline=(255, 99, 71), width=2)
    image.save(output_path)


def save_component_crops(rgba: np.ndarray, components: list[ComponentMask], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, component in enumerate(components):
        x, y, width, height = component.bbox
        crop = rgba[y : y + height, x : x + width].copy()
        local_mask = component.mask[y : y + height, x : x + width]
        crop[:, :, 3] = np.where(local_mask, crop[:, :, 3], 0)
        Image.fromarray(crop, mode="RGBA").save(output_dir / f"component_{index}.png")
