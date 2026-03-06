from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import Image2PptxConfig
from .detection import DetectedRegion, classify_region
from .mask_postprocess import ComponentMask
from .preprocess import PreprocessResult
from .scene_graph import BoundingBox, FreeformNode, PictureAssetNode, Point, PrimitiveNode, SceneGraph, SvgAssetNode

try:
    import vtracer
except ImportError:  # pragma: no cover - optional at runtime
    vtracer = None


def build_scene_graph(
    preprocessed: PreprocessResult,
    regions: list[DetectedRegion],
    config: Image2PptxConfig,
    artifacts_dir: Path,
    text_nodes: list[PrimitiveNode] | None = None,
    text_mask: np.ndarray | None = None,
) -> SceneGraph:
    svg_dir = artifacts_dir / "svg"
    fallback_dir = artifacts_dir / "fallback_png"
    svg_dir.mkdir(parents=True, exist_ok=True)
    fallback_dir.mkdir(parents=True, exist_ok=True)
    canvas_width, canvas_height = preprocessed.processed_size
    background_color = detect_background_color(preprocessed.segmented_rgba)

    graph = SceneGraph(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        background_color=background_color,
    )
    residual_rgba = np.zeros_like(preprocessed.rgba)
    kept_nodes = 0
    next_z_index = 100
    text_nodes = text_nodes or []

    for region in regions:
        if is_background_region(region, canvas_width, canvas_height, background_color):
            continue
        if region_overlaps_text(region.mask, text_mask):
            continue

        node = region_to_node(
            preprocessed=preprocessed,
            region=region,
            config=config,
            svg_dir=svg_dir,
            fallback_dir=fallback_dir,
            z_index=next_z_index,
        )
        if should_keep_node(node, region, config) and kept_nodes < config.vectorization.max_editable_nodes:
            graph.nodes.append(node)
            kept_nodes += 1
            next_z_index += 1
        else:
            paint_region_to_residual(
                residual_rgba=residual_rgba,
                source_rgba=preprocessed.rgba,
                mask=region.mask,
                text_mask=text_mask,
            )

    residual_nodes = split_residual_into_assets(
        residual_rgba=residual_rgba,
        fallback_dir=fallback_dir,
        max_components=config.vectorization.max_residual_components,
        min_component_area=config.vectorization.residual_component_min_area_px,
        merge_gap_px=config.vectorization.residual_merge_gap_px,
    )
    for index, node in enumerate(residual_nodes):
        node.z_index = index + 1
    graph.nodes.extend(residual_nodes)

    text_start_z = max([node.z_index for node in graph.nodes], default=0) + 1
    for offset, node in enumerate(text_nodes):
        node.z_index = text_start_z + offset
        graph.nodes.append(node)

    return graph


def build_scene_graph_from_components(
    preprocessed: PreprocessResult,
    components: list[ComponentMask],
    config: Image2PptxConfig,
    artifacts_dir: Path,
    text_nodes: list[PrimitiveNode] | None = None,
) -> SceneGraph:
    asset_dir = artifacts_dir / "component_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    graph = SceneGraph(
        canvas_width=preprocessed.processed_size[0],
        canvas_height=preprocessed.processed_size[1],
        background_color=detect_background_color(preprocessed.rgba),
    )

    sorted_components = sorted(components, key=lambda item: item.area, reverse=True)
    for index, component in enumerate(sorted_components):
        node = component_to_node(preprocessed, component, config, asset_dir, z_index=index + 1)
        if node is not None:
            graph.nodes.append(node)

    text_nodes = text_nodes or []
    next_z = max([node.z_index for node in graph.nodes], default=0) + 1
    for offset, node in enumerate(text_nodes):
        node.z_index = next_z + offset
        graph.nodes.append(node)

    return graph


def region_to_node(
    preprocessed: PreprocessResult,
    region: DetectedRegion,
    config: Image2PptxConfig,
    svg_dir: Path,
    fallback_dir: Path,
    z_index: int,
):
    x, y, width, height = region.bbox
    bbox = BoundingBox(x=float(x), y=float(y), width=float(width), height=float(height))
    fill_color = sample_region_color(preprocessed.rgba, region.mask)
    fallback_points = contour_to_points(region.approx_points, region.contour, config.vectorization.polygon_point_limit)

    if region.classification == "rect":
        return PrimitiveNode(
            id=region.id,
            primitive_type="rect",
            bbox=bbox,
            z_index=z_index,
            fill_color=fill_color,
        )

    if region.classification == "circle":
        return PrimitiveNode(
            id=region.id,
            primitive_type="circle",
            bbox=bbox,
            z_index=z_index,
            fill_color=fill_color,
        )

    if region.classification == "line":
        horizontal = width >= height
        start = Point(x=float(x), y=float(y + height / 2 if horizontal else y))
        end = Point(x=float(x + width if horizontal else x + width / 2), y=float(y + height / 2 if horizontal else y + height))
        return PrimitiveNode(
            id=region.id,
            primitive_type="line",
            bbox=bbox,
            z_index=z_index,
            fill_color=fill_color,
            stroke_color=fill_color,
            stroke_width=float(max(1, min(width, height))),
            start=start,
            end=end,
        )

    if should_emit_svg(region, config) and vtracer is not None:
        svg_node = try_build_svg_asset(
            preprocessed=preprocessed,
            region=region,
            bbox=bbox,
            fill_color=fill_color,
            svg_dir=svg_dir,
            fallback_dir=fallback_dir,
            z_index=z_index,
            path_precision=config.vectorization.path_precision,
        )
        if svg_node is not None:
            return svg_node

    return FreeformNode(
        id=region.id,
        bbox=bbox,
        z_index=z_index,
        fill_color=fill_color,
        points=fallback_points,
        closed=True,
        source_region_id=region.id,
    )


def component_to_node(
    preprocessed: PreprocessResult,
    component: ComponentMask,
    config: Image2PptxConfig,
    asset_dir: Path,
    z_index: int,
):
    contour, approx_points, bbox, area, classification = classify_component_mask(component.mask, config)
    if contour is None:
        return build_picture_asset(preprocessed, component, asset_dir, z_index)

    x, y, width, height = bbox
    bbox_model = BoundingBox(x=float(x), y=float(y), width=float(width), height=float(height))
    fill_color = sample_region_color(preprocessed.rgba, component.mask)

    if classification == "rect" and area >= config.vectorization.editable_rect_min_area_px:
        return PrimitiveNode(
            id=component.id,
            primitive_type="rect",
            bbox=bbox_model,
            z_index=z_index,
            fill_color=fill_color,
        )

    if classification == "circle" and area >= config.vectorization.editable_rect_min_area_px:
        return PrimitiveNode(
            id=component.id,
            primitive_type="circle",
            bbox=bbox_model,
            z_index=z_index,
            fill_color=fill_color,
        )

    if classification == "line" and max(width, height) >= config.vectorization.editable_line_min_length_px:
        horizontal = width >= height
        start = Point(x=float(x), y=float(y + height / 2 if horizontal else y))
        end = Point(x=float(x + width if horizontal else x + width / 2), y=float(y + height / 2 if horizontal else y + height))
        return PrimitiveNode(
            id=component.id,
            primitive_type="line",
            bbox=bbox_model,
            z_index=z_index,
            fill_color=fill_color,
            stroke_color=fill_color,
            stroke_width=float(max(1, min(width, height))),
            start=start,
            end=end,
        )

    if (
        classification == "freeform"
        and area >= config.vectorization.editable_freeform_min_area_px
        and len(approx_points) <= config.vectorization.editable_freeform_point_limit
    ):
        points = [Point(x=float(px), y=float(py)) for px, py in approx_points]
        return FreeformNode(
            id=component.id,
            bbox=bbox_model,
            z_index=z_index,
            fill_color=fill_color,
            points=points,
            closed=True,
            source_region_id=component.id,
        )

    return build_picture_asset(preprocessed, component, asset_dir, z_index)


def classify_component_mask(
    mask: np.ndarray,
    config: Image2PptxConfig,
) -> tuple[np.ndarray | None, list[tuple[int, int]], tuple[int, int, int, int], float, str]:
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, [], (0, 0, 0, 0), 0.0, "picture_asset"

    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    area = float(np.count_nonzero(mask_uint8))
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, config.detection.polygon_epsilon_ratio * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx_points = [(int(point[0][0]), int(point[0][1])) for point in approx]
    classification = classify_region(
        contour=contour,
        approx_points=approx_points,
        bbox=(int(x), int(y), int(width), int(height)),
        area=area,
        settings=config.detection,
    )
    return contour, approx_points, (int(x), int(y), int(width), int(height)), area, classification


def build_picture_asset(
    preprocessed: PreprocessResult,
    component: ComponentMask,
    asset_dir: Path,
    z_index: int,
) -> PictureAssetNode:
    x, y, width, height = component.bbox
    crop = preprocessed.rgba[y : y + height, x : x + width].copy()
    local_mask = component.mask[y : y + height, x : x + width]
    crop[:, :, 3] = np.where(local_mask, crop[:, :, 3], 0)
    image_path = asset_dir / f"{component.id}.png"
    Image.fromarray(crop, mode="RGBA").save(image_path)
    return PictureAssetNode(
        id=component.id,
        bbox=BoundingBox(x=float(x), y=float(y), width=float(width), height=float(height)),
        z_index=z_index,
        fill_color=None,
        image_path=str(image_path),
        source_region_id=component.id,
    )


def detect_background_color(rgba: np.ndarray) -> tuple[int, int, int, int]:
    colors, counts = np.unique(rgba.reshape(-1, 4), axis=0, return_counts=True)
    opaque = [(tuple(int(v) for v in color.tolist()), int(count)) for color, count in zip(colors, counts, strict=False) if int(color[3]) > 0]
    if not opaque:
        return (255, 255, 255, 255)
    opaque.sort(key=lambda item: item[1], reverse=True)
    return opaque[0][0]


def is_background_region(
    region: DetectedRegion,
    canvas_width: int,
    canvas_height: int,
    background_color: tuple[int, int, int, int],
) -> bool:
    x, y, width, height = region.bbox
    return (
        region.color == background_color
        and x == 0
        and y == 0
        and width == canvas_width
        and height == canvas_height
    )


def should_keep_node(node, region: DetectedRegion, config: Image2PptxConfig) -> bool:
    width = region.bbox[2]
    height = region.bbox[3]
    bbox_area = width * height

    if isinstance(node, PrimitiveNode):
        if node.primitive_type in {"rect", "circle"}:
            return region.area >= config.vectorization.editable_rect_min_area_px
        if node.primitive_type == "line":
            return (
                max(width, height) >= config.vectorization.editable_line_min_length_px
                and bbox_area >= config.vectorization.editable_line_min_bbox_area_px
            )
        return False

    if isinstance(node, FreeformNode):
        return (
            region.area >= config.vectorization.editable_freeform_min_area_px
            and len(node.points) <= config.vectorization.editable_freeform_point_limit
        )

    if isinstance(node, SvgAssetNode):
        return False

    return False


def region_overlaps_text(region_mask: np.ndarray, text_mask: np.ndarray | None, overlap_ratio: float = 0.35) -> bool:
    if text_mask is None or not np.any(text_mask):
        return False
    region_pixels = int(np.count_nonzero(region_mask))
    if region_pixels == 0:
        return False
    overlap = int(np.count_nonzero((region_mask > 0) & (text_mask > 0)))
    return (overlap / region_pixels) >= overlap_ratio


def sample_region_color(source_rgba: np.ndarray, mask: np.ndarray) -> tuple[int, int, int, int]:
    pixels = source_rgba[mask > 0]
    if pixels.size == 0:
        return (0, 0, 0, 255)
    color = np.median(pixels[:, :4], axis=0).astype(int)
    return (int(color[0]), int(color[1]), int(color[2]), int(color[3]))


def paint_region_to_residual(
    residual_rgba: np.ndarray,
    source_rgba: np.ndarray,
    mask: np.ndarray,
    text_mask: np.ndarray | None = None,
) -> None:
    effective_mask = mask > 0
    if text_mask is not None:
        effective_mask &= text_mask == 0
    residual_rgba[effective_mask] = source_rgba[effective_mask]


def split_residual_into_assets(
    residual_rgba: np.ndarray,
    fallback_dir: Path,
    max_components: int,
    min_component_area: int,
    merge_gap_px: int,
) -> list[SvgAssetNode]:
    alpha_mask = (residual_rgba[:, :, 3] > 0).astype(np.uint8)
    if not np.any(alpha_mask):
        return []

    grouped_mask = alpha_mask
    if merge_gap_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_gap_px, merge_gap_px))
        grouped_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grouped_mask, connectivity=8)
    component_specs: list[tuple[int, int, int, int, int]] = []
    spill_mask = np.zeros_like(alpha_mask)

    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        if area < min_component_area:
            spill_mask[labels == label] = 1
            continue
        component_specs.append((label, int(x), int(y), int(width), int(height)))

    component_specs.sort(key=lambda item: item[3] * item[4], reverse=True)
    kept_specs = component_specs[:max_components]
    for label, _, _, _, _ in component_specs[max_components:]:
        spill_mask[labels == label] = 1

    nodes: list[SvgAssetNode] = []
    for index, (label, x, y, width, height) in enumerate(kept_specs):
        component_mask = labels[y : y + height, x : x + width] == label
        crop = residual_rgba[y : y + height, x : x + width].copy()
        crop[:, :, 3] = np.where(component_mask, crop[:, :, 3], 0)
        asset_path = fallback_dir / f"residual_component_{index}.png"
        Image.fromarray(crop).save(asset_path)
        nodes.append(
            SvgAssetNode(
                id=f"residual-component-{index}",
                bbox=BoundingBox(x=float(x), y=float(y), width=float(width), height=float(height)),
                z_index=0,
                fill_color=None,
                svg_path="",
                fallback_image_path=str(asset_path),
                source_region_id=None,
            )
        )

    if np.any(spill_mask):
        ys, xs = np.where(spill_mask > 0)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        spill_crop = residual_rgba[y1:y2, x1:x2].copy()
        local_spill = spill_mask[y1:y2, x1:x2] > 0
        spill_crop[:, :, 3] = np.where(local_spill, spill_crop[:, :, 3], 0)
        asset_path = fallback_dir / "residual_spill.png"
        Image.fromarray(spill_crop).save(asset_path)
        nodes.append(
            SvgAssetNode(
                id="residual-spill",
                bbox=BoundingBox(x=float(x1), y=float(y1), width=float(x2 - x1), height=float(y2 - y1)),
                z_index=0,
                fill_color=None,
                svg_path="",
                fallback_image_path=str(asset_path),
                source_region_id=None,
            )
        )

    return nodes


def should_emit_svg(region: DetectedRegion, config: Image2PptxConfig) -> bool:
    return (
        config.vectorization.use_vtracer
        and region.classification == "svg_candidate"
        and region.area >= config.vectorization.svg_area_min_px
    )


def contour_to_points(
    approx_points: list[tuple[int, int]],
    contour: np.ndarray,
    point_limit: int,
) -> list[Point]:
    points = approx_points
    if len(points) > point_limit:
        step = max(1, len(contour) // point_limit)
        points = [(int(point[0][0]), int(point[0][1])) for point in contour[::step]]

    deduped: list[Point] = []
    seen: set[tuple[int, int]] = set()
    for px, py in points:
        key = (px, py)
        if key in seen:
            continue
        deduped.append(Point(x=float(px), y=float(py)))
        seen.add(key)
    return deduped


def try_build_svg_asset(
    preprocessed: PreprocessResult,
    region: DetectedRegion,
    bbox: BoundingBox,
    fill_color: tuple[int, int, int, int],
    svg_dir: Path,
    fallback_dir: Path,
    z_index: int,
    path_precision: int,
) -> SvgAssetNode | None:
    x, y, width, height = region.bbox
    crop = preprocessed.rgba[y : y + height, x : x + width].copy()
    local_mask = region.mask[y : y + height, x : x + width]
    crop[:, :, 3] = np.where(local_mask > 0, crop[:, :, 3], 0)

    fallback_image_path = fallback_dir / f"{region.id}.png"
    Image.fromarray(crop, mode="RGBA").save(fallback_image_path)

    svg_path = svg_dir / f"{region.id}.svg"
    try:
        vtracer.convert_image_to_svg_py(
            str(fallback_image_path),
            str(svg_path),
            colormode="color",
            hierarchical="stacked",
            mode="spline",
            path_precision=path_precision,
        )
    except Exception:
        return None

    return SvgAssetNode(
        id=region.id,
        bbox=bbox,
        z_index=z_index,
        fill_color=fill_color,
        svg_path=str(svg_path),
        fallback_image_path=str(fallback_image_path),
        source_region_id=region.id,
    )
