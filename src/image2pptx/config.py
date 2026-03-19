from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class PreprocessSettings(BaseModel):
    max_side_px: int = 1600
    quantize_colors: int = 8
    blur_radius: float = 0.0
    alpha_threshold: int = 8


class ShapeDetectionSettings(BaseModel):
    min_area_px: int = 48
    rect_fill_ratio_min: float = 0.78
    circle_circularity_min: float = 0.82
    line_aspect_ratio_min: float = 6.0
    polygon_epsilon_ratio: float = 0.015


class SamSettings(BaseModel):
    model_path: str = "sam2_t.pt"
    device: str = "auto"
    imgsz: int = 1024
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.92
    min_mask_area_px: int = 150
    max_mask_area_ratio: float = 0.85
    duplicate_iou_thresh: float = 0.9
    containment_ratio_thresh: float = 0.95
    text_overlap_ratio_thresh: float = 0.65
    sparse_mask_bbox_area_ratio: float = 0.12
    sparse_mask_fill_ratio: float = 0.6
    thin_component_min_aspect_ratio: float = 5.0
    thin_component_min_length_px: int = 24
    thin_component_min_area_px: int = 60
    merge_gap_px: int = 0
    max_components: int = 96
    mask_close_px: int = 1
    mask_min_hole_area_px: int = 48
    thin_line_bridge_px: int = 3
    edge_expand_px: int = 2
    edge_color_distance_thresh: float = 32.0
    edge_expand_max_area_ratio: float = 1.6


class VectorizationSettings(BaseModel):
    svg_area_min_px: int = 3200
    polygon_point_limit: int = 24
    path_precision: int = 3
    use_vtracer: bool = True
    editable_rect_min_area_px: int = 2500
    editable_line_min_length_px: int = 120
    editable_line_min_bbox_area_px: int = 3500
    editable_freeform_min_area_px: int = 40000
    editable_freeform_point_limit: int = 12
    max_editable_nodes: int = 120
    residual_component_min_area_px: int = 150
    max_residual_components: int = 48
    residual_merge_gap_px: int = 6


class OcrSettings(BaseModel):
    enabled: bool = True
    min_score: float = 0.5
    min_text_length: int = 1
    remove_text_from_residual: bool = True
    text_padding_px: int = 3
    max_text_boxes: int = 200
    text_background_sample_px: int = 4
    text_mask_min_color_distance: float = 28.0
    text_mask_min_luminance_delta: float = 20.0
    text_mask_dilate_px: int = 1
    rasterize_dense_text: bool = True
    rasterize_text_height_px: int = 22
    rasterize_cluster_gap_px: int = 18
    rasterize_cluster_padding_px: int = 6
    rasterize_cluster_min_boxes: int = 4
    heading_text_height_px: int = 28


class ExportSettings(BaseModel):
    slide_width_in: float = 10.0
    slide_height_in: float | None = None
    background_rgb: tuple[int, int, int] = (255, 255, 255)


class Image2PptxConfig(BaseModel):
    preprocess: PreprocessSettings = Field(default_factory=PreprocessSettings)
    detection: ShapeDetectionSettings = Field(default_factory=ShapeDetectionSettings)
    sam: SamSettings = Field(default_factory=SamSettings)
    vectorization: VectorizationSettings = Field(default_factory=VectorizationSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)

    def resolve_artifacts_dir(self, output_pptx: Path, artifacts_dir: Path | None) -> Path:
        if artifacts_dir is not None:
            return artifacts_dir
        return output_pptx.with_suffix("").with_name(f"{output_pptx.stem}_artifacts")
