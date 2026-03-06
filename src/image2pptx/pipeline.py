from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .com_exporter import PowerPointComExporter
from .config import Image2PptxConfig
from .detection import detect_regions
from .mask_postprocess import postprocess_component_masks, save_sam_debug_artifacts
from .ocr import build_text_mask, build_text_nodes, detect_text_boxes
from .pptx_writer import PptxWriter
from .preprocess import preprocess_image
from .sam_segmenter import segment_with_mobilesam
from .vectorize import build_scene_graph, build_scene_graph_from_components


@dataclass(slots=True)
class PipelineResult:
    output_pptx: Path
    artifacts_dir: Path
    scene_graph_json: Path


def run_pipeline(
    input_image: Path,
    output_pptx: Path,
    config: Image2PptxConfig | None = None,
    artifacts_dir: Path | None = None,
    segmentation_backend: str = "sam",
    exporter: str = "python-pptx",
    dump_scene_graph: bool = True,
) -> PipelineResult:
    config = config or Image2PptxConfig()
    if segmentation_backend not in {"cv", "sam"}:
        raise ValueError(f"Unsupported segmentation backend: {segmentation_backend}")

    artifacts_dir = config.resolve_artifacts_dir(output_pptx, artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    preprocessed = preprocess_image(input_image, config.preprocess)
    preprocessed.image.save(artifacts_dir / "preprocessed.png")
    preprocessed.segmented_image.save(artifacts_dir / "quantized.png")

    text_boxes = detect_text_boxes(preprocessed, config.ocr)
    text_nodes = build_text_nodes(preprocessed, text_boxes)
    text_mask = build_text_mask(preprocessed.rgba, text_boxes, config.ocr) if config.ocr.remove_text_from_residual else None
    if text_boxes:
        (artifacts_dir / "ocr_boxes.json").write_text(
            json.dumps(
                [
                    {
                        "text": box.text,
                        "score": box.score,
                        "bbox": box.bbox.model_dump(mode="json"),
                    }
                    for box in text_boxes
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if text_mask is not None:
        Image.fromarray((text_mask * 255).astype("uint8")).save(artifacts_dir / "text_mask.png")

    if segmentation_backend == "sam":
        _, raw_components = segment_with_mobilesam(
            rgba=preprocessed.rgba,
            settings=config.sam,
            text_mask=text_mask,
            artifacts_dir=artifacts_dir,
        )
        components = postprocess_component_masks(
            raw_components=raw_components,
            settings=config.sam,
            image_shape=preprocessed.rgba.shape[:2],
            text_mask=text_mask,
        )
        save_sam_debug_artifacts(
            rgba=preprocessed.rgba,
            raw_components=raw_components,
            final_components=components,
            artifacts_dir=artifacts_dir,
        )
        scene_graph = build_scene_graph_from_components(
            preprocessed=preprocessed,
            components=components,
            config=config,
            artifacts_dir=artifacts_dir,
            text_nodes=text_nodes,
        )
    else:
        regions = detect_regions(preprocessed.segmented_rgba, config.detection)
        scene_graph = build_scene_graph(
            preprocessed,
            regions,
            config,
            artifacts_dir,
            text_nodes=text_nodes,
            text_mask=text_mask,
        )

    scene_graph_json = artifacts_dir / "scene_graph.json"
    if dump_scene_graph:
        scene_graph_json.write_text(
            json.dumps(scene_graph.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

    selected_exporter = exporter
    if exporter == "auto":
        com_exporter = PowerPointComExporter(config.export)
        selected_exporter = "com" if com_exporter.is_available() else "python-pptx"

    if selected_exporter == "com":
        try:
            PowerPointComExporter(config.export).write(scene_graph, output_pptx)
        except Exception:
            if exporter != "auto":
                raise
            PptxWriter(config.export).write(scene_graph, output_pptx)
    elif selected_exporter == "python-pptx":
        PptxWriter(config.export).write(scene_graph, output_pptx)
    else:
        raise ValueError(f"Unsupported exporter: {selected_exporter}")

    return PipelineResult(
        output_pptx=output_pptx,
        artifacts_dir=artifacts_dir,
        scene_graph_json=scene_graph_json,
    )
