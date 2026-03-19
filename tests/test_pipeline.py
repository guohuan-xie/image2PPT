from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from image2pptx.config import Image2PptxConfig
from image2pptx.detection import detect_regions
from image2pptx.mask_postprocess import ComponentMask, postprocess_component_masks
from image2pptx.ocr import OcrBox, build_text_mask, build_text_nodes
from image2pptx.pipeline import PipelineResult
from image2pptx.pipeline import run_pipeline
from image2pptx.preprocess import PreprocessResult
from image2pptx.sam_segmenter import prepare_sam_input, resolve_sam_device
from image2pptx.scene_graph import BoundingBox
from image2pptx.vectorize import build_scene_graph, build_scene_graph_from_components


def test_detect_regions_classifies_rectangle() -> None:
    rgba = np.zeros((80, 120, 4), dtype=np.uint8)
    rgba[10:50, 20:90] = (255, 0, 0, 255)

    regions = detect_regions(rgba, Image2PptxConfig().detection)

    assert len(regions) == 1
    assert regions[0].classification == "rect"


def test_run_pipeline_writes_outputs(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    output_pptx = tmp_path / "output.pptx"
    artifacts_dir = tmp_path / "artifacts"

    image = Image.new("RGBA", (160, 120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 70, 60), fill=(255, 0, 0, 255))
    draw.ellipse((90, 20, 140, 70), fill=(0, 80, 255, 255))
    image.save(image_path)

    config = Image2PptxConfig()
    config.ocr.enabled = False

    result = run_pipeline(
        input_image=image_path,
        output_pptx=output_pptx,
        config=config,
        artifacts_dir=artifacts_dir,
        segmentation_backend="cv",
        exporter="python-pptx",
    )

    assert result.output_pptx.exists()
    assert (artifacts_dir / "preprocessed.png").exists()
    assert result.scene_graph_json.exists()


def test_auto_exporter_falls_back_when_com_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "input.png"
    output_pptx = tmp_path / "output.pptx"
    artifacts_dir = tmp_path / "artifacts"

    image = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 60, 60), fill=(255, 0, 0, 255))
    image.save(image_path)

    from image2pptx import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module.PowerPointComExporter, "is_available", lambda self: True)

    def failing_write(self, scene_graph, output_path):
        raise RuntimeError("com failed")

    monkeypatch.setattr(pipeline_module.PowerPointComExporter, "write", failing_write)

    calls: list[Path] = []

    def fake_write(self, scene_graph, output_path):
        calls.append(output_path)
        output_path.write_bytes(b"pptx")
        return output_path

    monkeypatch.setattr(pipeline_module.PptxWriter, "write", fake_write)

    config = Image2PptxConfig()
    config.ocr.enabled = False

    result = run_pipeline(
        input_image=image_path,
        output_pptx=output_pptx,
        config=config,
        artifacts_dir=artifacts_dir,
        segmentation_backend="cv",
        exporter="auto",
    )

    assert isinstance(result, PipelineResult)
    assert calls == [output_pptx]
    assert output_pptx.exists()


def test_build_scene_graph_uses_residual_overlay_for_small_fragments(tmp_path: Path) -> None:
    rgba = np.zeros((120, 120, 4), dtype=np.uint8)
    rgba[:, :] = (255, 255, 255, 255)
    rgba[10:70, 10:70] = (255, 0, 0, 255)
    for idx in range(6):
        y = 78 + (idx // 3) * 16
        x = 10 + (idx % 3) * 20
        rgba[y : y + 10, x : x + 10] = (0, 0, 0, 255)

    image = Image.fromarray(rgba, mode="RGBA")
    preprocessed = PreprocessResult(
        source_path=tmp_path / "synthetic.png",
        image=image,
        rgba=rgba,
        segmented_image=image,
        segmented_rgba=rgba,
        original_size=image.size,
        processed_size=image.size,
    )
    config = Image2PptxConfig()
    config.ocr.enabled = False
    regions = detect_regions(rgba, config.detection)

    graph = build_scene_graph(
        preprocessed=preprocessed,
        regions=regions,
        config=config,
        artifacts_dir=tmp_path / "artifacts",
    )

    assert any(node.id.startswith("residual-") for node in graph.nodes)
    assert len(graph.nodes) < len(regions)


def test_build_scene_graph_from_components_creates_component_nodes(tmp_path: Path) -> None:
    rgba = np.zeros((100, 120, 4), dtype=np.uint8)
    rgba[:, :] = (255, 255, 255, 255)
    rgba[10:60, 10:70] = (255, 0, 0, 255)
    rgba[20:50, 80:110] = (0, 80, 255, 255)

    image = Image.fromarray(rgba, mode="RGBA")
    preprocessed = PreprocessResult(
        source_path=tmp_path / "synthetic.png",
        image=image,
        rgba=rgba,
        segmented_image=image,
        segmented_rgba=rgba,
        original_size=image.size,
        processed_size=image.size,
    )
    config = Image2PptxConfig()
    config.ocr.enabled = False

    components = [
        ComponentMask(
            id="sam-rect",
            mask=(rgba[:, :, 0] == 255) & (rgba[:, :, 1] == 0),
            bbox=(10, 10, 60, 50),
            area=3000,
            score=0.99,
        ),
        ComponentMask(
            id="sam-picture",
            mask=(rgba[:, :, 2] == 255),
            bbox=(80, 20, 30, 30),
            area=900,
            score=0.95,
        ),
    ]

    graph = build_scene_graph_from_components(
        preprocessed=preprocessed,
        components=components,
        config=config,
        artifacts_dir=tmp_path / "artifacts",
        text_nodes=[],
    )

    kinds = [node.kind for node in graph.nodes]
    assert "primitive" in kinds
    assert "picture_asset" in kinds


def test_postprocess_component_masks_keeps_thin_arrow_like_masks() -> None:
    config = Image2PptxConfig()
    settings = config.sam
    settings.min_mask_area_px = 50
    settings.sparse_mask_bbox_area_ratio = 0.01
    settings.sparse_mask_fill_ratio = 0.85

    mask = np.zeros((120, 120), dtype=bool)
    mask[58:62, 10:90] = True
    mask[54:66, 82:94] = np.tri(12, 12, dtype=bool)[:, ::-1]
    component = ComponentMask(
        id="thin-arrow",
        mask=mask,
        bbox=(10, 54, 84, 12),
        area=int(np.count_nonzero(mask)),
        score=0.95,
    )

    components = postprocess_component_masks(
        raw_components=[component],
        settings=settings,
        image_shape=mask.shape,
        rgba=np.dstack([mask.astype(np.uint8) * 255] * 3 + [np.full(mask.shape, 255, dtype=np.uint8)]),
    )

    assert [item.id for item in components] == ["thin-arrow"]


def test_postprocess_component_masks_bridges_thin_line_gaps() -> None:
    config = Image2PptxConfig()
    settings = config.sam
    settings.min_mask_area_px = 20
    settings.thin_component_min_length_px = 12
    settings.thin_component_min_area_px = 20
    settings.thin_component_min_aspect_ratio = 4.0
    settings.thin_line_bridge_px = 2

    mask = np.zeros((40, 80), dtype=bool)
    mask[19:21, 10:28] = True
    mask[19:21, 31:50] = True
    rgba = np.zeros((40, 80, 4), dtype=np.uint8)
    rgba[mask] = (80, 80, 80, 255)
    rgba[19:21, 28:31] = (80, 80, 80, 255)
    component = ComponentMask(
        id="thin-gap",
        mask=mask,
        bbox=(10, 19, 40, 2),
        area=int(np.count_nonzero(mask)),
        score=0.99,
    )

    components = postprocess_component_masks(
        raw_components=[component],
        settings=settings,
        image_shape=mask.shape,
        rgba=rgba,
    )

    assert len(components) == 1
    assert components[0].mask[19, 29]


def test_postprocess_component_masks_recovers_missing_edge_pixels() -> None:
    config = Image2PptxConfig()
    settings = config.sam
    settings.min_mask_area_px = 10
    settings.edge_expand_px = 2
    settings.edge_color_distance_thresh = 5.0

    rgba = np.zeros((40, 40, 4), dtype=np.uint8)
    rgba[10:20, 10:20] = (120, 180, 220, 255)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:20, 10:18] = True
    component = ComponentMask(
        id="partial-cloud",
        mask=mask,
        bbox=(10, 10, 8, 10),
        area=int(np.count_nonzero(mask)),
        score=0.99,
    )

    components = postprocess_component_masks(
        raw_components=[component],
        settings=settings,
        image_shape=mask.shape,
        rgba=rgba,
    )

    assert len(components) == 1
    assert components[0].mask[12, 18]
    assert components[0].bbox[2] >= 9


def test_build_text_mask_targets_text_pixels_instead_of_full_box() -> None:
    rgba = np.full((40, 80, 4), 255, dtype=np.uint8)
    rgba[8:32, 10:70] = (180, 220, 250, 255)
    rgba[8:32, 10] = (40, 40, 40, 255)
    rgba[8:32, 69] = (40, 40, 40, 255)
    rgba[8, 10:70] = (40, 40, 40, 255)
    rgba[31, 10:70] = (40, 40, 40, 255)
    rgba[13:27, 28:32] = (0, 0, 0, 255)
    rgba[13:17, 24:40] = (0, 0, 0, 255)

    settings = Image2PptxConfig().ocr
    box = OcrBox(
        text="T",
        score=0.99,
        bbox=BoundingBox(x=22, y=11, width=20, height=18),
        polygon=((22.0, 11.0), (42.0, 11.0), (42.0, 29.0), (22.0, 29.0)),
    )

    text_mask = build_text_mask(rgba, [box], settings)

    assert text_mask[15, 30] == 1
    assert text_mask[8, 10] == 0
    assert text_mask[31, 69] == 0
    assert np.count_nonzero(text_mask) <= 170


def test_build_text_nodes_marks_centered_title_and_single_line() -> None:
    rgba = np.full((120, 400, 4), 255, dtype=np.uint8)
    image = Image.fromarray(rgba, mode="RGBA")
    preprocessed = PreprocessResult(
        source_path=Path("synthetic.png"),
        image=image,
        rgba=rgba,
        segmented_image=image,
        segmented_rgba=rgba,
        original_size=image.size,
        processed_size=image.size,
    )
    settings = Image2PptxConfig().ocr
    box = OcrBox(
        text="Centered Title",
        score=0.99,
        bbox=BoundingBox(x=80, y=10, width=240, height=36),
        polygon=((80.0, 10.0), (320.0, 10.0), (320.0, 46.0), (80.0, 46.0)),
    )

    nodes = build_text_nodes(preprocessed, [box], settings)

    assert len(nodes) == 1
    node = nodes[0]
    assert node.kind == "primitive"
    assert node.text_align == "center"
    assert node.single_line is True
    assert node.bold is True


def test_build_text_nodes_rasterizes_dense_small_text_clusters(tmp_path: Path) -> None:
    rgba = np.full((120, 200, 4), 255, dtype=np.uint8)
    rgba[20:32, 20:50] = (0, 0, 0, 255)
    rgba[20:32, 60:90] = (0, 0, 0, 255)
    rgba[40:52, 20:50] = (0, 0, 0, 255)
    rgba[40:52, 60:90] = (0, 0, 0, 255)
    image = Image.fromarray(rgba, mode="RGBA")
    preprocessed = PreprocessResult(
        source_path=tmp_path / "synthetic.png",
        image=image,
        rgba=rgba,
        segmented_image=image,
        segmented_rgba=rgba,
        original_size=image.size,
        processed_size=image.size,
    )
    settings = Image2PptxConfig().ocr
    boxes = [
        OcrBox(
            text=f"T{index}",
            score=0.99,
            bbox=BoundingBox(x=float(x), y=float(y), width=30.0, height=12.0),
            polygon=((float(x), float(y)), (float(x + 30), float(y)), (float(x + 30), float(y + 12)), (float(x), float(y + 12))),
        )
        for index, (x, y) in enumerate([(20, 20), (60, 20), (20, 40), (60, 40)], start=1)
    ]

    nodes = build_text_nodes(preprocessed, boxes, settings, tmp_path)

    assert any(node.kind == "picture_asset" for node in nodes)
    assert not any(getattr(node, "text", None) == "T1" for node in nodes)
    assert (tmp_path / "text_assets" / "text_cluster_0.png").exists()


def test_build_text_nodes_skips_tiny_editable_text() -> None:
    rgba = np.full((80, 120, 4), 255, dtype=np.uint8)
    image = Image.fromarray(rgba, mode="RGBA")
    preprocessed = PreprocessResult(
        source_path=Path("synthetic.png"),
        image=image,
        rgba=rgba,
        segmented_image=image,
        segmented_rgba=rgba,
        original_size=image.size,
        processed_size=image.size,
    )
    settings = Image2PptxConfig().ocr
    settings.min_editable_text_height_px = 20
    box = OcrBox(
        text="tiny",
        score=0.99,
        bbox=BoundingBox(x=10, y=10, width=30, height=14),
        polygon=((10.0, 10.0), (40.0, 10.0), (40.0, 24.0), (10.0, 24.0)),
    )

    nodes = build_text_nodes(preprocessed, [box], settings)

    assert nodes == []


def test_prepare_sam_input_inpaints_text_without_breaking_box_border() -> None:
    rgba = np.full((40, 80, 4), 255, dtype=np.uint8)
    rgba[8:32, 10:70] = (180, 220, 250, 255)
    rgba[8:32, 10] = (40, 40, 40, 255)
    rgba[8:32, 69] = (40, 40, 40, 255)
    rgba[8, 10:70] = (40, 40, 40, 255)
    rgba[31, 10:70] = (40, 40, 40, 255)
    rgba[13:27, 28:32] = (0, 0, 0, 255)
    rgba[13:17, 24:40] = (0, 0, 0, 255)

    settings = Image2PptxConfig().ocr
    box = OcrBox(
        text="T",
        score=0.99,
        bbox=BoundingBox(x=22, y=11, width=20, height=18),
        polygon=((22.0, 11.0), (42.0, 11.0), (42.0, 29.0), (22.0, 29.0)),
    )
    text_mask = build_text_mask(rgba, [box], settings)

    sam_input = prepare_sam_input(rgba, text_mask)

    assert tuple(sam_input[8, 10, :3]) == (40, 40, 40)
    fill_delta = np.abs(sam_input[15, 30, :3].astype(int) - np.array([180, 220, 250]))
    assert int(fill_delta.max()) <= 35


def test_resolve_sam_device_prefers_gpu_in_auto_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from image2pptx import sam_segmenter as sam_segmenter_module

    monkeypatch.setattr(sam_segmenter_module, "cuda_is_available", lambda: True)
    monkeypatch.setattr(sam_segmenter_module, "mps_is_available", lambda: False)

    assert resolve_sam_device("auto") == "cuda:0"


def test_resolve_sam_device_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from image2pptx import sam_segmenter as sam_segmenter_module

    monkeypatch.setattr(sam_segmenter_module, "cuda_is_available", lambda: False)
    monkeypatch.setattr(sam_segmenter_module, "mps_is_available", lambda: False)

    with pytest.warns(RuntimeWarning, match="Falling back to CPU"):
        assert resolve_sam_device("cuda") == "cpu"
