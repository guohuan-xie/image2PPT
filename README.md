# image2pptx

`image2pptx` converts a flat image into a PowerPoint slide that is as editable as the current pipeline can safely make it.

Today the project is optimized for infographic-like inputs: flat illustrations, icons, cards, arrows, diagrams, and Gemini-style layouts. The output is usually a mix of editable text, editable primitive shapes, editable freeforms, and transparent image assets for components that are still too risky to convert into native PowerPoint geometry.

For Chinese documentation, see [`README.zh-CN.md`](README.zh-CN.md).

## What The Current Pipeline Produces

- Editable text boxes from OCR detections.
- Editable `rect`, `circle`, and `line` nodes when the geometry is simple enough.
- Editable freeform polygons for some large, simple irregular regions.
- Transparent PNG assets for complex components that should stay visually correct even when they are not yet editable as native shapes.
- A sidecar artifacts directory for debugging every major stage.

The goal is practical reconstruction, not perfect vector fidelity. The current implementation is tuned to:

- keep major elements independently selectable
- preserve visual layering and approximate colors
- avoid generating unusable slides with hundreds or thousands of tiny shapes

## Tech Stack

Core runtime:

- Python `3.11+`
- `opencv-python`
- `Pillow`
- `numpy`
- `python-pptx`
- `typer`
- `pydantic`

Segmentation and OCR:

- `ultralytics` `SAM`
- `rapidocr-onnxruntime`
- `vtracer`

Optional and platform-specific:

- `pywin32` for Windows PowerPoint COM export
- `pytest` for local regression tests

## Local Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch

CPU-only installation:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you have a supported CUDA environment, install the matching GPU wheels instead.

### 3. Install the project

```bash
python -m pip install -e .
```

Optional Windows PowerPoint export support:

```bash
python -m pip install -e ".[windows]"
```

Optional development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## CLI

Basic usage:

```bash
image2pptx input.png output.pptx
```

Common variants:

```bash
image2pptx input.png output.pptx --artifacts-dir artifacts
image2pptx input.png output.pptx --segmentation-backend cv
image2pptx input.png output.pptx --sam-model sam_b.pt
image2pptx input.png output.pptx --exporter python-pptx
image2pptx input.png output.pptx --exporter com
image2pptx input.png output.pptx --no-dump-scene-graph
```

Current CLI defaults:

- `--segmentation-backend sam`
- `--sam-model sam_b.pt`
- `--exporter auto`
- artifacts are always written; if `--artifacts-dir` is omitted, the pipeline creates `<output_stem>_artifacts`
- `scene_graph.json` is written by default unless `--no-dump-scene-graph` is passed

On the first `sam` run, Ultralytics may download the SAM weights if the requested model file is not already available locally.

## Pipeline Overview

This section describes the pipeline as it is implemented today.

### 1. Resolve output paths and runtime config

The CLI entrypoint is `image2pptx`. It validates the backend and exporter choice, builds an `Image2PptxConfig`, overrides the configured SAM model with `--sam-model`, and resolves the artifacts directory before any processing starts.

### 2. Preprocess the source image

The input image is opened as `RGBA`.

The preprocess stage then:

- downsizes the image if its longest side exceeds `1600px`
- optionally applies Gaussian blur when `blur_radius > 0`
- creates an adaptive quantized copy with `8` colors by default

Artifacts written here:

- `preprocessed.png`
- `quantized.png`

`preprocessed.png` is the resized and optionally blurred working image. `quantized.png` is the reduced-palette version used directly by the `cv` backend and kept as a debugging view for the `sam` backend.

### 3. Run OCR and build a text mask

OCR runs on the preprocessed `RGBA` image through `rapidocr-onnxruntime`.

The current implementation:

- keeps detections with score `>= 0.5`
- adds padding around each OCR box
- converts text boxes into editable text nodes
- estimates text color from local image content
- estimates font size from OCR box height

When text masking is enabled, the pipeline also builds a pixel-level `text_mask` so that later stages do not accidentally absorb text into graphic components.

Artifacts written here:

- `ocr_boxes.json` when OCR finds text
- `text_mask.png` when text masking is enabled

If OCR is unavailable at runtime or returns no detections, the pipeline simply proceeds without text nodes.

### 4. Segment the graphics

The project currently supports two segmentation backends.

### 4A. `sam` backend

This is the current CLI default.

The `sam` path works on the preprocessed full-color image, not on the quantized copy.

Before SAM inference:

- the pipeline removes detected text from the SAM input with OpenCV inpainting
- the inpainted image is saved as `sam_input.png`

SAM inference then runs through `ultralytics.SAM`, and the raw masks are post-processed to remove masks that are:

- too small
- too large relative to the image
- too sparse inside their own bounding boxes
- mostly overlapping text
- near-duplicates of already kept masks
- contained by better masks

The final list is capped, then sorted by area.

Artifacts written by the `sam` path:

- `sam_input.png`
- `sam_masks.json`
- `sam_masks_overlay.png`
- `component_boxes.png`
- `component_crops/`

### 4B. `cv` backend

The `cv` backend works from the quantized `RGBA` image.

The current logic:

1. enumerate unique non-transparent colors
2. build a binary mask for each color
3. split each color into connected components
4. trace contours
5. classify each component as `rect`, `circle`, `line`, `freeform`, or `svg_candidate`

This backend is deterministic and easier to debug, but it depends heavily on the quality of the quantized image.

### 5. Convert segmented regions into scene-graph nodes

Both backends eventually produce a `SceneGraph` that describes the slide before export.

Current node types are:

- primitive shapes
- freeform polygons
- SVG asset nodes
- picture asset nodes
- text nodes

The two backends diverge here:

- In the `sam` path, large simple components become editable shapes, while complex components are cropped into transparent PNGs under `component_assets/`.
- In the `cv` path, large simple components become editable shapes, while everything else is painted into residual fallback imagery and split into assets under `fallback_png/`.

The code can also write `svg/` files in the `cv` path when `vtracer` is used, but the current implementation still relies mainly on fallback imagery rather than keeping those SVGs as editable slide objects in the final graph.

Text nodes are appended last so text renders above the graphic content.

If scene-graph dumping is enabled, this stage writes:

- `scene_graph.json`

### 6. Export to `PPTX`

The exporter choice is:

- `auto`: prefer PowerPoint COM on Windows when `pywin32` is installed, otherwise use `python-pptx`
- `python-pptx`: always use the pure Python writer
- `com`: require the Windows PowerPoint COM exporter

Current export behavior:

- slide width is `10in`
- slide height follows the source aspect ratio unless explicitly configured
- slide background comes from the detected scene-graph background color
- text uses `Microsoft YaHei`

The `python-pptx` exporter inserts native PowerPoint shapes where possible and uses image fallbacks for asset nodes. The COM exporter can insert Office shapes and can place `.svg` files when they exist on disk.

## Artifact Reference

Artifacts are always created, either in the directory passed by `--artifacts-dir` or in `<output_stem>_artifacts`.

Common artifacts:

- `preprocessed.png`: resized and optionally blurred working image
- `quantized.png`: reduced-palette image used by the `cv` backend
- `scene_graph.json`: serialized scene graph before export, unless disabled

OCR-related artifacts:

- `ocr_boxes.json`: OCR text boxes and scores
- `text_mask.png`: pixel mask used to protect text regions

`sam` backend artifacts:

- `sam_input.png`: text-inpainted image sent to SAM
- `sam_masks.json`: raw and post-processed mask metadata
- `sam_masks_overlay.png`: mask visualization over the image
- `component_boxes.png`: final component bounding boxes
- `component_crops/`: cropped RGBA previews for each final component
- `component_assets/`: transparent PNG assets actually referenced by the scene graph for complex components

`cv` backend artifacts:

- `fallback_png/`: residual image assets for regions that were not kept as editable nodes
- `svg/`: intermediate vectorization outputs when available

## Project Layout

```text
docs/                 Design notes and implementation guidance
examples/             Example assets and sample expectations
src/image2pptx/       CLI, pipeline, preprocess, OCR, segmentation, scene graph, and exporters
tests/                Regression tests
```

## Running Tests

```bash
python -m pytest
```

## Current Limitations

- The project is tuned for flat infographic-like images, not photo-realistic scenes.
- Dense text, especially mixed-language or small-font layouts, can still confuse OCR.
- Many complex components are still exported as transparent PNG assets rather than native PowerPoint vectors.
- Small decorations and thin connectors can still be fragmented or merged imperfectly.
- The `cv` backend is highly sensitive to color quantization quality.
- The `python-pptx` path uses image fallbacks for asset-style nodes instead of preserving native SVG editability.
