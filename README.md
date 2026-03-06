# image2pptx

`image2pptx` is a Python project that converts infographic-style images into editable or semi-editable `PPTX` slides.

The current implementation is optimized for:

- flat illustrations
- infographic panels
- icons and diagrams
- Gemini-generated layouts
- PowerPoint-friendly reconstruction with debug artifacts

For Chinese documentation, see [`README.zh-CN.md`](README.zh-CN.md).

## What it does

The pipeline tries to reconstruct an input image into a PowerPoint slide by combining several strategies:

1. OCR for text extraction and editable text boxes.
2. `MobileSAM` for component-level segmentation.
3. Heuristic shape classification for rectangles, circles, and lines.
4. Transparent image assets for complex components that are not yet safe to convert into native PowerPoint shapes.

The goal is not perfect vector reconstruction for every element. The current focus is:

- make major elements independently selectable
- keep colors close to the source image
- avoid generating unusable slides with thousands of tiny shapes

## Tech stack

Core runtime:

- Python `3.11+`
- `opencv-python`
- `Pillow`
- `numpy`
- `python-pptx`
- `typer`
- `pydantic`

Segmentation and OCR:

- `ultralytics` with `MobileSAM`
- `rapidocr-onnxruntime`

Optional and platform-specific:

- `pywin32` for Windows PowerPoint COM export
- `pytest` for local regression tests

## Local setup

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

## First run

The first `sam` run may automatically download the `mobile_sam.pt` weights through Ultralytics.

Basic usage:

```bash
image2pptx input.png output.pptx
```

Run with explicit artifacts directory:

```bash
image2pptx input.png output.pptx --artifacts-dir artifacts
```

Force the OpenCV fallback instead of `MobileSAM`:

```bash
image2pptx input.png output.pptx --segmentation-backend cv
```

Force pure `python-pptx` export:

```bash
image2pptx input.png output.pptx --exporter python-pptx
```

Use PowerPoint COM export on Windows when available:

```bash
image2pptx input.png output.pptx --exporter com
```

## Generated artifacts

When `--artifacts-dir` is used, the pipeline can emit debugging files such as:

- `preprocessed.png`
- `quantized.png`
- `text_mask.png`
- `sam_input.png`
- `ocr_boxes.json`
- `sam_masks.json`
- `sam_masks_overlay.png`
- `component_boxes.png`
- `component_crops/`
- `scene_graph.json`

These artifacts help diagnose whether failures come from OCR, SAM segmentation, mask post-processing, or export.

## Project layout

```text
docs/                 Design notes and guidance manuals
examples/             Example assets and sample expectations
src/image2pptx/       CLI, pipeline, exporters, OCR, SAM, and scene graph logic
tests/                Regression tests for the codebase
```

## Running tests

```bash
python -m pytest
```

## Current limitations

- OCR quality for dense Chinese infographic text is still imperfect.
- Many complex components are currently exported as independent transparent image assets rather than native PowerPoint vectors.
- `MobileSAM` improves component isolation, but some small decorative elements and thin connectors still need additional post-processing.
- The project is currently aimed at infographic-like images, not photo-realistic scenes.
