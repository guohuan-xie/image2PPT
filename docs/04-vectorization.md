# Vectorization

## Strategy

- simple regions map directly to PowerPoint primitives
- clean irregular regions become freeform polygons
- larger complex regions are cropped and traced to `SVG` via `VTracer`

## Why VTracer

- supports color input directly
- works well for flat illustrations and high-resolution icon assets
- produces compact vector output suitable for a sidecar asset pipeline

## Current fallback chain

1. primitive detection
2. freeform polygon conversion
3. `VTracer` SVG output for `svg_candidate`
4. PNG fallback when pure `python-pptx` cannot embed SVG
