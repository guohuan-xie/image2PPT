# Scene Graph

## Purpose

`SceneGraph` is the pipeline contract between parsing and exporting.

## Node types

- `PrimitiveNode`: `rect`, `circle`, `line`, `text`
- `FreeformNode`: polygonal region with editable points
- `SvgAssetNode`: preserved vector asset plus fallback image or points
- `GroupNode`: logical grouping and future layering hooks

## Required fields

- `bbox`
- `z_index`
- fill and stroke colors
- region source identifiers where relevant

## Why this matters

The scene graph lets the project swap detection or vectorization backends without rewriting the export layer.
