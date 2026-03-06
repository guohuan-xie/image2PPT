# Scope And Success

## Input scope

The MVP accepts:

- flat Gemini-generated illustrations
- icons and logos
- poster elements with limited colors
- assets with clean alpha edges

The MVP does not target:

- photo-realistic content
- dense textures
- heavy gradients
- OCR-grade text reconstruction

## Editability levels

Level 1:
Native PowerPoint shapes such as rectangles, circles, connectors, and text boxes.

Level 2:
PowerPoint freeform polygons for irregular but clean regions.

Level 3:
SVG sidecar assets preserved for the optional Windows PowerPoint exporter.

## MVP acceptance

- `image2pptx convert input.png output.pptx` works end to end.
- Main logo or icon elements can be selected independently in PowerPoint.
- Simple geometric regions stay as native objects instead of one raster background.
- Complex regions keep vector intent through SVG sidecars or shape fallback.
