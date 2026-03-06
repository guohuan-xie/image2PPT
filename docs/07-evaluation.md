# Evaluation

## Manual checks

- can the deck be opened in PowerPoint
- can major elements be selected independently
- do simple shapes remain native objects
- do colors and relative positions stay close to the source image

## Debug artifacts

Review:

- `preprocessed.png`
- `scene_graph.json`
- `svg/*.svg`
- `fallback_png/*.png`

## Suggested benchmark set

- one simple logo
- one flat icon
- one 2-4 color illustration
- one poster element with limited text

## What failure looks like

- everything collapses into one raster image
- primitive detection turns obvious shapes into noisy freeforms
- major icon parts disappear after segmentation
