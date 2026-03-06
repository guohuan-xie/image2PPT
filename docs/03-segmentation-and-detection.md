# Segmentation And Detection

## OpenCV backend

The default backend is `cv`:

1. enumerate unique quantized colors
2. build a binary mask per color
3. run connected-component analysis
4. extract contours for each component
5. classify the region as `rect`, `circle`, `line`, `freeform`, or `svg_candidate`

## Primitive heuristics

- `rect`: four corners with high fill ratio inside the bounding box
- `circle`: high contour circularity
- `line`: elongated region with a strong aspect ratio
- `freeform`: low-complexity irregular polygon
- `svg_candidate`: larger or more complex contour that should stay vectorized

## SAM roadmap

Future `sam` mode should:

- propose semantic masks for icon-like sub-objects
- merge overly fragmented masks
- smooth edges before vectorization
- remain optional, not mandatory for the base pipeline
