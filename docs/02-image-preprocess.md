# Image Preprocess

## Goals

- normalize all inputs to `RGBA`
- cap resolution for predictable runtime
- reduce color noise before segmentation
- preserve transparent edges when possible

## Current rules

- convert every input image to `RGBA`
- resize down when the largest side exceeds `max_side_px`
- optionally blur before quantization if noisy inputs appear
- quantize to a small palette so near-identical Gemini colors collapse together

## Notes

- alpha-aware segmentation matters more than anti-aliased edge fidelity in the MVP
- quantization count should stay low for icons and flat illustrations
- preprocessing artifacts are written to `preprocessed.png` for debugging
