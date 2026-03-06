# PPTX Export

## Export modes

### `python-pptx`

- always available
- writes native rectangles, circles, connectors, and freeforms
- falls back from SVG assets to polygon or PNG placement

### `com`

- Windows only
- requires installed PowerPoint and `pywin32`
- inserts native shapes and can place SVG files directly into PowerPoint

### `auto`

- prefers `com` when available
- otherwise falls back to `python-pptx`

## Practical boundary

Pure `python-pptx` cannot turn SVG into editable Office shapes directly, so the COM exporter is the long-term path for stronger editability on Windows.
