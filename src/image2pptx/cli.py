from __future__ import annotations

from pathlib import Path

import typer

from .config import Image2PptxConfig
from .pipeline import run_pipeline

app = typer.Typer(no_args_is_help=True, help="Convert flat images into editable PPTX slides.")


@app.command()
def convert(
    input_image: Path = typer.Argument(..., exists=True, readable=True, help="Source PNG/JPG image."),
    output_pptx: Path = typer.Argument(..., help="Destination PPTX file."),
    artifacts_dir: Path | None = typer.Option(None, help="Optional sidecar artifact directory."),
    segmentation_backend: str = typer.Option("sam", help="Segmentation backend: sam or cv."),
    sam_model: str = typer.Option("sam_b.pt", help="SAM model path or name, e.g. sam2_t.pt, sam_b.pt, mobile_sam.pt."),
    exporter: str = typer.Option("auto", help="Exporter: auto, python-pptx, or com."),
    dump_scene_graph: bool = typer.Option(True, "--dump-scene-graph/--no-dump-scene-graph", help="Write scene graph JSON."),
) -> None:
    """Run the image-to-PPTX conversion pipeline."""

    if segmentation_backend not in {"cv", "sam"}:
        raise typer.BadParameter("segmentation-backend must be one of: cv, sam")
    if exporter not in {"auto", "python-pptx", "com"}:
        raise typer.BadParameter("exporter must be one of: auto, python-pptx, com")

    config = Image2PptxConfig()
    config.sam.model_path = sam_model

    result = run_pipeline(
        input_image=input_image,
        output_pptx=output_pptx,
        config=config,
        artifacts_dir=artifacts_dir,
        segmentation_backend=segmentation_backend,
        exporter=exporter,
        dump_scene_graph=dump_scene_graph,
    )
    typer.echo(f"Wrote {result.output_pptx}")
    typer.echo(f"Artifacts: {result.artifacts_dir}")
