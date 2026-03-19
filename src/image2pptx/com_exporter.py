from __future__ import annotations

import sys
from pathlib import Path

from .config import ExportSettings
from .scene_graph import FreeformNode, PictureAssetNode, PrimitiveNode, SceneGraph, SvgAssetNode

POINTS_PER_INCH = 72.0
MSO_EDITING_CORNER = 1
MSO_SEGMENT_LINE = 0
MSO_SHAPE_RECTANGLE = 1
MSO_SHAPE_OVAL = 9
PP_LAYOUT_BLANK = 12
PP_ALIGN_LEFT = 1
PP_ALIGN_CENTER = 2
PP_ALIGN_RIGHT = 3

try:
    import win32com.client  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on local Office install
    win32com = None
else:  # pragma: no cover - depends on local Office install
    win32com = win32com.client


class PowerPointComExporter:
    """Windows-only exporter that inserts native Office shapes and SVG assets."""

    def __init__(self, export_settings: ExportSettings) -> None:
        self.export_settings = export_settings

    def is_available(self) -> bool:
        return sys.platform.startswith("win") and win32com is not None

    def write(self, scene_graph: SceneGraph, output_path: Path) -> Path:
        if not self.is_available():
            raise RuntimeError("PowerPoint COM exporter is only available on Windows with pywin32 installed.")

        app = win32com.Dispatch("PowerPoint.Application")
        presentation = None
        try:
            app.Visible = True
            presentation = app.Presentations.Add()

            slide_width_pt = self.export_settings.slide_width_in * POINTS_PER_INCH
            slide_height_in = self.export_settings.slide_height_in
            if slide_height_in is None:
                slide_height_in = self.export_settings.slide_width_in * scene_graph.canvas_height / scene_graph.canvas_width
            slide_height_pt = slide_height_in * POINTS_PER_INCH

            presentation.PageSetup.SlideWidth = slide_width_pt
            presentation.PageSetup.SlideHeight = slide_height_pt
            slide = presentation.Slides.Add(1, PP_LAYOUT_BLANK)
            points_per_pixel = slide_width_pt / scene_graph.canvas_width

            for node in sorted(scene_graph.nodes, key=lambda item: item.z_index):
                if isinstance(node, PrimitiveNode):
                    self._add_primitive(slide, node, points_per_pixel)
                elif isinstance(node, FreeformNode):
                    self._add_freeform(slide, node, points_per_pixel)
                elif isinstance(node, SvgAssetNode):
                    self._add_svg(slide, node, points_per_pixel)
                elif isinstance(node, PictureAssetNode):
                    self._add_picture(slide, node, points_per_pixel)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            presentation.SaveAs(str(output_path))
        finally:
            if presentation is not None:
                presentation.Close()
            app.Quit()
        return output_path

    def _add_primitive(self, slide, node: PrimitiveNode, points_per_pixel: float) -> None:
        if node.primitive_type == "line" and node.start and node.end:
            shape = slide.Shapes.AddLine(
                node.start.x * points_per_pixel,
                node.start.y * points_per_pixel,
                node.end.x * points_per_pixel,
                node.end.y * points_per_pixel,
            )
            self._apply_line(shape, node.stroke_color or node.fill_color, node.stroke_width * points_per_pixel)
            return

        if node.primitive_type == "text":
            shape = slide.Shapes.AddTextbox(
                1,
                node.bbox.x * points_per_pixel,
                node.bbox.y * points_per_pixel,
                node.bbox.width * points_per_pixel,
                node.bbox.height * points_per_pixel,
            )
            shape.Fill.Visible = False
            shape.Line.Visible = False
            if node.text:
                shape.TextFrame.TextRange.Text = node.text
                shape.TextFrame.WordWrap = not node.single_line
                shape.TextFrame.MarginLeft = 2
                shape.TextFrame.MarginRight = 2
                shape.TextFrame.MarginTop = 1
                shape.TextFrame.MarginBottom = 1
                shape.TextFrame.TextRange.ParagraphFormat.Alignment = self._paragraph_alignment(node.text_align)
                font = shape.TextFrame.TextRange.Font
                font.Name = "Microsoft YaHei"
                font.Bold = -1 if node.bold else 0
                font.Size = self._fit_font_size(node, points_per_pixel)
                color = node.text_color or (0, 0, 0, 255)
                font.Color.RGB = color[0] + (color[1] << 8) + (color[2] << 16)
            return

        if node.primitive_type == "circle":
            shape_type = MSO_SHAPE_OVAL
        else:
            shape_type = MSO_SHAPE_RECTANGLE

        shape = slide.Shapes.AddShape(
            shape_type,
            node.bbox.x * points_per_pixel,
            node.bbox.y * points_per_pixel,
            node.bbox.width * points_per_pixel,
            node.bbox.height * points_per_pixel,
        )
        self._apply_fill(shape, node.fill_color)
        self._apply_line(shape, node.stroke_color, node.stroke_width * points_per_pixel)

    def _add_freeform(self, slide, node: FreeformNode, points_per_pixel: float) -> None:
        if len(node.points) < 2:
            return
        builder = slide.Shapes.BuildFreeform(
            MSO_EDITING_CORNER,
            node.points[0].x * points_per_pixel,
            node.points[0].y * points_per_pixel,
        )
        for point in node.points[1:]:
            builder.AddNodes(
                MSO_SEGMENT_LINE,
                MSO_EDITING_CORNER,
                point.x * points_per_pixel,
                point.y * points_per_pixel,
            )
        if node.closed:
            builder.AddNodes(
                MSO_SEGMENT_LINE,
                MSO_EDITING_CORNER,
                node.points[0].x * points_per_pixel,
                node.points[0].y * points_per_pixel,
            )
        shape = builder.ConvertToShape()
        self._apply_fill(shape, node.fill_color)
        self._apply_line(shape, node.stroke_color, node.stroke_width * points_per_pixel)

    def _add_svg(self, slide, node: SvgAssetNode, points_per_pixel: float) -> None:
        source_path = node.svg_path if Path(node.svg_path).exists() else node.fallback_image_path
        if source_path is None:
            return
        slide.Shapes.AddPicture(
            str(source_path),
            False,
            True,
            node.bbox.x * points_per_pixel,
            node.bbox.y * points_per_pixel,
            node.bbox.width * points_per_pixel,
            node.bbox.height * points_per_pixel,
        )

    def _add_picture(self, slide, node: PictureAssetNode, points_per_pixel: float) -> None:
        slide.Shapes.AddPicture(
            str(node.image_path),
            False,
            True,
            node.bbox.x * points_per_pixel,
            node.bbox.y * points_per_pixel,
            node.bbox.width * points_per_pixel,
            node.bbox.height * points_per_pixel,
        )

    def _apply_fill(self, shape, color) -> None:
        if color is None:
            shape.Fill.Visible = False
            return
        shape.Fill.Visible = True
        shape.Fill.Solid()
        shape.Fill.ForeColor.RGB = color[0] + (color[1] << 8) + (color[2] << 16)
        if len(color) > 3:
            shape.Fill.Transparency = max(0.0, min(1.0, 1 - (color[3] / 255)))

    def _apply_line(self, shape, color, width_points: float) -> None:
        if color is None or width_points <= 0:
            shape.Line.Visible = False
            return
        shape.Line.Visible = True
        shape.Line.ForeColor.RGB = color[0] + (color[1] << 8) + (color[2] << 16)
        shape.Line.Weight = max(0.25, width_points)

    def _paragraph_alignment(self, text_align: str) -> int:
        if text_align == "center":
            return PP_ALIGN_CENTER
        if text_align == "right":
            return PP_ALIGN_RIGHT
        return PP_ALIGN_LEFT

    def _fit_font_size(self, node: PrimitiveNode, points_per_pixel: float) -> float:
        base_size = max(8.0, (node.font_size or node.bbox.height) * points_per_pixel)
        if not node.text or not node.single_line:
            return base_size

        usable_width_pt = max(18.0, (node.bbox.width * points_per_pixel) - 4.0)
        approx_char_width_em = 0.56 if all(ord(char) < 128 for char in node.text) else 0.9
        width_limited_size = usable_width_pt / max(1.0, len(node.text) * approx_char_width_em)
        return max(8.0, min(base_size, width_limited_size))
