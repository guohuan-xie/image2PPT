from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Color = tuple[int, int, int, int]
NodeKind = Literal["primitive", "freeform", "svg_asset", "picture_asset", "group"]
PrimitiveType = Literal["rect", "circle", "line", "text"]


class Point(BaseModel):
    x: float
    y: float


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class BaseNode(BaseModel):
    id: str
    kind: NodeKind
    bbox: BoundingBox
    z_index: int = 0
    fill_color: Color | None = None
    stroke_color: Color | None = None
    stroke_width: float = 0.0
    opacity: float = 1.0


class PrimitiveNode(BaseNode):
    kind: Literal["primitive"] = "primitive"
    primitive_type: PrimitiveType
    text: str | None = None
    text_color: Color | None = None
    font_size: float | None = None
    start: Point | None = None
    end: Point | None = None


class FreeformNode(BaseNode):
    kind: Literal["freeform"] = "freeform"
    points: list[Point] = Field(default_factory=list)
    closed: bool = True
    source_region_id: str | None = None


class SvgAssetNode(BaseNode):
    kind: Literal["svg_asset"] = "svg_asset"
    svg_path: str
    fallback_image_path: str | None = None
    fallback_points: list[Point] = Field(default_factory=list)
    source_region_id: str | None = None


class PictureAssetNode(BaseNode):
    kind: Literal["picture_asset"] = "picture_asset"
    image_path: str
    source_region_id: str | None = None


class GroupNode(BaseNode):
    kind: Literal["group"] = "group"
    children: list[str] = Field(default_factory=list)


Node = PrimitiveNode | FreeformNode | SvgAssetNode | PictureAssetNode | GroupNode


class SceneGraph(BaseModel):
    canvas_width: int
    canvas_height: int
    background_color: Color = (255, 255, 255, 255)
    nodes: list[Node] = Field(default_factory=list)
