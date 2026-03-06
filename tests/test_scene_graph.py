from image2pptx.scene_graph import BoundingBox, Point, PrimitiveNode, SceneGraph


def test_scene_graph_serializes_primitives() -> None:
    graph = SceneGraph(
        canvas_width=320,
        canvas_height=180,
        nodes=[
            PrimitiveNode(
                id="rect-1",
                primitive_type="rect",
                bbox=BoundingBox(x=10, y=20, width=100, height=50),
                fill_color=(255, 0, 0, 255),
            ),
            PrimitiveNode(
                id="line-1",
                primitive_type="line",
                bbox=BoundingBox(x=0, y=0, width=50, height=2),
                start=Point(x=0, y=1),
                end=Point(x=50, y=1),
                stroke_color=(0, 0, 0, 255),
                stroke_width=2,
            ),
        ],
    )

    payload = graph.model_dump(mode="json")

    assert payload["canvas_width"] == 320
    assert payload["nodes"][0]["primitive_type"] == "rect"
    assert payload["nodes"][1]["primitive_type"] == "line"
