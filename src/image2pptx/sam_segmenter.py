from __future__ import annotations

from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image

from .config import SamSettings
from .mask_postprocess import ComponentMask, bbox_from_mask

try:
    from ultralytics import SAM
except ImportError:  # pragma: no cover - optional runtime dependency
    SAM = None

try:
    import torch
except ImportError:  # pragma: no cover - ultralytics should provide torch at runtime
    torch = None

_SAM_MODELS: dict[tuple[str, str], object] = {}


def segment_with_mobilesam(
    rgba: np.ndarray,
    settings: SamSettings,
    text_mask: np.ndarray | None = None,
    artifacts_dir: Path | None = None,
) -> tuple[np.ndarray, list[ComponentMask]]:
    if SAM is None:
        raise RuntimeError("ultralytics is required for segmentation_backend='sam'.")

    device = resolve_sam_device(settings.device)
    sam_input = prepare_sam_input(rgba, text_mask)
    if artifacts_dir is not None:
        Image.fromarray(sam_input, mode="RGBA").save(artifacts_dir / "sam_input.png")

    model = get_sam_model(settings.model_path, device)
    results = model.predict(
        sam_input[:, :, :3],
        stream=False,
        verbose=False,
        imgsz=settings.imgsz,
        device=device,
    )
    if not results:
        return sam_input, []

    result = results[0]
    if result.masks is None or result.boxes is None:
        return sam_input, []

    mask_data = result.masks.data.cpu().numpy().astype(bool)
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(mask_data), dtype=float)

    components: list[ComponentMask] = []
    for index, (mask, score) in enumerate(zip(mask_data, scores, strict=False)):
        bbox = bbox_from_mask(mask)
        area = int(np.count_nonzero(mask))
        if area == 0:
            continue
        if (bbox[2] == 0 or bbox[3] == 0) and index < len(boxes_xyxy):
            bbox = xyxy_to_bbox(boxes_xyxy[index])
        components.append(
            ComponentMask(
                id=f"sam-{index}",
                mask=mask,
                bbox=bbox,
                area=area,
                score=float(score),
            )
        )

    return sam_input, components


def prepare_sam_input(rgba: np.ndarray, text_mask: np.ndarray | None) -> np.ndarray:
    sam_input = rgba.copy()
    if text_mask is None or not np.any(text_mask):
        return sam_input

    inpaint_mask = (text_mask > 0).astype(np.uint8) * 255
    inpainted_bgr = cv2.inpaint(
        cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR),
        inpaint_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA,
    )
    sam_input[:, :, :3] = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    return sam_input


def get_sam_model(model_path: str, device: str):
    cache_key = (model_path, device)
    model = _SAM_MODELS.get(cache_key)
    if model is None:
        model = SAM(model_path)
        _SAM_MODELS[cache_key] = model
    return model


def resolve_sam_device(requested_device: str) -> str:
    normalized_device = requested_device.strip().lower()

    if normalized_device in {"", "auto"}:
        return detect_best_sam_device()

    if normalized_device.startswith("cuda") and not cuda_is_available():
        warnings.warn(
            "CUDA was requested for SAM segmentation, but no GPU is available. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"

    if normalized_device == "mps" and not mps_is_available():
        warnings.warn(
            "MPS was requested for SAM segmentation, but it is unavailable. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"

    return requested_device


def detect_best_sam_device() -> str:
    if cuda_is_available():
        return "cuda:0"
    if mps_is_available():
        return "mps"
    return "cpu"


def cuda_is_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def mps_is_available() -> bool:
    return bool(
        torch is not None
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def xyxy_to_bbox(xyxy: np.ndarray) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy.tolist()
    return (
        int(round(x1)),
        int(round(y1)),
        max(0, int(round(x2 - x1))),
        max(0, int(round(y2 - y1))),
    )
