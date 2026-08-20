from __future__ import annotations

from PIL import Image


OPENCLIP_MEAN_RGB = (123, 117, 104)


def letterbox_to_square(image: Image.Image, fill: tuple[int, int, int] = OPENCLIP_MEAN_RGB) -> Image.Image:
    """Pad a PIL image to a square canvas without cropping the original content."""
    width, height = image.size
    side = max(width, height)
    if width == side and height == side:
        return image.copy()
    canvas = Image.new("RGB", (side, side), fill)
    left = (side - width) // 2
    top = (side - height) // 2
    canvas.paste(image, (left, top))
    return canvas


def apply_preprocess_mode(image: Image.Image, mode: str) -> Image.Image:
    if mode == "model":
        return image
    if mode == "letterbox":
        return letterbox_to_square(image)
    raise ValueError(f"Unknown preprocess mode: {mode}")
