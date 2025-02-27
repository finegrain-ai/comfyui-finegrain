from typing import Any

import torch
from PIL import ImageDraw

from .image import image_to_tensor, tensor_to_image

BoundingBox = tuple[int, int, int, int]


class CreateBoundingBox:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "xmin": ("INT", {}),
                "ymin": ("INT", {}),
                "xmax": ("INT", {}),
                "ymax": ("INT", {}),
            },
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)

    TITLE = "Bounding Box"
    DESCRIPTION = "Bounding box coordinates."
    CATEGORY = "Finegrain/bbox"
    FUNCTION = "process"

    def process(
        self,
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
    ) -> tuple[BoundingBox]:
        assert xmin <= xmax, "xmin must be less than xmax"
        assert ymin <= ymax, "ymin must be less than ymax"
        return ((xmin, ymin, xmax, ymax),)


class DrawBoundingBox:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The image to draw the bounding box on.",
                    },
                ),
                "bbox": (
                    "BBOX",
                    {
                        "tooltip": "The bounding box to draw.",
                    },
                ),
                "color": (
                    "STRING",
                    {
                        "default": "red",
                        "tooltip": "The color of the bounding box.",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "tooltip": "The width of the bounding box.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TITLE = "Draw Bounding Box"
    DESCRIPTION = "Draw a bounding box on an image."
    CATEGORY = "Finegrain/bbox"
    FUNCTION = "process"

    def process(
        self,
        image: torch.Tensor,
        bbox: BoundingBox,
        color: str,
        width: int,
    ) -> tuple[torch.Tensor]:
        pil_image = tensor_to_image(image.permute(0, 3, 1, 2))
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle(bbox, outline=color, width=width)
        image = image_to_tensor(pil_image).permute(0, 2, 3, 1)
        return (image,)


class ImageCropBoundingBox:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The image to crop.",
                    },
                ),
                "bbox": (
                    "BBOX",
                    {
                        "tooltip": "The bounding box to crop the image with.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TITLE = "Crop Image to Bounding Box"
    DESCRIPTION = "Crop an image using a bounding box."
    CATEGORY = "Finegrain/bbox"
    FUNCTION = "process"

    def process(
        self,
        image: torch.Tensor,
        bbox: BoundingBox,
    ) -> tuple[torch.Tensor]:
        image = image[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]
        return (image,)


class MaskCropBoundingBox:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": "The mask to crop.",
                    },
                ),
                "bbox": (
                    "BBOX",
                    {
                        "tooltip": "The bounding box to crop the mask with.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    TITLE = "Crop Mask to Bounding Box"
    DESCRIPTION = "Crop a mask using a bounding box."
    CATEGORY = "Finegrain/bbox"
    FUNCTION = "process"

    def process(
        self,
        mask: torch.Tensor,
        bbox: BoundingBox,
    ) -> tuple[torch.Tensor]:
        mask = mask[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]
        return (mask,)
