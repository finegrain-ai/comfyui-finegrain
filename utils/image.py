from typing import Any

import numpy as np
import torch
from PIL import Image


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    assert tensor.ndim == 4, f"Expected 4D tensor, got {tensor.ndim}D"
    assert tensor.shape[0] == 1, f"Expected batch size of 1, got {tensor.shape[0]}"

    num_channels = tensor.shape[1]
    tensor = tensor.clamp(0, 1).squeeze(0)
    tensor = tensor.to(torch.float32)  # to avoid numpy error with bfloat16

    match num_channels:
        case 1:
            tensor = tensor.squeeze(0)
        case 3 | 4:
            tensor = tensor.permute(1, 2, 0)
        case _:
            raise ValueError(f"Unsupported number of channels: {num_channels}")

    array = tensor.detach().cpu().numpy()  # type: ignore[reportUnknownType]
    array = (array * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(array)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.tensor(array, dtype=torch.float32)

    assert isinstance(image.mode, str)  # type: ignore
    match image.mode:
        case "L":
            tensor = tensor.unsqueeze(0)
        case "RGBA" | "RGB":
            tensor = tensor.permute(2, 0, 1)
        case _:
            raise ValueError(f"Unsupported image mode: {image.mode}")

    return tensor.unsqueeze(0)


class ApplyTransparencyMask:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The RGB image to apply the mask to.",
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "The mask to apply to the image.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TITLE = "Apply Transparency Mask to Image"
    DESCRIPTION = "Apply a transparency mask to an RGB image, combining them into a single RGBA image."
    CATEGORY = "Finegrain/image"
    FUNCTION = "process"

    def process(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        assert image.ndim == 4, "Image must be 4D"
        assert image.shape[-1] == 3, "Image must be in RGB mode"
        assert mask.ndim == 3, "Mask must be 3D"

        image = torch.cat((image, mask.unsqueeze(-1)), dim=-1)
        return (image,)
