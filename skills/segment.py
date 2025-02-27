from dataclasses import dataclass
from typing import Any

import torch

from ..utils.bbox import BoundingBox
from ..utils.context import EditorAPIContext
from ..utils.image import (
    image_to_bytes,
    image_to_tensor,
    tensor_to_image,
)


@dataclass(kw_only=True)
class Params:
    image: torch.Tensor
    bbox: BoundingBox
    cropped: bool


async def _process(
    ctx: EditorAPIContext,
    params: Params,
) -> torch.Tensor:
    # convert tensors to PIL images
    image_pil = tensor_to_image(params.image.permute(0, 3, 1, 2))

    # make some assertions
    assert image_pil.mode == "RGB", "Image must be RGB"

    # convert PIL images to BytesIO
    image_bytes = image_to_bytes(image_pil)

    # queue state/create
    stateid_image = await ctx.create_state(file=image_bytes)

    # queue skills/infer-bbox
    stateid_mask = await ctx.skill_segment(
        stateid_image=stateid_image,
        bbox=params.bbox,
    )

    # queue skills/crop
    if params.cropped:
        stateid_mask = await ctx.skill_crop(
            stateid_image=stateid_mask,
            bbox=params.bbox,
        )

    # queue state/download
    mask = await ctx.download_image(stateid_mask)

    # convert PIL image to tensor
    mask_tensor = image_to_tensor(mask).squeeze(0)
    return mask_tensor


class Segment:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "api": (
                    "FG_API",
                    {
                        "tooltip": "The Finegrain API context",
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The image to segment",
                    },
                ),
                "bbox": (
                    "BBOX",
                    {
                        "tooltip": "Bounding box of the object to segment",
                    },
                ),
            },
            "optional": {
                "cropped": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Crop the mask to the bounding box",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    TITLE = "Segment"
    DESCRIPTION = "Segment an object in an image."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        image: torch.Tensor,
        bbox: BoundingBox,
        cropped: bool,
    ) -> tuple[torch.Tensor]:
        return (
            api.run_one_sync(
                co=_process,
                params=Params(
                    image=image,
                    bbox=bbox,
                    cropped=cropped,
                ),
            ),
        )
