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
    cutout: torch.Tensor
    width: int
    height: int
    seed: int
    bgcolor: str | None
    bbox: BoundingBox | None


async def _process(ctx: EditorAPIContext, params: Params) -> torch.Tensor:
    assert 0 <= params.seed <= 999, "Seed must be an integer between 0 and 999"
    assert params.width >= 8, "Width must be at least 8"
    assert params.height >= 8, "Height must be at least 8"

    # convert tensors to PIL images
    cutout_pil = tensor_to_image(params.cutout.permute(0, 3, 1, 2))

    # make some assertions
    assert cutout_pil.mode == "RGBA", "Cutout must be RGBA"

    # convert PIL images to BytesIO
    cutout_bytes = image_to_bytes(cutout_pil)

    # queue state/create
    stateid_cutout = await ctx.create_state(file=cutout_bytes)

    # queue skills/shadow
    stateid_shadow = await ctx.skill_shadow(
        stateid_cutout=stateid_cutout,
        resolution=(params.width, params.height),
        bbox=params.bbox,
        seed=params.seed,
    )

    # queue skills/set-background-color
    if params.bgcolor and params.bgcolor != "transparent":
        stateid_shadow = await ctx.skill_set_bgcolor(
            stateid_image=stateid_shadow,
            color=params.bgcolor,
        )

    # queue state/download
    shadow_pil = await ctx.download_image(stateid_shadow)

    # convert PIL image to tensor
    shadow_tensor = image_to_tensor(shadow_pil).permute(0, 2, 3, 1)

    return shadow_tensor


class Shadow:
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
                "cutout": (
                    "IMAGE",
                    {
                        "tooltip": "The cutout to create a shadow packshot from",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 8,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Width of the output image.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 8,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Height of the output image.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 999,
                        "tooltip": "Seed for the random number generator.",
                    },
                ),
            },
            "optional": {
                "bgcolor": (
                    "STRING",
                    {
                        "default": "transparent",
                        "tooltip": "Background color of the shadow.",
                    },
                ),
                "bbox": (
                    "BBOX",
                    {
                        "tooltip": "Bounding box of where to place the object in the output image.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TITLE = "Shadow"
    DESCRIPTION = "Create a shadow packshot from a cutout."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        cutout: torch.Tensor,
        width: int,
        height: int,
        seed: int,
        bgcolor: str | None = None,
        bbox: BoundingBox | None = None,
    ) -> tuple[torch.Tensor]:
        return (
            api.run_one_sync(
                co=_process,
                params=Params(
                    cutout=cutout,
                    width=width,
                    height=height,
                    seed=seed,
                    bgcolor=bgcolor,
                    bbox=bbox,
                ),
            ),
        )
