from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext
from ..utils.image import (
    image_to_bytes,
    image_to_tensor,
    tensor_to_image,
)


@dataclass(kw_only=True)
class Params:
    image: torch.Tensor
    mask: torch.Tensor
    mode: str
    seed: int


async def _process(
    ctx: EditorAPIContext,
    params: Params,
) -> torch.Tensor:
    assert params.mode in ["express", "standard", "premium"], "Invalid mode"
    assert params.seed >= 0, "Seed must be a non-negative integer"

    # convert tensors to PIL images
    image_pil = tensor_to_image(params.image.permute(0, 3, 1, 2))
    mask_pil = tensor_to_image(params.mask.unsqueeze(0))

    # make some assertions
    assert image_pil.size == mask_pil.size, "Image and mask sizes do not match"
    assert image_pil.mode == "RGB", "Image must be RGB"
    assert mask_pil.mode == "L", "Mask must be grayscale"

    # convert PIL images to BytesIO
    image_bytes = image_to_bytes(image_pil)
    mask_bytes = image_to_bytes(mask_pil)

    # queue state/create
    stateid_image = await ctx.create_state(file=image_bytes)
    stateid_mask = await ctx.create_state(file=mask_bytes)

    # queue skills/erase
    stateid_erased = await ctx.skill_erase(
        stateid_image=stateid_image,
        stateid_mask=stateid_mask,
        mode=params.mode,
        seed=params.seed,
    )

    # queue state/download
    erased_image = await ctx.download_image(stateid_erased)

    # convert PIL image to tensor
    erased_tensor = image_to_tensor(erased_image).permute(0, 2, 3, 1)

    return erased_tensor


class Eraser:
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
                        "tooltip": "The image to erase an object from",
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "The mask of the object to erase",
                    },
                ),
                "mode": (
                    [
                        "express",
                        "standard",
                        "premium",
                    ],
                ),
            },
            "optional": {
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 999,
                        "tooltip": "Seed for the random number generator",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TITLE = "Eraser"
    DESCRIPTION = "Erase an object from an image using a mask."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        image: torch.Tensor,
        mask: torch.Tensor,
        mode: str,
        seed: int,
    ) -> tuple[torch.Tensor]:
        return (
            api.run_one_sync(
                co=_process,
                params=Params(
                    image=image,
                    mask=mask,
                    mode=mode,
                    seed=seed,
                ),
            ),
        )
