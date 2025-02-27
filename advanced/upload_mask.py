from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext, StateID
from ..utils.image import (
    image_to_bytes,
    tensor_to_image,
)


@dataclass(kw_only=True)
class Params:
    mask: torch.Tensor


async def _process(
    ctx: EditorAPIContext,
    params: Params,
) -> StateID:
    # convert tensors to PIL images
    mask_pil = tensor_to_image(params.mask.unsqueeze(0))

    # make some assertions
    assert mask_pil.mode == "L", "Mask must be L mode"

    # convert PIL images to BytesIO
    mask_bytes = image_to_bytes(mask_pil)

    # queue state/create
    stateid_mask = await ctx.create_state(file=mask_bytes)

    return stateid_mask


class UploadMask:
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
                "mask": (
                    "MASK",
                    {
                        "tooltip": "The mask to upload",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STATEID",)
    RETURN_NAMES = ("mask",)

    TITLE = "[Advanced] Upload Mask"
    DESCRIPTION = "Create a new state id from a mask."
    CATEGORY = "Finegrain/advanced"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        mask: torch.Tensor,
    ) -> tuple[StateID]:
        return (
            api.run_one_sync(
                co=_process,
                params=Params(mask=mask),
            ),
        )
