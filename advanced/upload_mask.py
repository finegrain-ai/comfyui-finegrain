from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext, StateID, _get_ctx
from ..utils.image import (
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
    pil_mask = tensor_to_image(params.mask.unsqueeze(0))

    # make some assertions
    assert pil_mask.mode == "L", "Mask must be L mode"

    # upload the mask to the API
    stateid_mask = await ctx.call_async.upload_pil_image(pil_mask)

    return stateid_mask


class UploadMask:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
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

    TITLE = "[Low level] Upload Mask"
    DESCRIPTION = "Create a new state id from a mask."
    CATEGORY = "Finegrain/advanced"
    FUNCTION = "process"

    def process(
        self,
        mask: torch.Tensor,
    ) -> tuple[StateID]:
        return (
            _get_ctx().run_one_sync(
                co=_process,
                params=Params(mask=mask),
            ),
        )
