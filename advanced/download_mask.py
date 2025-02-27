from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext, StateID
from ..utils.image import image_to_tensor


@dataclass(kw_only=True)
class Params:
    mask: StateID


async def _process(
    ctx: EditorAPIContext,
    params: Params,
) -> torch.Tensor:
    # queue state/create
    mask_pil = await ctx.download_image(params.mask)

    # convert to tensor
    mask_tensor = image_to_tensor(mask_pil).squeeze(0)

    return mask_tensor


class DownloadMask:
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
                    "STATEID",
                    {
                        "tooltip": "The mask stateid to download",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    TITLE = "[Advanced] Download Mask"
    DESCRIPTION = "Download an image from a state id."
    CATEGORY = "Finegrain/advanced"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        mask: StateID,
    ) -> tuple[torch.Tensor]:
        return (
            api.run_one_sync(
                co=_process,
                params=Params(mask=mask),
            ),
        )
