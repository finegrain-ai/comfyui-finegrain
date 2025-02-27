from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext, StateID
from ..utils.image import image_to_tensor


@dataclass(kw_only=True)
class Params:
    image: StateID


async def _process(
    ctx: EditorAPIContext,
    params: Params,
) -> torch.Tensor:
    # queue state/create
    image_pil = await ctx.download_image(params.image)

    # convert to tensor
    image_tensor = image_to_tensor(image_pil).permute(0, 2, 3, 1)

    return image_tensor


class DownloadImage:
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
                    "STATEID",
                    {
                        "tooltip": "The image stateid to download",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TITLE = "[Advanced] Download Image"
    DESCRIPTION = "Download a mask from a state id."
    CATEGORY = "Finegrain/advanced"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        image: StateID,
    ) -> tuple[torch.Tensor]:
        return (
            api.run_one_sync(
                co=_process,
                params=Params(image=image),
            ),
        )
