from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext, StateID, _get_ctx
from ..utils.image import image_to_tensor


@dataclass(kw_only=True)
class Params:
    image: StateID


async def _process(
    ctx: EditorAPIContext,
    params: Params,
) -> torch.Tensor:
    # queue state/create
    image_pil = await ctx.call_async.download_image(params.image)

    # convert to tensor
    image_tensor = image_to_tensor(image_pil).permute(0, 2, 3, 1)

    return image_tensor


class DownloadImage:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
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

    TITLE = "[Low level] Download Image"
    DESCRIPTION = "Download a mask from a state id."
    CATEGORY = "Finegrain/advanced"
    FUNCTION = "process"

    def process(
        self,
        image: StateID,
    ) -> tuple[torch.Tensor]:
        return (
            _get_ctx().run_one_sync(
                co=_process,
                params=Params(image=image),
            ),
        )
