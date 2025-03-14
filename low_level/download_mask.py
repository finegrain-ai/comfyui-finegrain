from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..utils.context import EditorAPIContext, StateID, _get_ctx
from ..utils.image import image_to_tensor


@dataclass(kw_only=True)
class Params:
    mask: StateID
    image_format: Literal["JPEG", "PNG", "WEBP", "AUTO"]
    resolution: Literal["FULL", "DISPLAY"]


class DownloadMask:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "mask": (
                    "STATEID",
                    {
                        "tooltip": "The mask stateid to download",
                    },
                ),
                "image_format": (
                    [
                        "AUTO",
                        "JPEG",
                        "PNG",
                        "WEBP",
                    ],
                ),
                "resolution": (
                    [
                        "FULL",
                        "DISPLAY",
                    ],
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    TITLE = "[Low level] Download Mask"
    DESCRIPTION = "Download an image from a state id."
    CATEGORY = "Finegrain/low-level"
    FUNCTION = "process"

    @staticmethod
    async def _process(
        ctx: EditorAPIContext,
        params: Params,
    ) -> torch.Tensor:
        # download the image from the API
        pil_mask = await ctx.call_async.download_pil_image(params.mask)

        # convert to tensor
        tensor_mask = image_to_tensor(pil_mask).squeeze(0)

        return tensor_mask

    def process(
        self,
        mask: StateID,
        image_format: Literal["JPEG", "PNG", "WEBP", "AUTO"],
        resolution: Literal["FULL", "DISPLAY"],
    ) -> tuple[torch.Tensor]:
        return (
            _get_ctx().run_one_sync(
                co=self._process,
                params=Params(
                    mask=mask,
                    image_format=image_format,
                    resolution=resolution,
                ),
            ),
        )
