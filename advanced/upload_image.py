from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext, StateID, _get_ctx
from ..utils.image import (
    image_to_bytes,
    tensor_to_image,
)


@dataclass(kw_only=True)
class Params:
    image: torch.Tensor


async def _process(
    ctx: EditorAPIContext,
    params: Params,
) -> StateID:
    # convert tensors to PIL images
    image_pil = tensor_to_image(params.image.permute(0, 3, 1, 2))

    # make some assertions
    assert image_pil.mode in ["RGB", "RGBA"], "Image must be RGB or RGBA"

    # convert PIL images to BytesIO
    image_bytes = image_to_bytes(image_pil)

    # queue state/create
    stateid_image = await ctx.call_async.upload_image(file=image_bytes)

    return stateid_image


class UploadImage:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The image to upload",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STATEID",)
    RETURN_NAMES = ("image",)

    TITLE = "[Low level] Upload Image"
    DESCRIPTION = "Create a new state id from an image."
    CATEGORY = "Finegrain/advanced"
    FUNCTION = "process"

    def process(
        self,
        image: torch.Tensor,
    ) -> tuple[StateID]:
        return (
            _get_ctx().run_one_sync(
                co=_process,
                params=Params(image=image),
            ),
        )
