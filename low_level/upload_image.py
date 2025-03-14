from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext, StateID, _get_ctx
from ..utils.image import tensor_to_image


@dataclass(kw_only=True)
class Params:
    image: torch.Tensor


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
    CATEGORY = "Finegrain/low-level"
    FUNCTION = "process"

    @staticmethod
    async def _process(
        ctx: EditorAPIContext,
        params: Params,
    ) -> StateID:
        # convert tensors to PIL images
        pil_image = tensor_to_image(params.image.permute(0, 3, 1, 2))

        # make some assertions
        assert pil_image.mode in ["RGB", "RGBA"], "Image must be RGB or RGBA"

        # upload the image to the API
        stateid_image = await ctx.call_async.upload_pil_image(pil_image)

        return stateid_image

    def process(
        self,
        image: torch.Tensor,
    ) -> tuple[StateID]:
        return (
            _get_ctx().run_one_sync(
                co=self._process,
                params=Params(image=image),
            ),
        )
