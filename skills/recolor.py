from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext, ErrorResult, _get_ctx
from ..utils.image import (
    image_to_bytes,
    image_to_tensor,
    tensor_to_image,
)


@dataclass(kw_only=True)
class Params:
    image: torch.Tensor
    mask: torch.Tensor
    color: str


class Recolor:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The image to recolor something in",
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "The mask of the object to recolor",
                    },
                ),
                "color": (
                    "STRING",
                    {
                        "default": "#ff0000",
                        "tooltip": "The color to recolor the object to",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TITLE = "Recolor"
    DESCRIPTION = "Recolor a masked object in an image."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    @staticmethod
    async def _process(ctx: EditorAPIContext, params: Params) -> torch.Tensor:
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

        # upload image and mask
        stateid_image = await ctx.call_async.upload_image(file=image_bytes)
        stateid_mask = await ctx.call_async.upload_image(file=mask_bytes)

        # call recolor skill
        result_recolor = await ctx.call_async.recolor(
            image_state_id=stateid_image,
            mask_state_id=stateid_mask,
            color=params.color,
        )
        if isinstance(result_recolor, ErrorResult):
            raise ValueError(f"Failed to recolor object: {result_recolor.error}")
        stateid_recolor = result_recolor.state_id

        # download output image
        recolored_image = await ctx.call_async.download_image(stateid_recolor)

        # convert PIL image to tensor
        recolored_tensor = image_to_tensor(recolored_image).permute(0, 2, 3, 1)

        return recolored_tensor

    def process(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        color: str,
    ) -> tuple[torch.Tensor]:
        return (
            _get_ctx().run_one_sync(
                co=self._process,
                params=Params(
                    image=image,
                    mask=mask,
                    color=color,
                ),
            ),
        )
