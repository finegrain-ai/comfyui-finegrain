from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext, ErrorResult, _get_ctx
from ..utils.image import image_to_tensor, tensor_to_image


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
    CATEGORY = "Finegrain/high-level"
    FUNCTION = "process"

    @staticmethod
    async def _process(
        ctx: EditorAPIContext,
        params: Params,
    ) -> torch.Tensor:
        # convert tensors to PIL images
        pil_image = tensor_to_image(params.image.permute(0, 3, 1, 2))
        pil_mask = tensor_to_image(params.mask.unsqueeze(0))

        # make some assertions
        assert pil_image.size == pil_mask.size, "Image and mask sizes do not match"
        assert pil_image.mode == "RGB", "Image must be RGB"
        assert pil_mask.mode == "L", "Mask must be grayscale"

        # upload image and mask
        stateid_image = await ctx.call_async.upload_pil_image(pil_image)
        stateid_mask = await ctx.call_async.upload_pil_image(pil_mask)

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
        pil_output = await ctx.call_async.download_pil_image(stateid_recolor)

        # convert PIL image to tensor
        tensor_output = image_to_tensor(pil_output).permute(0, 2, 3, 1)

        return tensor_output

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
