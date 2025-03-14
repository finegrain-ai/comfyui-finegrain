from dataclasses import dataclass
from typing import Any, get_args

import torch

from ..utils.context import EditorAPIContext, ErrorResult, Mode, _get_ctx
from ..utils.image import image_to_tensor, tensor_to_image


@dataclass(kw_only=True)
class Params:
    image: torch.Tensor
    mask: torch.Tensor
    mode: Mode
    seed: int


class Eraser:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The image to erase an object from",
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "The mask of the object to erase",
                    },
                ),
                "mode": (
                    [
                        "premium",
                        "standard",
                        "express",
                    ],
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 999,
                        "tooltip": "Seed for the random number generator",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TITLE = "Eraser"
    DESCRIPTION = "Erase an object from an image using a mask."
    CATEGORY = "Finegrain/high-level"
    FUNCTION = "process"

    @staticmethod
    async def _process(
        ctx: EditorAPIContext,
        params: Params,
    ) -> torch.Tensor:
        assert params.mode in get_args(Mode), f"Mode must be one of {get_args(Mode)}"
        assert 0 <= params.seed <= 999, "Seed must be an integer between 0 and 999"

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

        # call erase skill
        result_erase = await ctx.call_async.erase(
            image_state_id=stateid_image,
            mask_state_id=stateid_mask,
            mode=params.mode,
            seed=params.seed,
        )
        if isinstance(result_erase, ErrorResult):
            raise ValueError(f"Failed to erase object: {result_erase.error}")
        stateid_erase = result_erase.state_id

        # download output image
        pil_output = await ctx.call_async.download_pil_image(stateid_erase)

        # convert PIL image to tensor
        tensor_output = image_to_tensor(pil_output).permute(0, 2, 3, 1)

        return tensor_output

    def process(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        mode: Mode,
        seed: int,
    ) -> tuple[torch.Tensor]:
        return (
            _get_ctx().run_one_sync(
                co=self._process,
                params=Params(
                    image=image,
                    mask=mask,
                    mode=mode,
                    seed=seed,
                ),
            ),
        )
