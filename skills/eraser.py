from dataclasses import dataclass
from typing import Any, get_args

import torch

from ..utils.context import EditorAPIContext, ErrorResult, Mode, _get_ctx
from ..utils.image import image_to_bytes, image_to_tensor, tensor_to_image


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
            },
            "optional": {
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
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    @staticmethod
    async def _process(
        ctx: EditorAPIContext,
        params: Params,
    ) -> torch.Tensor:
        assert params.mode in get_args(Mode), f"Mode must be one of {get_args(Mode)}"
        assert params.seed >= 0, "Seed must be a non-negative integer"

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
        image_erase = await ctx.call_async.download_image(stateid_erase)

        # convert PIL image to tensor
        tensor_erase = image_to_tensor(image_erase).permute(0, 2, 3, 1)

        return tensor_erase

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
