from dataclasses import dataclass
from typing import Any

import torch

from ..utils.bbox import BoundingBox
from ..utils.context import EditorAPIContext, ErrorResult, _get_ctx
from ..utils.image import image_to_tensor, tensor_to_image


@dataclass(kw_only=True)
class Params:
    cutout: torch.Tensor
    width: int
    height: int
    seed: int
    bgcolor: str
    bbox: BoundingBox | None


class Shadow:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "cutout": (
                    "IMAGE",
                    {
                        "tooltip": "The cutout to create a shadow packshot from",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 8,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Width of the output image.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 8,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Height of the output image.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 999,
                        "tooltip": "Seed for the random number generator.",
                    },
                ),
                "bgcolor": (
                    "STRING",
                    {
                        "default": "transparent",
                        "tooltip": "Background color of the shadow.",
                    },
                ),
            },
            "optional": {
                "bbox": (
                    "BBOX",
                    {
                        "tooltip": "Bounding box of where to place the object in the output image.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TITLE = "Shadow"
    DESCRIPTION = "Create a shadow packshot from a cutout."
    CATEGORY = "Finegrain/high-level"
    FUNCTION = "process"

    @staticmethod
    async def _process(
        ctx: EditorAPIContext,
        params: Params,
    ) -> torch.Tensor:
        assert 0 <= params.seed <= 999, "Seed must be an integer between 0 and 999"
        assert params.width >= 8, "Width must be at least 8"
        assert params.height >= 8, "Height must be at least 8"

        # convert tensors to PIL images
        pil_cutout = tensor_to_image(params.cutout.permute(0, 3, 1, 2))

        # make some assertions
        assert pil_cutout.mode == "RGBA", "Cutout must be RGBA"

        # upload cutout
        stateid_cutout = await ctx.call_async.upload_pil_image(pil_cutout)

        # call shadow skill
        result_shadow = await ctx.call_async.shadow(
            state_id=stateid_cutout,
            resolution=(params.width, params.height),
            bbox=params.bbox,
            seed=params.seed,
        )
        if isinstance(result_shadow, ErrorResult):
            raise ValueError(f"Failed to create shadow: {result_shadow.error}")
        stateid_shadow = result_shadow.state_id

        if params.bgcolor != "transparent":
            # call set_background_color skill
            result_bgcolor = await ctx.call_async.set_background_color(
                state_id=stateid_shadow,
                background=params.bgcolor,
            )
            if isinstance(result_bgcolor, ErrorResult):
                raise ValueError(f"Failed to set background color: {result_bgcolor.error}")
            stateid_shadow = result_bgcolor.state_id

        # download output image
        pil_output = await ctx.call_async.download_pil_image(stateid_shadow)

        # convert PIL image to tensor
        tensor_output = image_to_tensor(pil_output).permute(0, 2, 3, 1)

        return tensor_output

    def process(
        self,
        cutout: torch.Tensor,
        width: int,
        height: int,
        seed: int,
        bgcolor: str,
        bbox: BoundingBox | None = None,
    ) -> tuple[torch.Tensor]:
        return (
            _get_ctx().run_one_sync(
                co=self._process,
                params=Params(
                    cutout=cutout,
                    width=width,
                    height=height,
                    seed=seed,
                    bgcolor=bgcolor,
                    bbox=bbox,
                ),
            ),
        )
