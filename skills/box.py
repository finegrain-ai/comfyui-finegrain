from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import BoundingBox, EditorAPIContext, ErrorResult, _get_ctx
from ..utils.image import (
    image_to_bytes,
    tensor_to_image,
)


@dataclass(kw_only=True)
class Params:
    image: torch.Tensor
    prompt: str


class Box:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The image to detect an object in",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "tooltip": "The product name to detect",
                    },
                ),
            },
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)

    TITLE = "Box"
    DESCRIPTION = "Box an object in an image."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    @staticmethod
    async def _process(
        ctx: EditorAPIContext,
        params: Params,
    ) -> BoundingBox:
        assert params.prompt, "Prompt must not be empty"

        # convert tensors to PIL images
        image_pil = tensor_to_image(params.image.permute(0, 3, 1, 2))

        # make some assertions
        assert image_pil.mode == "RGB", "Image must be RGB"

        # convert PIL images to BytesIO
        image_bytes = image_to_bytes(image_pil)

        # upload image
        stateid_image = await ctx.call_async.upload_image(file=image_bytes)

        # call bbox skill
        result_bbox = await ctx.call_async.infer_bbox(
            state_id=stateid_image,
            product_name=params.prompt,
        )
        if isinstance(result_bbox, ErrorResult):
            raise ValueError(f"Failed to detect object: {result_bbox.error}")

        return result_bbox.bbox

    def process(
        self,
        image: torch.Tensor,
        prompt: str,
    ) -> tuple[BoundingBox]:
        return (
            _get_ctx().run_one_sync(
                co=self._process,
                params=Params(
                    image=image,
                    prompt=prompt,
                ),
            ),
        )
