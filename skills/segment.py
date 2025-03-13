from dataclasses import dataclass
from typing import Any

import torch

from ..utils.bbox import BoundingBox
from ..utils.context import EditorAPIContext, ErrorResult, _get_ctx
from ..utils.image import (
    image_to_bytes,
    image_to_tensor,
    tensor_to_image,
)


@dataclass(kw_only=True)
class Params:
    image: torch.Tensor
    bbox: BoundingBox
    cropped: bool


class Segment:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The image to segment",
                    },
                ),
                "bbox": (
                    "BBOX",
                    {
                        "tooltip": "Bounding box of the object to segment",
                    },
                ),
            },
            "optional": {
                "cropped": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Crop the mask to the bounding box",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    TITLE = "Segment"
    DESCRIPTION = "Segment an object in an image."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    @staticmethod
    async def _process(
        ctx: EditorAPIContext,
        params: Params,
    ) -> torch.Tensor:
        # convert tensors to PIL images
        image_pil = tensor_to_image(params.image.permute(0, 3, 1, 2))

        # make some assertions
        assert image_pil.mode == "RGB", "Image must be RGB"

        # convert PIL images to BytesIO
        image_bytes = image_to_bytes(image_pil)

        # upload image
        stateid_image = await ctx.call_async.upload_image(file=image_bytes)

        # call segment skill
        result_segment = await ctx.call_async.segment(
            state_id=stateid_image,
            bbox=params.bbox,
        )
        if isinstance(result_segment, ErrorResult):
            raise ValueError(f"Failed to segment object: {result_segment.error}")
        stateid_mask = result_segment.state_id

        # call crop if needed
        if params.cropped:
            result_crop = await ctx.call_async.crop(
                state_id=stateid_mask,
                bbox=params.bbox,
            )
            if isinstance(result_crop, ErrorResult):
                raise ValueError(f"Failed to crop mask: {result_crop.error}")
            stateid_mask = result_crop.state_id

        # download mask
        mask = await ctx.call_async.download_image(stateid_mask)

        # convert PIL image to tensor
        mask_tensor = image_to_tensor(mask).squeeze(0)
        return mask_tensor

    def process(
        self,
        image: torch.Tensor,
        bbox: BoundingBox,
        cropped: bool = False,
    ) -> tuple[torch.Tensor]:
        return (
            _get_ctx().run_one_sync(
                co=self._process,
                params=Params(
                    image=image,
                    bbox=bbox,
                    cropped=cropped,
                ),
            ),
        )
