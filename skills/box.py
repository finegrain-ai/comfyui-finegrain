from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext
from ..utils.image import (
    image_to_bytes,
    tensor_to_image,
)


@dataclass(kw_only=True)
class Params:
    image: torch.Tensor
    prompt: str


async def _process(
    ctx: EditorAPIContext,
    params: Params,
) -> torch.Tensor:
    assert params.prompt, "Prompt must not be empty"

    # convert tensors to PIL images
    image_pil = tensor_to_image(params.image.permute(0, 3, 1, 2))

    # make some assertions
    assert image_pil.mode == "RGB", "Image must be RGB"

    # convert PIL images to BytesIO
    image_bytes = image_to_bytes(image_pil)

    # queue state/create
    stateid_image = await ctx.create_state(file=image_bytes)

    # queue skills/infer-bbox
    stateid_bbox = await ctx.skill_bbox(
        stateid_image=stateid_image,
        product_name=params.prompt,
    )

    # get bbox state/meta
    metadata_bbox = await ctx.get_meta(stateid_bbox)
    bounding_box = metadata_bbox["bbox"]

    return bounding_box


class Box:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "api": (
                    "FG_API",
                    {
                        "tooltip": "The Finegrain API context",
                    },
                ),
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

    def process(
        self,
        api: EditorAPIContext,
        image: torch.Tensor,
        prompt: str,
    ) -> tuple[torch.Tensor]:
        return (
            api.run_one_sync(
                co=_process,
                params=Params(
                    image=image,
                    prompt=prompt,
                ),
            ),
        )
