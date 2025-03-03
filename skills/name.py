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


class InferMainSubject:
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
                        "tooltip": "The image to guess the main subject of.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("subject",)

    TITLE = "Infer Main Subject"
    DESCRIPTION = "Infer the main subject in an image."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    @staticmethod
    async def _process(
        ctx: EditorAPIContext,
        params: Params,
    ) -> str:
        # convert tensors to PIL images
        image_pil = tensor_to_image(params.image.permute(0, 3, 1, 2))

        # make some assertions
        assert image_pil.mode == "RGB", "Image must be RGB"

        # convert PIL images to BytesIO
        image_bytes = image_to_bytes(image_pil)

        # queue state/create
        stateid_image = await ctx.create_state(file=image_bytes)

        # queue skills/infer-main-subject
        stateid_name = await ctx.skill_infer_main_subject(
            stateid_image=stateid_image,
        )

        # get name state/meta
        metadata_name = await ctx.get_meta(stateid_name)
        name = metadata_name["main_subject"]

        return name

    def process(
        self,
        api: EditorAPIContext,
        image: torch.Tensor,
    ) -> tuple[str]:
        return (
            api.run_one_sync(
                co=self._process,
                params=Params(
                    image=image,
                ),
            ),
        )
