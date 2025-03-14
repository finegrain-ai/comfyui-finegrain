from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext, ErrorResult, _get_ctx
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
    CATEGORY = "Finegrain/high-level"
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

        # upload image
        stateid_image = await ctx.call_async.upload_image(file=image_bytes)

        # call infer-main-subject skill
        result_subject = await ctx.call_async.infer_main_subject(state_id=stateid_image)
        if isinstance(result_subject, ErrorResult):
            raise ValueError(f"Failed to infer main subject: {result_subject.error}")

        return result_subject.main_subject

    def process(
        self,
        image: torch.Tensor,
    ) -> tuple[str]:
        return (
            _get_ctx().run_one_sync(
                co=self._process,
                params=Params(
                    image=image,
                ),
            ),
        )
