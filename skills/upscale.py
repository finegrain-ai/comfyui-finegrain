from dataclasses import dataclass
from typing import Any

import torch

from ..utils.context import EditorAPIContext
from ..utils.image import (
    image_to_bytes,
    image_to_tensor,
    tensor_to_image,
)


@dataclass(kw_only=True)
class Params:
    image: torch.Tensor
    preprocess: bool
    scale_factor: int
    resemblance: float
    decay: float
    creativity: float
    seed: int


async def _process(ctx: EditorAPIContext, params: Params) -> torch.Tensor:
    assert 0 <= params.seed <= 999, "Seed must be an integer between 0 and 999"
    assert params.scale_factor in [1, 2, 4], "Scale factor must be 1, 2 or 4"
    assert 0 <= params.resemblance <= 1.5, "Resemblance must be a float between 0 and 1.5"
    assert 0.5 <= params.decay <= 1.5, "Decay must be a float between 0.5 and 1.5"
    assert 0 <= params.creativity <= 1, "Creativity must be a float between 0 and 1"

    # convert tensors to PIL images
    image_pil = tensor_to_image(params.image.permute(0, 3, 1, 2))

    # make some assertions
    assert image_pil.mode == "RGB", "Image must be RGB"

    # convert PIL images to BytesIO
    image_bytes = image_to_bytes(image_pil)

    # queue state/create
    stateid_image = await ctx.create_state(file=image_bytes)

    # queue skills/shadow
    stateid_shadow = await ctx.skill_upscale(
        stateid_image=stateid_image,
        preprocess=params.preprocess,
        scale_factor=params.scale_factor,
        resemblance=params.resemblance,
        decay=params.decay,
        creativity=params.creativity,
        seed=params.seed,
    )

    # queue state/download
    upscaled_pil = await ctx.download_image(stateid_shadow)

    # convert PIL image to tensor
    upscaled_tensor = image_to_tensor(upscaled_pil).permute(0, 2, 3, 1)

    return upscaled_tensor


class Upscale:
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
                        "tooltip": "The image to upscale",
                    },
                ),
            },
            "optional": {
                "preprocess": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use two-step upscaling. This removes some defects but is a bit slower and can smooth textures.",
                    },
                ),
                "scale_factor": (["2", "1", "4"],),
                "resemblance": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0,
                        "max": 1.5,
                        "tooltip": "Higher values make the output closer to the output.",
                    },
                ),
                "decay": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 1.5,
                        "tooltip": "Technical parameter that influences resemblance. Leave it set to 1 in general. Lowering it a bit can increase details generation.",
                    },
                ),
                "creativity": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0,
                        "max": 1,
                        "tooltip": "Increasing this will make images less like the input.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 999,
                        "tooltip": "The seed for the random number generator.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TITLE = "Upscale"
    DESCRIPTION = "Upscale an image."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        image: torch.Tensor,
        preprocess: bool,
        scale_factor: str,
        resemblance: float,
        decay: float,
        creativity: float,
        seed: int,
    ) -> tuple[torch.Tensor]:
        return (
            api.run_one_sync(
                co=_process,
                params=Params(
                    image=image,
                    preprocess=preprocess,
                    scale_factor=int(scale_factor),
                    resemblance=resemblance,
                    decay=decay,
                    creativity=creativity,
                    seed=seed,
                ),
            ),
        )
