from dataclasses import dataclass
from typing import Any, get_args

import torch

from ..utils.bbox import BoundingBox
from ..utils.context import EditorAPIContext, ErrorResult, Mode, _get_ctx
from ..utils.image import (
    image_to_bytes,
    image_to_tensor,
    tensor_to_image,
)


@dataclass(kw_only=True)
class Params:
    scene: torch.Tensor
    cutout: torch.Tensor
    bbox: BoundingBox
    flip: bool
    rotation_angle: float
    mode: Mode
    seed: int


class Blender:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "scene": (
                    "IMAGE",
                    {
                        "tooltip": "The background scene to blend the cutout into.",
                    },
                ),
                "cutout": (
                    "IMAGE",
                    {
                        "tooltip": "The object cutout to blend into the scene.",
                    },
                ),
                "bbox": (
                    "BBOX",
                    {
                        "tooltip": "Bounding box of where to place the cutout in the scene.",
                    },
                ),
                "mode": (
                    [
                        "standard",
                        "express",
                    ],
                ),
                "flip": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Flip the cutout horizontally before blending.",
                    },
                ),
                "rotation_angle": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -360.0,
                        "max": 360.0,
                        "tooltip": "Rotate the cutout by the specified angle before blending.",
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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TITLE = "Blender"
    DESCRIPTION = "Blend an object cutout into a scene."
    CATEGORY = "Finegrain/high-level"
    FUNCTION = "process"

    @staticmethod
    async def _process(
        ctx: EditorAPIContext,
        params: Params,
    ) -> torch.Tensor:
        assert params.mode in get_args(Mode), f"Mode must be one of {get_args(Mode)}"
        assert 0 <= params.seed <= 999, "Seed must be an integer between 0 and 999"
        assert -360 <= params.rotation_angle <= 360, "Rotation angle must be between -360 and 360"

        # convert tensors to PIL images
        scene_pil = tensor_to_image(params.scene.permute(0, 3, 1, 2))
        cutout_pil = tensor_to_image(params.cutout.permute(0, 3, 1, 2))

        # make some assertions
        assert scene_pil.mode == "RGB", "Background must be RGB"
        assert cutout_pil.mode == "RGBA", "Cutout must be RGBA"

        # convert PIL images to BytesIO
        scene_bytes = image_to_bytes(scene_pil)
        cutout_bytes = image_to_bytes(cutout_pil)

        # upload image and cutout
        stateid_scene = await ctx.call_async.upload_image(file=scene_bytes)
        stateid_cutout = await ctx.call_async.upload_image(file=cutout_bytes)

        # call blend skill
        result_blend = await ctx.call_async.blend(
            image_state_id=stateid_scene,
            mask_state_id=stateid_cutout,
            bbox=params.bbox,
            flip=params.flip,
            rotation_angle=params.rotation_angle,
            mode=params.mode,
            seed=params.seed,
        )
        if isinstance(result_blend, ErrorResult):
            raise ValueError(f"Failed to blend: {result_blend.error}")
        stateid_blend = result_blend.state_id

        # download output image
        image_erase = await ctx.call_async.download_image(stateid_blend)

        # convert PIL image to tensor
        tensor_erase = image_to_tensor(image_erase).permute(0, 2, 3, 1)

        return tensor_erase

    def process(
        self,
        scene: torch.Tensor,
        cutout: torch.Tensor,
        bbox: BoundingBox,
        flip: bool,
        rotation_angle: float,
        mode: Mode,
        seed: int,
    ) -> tuple[torch.Tensor]:
        return (
            _get_ctx().run_one_sync(
                co=self._process,
                params=Params(
                    scene=scene,
                    cutout=cutout,
                    flip=flip,
                    rotation_angle=rotation_angle,
                    bbox=bbox,
                    mode=mode,
                    seed=seed,
                ),
            ),
        )
