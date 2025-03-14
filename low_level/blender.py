from dataclasses import dataclass
from typing import Any, get_args

from ..utils.bbox import BoundingBox
from ..utils.context import EditorAPIContext, ErrorResult, Mode, StateID


@dataclass(kw_only=True)
class Params:
    scene: StateID
    cutout: StateID
    bbox: BoundingBox
    flip: bool
    rotation_angle: float
    mode: Mode
    seed: int


async def _process(ctx: EditorAPIContext, params: Params) -> StateID:
    assert params.mode in get_args(Mode), f"Mode must be one of {get_args(Mode)}"
    assert 0 <= params.seed <= 999, "Seed must be an integer between 0 and 999"
    assert -360 <= params.rotation_angle <= 360, "Rotation angle must be between -360 and 360"

    # call blend skill
    result_erase = await ctx.call_async.blend(
        image_state_id=params.scene,
        mask_state_id=params.cutout,
        bbox=params.bbox,
        flip=params.flip,
        rotation_angle=params.rotation_angle,
        mode=params.mode,
        seed=params.seed,
    )
    if isinstance(result_erase, ErrorResult):
        raise ValueError(f"Failed to blend object: {result_erase.error}")
    stateid_erase = result_erase.state_id

    return stateid_erase


class AdvancedBlender:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "scene": (
                    "STATEID",
                    {
                        "tooltip": "The background scene stateid to blend the cutout into.",
                    },
                ),
                "cutout": (
                    "STATEID",
                    {
                        "tooltip": "The object cutout stateid to blend into the scene.",
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

    RETURN_TYPES = ("STATEID",)
    RETURN_NAMES = ("image",)

    TITLE = "[Low level] Blender"
    DESCRIPTION = "Blend an object cutout into a scene."
    CATEGORY = "Finegrain/low-level"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        scene: StateID,
        cutout: StateID,
        bbox: BoundingBox,
        flip: bool,
        rotation_angle: float,
        mode: Mode,
        seed: int,
    ) -> tuple[StateID]:
        return (
            api.run_one_sync(
                co=_process,
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
