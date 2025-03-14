from dataclasses import dataclass
from typing import Any

from ..utils.bbox import BoundingBox
from ..utils.context import EditorAPIContext, ErrorResult, StateID, _get_ctx


@dataclass(kw_only=True)
class Params:
    cutout: StateID
    width: int
    height: int
    seed: int
    bgcolor: str
    bbox: BoundingBox | None


async def _process(ctx: EditorAPIContext, params: Params) -> StateID:
    assert 0 <= params.seed <= 999, "Seed must be an integer between 0 and 999"
    assert params.width >= 8, "Width must be at least 8"
    assert params.height >= 8, "Height must be at least 8"

    # call shadow skill
    result_shadow = await ctx.call_async.shadow(
        state_id=params.cutout,
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

    return stateid_shadow


class AdvancedShadow:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "cutout": (
                    "STATEID",
                    {
                        "tooltip": "The cutout stateid to create a shadow packshot from",
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

    RETURN_TYPES = ("STATEID",)
    RETURN_NAMES = ("image",)

    TITLE = "[Low level] Shadow"
    DESCRIPTION = "Create a shadow packshot from a cutout."
    CATEGORY = "Finegrain/low-level"
    FUNCTION = "process"

    def process(
        self,
        cutout: StateID,
        width: int,
        height: int,
        seed: int,
        bgcolor: str,
        bbox: BoundingBox | None = None,
    ) -> tuple[StateID]:
        return (
            _get_ctx().run_one_sync(
                co=_process,
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
