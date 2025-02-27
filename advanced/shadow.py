from dataclasses import dataclass
from typing import Any

from ..utils.bbox import BoundingBox
from ..utils.context import EditorAPIContext, StateID


@dataclass(kw_only=True)
class Params:
    cutout: StateID
    width: int
    height: int
    seed: int
    bgcolor: str | None
    bbox: BoundingBox | None


async def _process(ctx: EditorAPIContext, params: Params) -> StateID:
    assert 0 <= params.seed <= 999, "Seed must be an integer between 0 and 999"
    assert params.width >= 8, "Width must be at least 8"
    assert params.height >= 8, "Height must be at least 8"

    # queue skills/shadow
    stateid_shadow = await ctx.skill_shadow(
        stateid_cutout=params.cutout,
        resolution=(params.width, params.height),
        bbox=params.bbox,
        seed=params.seed,
    )

    # queue skills/set-background-color
    if params.bgcolor and params.bgcolor != "transparent":
        stateid_shadow = await ctx.skill_set_bgcolor(
            stateid_image=stateid_shadow,
            color=params.bgcolor,
        )

    return stateid_shadow


class AdvancedShadow:
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
            },
            "optional": {
                "bgcolor": (
                    "STRING",
                    {
                        "default": "transparent",
                        "tooltip": "Background color of the shadow.",
                    },
                ),
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

    TITLE = "[Advanced] Shadow"
    DESCRIPTION = "Create a shadow packshot from a cutout."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        cutout: StateID,
        width: int,
        height: int,
        seed: int,
        bgcolor: str | None = None,
        bbox: BoundingBox | None = None,
    ) -> tuple[StateID]:
        return (
            api.run_one_sync(
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
