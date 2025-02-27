from dataclasses import dataclass
from typing import Any

from ..utils.context import EditorAPIContext, StateID


@dataclass(kw_only=True)
class Params:
    image: StateID
    mask: StateID
    mode: str
    seed: int


async def _process(
    ctx: EditorAPIContext,
    params: Params,
) -> StateID:
    assert params.mode in ["express", "standard", "premium"], "Invalid mode"
    assert params.seed >= 0, "Seed must be a non-negative integer"

    # queue skills/erase
    stateid_erased = await ctx.skill_erase(
        stateid_image=params.image,
        stateid_mask=params.mask,
        mode=params.mode,
        seed=params.seed,
    )

    return stateid_erased


class AdvancedEraser:
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
                    "STATEID",
                    {
                        "tooltip": "The image statid to erase an object from",
                    },
                ),
                "mask": (
                    "STATEID",
                    {
                        "tooltip": "The mask stateid of the object to erase",
                    },
                ),
            },
            "optional": {
                "mode": (
                    [
                        "express",
                        "standard",
                        "premium",
                    ],
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 999,
                        "tooltip": "Seed for the random number generator",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STATEID",)
    RETURN_NAMES = ("image",)

    TITLE = "[Advanced] Eraser"
    DESCRIPTION = "Erase an object from an image using a mask."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        image: StateID,
        mask: StateID,
        mode: str,
        seed: int,
    ) -> tuple[StateID]:
        return (
            api.run_one_sync(
                co=_process,
                params=Params(
                    image=image,
                    mask=mask,
                    mode=mode,
                    seed=seed,
                ),
            ),
        )
