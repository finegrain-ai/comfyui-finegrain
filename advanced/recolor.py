from dataclasses import dataclass
from typing import Any

from ..utils.context import EditorAPIContext, StateID


@dataclass(kw_only=True)
class Params:
    image: StateID
    mask: StateID
    color: str


async def _process(ctx: EditorAPIContext, params: Params) -> StateID:
    # queue skills/shadow
    stateid_recolor = await ctx.skill_recolor(
        stateid_image=params.image,
        stateid_mask=params.mask,
        color=params.color,
    )

    return stateid_recolor


class AdvancedRecolor:
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
                        "tooltip": "The image stateid to recolor something in",
                    },
                ),
                "mask": (
                    "STATEID",
                    {
                        "tooltip": "The mask stateid of the object to recolor",
                    },
                ),
                "color": (
                    "STRING",
                    {
                        "default": "#ff0000",
                        "tooltip": "The color to recolor the object to",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STATEID",)
    RETURN_NAMES = ("image",)

    TITLE = "[Advanced] Recolor"
    DESCRIPTION = "Recolor a masked object in an image."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        image: StateID,
        mask: StateID,
        color: str,
    ) -> tuple[StateID]:
        return (
            api.run_one_sync(
                co=_process,
                params=Params(
                    image=image,
                    mask=mask,
                    color=color,
                ),
            ),
        )
