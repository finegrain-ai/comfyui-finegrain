from dataclasses import dataclass
from typing import Any

from ..utils.context import EditorAPIContext, ErrorResult, StateID, _get_ctx


@dataclass(kw_only=True)
class Params:
    image: StateID
    mask: StateID
    color: str


async def _process(ctx: EditorAPIContext, params: Params) -> StateID:
    # call recolor skill
    result_recolor = await ctx.call_async.recolor(
        image_state_id=params.image,
        mask_state_id=params.mask,
        color=params.color,
    )
    if isinstance(result_recolor, ErrorResult):
        raise ValueError(f"Failed to recolor object: {result_recolor.error}")
    stateid_recolor = result_recolor.state_id

    return stateid_recolor


class AdvancedRecolor:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
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

    TITLE = "[Low level] Recolor"
    DESCRIPTION = "Recolor a masked object in an image."
    CATEGORY = "Finegrain/low-level"
    FUNCTION = "process"

    def process(
        self,
        image: StateID,
        mask: StateID,
        color: str,
    ) -> tuple[StateID]:
        return (
            _get_ctx().run_one_sync(
                co=_process,
                params=Params(
                    image=image,
                    mask=mask,
                    color=color,
                ),
            ),
        )
