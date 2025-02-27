from dataclasses import dataclass
from typing import Any

from ..utils.bbox import BoundingBox
from ..utils.context import EditorAPIContext, StateID


@dataclass(kw_only=True)
class Params:
    image: StateID
    bbox: BoundingBox
    cropped: bool


async def _process(
    ctx: EditorAPIContext,
    params: Params,
) -> StateID:
    # queue skills/infer-bbox
    stateid_mask = await ctx.skill_segment(
        stateid_image=params.image,
        bbox=params.bbox,
    )

    # queue skills/crop
    if params.cropped:
        stateid_mask = await ctx.skill_crop(
            stateid_image=stateid_mask,
            bbox=params.bbox,
        )

    return stateid_mask


class AdvancedSegment:
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
                        "tooltip": "The image state tido segment",
                    },
                ),
                "bbox": (
                    "BBOX",
                    {
                        "tooltip": "Bounding box of the object to segment",
                    },
                ),
            },
            "optional": {
                "cropped": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Crop the mask to the bounding box",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STATEID",)
    RETURN_NAMES = ("mask",)

    TITLE = "[Advanced] Segment"
    DESCRIPTION = "Segment an object in an image."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        image: StateID,
        bbox: BoundingBox,
        cropped: bool,
    ) -> tuple[StateID]:
        return (
            api.run_one_sync(
                co=_process,
                params=Params(
                    image=image,
                    bbox=bbox,
                    cropped=cropped,
                ),
            ),
        )
