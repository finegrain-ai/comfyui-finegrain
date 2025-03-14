from dataclasses import dataclass
from typing import Any

from ..utils.bbox import BoundingBox
from ..utils.context import EditorAPIContext, ErrorResult, StateID, _get_ctx


@dataclass(kw_only=True)
class Params:
    image: StateID
    bbox: BoundingBox
    cropped: bool


class Segment:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
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

    TITLE = "[Low level] Segment"
    DESCRIPTION = "Segment an object in an image."
    CATEGORY = "Finegrain/low-level"
    FUNCTION = "process"

    @staticmethod
    async def _process(
        ctx: EditorAPIContext,
        params: Params,
    ) -> StateID:
        # call segment skill
        result_segment = await ctx.call_async.segment(
            state_id=params.image,
            bbox=params.bbox,
        )
        if isinstance(result_segment, ErrorResult):
            raise ValueError(f"Failed to segment object: {result_segment.error}")
        stateid_segment = result_segment.state_id

        if params.cropped:
            # call crop skill
            result_segment = await ctx.call_async.crop(
                state_id=stateid_segment,
                bbox=params.bbox,
            )
            if isinstance(result_segment, ErrorResult):
                raise ValueError(f"Failed to crop object: {result_segment.error}")
            stateid_segment = result_segment.state_id

        return stateid_segment

    def process(
        self,
        image: StateID,
        bbox: BoundingBox,
        cropped: bool,
    ) -> tuple[StateID]:
        return (
            _get_ctx().run_one_sync(
                co=self._process,
                params=Params(
                    image=image,
                    bbox=bbox,
                    cropped=cropped,
                ),
            ),
        )
