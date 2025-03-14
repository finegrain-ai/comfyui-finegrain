from dataclasses import dataclass
from typing import Any

from ..utils.bbox import BoundingBox
from ..utils.context import EditorAPIContext, ErrorResult, StateID, _get_ctx


@dataclass(kw_only=True)
class Params:
    stateid_image: StateID
    prompt: str


async def _process(
    ctx: EditorAPIContext,
    params: Params,
) -> BoundingBox:
    assert params.prompt, "Prompt must not be empty"

    # call infer-bbox skill
    result_bbox = await ctx.call_async.infer_bbox(
        state_id=params.stateid_image,
        product_name=params.prompt,
    )
    if isinstance(result_bbox, ErrorResult):
        raise ValueError(f"Failed to infer bounding box: {result_bbox.error}")
    bbox = result_bbox.bbox

    return bbox


class AdvancedBox:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": (
                    "STATEID",
                    {
                        "tooltip": "The image stateid to detect an object in",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "tooltip": "The product name to detect",
                    },
                ),
            },
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)

    TITLE = "[Low level] Box"
    DESCRIPTION = "Box an object in an image."
    CATEGORY = "Finegrain/low-level"
    FUNCTION = "process"

    def process(
        self,
        image: StateID,
        prompt: str,
    ) -> tuple[BoundingBox]:
        return (
            _get_ctx().run_one_sync(
                co=_process,
                params=Params(
                    stateid_image=image,
                    prompt=prompt,
                ),
            ),
        )
