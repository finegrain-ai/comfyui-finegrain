from dataclasses import dataclass
from typing import Any

from ..utils.bbox import BoundingBox
from ..utils.context import EditorAPIContext, StateID


@dataclass(kw_only=True)
class Params:
    stateid_image: StateID
    prompt: str


async def _process(
    ctx: EditorAPIContext,
    params: Params,
) -> BoundingBox:
    assert params.prompt, "Prompt must not be empty"

    # queue skills/infer-bbox
    stateid_bbox = await ctx.skill_bbox(
        stateid_image=params.stateid_image,
        product_name=params.prompt,
    )

    # get bbox state/meta
    metadata_bbox = await ctx.get_meta(stateid_bbox)
    bounding_box = metadata_bbox["bbox"]

    return bounding_box


class AdvancedBox:
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

    TITLE = "[Advanced] Box"
    DESCRIPTION = "Box an object in an image."
    CATEGORY = "Finegrain/skills"
    FUNCTION = "process"

    def process(
        self,
        api: EditorAPIContext,
        image: StateID,
        prompt: str,
    ) -> tuple[BoundingBox]:
        return (
            api.run_one_sync(
                co=_process,
                params=Params(
                    stateid_image=image,
                    prompt=prompt,
                ),
            ),
        )
