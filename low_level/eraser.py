from dataclasses import dataclass
from typing import Any, get_args

from ..utils.context import EditorAPIContext, ErrorResult, Mode, StateID, _get_ctx


@dataclass(kw_only=True)
class Params:
    image: StateID
    mask: StateID
    mode: Mode
    seed: int


class Eraser:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": (
                    "STATEID",
                    {
                        "tooltip": "The image stateid to erase an object from",
                    },
                ),
                "mask": (
                    "STATEID",
                    {
                        "tooltip": "The mask stateid of the object to erase",
                    },
                ),
                "mode": (
                    [
                        "premium",
                        "standard",
                        "express",
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

    TITLE = "[LOW-LEVEL] Eraser"
    DESCRIPTION = "Erase an object from an image using a mask."
    CATEGORY = "Finegrain/low-level"
    FUNCTION = "process"

    @staticmethod
    async def _process(
        ctx: EditorAPIContext,
        params: Params,
    ) -> StateID:
        assert params.mode in get_args(Mode), f"Mode must be one of {get_args(Mode)}"
        assert params.seed >= 0, "Seed must be a non-negative integer"

        # call erase skill
        result_erase = await ctx.call_async.erase(
            image_state_id=params.image,
            mask_state_id=params.mask,
            mode=params.mode,
            seed=params.seed,
        )
        if isinstance(result_erase, ErrorResult):
            raise ValueError(f"Failed to erase object: {result_erase.error}")
        stateid_erase = result_erase.state_id

        return stateid_erase

    def process(
        self,
        image: StateID,
        mask: StateID,
        mode: Mode,
        seed: int,
    ) -> tuple[StateID]:
        return (
            _get_ctx().run_one_sync(
                co=self._process,
                params=Params(
                    image=image,
                    mask=mask,
                    mode=mode,
                    seed=seed,
                ),
            ),
        )
