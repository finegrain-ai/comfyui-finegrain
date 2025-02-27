import io
from typing import Any

from finegrain import EditorAPIContext as _EditorAPIContext
from finegrain import Priority
from PIL import Image

StateID = str


class EditorAPIContext(_EditorAPIContext):
    async def create_state(
        self,
        file_url: str | None = None,
        file: io.BytesIO | None = None,
        timeout: float | None = None,
    ) -> str:
        assert file_url or file, "file_url or file is required"

        response = await self.request(
            method="POST",
            url="state/create",
            json={"priority": self.priority},
            data={"file_url": file_url} if file_url else None,
            files={"file": file} if file else None,
        )
        state_id = response.json()["state"]
        status = await self.sse_await(state_id, timeout=timeout)
        if status:
            return state_id
        meta = await self.get_meta(state_id)
        raise RuntimeError(f"create_state failed with {state_id}: {meta}")

    async def download_image(
        self,
        stateid_image: str,
        image_format: str = "PNG",
        image_resolution: str = "FULL",
    ) -> Image.Image:
        response = await self.request(
            method="GET",
            url=f"state/image/{stateid_image}",
            params={
                "format": image_format,
                "resolution": image_resolution,
            },
        )
        return Image.open(io.BytesIO(response.content))

    async def skill_infer_main_subject(
        self,
        stateid_image: StateID,
    ) -> StateID:
        return await self.ensure_skill(
            url=f"infer-main-subject/{stateid_image}",
        )

    async def skill_bbox(
        self,
        stateid_image: StateID,
        product_name: str,
    ) -> StateID:
        return await self.ensure_skill(
            url=f"infer-bbox/{stateid_image}",
            params={"product_name": product_name},
        )

    async def skill_segment(
        self,
        stateid_image: StateID,
        bbox: tuple[int, int, int, int],
    ) -> StateID:
        return await self.ensure_skill(
            url=f"segment/{stateid_image}",
            params={"bbox": bbox},
        )

    async def skill_crop(
        self,
        stateid_image: StateID,
        bbox: tuple[int, int, int, int],
    ) -> StateID:
        return await self.ensure_skill(
            url=f"crop/{stateid_image}",
            params={"bbox": bbox},
        )

    async def skill_erase(
        self,
        stateid_image: StateID,
        stateid_mask: StateID,
        mode: str,
        seed: int,
    ) -> StateID:
        return await self.ensure_skill(
            url=f"erase/{stateid_image}/{stateid_mask}",
            params={
                "mode": mode,
                "seed": seed,
            },
        )

    async def skill_blend(
        self,
        stateid_scene: StateID,
        stateid_cutout: StateID,
        bbox: tuple[int, int, int, int],
        mode: str,
        rotation_angle: float = 0.0,
        flip: bool = False,
        seed: int | None = None,
    ) -> StateID:
        return await self.ensure_skill(
            url=f"blend/{stateid_scene}/{stateid_cutout}",
            params={
                "bbox": bbox,
                "flip": flip,
                "rotation_angle": rotation_angle,
                "mode": mode,
                "seed": seed,
            },
        )

    async def skill_recolor(
        self,
        stateid_image: StateID,
        stateid_mask: StateID,
        color: str,
    ) -> StateID:
        return await self.ensure_skill(
            url=f"recolor/{stateid_image}/{stateid_mask}",
            params={"color": color},
        )

    async def skill_shadow(
        self,
        stateid_cutout: StateID,
        resolution: tuple[int, int],
        background_color: str = "transparent",
        bbox: tuple[int, int, int, int] | None = None,
        seed: int | None = None,
    ) -> StateID:
        return await self.ensure_skill(
            url=f"shadow/{stateid_cutout}",
            params={
                "resolution": resolution,
                "background": background_color,
                "bbox": bbox,
                "seed": seed,
            },
        )

    async def skill_upscale(
        self,
        stateid_image: StateID,
        preprocess: bool,
        scale_factor: int,
        resemblance: float,
        decay: float,
        creativity: float,
        seed: int,
    ) -> StateID:
        return await self.ensure_skill(
            url=f"upscale/{stateid_image}",
            params={
                "preprocess": preprocess,
                "scale_factor": scale_factor,
                "resemblance": resemblance,
                "decay": decay,
                "creativity": creativity,
                "seed": seed,
            },
            timeout=max(300, self.default_timeout),
        )

    async def skill_set_bgcolor(
        self,
        stateid_image: StateID,
        color: str,
    ) -> StateID:
        return await self.ensure_skill(
            url=f"set-background-color/{stateid_image}",
            params={"background": color},
        )


class API:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "username": (
                    "STRING",
                    {
                        "tooltip": "The Finegrain API username",
                    },
                ),
                "password": (
                    "STRING",
                    {
                        "tooltip": "The Finegrain API password",
                    },
                ),
                "priority": (
                    [
                        "standard",
                        "low",
                        "high",
                    ],
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 120,
                        "tooltip": "The default timeout in seconds for each HTTP requests",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FG_API",)
    RETURN_NAMES = ("api",)

    TITLE = "Finegrain API"
    DESCRIPTION = "Connect to the Finegrain API."
    CATEGORY = "Finegrain"
    FUNCTION = "process"

    def process(
        self,
        username: str,
        password: str,
        priority: Priority,
        timeout: int,
    ) -> tuple[EditorAPIContext]:
        return (
            EditorAPIContext(
                user=username,
                password=password,
                priority=priority,
                default_timeout=timeout,
            ),
        )
