from typing import Any

from .advanced.blender import AdvancedBlender
from .advanced.box import AdvancedBox
from .advanced.download_image import DownloadImage
from .advanced.download_mask import DownloadMask
from .advanced.eraser import AdvancedEraser
from .advanced.recolor import AdvancedRecolor
from .advanced.segment import AdvancedSegment
from .advanced.shadow import AdvancedShadow
from .advanced.upload_image import UploadImage
from .advanced.upload_mask import UploadMask
from .advanced.upscale import AdvancedUpscale
from .skills.blender import Blender
from .skills.box import Box
from .skills.eraser import Eraser
from .skills.name import InferMainSubject
from .skills.recolor import Recolor
from .skills.segment import Segment
from .skills.shadow import Shadow
from .skills.upscale import Upscale
from .utils.bbox import CreateBoundingBox, DrawBoundingBox, ImageCropBoundingBox, MaskCropBoundingBox
from .utils.context import API
from .utils.image import ApplyTransparencyMask

NODE_CLASS_MAPPINGS: dict[str, Any] = {
    c.TITLE: c
    for c in [
        # low level nodes
        AdvancedBlender,
        AdvancedBox,
        AdvancedEraser,
        AdvancedRecolor,
        AdvancedSegment,
        AdvancedShadow,
        AdvancedUpscale,
        DownloadImage,
        DownloadMask,
        UploadImage,
        UploadMask,
        # high level nodes
        Blender,
        Box,
        Eraser,
        InferMainSubject,
        Recolor,
        Segment,
        Shadow,
        Upscale,
        # utils nodes
        CreateBoundingBox,
        DrawBoundingBox,
        ImageCropBoundingBox,
        MaskCropBoundingBox,
        ApplyTransparencyMask,
        API,
    ]
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k, _ in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
