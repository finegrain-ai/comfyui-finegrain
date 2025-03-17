from typing import Any

from .high_level.blender import Blender as HighLevelBlender
from .high_level.box import Box as HighLevelBox
from .high_level.eraser import Eraser as HighLevelEraser
from .high_level.name import InferMainSubject as HighLevelInferMainSubject
from .high_level.recolor import Recolor as HighLevelRecolor
from .high_level.segment import Segment as HighLevelSegment
from .high_level.shadow import Shadow as HighLevelShadow
from .low_level.blender import Blender as LowLevelBlender
from .low_level.box import Box as LowLevelBox
from .low_level.download_image import DownloadImage as LowLevelDownloadImage
from .low_level.download_mask import DownloadMask as LowLevelDownloadMask
from .low_level.eraser import Eraser as LowLevelEraser
from .low_level.recolor import Recolor as LowLevelRecolor
from .low_level.segment import Segment as LowLevelSegment
from .low_level.shadow import Shadow as LowLevelShadow
from .low_level.upload_image import UploadImage as LowLevelUploadImage
from .low_level.upload_mask import UploadMask as LowLevelUploadMask
from .utils.bbox import CreateBoundingBox, DrawBoundingBox, ImageCropBoundingBox, MaskCropBoundingBox
from .utils.image import ApplyTransparencyMask

NODE_CLASS_MAPPINGS: dict[str, Any] = {
    c.TITLE: c
    for c in [
        # low level nodes
        LowLevelBlender,
        LowLevelBox,
        LowLevelDownloadImage,
        LowLevelDownloadMask,
        LowLevelEraser,
        LowLevelRecolor,
        LowLevelSegment,
        LowLevelShadow,
        LowLevelUploadImage,
        LowLevelUploadMask,
        # high level nodes
        HighLevelBlender,
        HighLevelBox,
        HighLevelEraser,
        HighLevelInferMainSubject,
        HighLevelRecolor,
        HighLevelSegment,
        HighLevelShadow,
        # utils nodes
        CreateBoundingBox,
        DrawBoundingBox,
        ImageCropBoundingBox,
        MaskCropBoundingBox,
        ApplyTransparencyMask,
    ]
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k, _ in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
