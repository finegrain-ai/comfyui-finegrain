from typing import Any

from .high_level.blender import Blender
from .high_level.box import Box
from .high_level.eraser import Eraser
from .high_level.name import InferMainSubject
from .high_level.recolor import Recolor
from .high_level.segment import Segment
from .high_level.shadow import Shadow
from .low_level.blender import AdvancedBlender
from .low_level.box import AdvancedBox
from .low_level.download_image import DownloadImage
from .low_level.download_mask import DownloadMask
from .low_level.eraser import AdvancedEraser
from .low_level.recolor import AdvancedRecolor
from .low_level.segment import AdvancedSegment
from .low_level.shadow import AdvancedShadow
from .low_level.upload_image import UploadImage
from .low_level.upload_mask import UploadMask
from .utils.bbox import CreateBoundingBox, DrawBoundingBox, ImageCropBoundingBox, MaskCropBoundingBox
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
