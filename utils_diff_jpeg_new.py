import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diff_jpeg import diff_jpeg_coding

def differentiable_jpeg(image, quality=10):
    """
    Wrapper for the 'diff-jpeg' library (necla-ml).
    
    This function calls 'diff_jpeg_coding' and handles:
    1. Range conversion from [-1, 1] (model output) to [0, 1] (library input).
    2. Automatic padding to a multiple of 8 (required by the library).
    3. Creation of the quality tensor to match the batch size.
    
    :param image: Input image tensor, range [-1, 1].
    :param quality: JPEG quality factor (a Python number, e.g., 10 or 75).
    :return: Simulated JPEG compressed image, range [-1, 1].
    """
    
    # 1. Convert range from [-1, 1] to [0, 1] (library requirement)
    image_0_1 = (image + 1) / 2.0
    
    # 2. Get batch size and create a quality tensor
    #    that matches the batch size and device.
    B, C, H, W = image.shape
    quality_tensor = torch.full((B,), float(quality), device=image.device)

    # 3. Pad image dimensions to be a multiple of 8 (library requirement)
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    
    if pad_h != 0 or pad_w != 0:
        # Pad with 'replicate' mode
        image_0_1 = F.pad(image_0_1, (0, pad_w, 0, pad_h), mode='replicate')

    # 4. Call the diff_jpeg library (operates in [0, 1] range)
    # This uses ste=False by default (soft simulation)
    jpeg_image_0_1 = diff_jpeg_coding(image_0_1, quality_tensor)
    
    # 5. Crop back to original size if padding was added
    if pad_h != 0 or pad_w != 0:
        jpeg_image_0_1 = jpeg_image_0_1[:, :, :H, :W]

    # 6. Convert range back to [-1, 1]
    jpeg_image = jpeg_image_0_1 * 2.0 - 1.0
    
    return jpeg_image
